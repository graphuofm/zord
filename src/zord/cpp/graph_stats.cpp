// zord graph-stats kernel in C++ (cheap O(E) / O(E log E) structural counts; never networkx).
// =====================================================================================
// Build:  g++ -O3 -std=c++17 -fopenmp src/zord/cpp/graph_stats.cpp -o build/graph_stats
//
// WHAT IT DOES
// ------------
// The EXACT-COUNTS half of frontend.ingest.GraphStats over the FULL timestamped edge
// stream of a temporal graph (the structural passes that touch every edge at 100M-1B
// scale, which numpy cannot do without allocating ~5x M int64 and hanging). It computes
// the three exact-count arrays the Python prober/ingest derive everything else from:
//   1. deg[v]                 -- UNDIRECTED degree of vertex v (each edge bumps src AND dst,
//                                exactly like cpp_kernel.node_degree = bincount(src)+bincount(dst)).
//                                Python derives avg_degree (E/N), max_degree, deg_p99 from this.
//   2. per_snapshot_nodes[s]  -- number of DISTINCT active vertices in snapshot s (unique
//                                (snapshot, vertex) cells bucketed by snapshot). Python derives
//                                mean_snapshot_nodes (over NON-EMPTY snapshots) and max_snapshot_nodes.
//                                Mirrors ingest.graph_stats: cell = snap*N + v; unique; bucket by snap.
//   3. Tv[v]                  -- |T_v|, the number of DISTINCT snapshots vertex v appears in
//                                (over both endpoints). Python derives the persistence
//                                rho = mean over active vertices of (Tv[v]-1)/(S-1) (THEORY 9.4).
//                                Mirrors ingest._persistence: key = v*S + s; unique; bincount by v.
//
// BINARY INPUT FORMAT (little-endian; SAME prefix as supra_solver.cpp / supra_build.cpp)
// -----------------------------------------------------------------------------------
//   int64   N            number of vertices
//   int64   S            number of snapshots
//   int64   M            number of timestamped edges
//   int32   triples[3*M] M triples (src, dst, snapshot), snapshot clamped to [0,S)
//
// BINARY OUTPUT FORMAT (little-endian)
// ------------------------------------
//   int64   N                       (== input N)
//   int32   deg[N]                  undirected degree per vertex
//   int64   S                       (== input S)
//   int32   per_snapshot_nodes[S]   distinct active-node count per snapshot
//   int64   N2                      (== input N; lets the reader sanity-check)
//   int32   Tv[N]                   |T_v| = distinct snapshots per vertex
// To stderr: load/compute timings + a one-line summary.
//
// COMPLEXITY: deg O(E); per-snapshot + |T_v| via two unique passes O(E log E). Memory O(N+E).
// Both unique passes use a sort over packed int64 cell keys -> they reproduce numpy's
// np.unique result bit-for-bit (same set of distinct cells), so the C++ and numpy paths
// produce IDENTICAL per_snapshot_nodes / Tv arrays.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
using i64 = int64_t;
using i32 = int32_t;
using u64 = uint64_t;

static double now() {
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    const char* inpath = argv[1];
    const char* outpath = argv[2];
    double t0 = now();

    // ---- read header + edges (SAME prefix as supra_solver.cpp) ----------------------------
    FILE* f = fopen(inpath, "rb");
    if (!f) { fprintf(stderr, "[cpp] open fail %s\n", inpath); return 1; }
    i64 N = 0, S = 0, M = 0;
    if (fread(&N, 8, 1, f) != 1 || fread(&S, 8, 1, f) != 1 || fread(&M, 8, 1, f) != 1) {
        fprintf(stderr, "[cpp] header read fail\n"); fclose(f); return 1;
    }
    if (N < 0) N = 0;
    if (S < 1) S = 1;
    vector<i32> trip((size_t)3 * (M > 0 ? M : 0));
    if (M > 0 && fread(trip.data(), 4, (size_t)3 * M, f) != (size_t)3 * M) {
        fprintf(stderr, "[cpp] triple read fail\n"); fclose(f); return 1;
    }
    fclose(f);
    fprintf(stderr, "[cpp] graph_stats loaded N=%lld S=%lld M=%lld in %.2fs\n",
            (long long)N, (long long)S, (long long)M, now() - t0);

    double tc = now();

    // ---- (1) undirected degree: each edge bumps both endpoints --------------------------
    // == cpp_kernel.node_degree = bincount(src,minlength=N) + bincount(dst,minlength=N).
    vector<i32> deg((size_t)N, 0);
    for (i64 k = 0; k < M; k++) {
        i32 s = trip[3*k], d = trip[3*k+1];
        if (s >= 0 && s < N) deg[s]++;
        if (d >= 0 && d < N) deg[d]++;
    }

    // ---- build the (vertex, snapshot) cell key list over BOTH endpoints ------------------
    // key = v * S + t  (vertex-major, snapshot-minor). This single packed-key list serves
    // BOTH the per-snapshot distinct count and the |T_v| distinct count after a sort+unique,
    // exactly reproducing the two np.unique passes in ingest.py (which use snap*N+v and v*S+t
    // respectively -- both are just unique over the SAME set of distinct (v,t) cells).
    vector<u64> keys;
    keys.reserve((size_t)2 * (M > 0 ? M : 0));
    for (i64 k = 0; k < M; k++) {
        i32 s = trip[3*k], d = trip[3*k+1], t = trip[3*k+2];
        if (t < 0) t = 0;
        if (t >= S) t = (i32)(S - 1);
        if (s >= 0 && s < N) keys.push_back((u64)s * (u64)S + (u64)t);
        if (d >= 0 && d < N) keys.push_back((u64)d * (u64)S + (u64)t);
    }
    vector<i32>().swap(trip);                 // free the edge buffer before the sort
    sort(keys.begin(), keys.end());
    keys.erase(unique(keys.begin(), keys.end()), keys.end());   // distinct (v,t) cells
    i64 C = (i64)keys.size();

    // ---- (2) per-snapshot distinct active-node count -------------------------------------
    // For each distinct cell key, t = key % S -> bump per_snapshot_nodes[t].
    vector<i32> per_snap((size_t)S, 0);
    // ---- (3) |T_v| = distinct snapshots per vertex ---------------------------------------
    // For each distinct cell key, v = key / S -> bump Tv[v]. Because keys are DISTINCT
    // (v,t) cells, each cell contributes exactly one distinct snapshot to its vertex.
    vector<i32> Tv((size_t)N, 0);
    for (i64 c = 0; c < C; c++) {
        u64 key = keys[c];
        i64 t = (i64)(key % (u64)S);
        i64 v = (i64)(key / (u64)S);
        if (t >= 0 && t < S) per_snap[t]++;
        if (v >= 0 && v < N) Tv[v]++;
    }
    vector<u64>().swap(keys);

    fprintf(stderr, "[cpp] graph_stats computed (cells=%lld) in %.2fs (load+%.2fs total)\n",
            (long long)C, now() - tc, now() - t0);

    // ---- write output (little-endian) ----------------------------------------------------
    FILE* o = fopen(outpath, "wb");
    if (!o) { fprintf(stderr, "[cpp] open out fail %s\n", outpath); return 1; }
    fwrite(&N, 8, 1, o);
    if (N > 0) fwrite(deg.data(), 4, (size_t)N, o);
    fwrite(&S, 8, 1, o);
    if (S > 0) fwrite(per_snap.data(), 4, (size_t)S, o);
    fwrite(&N, 8, 1, o);                      // N2 (== N), lets the reader sanity-check
    if (N > 0) fwrite(Tv.data(), 4, (size_t)N, o);
    fclose(o);
    return 0;
}
