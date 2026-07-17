// zord FRONT/MIDDLE kernel: build the active-cell table + spatial/temporal cell-pair lists
// from a timestamped edge stream, at 100M-1B edge scale (C++17, the HOT structural pass).
// =====================================================================================
// Build:  g++ -O3 -std=c++17 -fopenmp src/zord/cpp/supra_build.cpp -o build/supra_build
//
// WHAT IT DOES
// ------------
// A temporal graph is a SUPRA-GRAPH over CELLS (v, t) = "vertex v at snapshot t". A cell is
// ACTIVE iff v has an incident edge in snapshot t. This kernel materialises, at scale, the
// exact arrays that scripts/supra_solve.py::build_supra_cells produces in numpy -- but the
// numpy concat+unique allocates ~5x M int64 and OOMs/slows at billion-edge, so the hot
// O(E log E) sort/unique + searchsorted moves here:
//   - cell table   : unique (v*S + t) over BOTH endpoints of every edge, sorted -> cell-id
//                     (vertex-major, snapshot-minor == supra_solver.cpp's canonical order, so
//                     a later cell_device[] from supra_solver lines up index-for-index).
//   - spatial pairs: per input edge (s,d,t) -> (cell(s,t), cell(d,t)), DROPPING a==b
//                     (self-loops / same-cell). Parallel edges are NOT deduplicated --
//                     identical to the numpy reference (it keeps duplicate spatial pairs).
//   - temporal pairs: consecutive cells of the SAME vertex in cell-id order (cells are
//                     vertex-major, time-minor, so each vertex's active snapshots are a
//                     contiguous ascending run) -> (c-1, c) when cell_v[c]==cell_v[c-1].
//
// BINARY INPUT FORMAT (little-endian) -- IDENTICAL prefix to supra_solver.cpp (shared writer)
// -------------------------------------------------------------------------------------------
//   int64   N            number of vertices
//   int64   S            number of snapshots
//   int64   M            number of timestamped edges
//   int32   triples[3*M] M triples (src, dst, snapshot), snapshot in [0,S)
//
// BINARY OUTPUT FORMAT (little-endian)
// ------------------------------------
//   int64   C            number of ACTIVE cells (rows of the cell table)
//   int32   cell_v[C]    vertex coordinate per cell  (cell-id order)
//   int32   cell_t[C]    snapshot coordinate per cell (cell-id order)
//   int64   nSpat        number of spatial cell-pairs
//   int32   sp[2*nSpat]  spatial pairs (cell-id a,b interleaved; a!=b)
//   int64   nTemp        number of temporal cell-pairs
//   int32   tp[2*nTemp]  temporal pairs (cell-id a,b interleaved; consecutive same-vertex)
//
// COMPLEXITY: build O(M log M) (the unique sort of 2M keys), mem O(C + E_S + E_T).
// Mirrors supra_solve.py::build_supra_cells; reuses supra_solver.cpp's cell-discovery verbatim.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
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

    // ---- read header + edges (same prefix as supra_solver.cpp) -------------------------
    FILE* f = fopen(inpath, "rb");
    if (!f) { fprintf(stderr, "[cpp] open fail %s\n", inpath); return 1; }
    i64 N = 0, S = 0, M = 0;
    if (fread(&N, 8, 1, f) != 1 || fread(&S, 8, 1, f) != 1 || fread(&M, 8, 1, f) != 1) {
        fprintf(stderr, "[cpp] header read fail\n"); fclose(f); return 1;
    }
    if (S < 1) S = 1;
    vector<i32> trip((size_t)3 * (M > 0 ? M : 0));
    if (M > 0 && fread(trip.data(), 4, (size_t)3 * M, f) != (size_t)3 * M) {
        fprintf(stderr, "[cpp] triple read fail\n"); fclose(f); return 1;
    }
    fclose(f);
    fprintf(stderr, "[cpp] loaded N=%lld S=%lld M=%lld in %.2fs\n",
            (long long)N, (long long)S, (long long)M, now() - t0);

    // ---- clamp helper for snapshot (matches supra_solver.cpp) --------------------------
    auto clamp_t = [&](i32 t) -> i32 {
        if (t < 0) return 0;
        if (t >= (i32)S) return (i32)(S - 1);
        return t;
    };

    // ---- discover ACTIVE cells: unique(v*S + t) over both endpoints --------------------
    // keys[] sorted ascending == cell ids 0..C-1, canonical (vertex-major, snapshot-minor).
    vector<u64> keys;
    keys.reserve((size_t)2 * (M > 0 ? M : 0));
    for (i64 k = 0; k < M; k++) {
        i32 s = trip[3*k], d = trip[3*k+1], t = clamp_t(trip[3*k+2]);
        keys.push_back((u64)s * (u64)S + (u64)t);
        keys.push_back((u64)d * (u64)S + (u64)t);
    }
    sort(keys.begin(), keys.end());
    keys.erase(unique(keys.begin(), keys.end()), keys.end());
    i64 C = (i64)keys.size();
    fprintf(stderr, "[cpp] active cells C=%lld (%.2fs)\n", (long long)C, now() - t0);

    // per-cell coordinates
    vector<i32> cell_v(C), cell_t(C);
    for (i64 c = 0; c < C; c++) {
        cell_v[c] = (i32)(keys[c] / (u64)S);
        cell_t[c] = (i32)(keys[c] % (u64)S);
    }

    auto cell_of = [&](u64 key) -> i64 {
        return (i64)(lower_bound(keys.begin(), keys.end(), key) - keys.begin());
    };

    // ---- spatial pairs (per edge, in EDGE order, dropping a==b; no dedup of parallels) --
    // Matches numpy: a = searchsorted(keys, src*S+t); b = searchsorted(keys, dst*S+t);
    //                m = a != b; sp_a, sp_b = a[m], b[m].  (edge order preserved)
    vector<i32> sp;  sp.reserve((size_t)2 * (M > 0 ? M : 0));
    for (i64 k = 0; k < M; k++) {
        i32 s = trip[3*k], d = trip[3*k+1], t = clamp_t(trip[3*k+2]);
        i64 a = cell_of((u64)s * (u64)S + (u64)t);
        i64 b = cell_of((u64)d * (u64)S + (u64)t);
        if (a != b) { sp.push_back((i32)a); sp.push_back((i32)b); }
    }
    vector<i32>().swap(trip);
    i64 nSpat = (i64)sp.size() / 2;

    // ---- temporal pairs: consecutive same-vertex cells in cell-id order ----------------
    // Matches numpy: same_v = cell_v[1:]==cell_v[:-1]; idx=nonzero(same_v); tp=(idx, idx+1).
    vector<i32> tp;  tp.reserve((size_t)2 * (C > 0 ? C : 0));
    for (i64 c = 1; c < C; c++) {
        if (cell_v[c] == cell_v[c-1]) { tp.push_back((i32)(c - 1)); tp.push_back((i32)c); }
    }
    i64 nTemp = (i64)tp.size() / 2;
    fprintf(stderr, "[cpp] spatial_pairs=%lld temporal_pairs=%lld (%.2fs)\n",
            (long long)nSpat, (long long)nTemp, now() - t0);

    // ---- write output ------------------------------------------------------------------
    FILE* o = fopen(outpath, "wb");
    if (!o) { fprintf(stderr, "[cpp] open out fail %s\n", outpath); return 1; }
    fwrite(&C, 8, 1, o);
    if (C > 0) {
        fwrite(cell_v.data(), 4, (size_t)C, o);
        fwrite(cell_t.data(), 4, (size_t)C, o);
    }
    fwrite(&nSpat, 8, 1, o);
    if (nSpat > 0) fwrite(sp.data(), 4, (size_t)(2 * nSpat), o);
    fwrite(&nTemp, 8, 1, o);
    if (nTemp > 0) fwrite(tp.data(), 4, (size_t)(2 * nTemp), o);
    fclose(o);
    fprintf(stderr, "[cpp] wrote C=%lld nSpat=%lld nTemp=%lld in %.2fs total\n",
            (long long)C, (long long)nSpat, (long long)nTemp, now() - t0);
    return 0;
}
