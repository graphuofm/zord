// zord ONLINE kernel: the event-dependency DAG / CHANGED-CONE closure (C++17; BACKLOG P1).
// =====================================================================================
// Build:  g++ -O3 -std=c++17 -fopenmp src/zord/cpp/changed_cone.cpp -o build/changed_cone
//
// WHAT IT DOES  (the C++ HOT PATH behind schedule/dynamic_online.build_event_dependency)
// --------------------------------------------------------------------------------------
// Given a TIME-SORTED edge stream and a cut point `new_edge_lo`, the events at offset
// >= new_edge_lo are the NEW events of the current window. For each new event e we build
// its NeutronStream-style temporal "ear": the LOCAL index (within the new-event block) of
// the most-recent EARLIER new event that shares an endpoint (src or dst) with e -- the
// single event whose state e most-directly depends on. -1 marks a dependency root. The
// `depth[i] = depth[ear[i]] + 1` is the longest dependency chain ending at e (its
// topological depth in the DAG). The CHANGED CONE is the set of unique vertices touched by
// the new events -- the O(|cone(delta)|) work the incremental re-arrangement must re-place.
//
// This reproduces dynamic_online.build_event_dependency EXACTLY:
//   * `ear` is a LOCAL index (0..n_new-1) into the new-event block, not a global edge id.
//   * the immediate predecessor is the MORE-RECENT (larger local index) of the two
//     endpoints' last-seen new events: p = max(last_seen[u], last_seen[v]).
//   * `cone` = sorted-unique(src[lo:] concat dst[lo:]) (np.unique returns sorted).
// The Python loop is O(n_new) but Python-slow for large windows (a dict walk per event);
// here it is a flat C++ forward pass with an open-addressing hash map for last_seen[v]
// over only the vertices the window actually touches -- so memory is O(distinct touched
// vertices), NOT O(N), and it scales to large windows of a billion-edge stream.
//
// BINARY INPUT FORMAT  (little-endian, exactly this order; documented for dynamic_online.py)
// ------------------------------------------------------------------------------------------
//   int64   E             number of edges in the (time-sorted) view
//   int64   new_edge_lo   offset of the first NEW event (events [lo, E) are "new")
//   int32   src[E]        source vertex of each edge, time-sorted
//   int32   dst[E]        destination vertex of each edge, time-sorted
//
// BINARY OUTPUT FORMAT  (little-endian)
// -------------------------------------
//   int64   n_new         number of new events = max(0, E - lo)
//   int64   ear[n_new]    local index of the immediate temporal predecessor (-1 = root)
//   int64   depth[n_new]  topological depth (longest dependency chain ending at e)
//   int64   k             changed-cone size (# unique vertices touched by the new events)
//   int64   cone[k]       the changed cone: sorted unique vertices touched by new events
// Matches EventDependencyGraph{ear, cone, depth} field-for-field.
//
// COMPLEXITY: O(n_new) forward pass (amortised O(1) hash probe per endpoint) + O(k log k)
// to sort the cone. Memory O(n_new + distinct-touched-vertices).
#include <cstdio>
#include <cstdint>
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

// Open-addressing hash map  vertex(i64) -> last-seen local index(i64), linear probing.
// Stores only the vertices the window touches, so memory is O(distinct touched), not O(N).
// This mirrors the Python `last_seen: dict[int,int]` but without the per-event Python overhead.
struct LastSeen {
    vector<i64> key;   // vertex id, or EMPTY for an unused slot
    vector<i64> val;   // last-seen local index
    u64 mask = 0;      // capacity-1 (capacity is a power of two)
    i64 count = 0;
    static constexpr i64 EMPTY = -1;

    void init(i64 expected) {
        // capacity = next power of two >= 2*expected (load factor <= 0.5), min 16.
        u64 cap = 16;
        while ((i64)cap < 2 * expected + 1) cap <<= 1;
        key.assign(cap, EMPTY);
        val.assign(cap, -1);
        mask = cap - 1;
        count = 0;
    }
    static inline u64 hash(i64 x) {
        // splitmix64 finalizer -- good distribution for sequential / clustered vertex ids.
        u64 z = (u64)x + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    // Returns the last-seen local index of v, or -1 if v has not been seen yet.
    inline i64 get(i64 v) const {
        u64 i = hash(v) & mask;
        while (key[i] != EMPTY) {
            if (key[i] == v) return val[i];
            i = (i + 1) & mask;
        }
        return -1;
    }
    void grow() {
        vector<i64> ok = key, ov = val;
        u64 ncap = (mask + 1) << 1;
        key.assign(ncap, EMPTY); val.assign(ncap, -1); mask = ncap - 1; count = 0;
        for (size_t j = 0; j < ok.size(); j++) if (ok[j] != EMPTY) set(ok[j], ov[j]);
    }
    // Insert/update v -> idx.
    inline void set(i64 v, i64 idx) {
        if (2 * (count + 1) > (i64)(mask + 1)) grow();
        u64 i = hash(v) & mask;
        while (key[i] != EMPTY) {
            if (key[i] == v) { val[i] = idx; return; }
            i = (i + 1) & mask;
        }
        key[i] = v; val[i] = idx; count++;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]); return 1; }
    const char* inpath = argv[1];
    const char* outpath = argv[2];
    double t0 = now();

    // ---- read header + time-sorted endpoints ------------------------------------------
    FILE* f = fopen(inpath, "rb");
    if (!f) { fprintf(stderr, "[cpp] open fail %s\n", inpath); return 1; }
    i64 E = 0, lo = 0;
    if (fread(&E, 8, 1, f) != 1 || fread(&lo, 8, 1, f) != 1) {
        fprintf(stderr, "[cpp] header read fail\n"); fclose(f); return 1;
    }
    if (E < 0) E = 0;
    vector<i32> src((size_t)E), dst((size_t)E);
    if (E > 0) {
        if (fread(src.data(), 4, (size_t)E, f) != (size_t)E) {
            fprintf(stderr, "[cpp] src read fail\n"); fclose(f); return 1;
        }
        if (fread(dst.data(), 4, (size_t)E, f) != (size_t)E) {
            fprintf(stderr, "[cpp] dst read fail\n"); fclose(f); return 1;
        }
    }
    fclose(f);

    if (lo < 0) lo = 0;
    if (lo > E) lo = E;
    i64 n_new = E - lo;
    fprintf(stderr, "[cpp] changed_cone: E=%lld new_edge_lo=%lld n_new=%lld loaded in %.2fs\n",
            (long long)E, (long long)lo, (long long)n_new, now() - t0);

    // ---- empty window: write all-empty arrays -----------------------------------------
    if (n_new <= 0) {
        FILE* o = fopen(outpath, "wb");
        if (!o) { fprintf(stderr, "[cpp] out open fail %s\n", outpath); return 1; }
        i64 z = 0;
        fwrite(&z, 8, 1, o);   // n_new
        fwrite(&z, 8, 1, o);   // k (cone size)
        fclose(o);
        return 0;
    }

    // ---- forward pass: ear[i] = max(last_seen[src], last_seen[dst]) over earlier events --
    vector<i64> ear((size_t)n_new), depth((size_t)n_new, 0);
    LastSeen last;
    last.init(n_new);          // at most 2*n_new distinct vertices touched
    double tc = now();
    for (i64 i = 0; i < n_new; i++) {
        i64 u = (i64)src[lo + i];
        i64 v = (i64)dst[lo + i];
        i64 pu = last.get(u);
        i64 pv = last.get(v);
        // immediate predecessor = the more-recent (larger local index) of the two endpoints'
        // last events; identical to the Python `p = pu if pu > pv else pv`.
        i64 p = (pu > pv) ? pu : pv;
        ear[i] = p;
        depth[i] = (p >= 0) ? depth[p] + 1 : 0;
        last.set(u, i);
        last.set(v, i);
    }

    // ---- changed cone = sorted unique(src[lo:] concat dst[lo:]) (np.unique order) -------
    vector<i64> cone;
    cone.reserve((size_t)2 * n_new);
    for (i64 i = lo; i < E; i++) { cone.push_back((i64)src[i]); cone.push_back((i64)dst[i]); }
    sort(cone.begin(), cone.end());
    cone.erase(unique(cone.begin(), cone.end()), cone.end());
    i64 k = (i64)cone.size();
    fprintf(stderr, "[cpp] changed_cone: cone=%lld vertices, max-depth=%lld, computed in %.2fs (total %.2fs)\n",
            (long long)k, (long long)(n_new ? *max_element(depth.begin(), depth.end()) : 0),
            now() - tc, now() - t0);

    // ---- write output -------------------------------------------------------------------
    FILE* o = fopen(outpath, "wb");
    if (!o) { fprintf(stderr, "[cpp] out open fail %s\n", outpath); return 1; }
    fwrite(&n_new, 8, 1, o);
    fwrite(ear.data(), 8, (size_t)n_new, o);
    fwrite(depth.data(), 8, (size_t)n_new, o);
    fwrite(&k, 8, 1, o);
    if (k > 0) fwrite(cone.data(), 8, (size_t)k, o);
    fclose(o);
    return 0;
}
