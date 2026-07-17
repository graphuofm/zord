// zord graph-algorithm kernels in C++ (fast; never networkx). Computes node ORDERINGS that improve
// the locality of the memory-bound GNN aggregation -- a process speedup that does NOT change results.
// Build:  g++ -O3 -std=c++17 -fopenmp src/zord/cpp/graph_algos.cpp -o build/graph_algos
// Input  (binary): int64 N, int64 M, then 2*M int32 interleaved (src,dst).
// Output (binary): int64 N, then N int32  newid[old_node] = rank (the new contiguous id).
//   EXCEPTION -- mode `kcorevals`: output is int64 N, then N int32 core[v] (the per-vertex
//   CORE NUMBER in vertex-id order), NOT a reordering. Same input binary format.
// Modes: degree | kcore | kcorevals | bfs | lpa | dfs | slashburn | gorder
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstring>
#include <string>
#include <queue>
#include <utility>
using namespace std;
using i64 = int64_t;
using i32 = int32_t;

static double now() {
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <edges.bin> <degree|kcore|kcorevals|bfs|lpa|dfs|slashburn|gorder> <out.bin>\n", argv[0]); return 1; }
    const char* inpath = argv[1]; string mode = argv[2]; const char* outpath = argv[3];
    double t0 = now();
    FILE* f = fopen(inpath, "rb"); if (!f) { fprintf(stderr, "open fail %s\n", inpath); return 1; }
    i64 N = 0, M = 0; if (fread(&N, 8, 1, f) != 1 || fread(&M, 8, 1, f) != 1) { fprintf(stderr, "hdr fail\n"); return 1; }
    vector<i32> e((size_t)2 * M);
    if (fread(e.data(), 4, (size_t)2 * M, f) != (size_t)2 * M) { fprintf(stderr, "edge read fail\n"); return 1; }
    fclose(f);
    fprintf(stderr, "[cpp] loaded N=%lld M=%lld in %.2fs\n", (long long)N, (long long)M, now() - t0);

    // undirected CSR
    vector<i64> deg(N, 0);
    for (i64 k = 0; k < M; k++) { deg[e[2*k]]++; deg[e[2*k+1]]++; }
    vector<i64> off(N + 1, 0);
    for (i64 i = 0; i < N; i++) off[i+1] = off[i] + deg[i];
    vector<i32> adj(off[N]);
    { vector<i64> pos(off.begin(), off.end());
      for (i64 k = 0; k < M; k++) { i32 u = e[2*k], v = e[2*k+1]; adj[pos[u]++] = v; adj[pos[v]++] = u; } }
    vector<i32>().swap(e);

    vector<i32> newid(N);
    vector<i32> coreval;        // populated ONLY by the `kcorevals` mode (per-vertex core number)
    double tc = now();

    if (mode == "degree") {
        vector<i32> order(N); iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](i32 a, i32 b){ return deg[a] > deg[b]; });
        for (i64 r = 0; r < N; r++) newid[order[r]] = (i32)r;

    } else if (mode == "kcore") {
        // Batagelj-Zaversnik O(V+E) core decomposition; degeneracy order = processing order.
        i64 md = 0; for (i64 i = 0; i < N; i++) md = max(md, deg[i]);
        vector<i64> d(deg);
        vector<i64> bin(md + 2, 0);
        for (i64 i = 0; i < N; i++) bin[d[i]]++;
        i64 start = 0; for (i64 dd = 0; dd <= md; dd++) { i64 num = bin[dd]; bin[dd] = start; start += num; }
        vector<i64> pos(N), vert(N);
        for (i64 i = 0; i < N; i++) { pos[i] = bin[d[i]]; vert[pos[i]] = i; bin[d[i]]++; }
        for (i64 dd = md; dd >= 1; dd--) bin[dd] = bin[dd-1]; bin[0] = 0;
        for (i64 i = 0; i < N; i++) {
            i32 v = (i32)vert[i];
            newid[v] = (i32)i;                          // degeneracy order = good locality grouping
            for (i64 p = off[v]; p < off[v+1]; p++) {
                i32 u = adj[p];
                if (d[u] > d[v]) {
                    i64 du = d[u], pu = pos[u], pw = bin[du]; i32 w = (i32)vert[pw];
                    if (u != (i64)w) { vert[pu] = w; pos[w] = pu; vert[pw] = u; pos[u] = pw; }
                    bin[du]++; d[u]--;
                }
            }
        }
        fprintf(stderr, "[cpp] kcore: max_core=%lld\n", (long long)(*max_element(d.begin(), d.end())));

    } else if (mode == "kcorevals") {
        // SAME Batagelj-Zaversnik O(V+E) core decomposition as `kcore`, but instead of the
        // degeneracy ORDER we OUTPUT each vertex's CORE NUMBER in vertex-id order. In BZ, the
        // core number of v is exactly the value of d[v] at the moment v is processed (removed):
        // once v is popped from the bin-sorted `vert[]` it is never decremented again (only
        // neighbors u with d[u] > d[v], which sit later in the order, are decremented), so the
        // d[v] read at processing time is final. We capture it into coreval[v]. This reproduces
        // the standard core-number definition (largest k s.t. v is in a non-empty k-core) and
        // matches the numpy peel exactly.
        coreval.assign(N, 0);
        i64 md = 0; for (i64 i = 0; i < N; i++) md = max(md, deg[i]);
        vector<i64> d(deg);
        vector<i64> bin(md + 2, 0);
        for (i64 i = 0; i < N; i++) bin[d[i]]++;
        i64 start = 0; for (i64 dd = 0; dd <= md; dd++) { i64 num = bin[dd]; bin[dd] = start; start += num; }
        vector<i64> pos(N), vert(N);
        for (i64 i = 0; i < N; i++) { pos[i] = bin[d[i]]; vert[pos[i]] = i; bin[d[i]]++; }
        for (i64 dd = md; dd >= 1; dd--) bin[dd] = bin[dd-1]; bin[0] = 0;
        for (i64 i = 0; i < N; i++) {
            i32 v = (i32)vert[i];
            coreval[v] = (i32)d[v];                     // core number of v = its degree when removed
            for (i64 p = off[v]; p < off[v+1]; p++) {
                i32 u = adj[p];
                if (d[u] > d[v]) {
                    i64 du = d[u], pu = pos[u], pw = bin[du]; i32 w = (i32)vert[pw];
                    if (u != (i64)w) { vert[pu] = w; pos[w] = pu; vert[pw] = u; pos[u] = pw; }
                    bin[du]++; d[u]--;
                }
            }
        }
        fprintf(stderr, "[cpp] kcorevals: max_core=%lld\n",
                (long long)(coreval.empty() ? 0 : *max_element(coreval.begin(), coreval.end())));

    } else if (mode == "lpa") {
        // Label propagation clustering (K iters), then order nodes by cluster -> locality WITHOUT
        // ground-truth communities (the practical version of the community-oracle ordering).
        vector<i32> lab(N); iota(lab.begin(), lab.end(), 0);
        int K = 5; vector<i32> tmp;
        for (int it = 0; it < K; it++) {
            for (i64 v = 0; v < N; v++) {
                i64 s = off[v], en = off[v+1]; if (en == s) continue;
                tmp.clear(); for (i64 p = s; p < en; p++) tmp.push_back(lab[adj[p]]);
                sort(tmp.begin(), tmp.end());
                i32 best = tmp[0]; int bestc = 1, curc = 1; i32 cur = tmp[0];
                for (size_t i = 1; i < tmp.size(); i++) {
                    if (tmp[i] == cur) curc++; else { cur = tmp[i]; curc = 1; }
                    if (curc > bestc) { bestc = curc; best = cur; }
                }
                lab[v] = best;
            }
        }
        vector<i32> order(N); iota(order.begin(), order.end(), 0);
        stable_sort(order.begin(), order.end(), [&](i32 a, i32 b){ return lab[a] < lab[b]; });
        for (i64 r = 0; r < N; r++) newid[order[r]] = (i32)r;
        { vector<i32> u(lab); sort(u.begin(), u.end()); u.erase(unique(u.begin(), u.end()), u.end());
          fprintf(stderr, "[cpp] lpa: %zu clusters after %d iters\n", u.size(), K); }

    } else if (mode == "bfs") {
        // RCM-like locality: BFS from the highest-degree node, order by visit.
        i32 s = 0; i64 best = -1; for (i64 i = 0; i < N; i++) if (deg[i] > best) { best = deg[i]; s = (i32)i; }
        vector<char> seen(N, 0); queue<i32> q; i64 r = 0;
        for (i64 st = 0; st < N; st++) {
            i32 root = (i32)((s + st) % N);
            if (seen[root]) continue;
            seen[root] = 1; q.push(root);
            while (!q.empty()) { i32 v = q.front(); q.pop(); newid[v] = (i32)r++;
                for (i64 p = off[v]; p < off[v+1]; p++) { i32 u = adj[p]; if (!seen[u]) { seen[u] = 1; q.push(u); } } }
        }
    } else if (mode == "dfs") {
        // DFS visit order from the highest-degree node (depth-first analogue of bfs above).
        // Explicit stack to avoid recursion-depth blowup on large graphs. Handle disconnected
        // components exactly like bfs: cycle roots starting at the seed so every node gets a rank.
        i32 s = 0; i64 best = -1; for (i64 i = 0; i < N; i++) if (deg[i] > best) { best = deg[i]; s = (i32)i; }
        vector<char> seen(N, 0); vector<i32> stk; i64 r = 0;
        for (i64 st = 0; st < N; st++) {
            i32 root = (i32)((s + st) % N);
            if (seen[root]) continue;
            seen[root] = 1; stk.push_back(root);
            while (!stk.empty()) {
                i32 v = stk.back(); stk.pop_back();
                newid[v] = (i32)r++;
                // Push neighbors; they are popped LIFO so DFS descends into the first unseen one.
                // Mark on push (not on pop) so each node is enqueued at most once.
                for (i64 p = off[v]; p < off[v+1]; p++) { i32 u = adj[p]; if (!seen[u]) { seen[u] = 1; stk.push_back(u); } }
            }
        }

    } else if (mode == "slashburn") {
        // SlashBurn-style hub ordering (tractable approximation). Each round we remove the top-k
        // (k = max(1, N/1000)) highest *remaining-degree* nodes and assign them the next LOW ranks
        // (placed first). Removing a node decrements its still-present neighbors' remaining degree,
        // which approximates the disconnecting effect of SlashBurn's hub removal. We repeat until
        // every node with remaining degree > 0 is consumed; the remaining tail (now degree-0 in the
        // peeled graph, e.g. the periphery / spokes) is appended at the end ordered by original
        // degree (descending) so the densest tail nodes still land earlier.
        // APPROXIMATION: we do NOT recompute the exact greatest-connected-component each round (that
        // would need a union-find/BFS per round); degree-peeling of top hubs is the accepted
        // tractable substitute and preserves the "hubs first, periphery last" locality intent.
        vector<i64> d(deg);                 // remaining degree, mutated as hubs are removed
        vector<char> removed(N, 0);
        i64 k = max((i64)1, N / 1000);
        i64 r = 0;
        // Lazy max-heap keyed on remaining degree: entries are (degree, node). When a node is
        // popped we verify its stored degree still matches d[node] (else it is a stale entry and we
        // skip it) and that it is not already removed. We re-push a neighbor only when its degree
        // changes, so total heap ops are O((N+E) log N) -- far cheaper than re-scanning all N nodes
        // every round. This is the standard lazy-deletion priority-queue pattern.
        // pair<degree,node>; default pair compares first then second -> a true max-heap on degree.
        priority_queue<pair<i64,i32>> pq;
        for (i64 i = 0; i < N; i++) if (d[i] > 0) pq.push({d[i], (i32)i});
        // Each "round" pops up to k currently-valid top hubs, removes them, then refreshes the heap
        // for affected neighbors. Heap ordering already gives global top-degree, so popping k valid
        // entries == taking this round's top-k highest-remaining-degree nodes.
        while (!pq.empty()) {
            i64 take = 0;
            while (take < k && !pq.empty()) {
                auto top = pq.top(); pq.pop();
                i32 v = top.second; i64 dv = top.first;
                if (removed[v] || dv != d[v]) continue;   // stale / already removed
                removed[v] = 1; newid[v] = (i32)r++;
                take++;
                // Decrement remaining degree of still-present neighbors and re-push their new key.
                for (i64 p = off[v]; p < off[v+1]; p++) {
                    i32 u = adj[p];
                    if (!removed[u] && d[u] > 0) { d[u]--; if (d[u] > 0) pq.push({d[u], u}); }
                }
            }
            if (take == 0) break;   // only stale entries remained
        }
        // Tail: everything not yet ranked (peeled to degree 0, plus original isolated nodes),
        // ordered by original degree descending so the densest periphery still groups early.
        vector<i32> tail;
        for (i64 i = 0; i < N; i++) if (!removed[i]) tail.push_back((i32)i);
        sort(tail.begin(), tail.end(), [&](i32 a, i32 b){ return deg[a] > deg[b]; });
        for (i32 v : tail) newid[v] = (i32)r++;
        fprintf(stderr, "[cpp] slashburn: k=%lld, tail=%zu\n", (long long)k, tail.size());

    } else if (mode == "gorder") {
        // Gorder-lite: greedy sliding-window neighbor co-occurrence ordering. Start from the
        // highest-degree node; keep a window of the last W=5 placed nodes. Repeatedly pick the
        // unplaced candidate that shares the most edges with currently-placed window nodes
        // (score = number of edges from the candidate to the W window members), tie-break by
        // higher original degree. Candidates are restricted to neighbors-of-the-window (plus a
        // fallback when that set is empty), so per-step cost is bounded by window-neighborhood
        // size rather than N -- keeping it tractable on ~100M-edge graphs.
        const int W = 5;
        vector<char> placed(N, 0);
        // window holds up to W most-recently-placed node ids (ring usage via vector + index)
        vector<i32> window; window.reserve(W);
        i64 r = 0;
        // score[c] = current co-occurrence score of candidate c with the window; cand_list tracks
        // which nodes currently have nonzero score so we can reset them cheaply each step.
        vector<i32> score(N, 0);
        vector<i32> cand_list;            // distinct candidates with score>0
        vector<char> in_cand(N, 0);
        // Helper-free inline: add a node's contribution (+1 to each unplaced neighbor's score).
        auto add_contrib = [&](i32 v) {
            for (i64 p = off[v]; p < off[v+1]; p++) {
                i32 u = adj[p];
                if (placed[u]) continue;
                if (!in_cand[u]) { in_cand[u] = 1; cand_list.push_back(u); }
                score[u]++;
            }
        };
        auto sub_contrib = [&](i32 v) {
            for (i64 p = off[v]; p < off[v+1]; p++) {
                i32 u = adj[p];
                if (placed[u]) continue;
                if (score[u] > 0) score[u]--;
            }
        };
        // Seed with the highest-degree node.
        i32 s = 0; i64 best = -1; for (i64 i = 0; i < N; i++) if (deg[i] > best) { best = deg[i]; s = (i32)i; }
        i32 next_scan = 0;                // cursor for finding an unplaced node when window empties
        for (i64 step = 0; step < N; step++) {
            i32 pick = -1;
            if (step == 0) {
                pick = s;
            } else {
                // Choose best candidate from cand_list (neighbors of window with score>0).
                i64 bscore = -1, bdeg = -1;
                for (i32 c : cand_list) {
                    if (placed[c]) continue;
                    i32 sc = score[c];
                    if (sc <= 0) continue;
                    if (sc > bscore || (sc == bscore && deg[c] > bdeg)) { bscore = sc; bdeg = deg[c]; pick = c; }
                }
                if (pick == -1) {
                    // Window neighborhood exhausted (disconnected / window all low-degree).
                    // Fall back to the next unplaced node, preferring a remaining high-degree
                    // start: linear cursor scan keeps this O(N) total across all fallbacks.
                    while (next_scan < N && placed[next_scan]) next_scan++;
                    if (next_scan >= N) break;   // nothing left (shouldn't happen before step==N)
                    pick = next_scan;
                }
            }
            // Place pick.
            placed[pick] = 1; newid[pick] = (i32)r++;
            // If pick was a scored candidate, clear its score bookkeeping.
            if (in_cand[pick]) { in_cand[pick] = 0; score[pick] = 0; }
            // Slide the window: evicting node loses its contribution, new node adds its contribution.
            if ((i64)window.size() == W) {
                i32 evict = window.front();
                window.erase(window.begin());
                sub_contrib(evict);
            }
            window.push_back(pick);
            add_contrib(pick);
            // Compact cand_list lazily: drop entries that are now placed or zero-scored so it does
            // not grow unbounded (placed nodes can linger after eviction zeroed their score).
            if (cand_list.size() > (size_t)(8 * (off[N] / max((i64)1, N)) * W + 64)) {
                vector<i32> keep; keep.reserve(cand_list.size());
                for (i32 c : cand_list) {
                    if (!placed[c] && score[c] > 0) keep.push_back(c);
                    else if (placed[c]) in_cand[c] = 0;
                    else if (score[c] == 0) in_cand[c] = 0;
                }
                cand_list.swap(keep);
            }
        }
        fprintf(stderr, "[cpp] gorder: W=%d placed=%lld\n", W, (long long)r);

    } else { fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 1; }

    fprintf(stderr, "[cpp] ordering(%s) computed in %.2fs (load+%.2fs total)\n", mode.c_str(), now() - tc, now() - t0);
    FILE* o = fopen(outpath, "wb");
    fwrite(&N, 8, 1, o);
    // `kcorevals` writes per-vertex CORE NUMBERS (coreval[v]); all other modes write the
    // reordering newid[old]->rank. Both are N int32 after the int64 N header.
    if (mode == "kcorevals") fwrite(coreval.data(), 4, N, o);
    else                     fwrite(newid.data(), 4, N, o);
    fclose(o);
    return 0;
}
