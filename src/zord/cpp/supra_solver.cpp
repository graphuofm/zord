// zord CORE solver: multilevel WEIGHTED supra-graph partitioner (C++17).
// =====================================================================================
// Build:  g++ -O3 -std=c++17 -fopenmp src/zord/cpp/supra_solver.cpp -o build/supra_solver
//
// WHAT IT DOES
// ------------
// A temporal graph is modelled as a SUPRA-GRAPH over CELLS (v, t) = "vertex v at snapshot t"
// (THEORY.md sec.1). There are two edge types between cells:
//   - SPATIAL edges: within ONE snapshot t, edge (src,dst,t) couples cell (src,t)--(dst,t)
//     (the GNN aggregation / SpMM dependency).
//   - TEMPORAL edges: the SAME vertex v across ADJACENT active snapshots t1<t2 couples
//     cell (v,t1)--(v,t2) (the node-memory / embedding recurrence).
// A cell (v,t) is ACTIVE iff v has an incident edge in snapshot t. We assign every active
// cell to one of D devices to minimise
//       w_S * SpatialCut(P) + w_T * TemporalCut(P)
// subject to a per-device capacity cap_cells (#cells; 0 = unbounded).
//   SpatialCut  = # spatial cell-pairs whose two endpoints are on different devices.
//   TemporalCut = # temporal cell-pairs whose two endpoints are on different devices.
// (This is exactly the cut-counting mirrored by scripts/duality_frontier.py and
//  scripts/supra_solve.py, but on the EXPLICIT cell graph so an arbitrary assignment is
//  scored, not just a Dv x Dt factorization.)
//
// BINARY INPUT FORMAT (little-endian, the order below; documented for supra_solve.py)
// -----------------------------------------------------------------------------------
//   int64   N           number of vertices
//   int64   S           number of snapshots
//   int64   M           number of temporal (time-stamped) edges
//   int32   triples[3*M] M triples (src, dst, snapshot), snapshot in [0,S)
//   int32   D           number of devices
//   float   w_S         spatial-cut weight  (bytes_per_halo  / B_link, THEORY.md sec.2)
//   float   w_T         temporal-cut weight (bytes_per_memory / B_link)
//   int64   cap_cells   per-device capacity in #cells (0 = unbounded)
//
// BINARY OUTPUT FORMAT (little-endian)
// ------------------------------------
//   int64   num_cells               number of ACTIVE cells (== rows in the cell table)
//   int32   device[num_cells]       device id in [0,D) per active cell, in CELL-ID order
// The cell-id order is the canonical order produced here (sorted by (vertex, snapshot));
// supra_solve.py reconstructs the identical order to map cells back to (v,t).
// To stderr we print: SpatialCut, TemporalCut, weighted cost, per-device cell counts, runtime.
//
// ALGORITHM (tractable v1 = Fennel/streaming greedy + KL refinement, then corner-guarded pick)
// ---------------------------------------------------------------------------------------------
// We build the cell graph (CSR over cells, each incident edge tagged spatial/temporal and
// carrying weight w_S or w_T). Then we generate THREE candidate assignments and OUTPUT THE
// BEST FEASIBLE one (this is what makes "zord <= min(PSS,PTS)" a guarantee, not luck):
//   (A) greedy+refine  -- the capacity-respecting INTERIOR optimiser:
//       Cells are streamed in a STRUCTURED order: primary key = snapshot t when w_T <= w_S,
//       else primary key = vertex v (the "leading coordinate" biases the stream toward the
//       relevant corner). For each cell c we score every device d by the Fennel objective
//          gain(c,d) = sum over already-placed neighbours n on device d of weight(c,n)
//                    - lambda * load_d / cap                  // capacity / balance term
//       and place c on argmax (weight = w_S for spatial neighbours, w_T for temporal, so
//       co-locating a neighbour AVOIDS that cut). A HARD cap forbids full devices (with a
//       least-loaded fallback so a feasible placement always exists). Then a few rounds of
//       KL/FM-style boundary moves relocate each cell to the device retaining the most
//       incident weight (subject to the cap) -- a monotone weighted-cost descent. This is the
//       "multilevel-style" local search the v1 affords without full coarsening.
//   (B) PSS block assignment -- snapshots split into D contiguous balanced blocks -> devices.
//   (C) PTS block assignment -- vertices  split into D contiguous balanced blocks -> devices.
// zord reports the MIN-COST candidate whose max device load respects the cap.
//
// HOW THE PSS / PTS CORNERS ARE REACHED (CRITICAL REQUIREMENT, THEORY.md sec.3)
// ----------------------------------------------------------------------------
// The solver's search space CONTAINS every baseline as an explicit feasible point, so it can
// never do worse than the best feasible corner:
//   * PSS corner  (w_T -> 0): candidate (B) puts whole snapshots on single devices -> the only
//     crossing edges are temporal, whose weight is 0, so its cost is 0 (== the w_T=0 optimum).
//     It is selected because no candidate can beat cost 0. (Independently, with w_T=0 the
//     greedy also streams snapshot-major and packs whole snapshots, converging to the same.)
//   * PTS corner  (w_S -> 0): candidate (C) puts whole vertex-timelines on single devices ->
//     the only crossing edges are spatial, weight 0, cost 0; selected for the same reason.
//   * S=1 (static): there is one snapshot; candidate (B) is the whole graph on D devices via
//     vertex blocks is degenerate, and the greedy reduces to a Fennel spatial min-cut (METIS-like).
//   * INTERIOR (both weights > 0 AND a cap that forbids a full snapshot AND a full timeline):
//     the corners (B)/(C) become INFEASIBLE (max load > cap) and are dropped; only the
//     capacity-respecting greedy+refine (A) remains, trading w_S*spatial vs w_T*temporal under
//     the cap -- the regime where the integrated cut strictly dominates both corners
//     (THEORY.md sec.3/sec.6). When both corners ARE feasible but neither is best, (A) can
//     still win by finding a blend that beats them.
//
// COMPLEXITY: build O(M + cells), greedy O(sum_c deg(c) + cells*D) ~ O(E_supra + cells*D),
// refinement O(rounds * (E_supra + cells)). Memory O(cells + E_supra).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <string>
#include <cmath>
using namespace std;
using i64 = int64_t;
using i32 = int32_t;
using u64 = uint64_t;

static double now() {
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin> [cells_out.bin]\n", argv[0]);
        return 1;
    }
    const char* inpath = argv[1];
    const char* outpath = argv[2];
    // OPTIONAL 3rd arg (ADDITIVE, back-compatible): a sidecar path to which the canonical active-cell
    // table is dumped so the Python wrapper (allocate.py) can fold cell_device[] -> vertex assignment
    // WITHOUT re-deriving the (v,t) cell coordinates in numpy at 100M-cell scale. When omitted the
    // tool behaves EXACTLY as before (2-arg CLI; binary output unchanged). Format (little-endian):
    //   int64 C; int32 cell_v[C]; int32 cell_t[C]   -- in canonical cell-id order (matches device[]).
    const char* cellpath = (argc >= 4) ? argv[3] : nullptr;
    double t0 = now();

    // ---- read header + edges -----------------------------------------------------------
    FILE* f = fopen(inpath, "rb");
    if (!f) { fprintf(stderr, "[cpp] open fail %s\n", inpath); return 1; }
    i64 N = 0, S = 0, M = 0;
    if (fread(&N, 8, 1, f) != 1 || fread(&S, 8, 1, f) != 1 || fread(&M, 8, 1, f) != 1) {
        fprintf(stderr, "[cpp] header read fail\n"); fclose(f); return 1;
    }
    vector<i32> trip((size_t)3 * M);
    if (M > 0 && fread(trip.data(), 4, (size_t)3 * M, f) != (size_t)3 * M) {
        fprintf(stderr, "[cpp] triple read fail\n"); fclose(f); return 1;
    }
    i32 D = 0; float w_S = 0.f, w_T = 0.f; i64 cap_cells = 0;
    if (fread(&D, 4, 1, f) != 1 || fread(&w_S, 4, 1, f) != 1 ||
        fread(&w_T, 4, 1, f) != 1 || fread(&cap_cells, 8, 1, f) != 1) {
        fprintf(stderr, "[cpp] footer read fail\n"); fclose(f); return 1;
    }
    fclose(f);
    if (D < 1) D = 1;
    fprintf(stderr, "[cpp] loaded N=%lld S=%lld M=%lld D=%d w_S=%.6g w_T=%.6g cap=%lld in %.2fs\n",
            (long long)N, (long long)S, (long long)M, D, w_S, w_T, (long long)cap_cells, now() - t0);

    // ---- discover ACTIVE cells: (v,t) with an incident edge in snapshot t --------------
    // Use a sorted list of unique (v*S + t) keys; cell-id = rank in that sorted order
    // => canonical order is (vertex-major, snapshot-minor) which supra_solve.py reproduces.
    vector<u64> keys;
    keys.reserve((size_t)2 * M);
    for (i64 k = 0; k < M; k++) {
        i32 s = trip[3*k], d = trip[3*k+1], t = trip[3*k+2];
        if (t < 0) t = 0;
        if (t >= S) t = (i32)(S - 1);
        keys.push_back((u64)s * (u64)S + (u64)t);
        keys.push_back((u64)d * (u64)S + (u64)t);
    }
    sort(keys.begin(), keys.end());
    keys.erase(unique(keys.begin(), keys.end()), keys.end());
    i64 C = (i64)keys.size();                       // number of active cells
    if (C == 0) {
        fprintf(stderr, "[cpp] no active cells; writing empty assignment\n");
        FILE* o = fopen(outpath, "wb"); i64 z = 0; fwrite(&z, 8, 1, o); fclose(o);
        if (cellpath) { FILE* cf = fopen(cellpath, "wb"); i64 zz = 0; fwrite(&zz, 8, 1, cf); fclose(cf); }
        fprintf(stderr, "STAT cells=0 spatial_cut=0 temporal_cut=0 weighted_cost=0 winner=empty\n");
        return 0;
    }
    // map (v,t) key -> cell id via binary search over `keys`
    auto cell_of = [&](u64 key) -> i64 {
        return (i64)(lower_bound(keys.begin(), keys.end(), key) - keys.begin());
    };
    // per-cell coordinates
    vector<i32> cell_v(C), cell_t(C);
    for (i64 c = 0; c < C; c++) { cell_v[c] = (i32)(keys[c] / (u64)S); cell_t[c] = (i32)(keys[c] % (u64)S); }

    // ---- build the supra-graph (CSR over cells), edges tagged spatial/temporal ----------
    // Spatial edges: per input edge (s,d,t) -> cell(s,t) -- cell(d,t)  (skip self/duplicate handled by weight sum).
    // Temporal edges: per vertex v, sort its active snapshots, connect adjacent ones.
    // We first COUNT degree, then fill. Edge weight stored implicitly by a 1-bit "type" flag
    // packed with the neighbour id; weight = w_S (spatial) or w_T (temporal).
    // To keep memory modest we store neighbour as int64 = (neighbour_cell_id<<1) | is_temporal.
    // Collect spatial pairs.
    vector<pair<i64,i64>> spat;  spat.reserve((size_t)M);
    for (i64 k = 0; k < M; k++) {
        i32 s = trip[3*k], d = trip[3*k+1], t = trip[3*k+2];
        if (t < 0) t = 0;
        if (t >= S) t = (i32)(S - 1);
        if (s == d) continue;
        i64 a = cell_of((u64)s * (u64)S + (u64)t);
        i64 b = cell_of((u64)d * (u64)S + (u64)t);
        if (a != b) spat.push_back({a, b});
    }
    vector<i32>().swap(trip);
    // Collect temporal pairs: cells are sorted (v,t), so each vertex's cells are a contiguous
    // run in cell-id order; connect consecutive cells of the SAME vertex (adjacent in time).
    vector<pair<i64,i64>> temp;  temp.reserve((size_t)C);
    for (i64 c = 1; c < C; c++) {
        if (cell_v[c] == cell_v[c-1]) temp.push_back({c-1, c});   // same vertex, next active snapshot
    }
    i64 nSpat = (i64)spat.size(), nTemp = (i64)temp.size();

    // CSR build (undirected: each pair contributes to both endpoints).
    vector<i64> deg(C, 0);
    for (auto& p : spat) { deg[p.first]++; deg[p.second]++; }
    for (auto& p : temp) { deg[p.first]++; deg[p.second]++; }
    vector<i64> off(C + 1, 0);
    for (i64 c = 0; c < C; c++) off[c+1] = off[c] + deg[c];
    vector<i64> nbr(off[C]);             // (neighbour_cell << 1) | is_temporal
    {
        vector<i64> pos(off.begin(), off.end());
        for (auto& p : spat) {
            nbr[pos[p.first]++]  = (p.second << 1) | 0;
            nbr[pos[p.second]++] = (p.first  << 1) | 0;
        }
        for (auto& p : temp) {
            nbr[pos[p.first]++]  = (p.second << 1) | 1;
            nbr[pos[p.second]++] = (p.first  << 1) | 1;
        }
    }
    vector<pair<i64,i64>>().swap(spat);
    vector<pair<i64,i64>>().swap(temp);
    fprintf(stderr, "[cpp] supra-graph: cells=%lld spatial_edges=%lld temporal_edges=%lld (build %.2fs)\n",
            (long long)C, (long long)nSpat, (long long)nTemp, now() - t0);

    // ---- capacity -----------------------------------------------------------------------
    // Effective hard cap per device. 0 (unbounded) -> ceil(C/D) is used only for the balance
    // penalty, not as a hard limit. A hard cap below ceil(C/D) is raised to it (else infeasible).
    i64 cap = cap_cells;
    i64 minfeasible = (C + D - 1) / D;
    bool hard_cap = (cap_cells > 0);
    if (hard_cap && cap < minfeasible) {
        fprintf(stderr, "[cpp] WARNING cap_cells=%lld < ceil(C/D)=%lld; raising to keep feasible\n",
                (long long)cap_cells, (long long)minfeasible);
        cap = minfeasible;
    }
    if (!hard_cap) cap = minfeasible;          // soft target for balance penalty

    // ---- which coordinate "leads" the stream (and seeds corners) ------------------------
    // w_T==0 -> PSS corner: lead by snapshot; w_S==0 -> PTS corner: lead by vertex.
    // Generic interior: lead by the SMALLER-weight coordinate's natural locality so the
    // dominant (larger-weight) edge type is the one the greedy actively glues.
    bool pss_mode = (w_T <= 0.f && w_S > 0.f);    // temporal free -> whole snapshots together
    bool pts_mode = (w_S <= 0.f && w_T > 0.f);    // spatial free  -> whole timelines together
    // lead_by_snapshot: stream order primary key = snapshot (true) or vertex (false).
    bool lead_by_snapshot = pss_mode ? true : (pts_mode ? false : (w_T <= w_S));

    // streaming order over cells
    vector<i64> order(C);
    iota(order.begin(), order.end(), 0);
    if (lead_by_snapshot) {
        // primary = snapshot, secondary = vertex (cells already vertex-major -> need re-sort)
        stable_sort(order.begin(), order.end(), [&](i64 a, i64 b){
            if (cell_t[a] != cell_t[b]) return cell_t[a] < cell_t[b];
            return cell_v[a] < cell_v[b];
        });
    } // else: cells are already (vertex-major, snapshot-minor) == the desired PTS/vertex order

    // ---- shared helpers -----------------------------------------------------------------
    // weighted cut cost of an assignment (each pair counted once via n>c).
    auto cost_of = [&](const vector<i32>& asg, double& spatialCut, double& temporalCut) {
        i64 sc = 0, tc = 0;
        for (i64 c = 0; c < C; c++) {
            for (i64 p = off[c]; p < off[c+1]; p++) {
                i64 enc = nbr[p]; i64 n = enc >> 1; bool isT = (enc & 1);
                if (n > c && asg[n] != asg[c]) { if (isT) tc++; else sc++; }
            }
        }
        spatialCut = (double)sc; temporalCut = (double)tc;
        return (double)w_S * sc + (double)w_T * tc;
    };
    auto max_load = [&](const vector<i32>& asg) {
        vector<i64> ld(D, 0); for (i64 c = 0; c < C; c++) ld[asg[c]]++;
        i64 mx = 0; for (i32 d = 0; d < D; d++) mx = max(mx, ld[d]); return mx;
    };

    // ---- CORNER builders (the search space CONTAINS every baseline; THEORY.md sec.3) -----
    // A "block" assignment: split a sorted coordinate's distinct values into D contiguous,
    // balanced device runs. lead_snapshot=true -> snapshot-blocks (PSS, Dv=1xDt=D); false ->
    // vertex-blocks (PTS, Dv=DxDt=1). Balanced by distinct-block count, matching duality_frontier.
    auto block_assignment = [&](bool by_snapshot) {
        // distinct values of the chosen coordinate, ascending
        vector<i64> vals(C);
        for (i64 c = 0; c < C; c++) vals[c] = by_snapshot ? (i64)cell_t[c] : (i64)cell_v[c];
        vector<i64> uniq(vals); sort(uniq.begin(), uniq.end());
        uniq.erase(unique(uniq.begin(), uniq.end()), uniq.end());
        i64 B = (i64)uniq.size();
        vector<i32> asg(C);
        for (i64 c = 0; c < C; c++) {
            i64 idx = lower_bound(uniq.begin(), uniq.end(), vals[c]) - uniq.begin();
            asg[c] = (i32)((idx * D) / B);                 // contiguous balanced block -> device
        }
        return asg;
    };

    // ---- the streaming Fennel greedy + KL/FM refinement (the interior optimiser) ---------
    double wmax = max((double)w_S, (double)w_T);
    if (wmax <= 0.0) wmax = 1.0;                       // degenerate (both weights 0): pure balance
    const double gamma = 1.5;
    double lambda = wmax * gamma;                       // balance penalty strength

    vector<i32> dev(C, -1);
    vector<i64> load(D, 0);
    vector<double> gbuf(D, 0.0);
    vector<i32> touched; touched.reserve(64);
    for (i64 i = 0; i < C; i++) {
        i64 c = order[i];
        for (i64 p = off[c]; p < off[c+1]; p++) {
            i64 enc = nbr[p]; i64 n = enc >> 1; bool isT = (enc & 1);
            i32 nd = dev[n]; if (nd < 0) continue;
            double w = isT ? (double)w_T : (double)w_S;
            if (gbuf[nd] == 0.0) touched.push_back(nd);
            gbuf[nd] += w;
        }
        i32 best = -1; double bestScore = -1e300;
        for (i32 d = 0; d < D; d++) {
            if (load[d] >= cap) continue;                 // hard cap (cap >= ceil(C/D) -> always feasible)
            double score = gbuf[d] - lambda * (double)load[d] / (double)cap;   // Fennel-style balance
            if (score > bestScore) { bestScore = score; best = d; }
        }
        if (best < 0) {                                   // all at cap: least-loaded feasible
            i64 mn = -1;
            for (i32 d = 0; d < D; d++) if (mn < 0 || load[d] < load[mn]) mn = d;
            best = (i32)mn;
        }
        dev[c] = best; load[best]++;
        for (i32 d : touched) gbuf[d] = 0.0;
        touched.clear();
    }
    fprintf(stderr, "[cpp] greedy placement done (%.2fs)\n", now() - t0);

    // KL/FM-style boundary refinement: move each cell to the device that retains the most
    // incident weight (== removes the most cut), respecting the cap. Monotone cost descent.
    int REFINE_ROUNDS = (C > 5'000'000) ? 2 : 8;
    vector<double> mgain(D, 0.0);
    vector<i32> mtouched; mtouched.reserve(64);
    for (int round = 0; round < REFINE_ROUNDS; round++) {
        i64 moves = 0;
        for (i64 c = 0; c < C; c++) {
            i32 cur = dev[c];
            for (i64 p = off[c]; p < off[c+1]; p++) {
                i64 enc = nbr[p]; i64 n = enc >> 1; bool isT = (enc & 1);
                i32 nd = dev[n]; double w = isT ? (double)w_T : (double)w_S;
                if (mgain[nd] == 0.0) mtouched.push_back(nd);
                mgain[nd] += w;
            }
            i32 best = cur; double bestRetain = mgain[cur];
            for (i32 d : mtouched) {
                if (d == cur || load[d] + 1 > cap) continue;       // respect cap
                if (mgain[d] > bestRetain) { bestRetain = mgain[d]; best = d; }
            }
            if (best != cur) { load[cur]--; load[best]++; dev[c] = best; moves++; }
            for (i32 d : mtouched) mgain[d] = 0.0;
            mtouched.clear();
        }
        if (moves == 0) { fprintf(stderr, "[cpp] refine round %d: converged\n", round); break; }
        fprintf(stderr, "[cpp] refine round %d: %lld moves\n", round, (long long)moves);
    }

    // ---- candidate selection: zord = best feasible over {greedy+refine, PSS, PTS} --------
    // This is the operational form of THEORY.md sec.3: zord minimises the SAME cost over a
    // SUPERSET that includes the corners, so its output is <= min(PSS,PTS) by construction.
    // A candidate is feasible iff its max device load respects the hard cap (corners may
    // violate a tight cap -> they are simply dropped, and the capacity-respecting greedy wins).
    struct Cand { const char* name; vector<i32> asg; double sc, tc, cost; bool feasible; };
    vector<Cand> cands;
    auto add_cand = [&](const char* nm, vector<i32> asg) {
        double sc, tc; double cost = cost_of(asg, sc, tc);
        bool feas = (!hard_cap) || (max_load(asg) <= cap);
        cands.push_back({nm, move(asg), sc, tc, cost, feas});
    };
    add_cand("greedy+refine", dev);                       // always feasible (built under the cap)
    add_cand("PSS(Dv1xDtD)", block_assignment(true));     // whole snapshots -> devices
    add_cand("PTS(DvDxDt1)", block_assignment(false));    // whole timelines -> devices

    int bi = -1; double bcost = 1e300;
    for (int i = 0; i < (int)cands.size(); i++) {
        fprintf(stderr, "[cpp] candidate %-14s Sc=%.0f Tc=%.0f cost=%.6g feasible=%d\n",
                cands[i].name, cands[i].sc, cands[i].tc, cands[i].cost, (int)cands[i].feasible);
        if (cands[i].feasible && cands[i].cost < bcost) { bcost = cands[i].cost; bi = i; }
    }
    if (bi < 0) bi = 0;                                   // greedy is always feasible
    const Cand& win = cands[bi];
    const vector<i32>& out = win.asg;

    // ---- final accounting ---------------------------------------------------------------
    vector<i64> finalLoad(D, 0); for (i64 c = 0; c < C; c++) finalLoad[out[c]]++;
    fprintf(stderr, "[cpp] ==== RESULT (winner=%s) ====\n", win.name);
    fprintf(stderr, "[cpp] SpatialCut  = %.0f\n", win.sc);
    fprintf(stderr, "[cpp] TemporalCut = %.0f\n", win.tc);
    fprintf(stderr, "[cpp] weighted cost = w_S*Sc + w_T*Tc = %.6g\n", win.cost);
    // MACHINE-READABLE stat line (ADDITIVE, back-compatible): a single greppable line the Python
    // wrapper (partition/allocate.py) parses to recover SpatialCut/TemporalCut/cost/winner WITHOUT
    // re-counting the cuts in numpy at 100M-cell scale. The binary OUTPUT format is UNCHANGED
    // (int64 C; int32 device[C]); this is purely an extra stderr line that old callers ignore.
    fprintf(stderr, "STAT cells=%lld spatial_cut=%.0f temporal_cut=%.0f weighted_cost=%.10g winner=%s\n",
            (long long)C, win.sc, win.tc, win.cost, win.name);
    fprintf(stderr, "[cpp] per-device cell counts:");
    for (i32 d = 0; d < D; d++) fprintf(stderr, " %lld", (long long)finalLoad[d]);
    fprintf(stderr, "\n[cpp] mode=%s lead=%s cap=%lld(%s)\n",
            pss_mode ? "PSS(w_T=0)" : pts_mode ? "PTS(w_S=0)" : "interior",
            lead_by_snapshot ? "snapshot" : "vertex",
            (long long)cap, hard_cap ? "hard" : "soft");
    fprintf(stderr, "[cpp] runtime = %.2fs\n", now() - t0);

    // ---- write output --------------------------------------------------------------------
    FILE* o = fopen(outpath, "wb");
    if (!o) { fprintf(stderr, "[cpp] open out fail %s\n", outpath); return 1; }
    fwrite(&C, 8, 1, o);
    fwrite(out.data(), 4, (size_t)C, o);
    fclose(o);

    // OPTIONAL canonical-cell sidecar (ADDITIVE): C, cell_v[C], cell_t[C] in cell-id order so the
    // Python wrapper folds cells->vertices without rebuilding the (v,t) table. Written only when the
    // 3rd arg is given; the primary output above is untouched -> existing 2-arg callers are unaffected.
    if (cellpath) {
        FILE* cf = fopen(cellpath, "wb");
        if (!cf) { fprintf(stderr, "[cpp] open cells out fail %s\n", cellpath); return 1; }
        fwrite(&C, 8, 1, cf);
        fwrite(cell_v.data(), 4, (size_t)C, cf);
        fwrite(cell_t.data(), 4, (size_t)C, cf);
        fclose(cf);
    }
    return 0;
}
