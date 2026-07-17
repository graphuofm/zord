// zord MULTILEVEL k-way graph partitioner (C++17) -- zord's OWN min-cut, no pymetis dependency.
// =====================================================================================
// Build:  g++ -O3 -std=c++17 -fopenmp src/zord/cpp/multilevel.cpp -o build/multilevel
//
// WHY: arrange's "zord <= METIS" floor leaned on pymetis (Python, superlinear, GATED OFF above
// 20M edges -> at scale the floor silently degraded to a cheap lpa-proxy). This is zord's OWN
// multilevel partitioner -- the classic coarsen / initial-partition / uncoarsen+refine scheme
// (Karypis-Kumar) in C++ -- so a real balanced min-cut runs at 100M-1B edges with no Python and
// no superlinear blowup. PROCESS-only: a partition is a result-preserving placement.
//
// ALGORITHM (direct k-way multilevel):
//   1. COARSEN: heavy-edge matching (HEM) collapses each vertex with its heaviest unmatched
//      neighbour into a supernode (vertex+edge weights summed); recurse until <= COARSEST_MULT*D
//      vertices or COARSEN_STALL levels with little shrink.
//   2. INITIAL k-way: greedy balanced growth on the coarsest graph -- vertices in descending
//      weight are placed on the part that maximises retained edge weight subject to the balance
//      cap (ceil(totalW/D)*ubfactor), least-loaded fallback.
//   3. UNCOARSEN + REFINE: project the partition to each finer level and run greedy boundary
//      k-way FM (move a boundary vertex to the neighbouring part with the largest cut reduction
//      that keeps balance; a few passes per level, monotone cut descent).
//
// BINARY INPUT (LE):  i64 N, M ; i32 src[M], dst[M] ; i32 D ; double ubfactor (e.g. 1.03)
//   [optional] i32 has_ratio ; if 1: f64 ratio[D]  (per-part target share, heterogeneity-aware;
//   absent/0 -> equal split. cap[d] = totalW * ratio[d]/sum * ubfactor.)
//   [optional] i32 has_vwgt  ; if 1: i64 vwgt[N]   (per-node FEATURE BYTES F_v*4 -> balance feature MEMORY)
//   [optional] i32 has_ewgt  ; if 1: i64 ewgt[M]   (per-edge FEATURE BYTES F_e   -> feature-comm-weighted cut)
// BINARY OUTPUT (LE): i64 N ; i32 part[N]
// stderr: edgecut, balance (max_part_w / avg), per-level shrink, runtime.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <chrono>
using namespace std;
using i64=int64_t; using i32=int32_t; using u64=uint64_t;
static double now_s(){ return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }

// weighted undirected CSR
struct Graph {
    i64 n=0;
    vector<i64> xadj;       // n+1
    vector<i32> adj;        // m (neighbours)
    vector<i64> ewgt;       // m (edge weights, parallel to adj)
    vector<i64> vwgt;       // n (vertex weights)
    i64 totalw=0;
};

// build CSR from an edge list (dedupe parallel edges by summing weight; drop self-loops)
static Graph build_csr(i64 n, const vector<i32>& src, const vector<i32>& dst, const vector<i64>& ew){
    Graph g; g.n=n; g.vwgt.assign(n,1); g.totalw=n;
    i64 m=src.size();
    vector<i64> deg(n,0);
    for(i64 i=0;i<m;i++){ if(src[i]!=dst[i]){ deg[src[i]]++; deg[dst[i]]++; } }
    g.xadj.assign(n+1,0); for(i64 v=0;v<n;v++) g.xadj[v+1]=g.xadj[v]+deg[v];
    vector<i64> pos(g.xadj.begin(),g.xadj.end());
    vector<i32> a(g.xadj[n]); vector<i64> w(g.xadj[n]);
    for(i64 i=0;i<m;i++){ if(src[i]==dst[i]) continue; i64 e=ew.empty()?1:ew[i];
        a[pos[src[i]]]=dst[i]; w[pos[src[i]]++]=e; a[pos[dst[i]]]=src[i]; w[pos[dst[i]]++]=e; }
    // dedupe parallel edges per vertex (sort by neighbour, sum weights)
    g.adj.reserve(a.size()); g.ewgt.reserve(a.size());
    vector<i64> nx(n+1,0);
    for(i64 v=0; v<n; v++){
        i64 s=g.xadj[v], t=g.xadj[v+1];
        vector<pair<i32,i64>> nb; nb.reserve(t-s);
        for(i64 p=s;p<t;p++) nb.push_back({a[p],w[p]});
        sort(nb.begin(),nb.end());
        for(size_t k=0;k<nb.size();){ size_t j=k; i64 acc=0; while(j<nb.size()&&nb[j].first==nb[k].first){ acc+=nb[j].second; j++; }
            g.adj.push_back(nb[k].first); g.ewgt.push_back(acc); k=j; }
        nx[v+1]=(i64)g.adj.size();
    }
    g.xadj=nx;
    return g;
}

// HEM coarsening: returns coarser graph + cmap (fine vertex -> coarse vertex)
static Graph coarsen(const Graph& g, vector<i64>& cmap, i64 maxvw, bool use_2hop){
    i64 n=g.n; vector<i32> match(n,-1); cmap.assign(n,-1);
    vector<i64> order(n); iota(order.begin(),order.end(),0);
    // visit lower-degree first (tends to match leaves -> better balance)
    stable_sort(order.begin(),order.end(),[&](i64 a,i64 b){ return (g.xadj[a+1]-g.xadj[a])<(g.xadj[b+1]-g.xadj[b]); });
    // pass 1: HEM (heavy-edge matching, weight-capped)
    for(i64 idx=0; idx<n; idx++){ i64 v=order[idx]; if(match[v]>=0) continue;
        i64 best=-1, bestw=-1;
        for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++){ i32 u=g.adj[p];
            // METIS-style: refuse a match that would make the supernode too heavy -> keeps enough
            // coarse vertices for initial_kway to BALANCE (the hub-graph imbalance fix).
            if(match[u]<0 && u!=v && g.ewgt[p]>bestw && g.vwgt[v]+g.vwgt[u]<=maxvw){ bestw=g.ewgt[p]; best=u; } }
        if(best>=0){ match[v]=(i32)best; match[best]=(i32)v; }
    }
    // pass 2 (optional): 2-HOP matching -- pair still-unmatched vertices that SHARE a neighbor
    // (collapses hub fans where HEM stalls on power-law graphs; KaMinPar-style). Measured on real
    // data: clear WIN for the attribute-weighted mode (wiki-talk aware cut -24%, jodie -10..0%),
    // mixed for unit weights -> the caller enables it for ATTRIBUTE runs (a choice, not a replace).
    if(use_2hop) for(i64 u=0; u<n; u++){
        i64 prev=-1;
        for(i64 p=g.xadj[u]; p<g.xadj[u+1]; p++){ i32 w=g.adj[p];
            if(match[w]>=0 || (i64)w==u) continue;
            if(prev<0){ prev=w; continue; }
            if(g.vwgt[prev]+g.vwgt[w]<=maxvw){ match[prev]=(i32)w; match[w]=(i32)prev; prev=-1; }
            else prev=w;
        }
    }
    // number supernodes (pairs share one id; leftovers are singletons)
    i64 nc=0;
    for(i64 v=0; v<n; v++){
        if(cmap[v]>=0) continue;
        if(match[v]>=0 && match[v]!=(i32)v){ cmap[v]=cmap[match[v]]=nc++; }
        else cmap[v]=nc++;
    }
    // build coarse graph
    Graph c; c.n=nc; c.vwgt.assign(nc,0);
    for(i64 v=0; v<n; v++) c.vwgt[cmap[v]]+=g.vwgt[v];
    c.totalw=g.totalw;
    // accumulate coarse edges (map fine neighbours through cmap, sum weights, drop intra-supernode)
    vector<i32> csrc, cdst; vector<i64> cew; csrc.reserve(g.adj.size()); cdst.reserve(g.adj.size()); cew.reserve(g.adj.size());
    for(i64 v=0; v<n; v++){ i64 cv=cmap[v];
        for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++){ i64 cu=cmap[g.adj[p]]; if(cu!=cv && v<g.adj[p]){ csrc.push_back((i32)cv); cdst.push_back((i32)cu); cew.push_back(g.ewgt[p]); } }
    }
    Graph cg=build_csr(nc,csrc,cdst,cew); cg.vwgt=c.vwgt; cg.totalw=c.totalw;
    return cg;
}

// initial k-way on a small graph: greedy weighted growth with balance cap
static vector<i32> initial_kway(const Graph& g, i32 D, const vector<double>& cap, u64 seed=0){
    i64 n=g.n; vector<i32> part(n,-1); vector<i64> load(D,0);
    vector<i64> ord(n); iota(ord.begin(),ord.end(),0);
    if(seed){ // multi-start: shuffle ties before the weight sort (seed 0 = original deterministic order)
        u64 s=seed; for(i64 i=n-1;i>0;i--){ s^=s<<13; s^=s>>7; s^=s<<17; swap(ord[i],ord[s%(u64)(i+1)]); } }
    stable_sort(ord.begin(),ord.end(),[&](i64 a,i64 b){ return g.vwgt[a]>g.vwgt[b]; });
    vector<i64> gain(D,0); vector<i32> touched;
    for(i64 idx=0; idx<n; idx++){ i64 v=ord[idx];
        for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++){ i32 u=g.adj[p]; if(part[u]>=0){ if(gain[part[u]]==0) touched.push_back(part[u]); gain[part[u]]+=g.ewgt[p]; } }
        i32 best=-1; i64 bg=-1;
        for(i32 d=0; d<D; d++){ if(load[d]+g.vwgt[v]>cap[d]) continue;   // hard-skip full parts (per-part cap)
            if(best<0 || gain[d]>bg || (gain[d]==bg && load[d]<load[best])){ bg=gain[d]; best=d; } }
        // all full -> least RELATIVE-loaded (load/cap): with uniform caps this is plain least-loaded
        if(best<0){ best=0; for(i32 d=1;d<D;d++) if(load[d]*cap[best] < load[best]*cap[d]) best=d; }
        part[v]=best; load[best]+=g.vwgt[v];
        for(i32 d:touched) gain[d]=0; touched.clear();
    }
    return part;
}

// greedy boundary k-way FM refinement (a few passes; balance-capped; monotone cut descent)
static void refine_kway(const Graph& g, vector<i32>& part, i32 D, const vector<double>& cap, int passes){
    i64 n=g.n; vector<i64> load(D,0); for(i64 v=0;v<n;v++) load[part[v]]+=g.vwgt[v];
    vector<i64> conn(D,0); vector<i32> touched;
    vector<i64> boundary; boundary.reserve(1024);
    for(int it=0; it<passes; it++){
        // BOUNDARY-ONLY sweep: interior vertices can never gain (all neighbours co-resident),
        // so scan once for boundary vertices and refine just those -- much cheaper per pass,
        // which buys MORE passes at the same budget (the hub-graph cut-quality lever).
        boundary.clear();
        for(i64 v=0; v<n; v++){ i32 cur=part[v];
            for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++) if(part[g.adj[p]]!=cur){ boundary.push_back(v); break; } }
        i64 moves=0;
        for(i64 b=0; b<(i64)boundary.size(); b++){ i64 v=boundary[b]; i32 cur=part[v];
            for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++){ i32 d=part[g.adj[p]]; if(conn[d]==0) touched.push_back(d); conn[d]+=g.ewgt[p]; }
            i32 best=cur; i64 bestgain=0; i64 curconn=conn[cur];
            for(i32 d: touched){ if(d==cur) continue; if(load[d]+g.vwgt[v]>cap[d]) continue;
                i64 gn=conn[d]-curconn; if(gn>bestgain){ bestgain=gn; best=d; } }
            if(best!=cur){ load[cur]-=g.vwgt[v]; load[best]+=g.vwgt[v]; part[v]=best; moves++; }
            for(i32 d:touched) conn[d]=0; touched.clear();
        }
        if(moves==0) break;
    }
}

// recursive-bisection initial partition (METIS-style): split the coarsest vertex set 2-ways with
// an FM-refined bipartition, recursing until D parts. Stronger init than greedy growth on graphs
// with global structure (citation cores); added as ONE MORE multi-start candidate, not a replace.
static void rb_split(const Graph& g, vector<i32>& part, vector<i64>& vset, i32 p0, i32 Dk,
                     const vector<double>& cap){
    if(Dk==1){ for(i64 v: vset) part[v]=p0; return; }
    i32 DL=Dk/2, DR=Dk-DL;
    double capL=0, capR=0;
    for(i32 d=0; d<DL; d++) capL+=cap[p0+d];
    for(i32 d=DL; d<Dk; d++) capR+=cap[p0+d];
    // induced subgraph on vset
    i64 ns=vset.size();
    vector<i64> gidx(g.n,-1); for(i64 i=0;i<ns;i++) gidx[vset[i]]=i;
    Graph s; s.n=ns; s.vwgt.resize(ns); s.xadj.assign(ns+1,0);
    i64 totw=0;
    for(i64 i=0;i<ns;i++){ s.vwgt[i]=g.vwgt[vset[i]]; totw+=s.vwgt[i];
        for(i64 p=g.xadj[vset[i]]; p<g.xadj[vset[i]+1]; p++) if(gidx[g.adj[p]]>=0) s.xadj[i+1]++; }
    for(i64 i=0;i<ns;i++) s.xadj[i+1]+=s.xadj[i];
    s.adj.resize(s.xadj[ns]); s.ewgt.resize(s.xadj[ns]); s.totalw=totw;
    { vector<i64> pos(s.xadj.begin(),s.xadj.end());
      for(i64 i=0;i<ns;i++) for(i64 p=g.xadj[vset[i]]; p<g.xadj[vset[i]+1]; p++){
          i64 j=gidx[g.adj[p]]; if(j>=0){ s.adj[pos[i]]=(i32)j; s.ewgt[pos[i]++]=g.ewgt[p]; } } }
    // greedy BFS growth to the LEFT target weight, then 2-way FM refine
    double tgtL=(double)totw*capL/max(1e-12,capL+capR);
    vector<i32> bp(ns,1); vector<char> seen(ns,0); vector<i64> q; q.reserve(ns);
    i64 wL=0, qh=0;
    for(i64 st=0; st<ns && (double)wL<tgtL; st++){
        if(seen[st]) continue; seen[st]=1; q.push_back(st);
        while(qh<(i64)q.size() && (double)wL<tgtL){ i64 v=q[qh++];
            bp[v]=0; wL+=s.vwgt[v];
            for(i64 p=s.xadj[v]; p<s.xadj[v+1]; p++){ i32 u=s.adj[p]; if(!seen[u]){ seen[u]=1; q.push_back(u); } } }
    }
    vector<double> cap2={tgtL*1.05, ((double)totw-tgtL)*1.05};
    refine_kway(s, bp, 2, cap2, 12);
    vector<i64> vL, vR;
    for(i64 i=0;i<ns;i++) (bp[i]==0?vL:vR).push_back(vset[i]);
    rb_split(g, part, vL, p0, DL, cap);
    rb_split(g, part, vR, p0+DL, DR, cap);
}

static vector<i32> initial_rb(const Graph& g, i32 D, const vector<double>& cap){
    vector<i32> part(g.n,0); vector<i64> all(g.n);
    iota(all.begin(), all.end(), 0);
    rb_split(g, part, all, 0, D, cap);
    return part;
}

// COMMUNICATION-VOLUME refinement (the executor's real objective): total halo rows =
// sum_u #{parts p != part(u) that contain a neighbor of u} -- each (receiving part, remote row)
// pair counts ONCE, which is exactly the feature rows a GPU copies per layer. Edge-cut over-counts
// parallel boundary edges, so FM on edge-cut can lose steps while "winning" the cut. We maintain
// cnt[u][d] = #neighbors of u in part d (N*D i32; fine for D<=64) for O(1) move deltas.
static void refine_volume(const Graph& g, vector<i32>& part, i32 D, const vector<double>& cap, int passes){
    i64 n=g.n;
    if(n*(i64)D > (i64)2'000'000'000) return;               // memory guard: fall back silently
    vector<i32> cnt((size_t)n*D, 0);
    vector<i64> load(D,0);
    for(i64 v=0;v<n;v++){ load[part[v]]+=g.vwgt[v];
        for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++) cnt[(size_t)g.adj[p]*D + part[v]]++; }
    vector<i64> boundary; boundary.reserve(1024);
    for(int it=0; it<passes; it++){
        boundary.clear();
        for(i64 v=0; v<n; v++){ i32 cur=part[v]; i32* cv=&cnt[(size_t)v*D];
            for(i32 d=0; d<D; d++) if(d!=cur && cv[d]>0){ boundary.push_back(v); break; } }
        i64 moves=0;
        for(i64 b=0; b<(i64)boundary.size(); b++){ i64 v=boundary[b]; i32 a=part[v];
            i32* cv=&cnt[(size_t)v*D];
            // c(v) if homed at x: #{d != x : cv[d]>0}
            i32 nz=0; for(i32 d=0; d<D; d++) if(cv[d]>0) nz++;
            i32 best=a; i64 bestdelta=0;
            for(i32 d=0; d<D; d++){
                if(d==a || cv[d]==0) continue;               // only move toward parts we touch
                if(load[d]+g.vwgt[v]>cap[d]) continue;
                i64 delta = (i64)(nz - (cv[d]>0?1:0)) - (i64)(nz - (cv[a]>0?1:0)); // c(v) change
                for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++){ i32 u=g.adj[p]; i32 pu=part[u];
                    i32* cu=&cnt[(size_t)u*D];
                    if(cu[a]==1 && a!=pu) delta--;           // u stops receiving from part a
                    if(cu[d]==0 && d!=pu) delta++;           // u starts being received by part d
                }
                if(delta<bestdelta || (delta==bestdelta && d!=a && load[d]*1.0/cap[d] < load[best]*1.0/cap[best] && best!=a)){
                    bestdelta=delta; best=d; }
            }
            if(best!=a && bestdelta<0){
                load[a]-=g.vwgt[v]; load[best]+=g.vwgt[v]; part[v]=best;
                for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++){ i32 u=g.adj[p];
                    cnt[(size_t)u*D + a]--; cnt[(size_t)u*D + best]++; }
                moves++;
            }
        }
        if(moves==0) break;
    }
}

// STRAGGLER-AWARE (makespan) refinement: the executor's step time is NOT total volume but
//   max_d [ R[d]*F*FEATURE_ROW_BYTES/link + I[d]*BYTES_PER_EDGE_TRAVERSAL/bw ]   (arrange.py model)
// where R[d] = #distinct remote rows device d RECEIVES per layer (its halo, exactly what run_cell
// copies) and I[d] = incident edge traversals of d's vertices (its SpMM nnz). refine_volume
// minimizes sum_d R[d]; METIS can still win the STEP by spreading R more evenly (the measured
// stackoverflow 8xRTX6000 loss: fewer total halo rows, slower step). This pass does monotone
// MAKESPAN descent: only vertices homed on the current ARGMAX-cost device may move, only to a
// feasible part, and only if the max over the two affected devices strictly drops. Costs are
// scored as cost_d = alpha*R[d] + I[d] with alpha = (row cost)/(edge-traversal cost); measured on
// 8xRTX6000 (results/gpu_e2e_prof.csv): ~85ns/halo row over PCIe vs ~0.7ns/edge cuSPARSE SpMM
// -> default alpha=128, override via env ZORD_MAKESPAN_ALPHA (=0 disables). Additional pass after
// the existing descents (a choice, not a replace).
// PARETO CONSTRAINT (measured, job 144006: unconstrained max-descent raised TOTAL halo rows and
// the realized step -- the shared PCIe pays for the SUM as well as the MAX): a move v: a->b only
// changes R[a],R[b] (a receive set is per-device distinct rows; devices outside {a,b} cannot gain
// or lose a received row), so the total-volume delta is exactly daR+dbR. Accept only moves with
// daR+dbR <= 0 AND a strict drop of max(cost_a,cost_b): total volume never rises, the straggler
// strictly shrinks -- the pass improves the max or stops, and can never regress the sum.
// slack (env ZORD_MAKESPAN_SLACK, default 0) relaxes this to a CUMULATIVE budget: total volume
// may rise by at most slack * initial_total across the whole pass (0 = strict Pareto).
static void refine_makespan(const Graph& g, vector<i32>& part, i32 D, const vector<double>& cap,
                            int sweeps, double alpha, double slack){
    i64 n=g.n;
    if(D<2 || alpha<=0) return;
    if(n*(i64)D > (i64)2'000'000'000) return;               // memory guard: fall back silently
    vector<i32> cnt((size_t)n*D, 0);
    vector<i64> load(D,0), R(D,0), I(D,0), wdeg(n,0);
    for(i64 v=0; v<n; v++){ i64 s=0;
        for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++){ s+=g.ewgt[p]; cnt[(size_t)g.adj[p]*D+part[v]]++; }
        wdeg[v]=s; load[part[v]]+=g.vwgt[v]; I[part[v]]+=s;   // ewgt sums = executor nnz (unit-weight calls)
    }
    for(i64 v=0; v<n; v++){ i32 pv=part[v]; i32* cv=&cnt[(size_t)v*D];
        for(i32 d=0; d<D; d++) if(d!=pv && cv[d]>0) R[d]++; }
    auto cost=[&](i32 d){ return alpha*(double)R[d] + (double)I[d]; };
    i32 a0=0; for(i32 d=1; d<D; d++) if(cost(d)>cost(a0)) a0=d;
    double max0=cost(a0);
    i64 tot0=0; for(i32 d=0; d<D; d++) tot0+=R[d];
    i64 budget=(i64)(slack*(double)tot0), tot_delta=0;      // cumulative volume-rise budget
    i64 moved_total=0; vector<i64> cand; cand.reserve(1<<16);
    for(int it=0; it<sweeps; it++){
        i32 a=0; for(i32 d=1; d<D; d++) if(cost(d)>cost(a)) a=d;   // current straggler
        double curmax=cost(a);
        cand.clear();                                             // its boundary, heaviest first
        for(i64 v=0; v<n; v++) if(part[v]==a){ i32* cv=&cnt[(size_t)v*D];
            for(i32 d=0; d<D; d++) if(d!=a && cv[d]>0){ cand.push_back(v); break; } }
        stable_sort(cand.begin(), cand.end(), [&](i64 x, i64 y){ return wdeg[x]>wdeg[y]; });
        i64 moves=0; bool argmax_changed=false;
        for(i64 ci=0; ci<(i64)cand.size() && !argmax_changed; ci++){ i64 v=cand[ci];
            if(part[v]!=a) continue;
            i32* cv=&cnt[(size_t)v*D];
            i32 best=-1; double bestmax=curmax, bestb=0; i64 daR_best=0, dbR_best=0, besttot=0;
            for(i32 b=0; b<D; b++){
                if(b==a || cv[b]==0) continue;                    // only toward parts v touches
                if(load[b]+g.vwgt[v]>cap[b]) continue;
                i64 daR=(cv[a]>0)?1:0;                            // v becomes a row a must receive
                i64 dbR=(cv[b]>0)?-1:0;                           // v stops being a halo row for b
                for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++){ i32 u=g.adj[p]; i32 pu=part[u];
                    i32* cu=&cnt[(size_t)u*D];
                    if(cu[a]==1 && pu!=a) daR--;                  // u no longer received by a
                    if(cu[b]==0 && pu!=b) dbR++;                  // u newly received by b
                }
                if(tot_delta+daR+dbR>budget) continue;            // PARETO: cumulative total capped
                double na=alpha*(double)(R[a]+daR)+(double)(I[a]-wdeg[v]);
                double nb=alpha*(double)(R[b]+dbR)+(double)(I[b]+wdeg[v]);
                double nm=max(na,nb);
                if(nm<bestmax-1e-9 || (best>=0 && fabs(nm-bestmax)<=1e-9 &&
                                       (daR+dbR<besttot || (daR+dbR==besttot && nb<bestb)))){
                    bestmax=nm; best=b; bestb=nb; daR_best=daR; dbR_best=dbR; besttot=daR+dbR; }
            }
            if(best<0) continue;
            i32 b=best;
            load[a]-=g.vwgt[v]; load[b]+=g.vwgt[v];
            I[a]-=wdeg[v]; I[b]+=wdeg[v]; R[a]+=daR_best; R[b]+=dbR_best;
            tot_delta+=daR_best+dbR_best;
            part[v]=b;
            for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++){ i32 u=g.adj[p];
                cnt[(size_t)u*D+a]--; cnt[(size_t)u*D+b]++; }
            moves++; moved_total++;
            i32 a2=0; for(i32 d=1; d<D; d++) if(cost(d)>cost(a2)) a2=d;
            curmax=cost(a2);
            if(a2!=a) argmax_changed=true;                        // straggler moved: rebuild list
        }
        if(moves==0) break;                                       // no strictly-improving move left
    }
    i32 a1=0; for(i32 d=1; d<D; d++) if(cost(d)>cost(a1)) a1=d;
    fprintf(stderr,"[ml] MAKESPAN alpha=%.1f slack=%.3f moves=%lld maxcost %.4g -> %.4g dvol=%lld"
            " (dev %d: R=%lld I=%lld)\n", alpha,slack,(long long)moved_total,max0,cost(a1),
            (long long)tot_delta,a1,(long long)R[a1],(long long)I[a1]);
}

// explicit BALANCE pass (skewed-vwgt fix): move boundary vertices OUT of overfull parts into the
// least-loaded fitting part with MINIMUM cut damage. Restores per-part caps that the loose
// intermediate-level refinement (cap*slack) intentionally allowed to drift.
static void balance_kway(const Graph& g, vector<i32>& part, i32 D, const vector<double>& cap){
    i64 n=g.n; vector<i64> load(D,0); for(i64 v=0;v<n;v++) load[part[v]]+=g.vwgt[v];
    vector<i64> conn(D,0); vector<i32> touched;
    for(int pass=0; pass<4; pass++){
        bool any=false;
        for(i64 v=0; v<n; v++){ i32 cur=part[v];
            if((double)load[cur] <= cap[cur]) continue;             // only drain OVERFULL parts
            for(i64 p=g.xadj[v]; p<g.xadj[v+1]; p++){ i32 d=part[g.adj[p]]; if(conn[d]==0) touched.push_back(d); conn[d]+=g.ewgt[p]; }
            i32 best=-1; i64 bestloss=(i64)1<<62;
            for(i32 d=0; d<D; d++){ if(d==cur) continue; if(load[d]+g.vwgt[v]>cap[d]) continue;
                i64 loss=conn[cur]-conn[d];                          // cut increase if v moves to d
                if(loss<bestloss || (loss==bestloss && best>=0 && load[d]*cap[best] < load[best]*cap[d])){ bestloss=loss; best=d; } }
            if(best>=0){ load[cur]-=g.vwgt[v]; load[best]+=g.vwgt[v]; part[v]=best; any=true; }
            for(i32 d:touched) conn[d]=0; touched.clear();
        }
        if(!any) break;
    }
}

static i64 edgecut(const Graph& g, const vector<i32>& part){
    i64 c=0;
    #pragma omp parallel for reduction(+:c) schedule(dynamic,8192)
    for(i64 v=0;v<g.n;v++){ i64 local=0; for(i64 p=g.xadj[v];p<g.xadj[v+1];p++){ i32 u=g.adj[p]; if(u>v && part[u]!=part[v]) local+=g.ewgt[p]; } c+=local; }
    return c;
}

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s in.bin out.bin\n",argv[0]); return 1; }
    double t0=now_s();
    FILE* f=fopen(argv[1],"rb"); if(!f){ fprintf(stderr,"[ml] open in fail\n"); return 1; }
    i64 N=0,M=0; auto rd=[&](void*p,size_t s,size_t k){ if(fread(p,s,k,f)!=k){ fprintf(stderr,"[ml] read fail\n"); exit(1);} };
    rd(&N,8,1); rd(&M,8,1);
    vector<i32> src(M),dst(M); rd(src.data(),4,M); rd(dst.data(),4,M);
    i32 D=0; double ub=1.03; i32 has_ratio=0; vector<double> ratio;
    rd(&D,4,1); if(fread(&ub,8,1,f)!=1) ub=1.03;
    // OPTIONAL per-part target RATIO (additive, heterogeneity-aware): cap[d] proportional to ratio[d]
    // (e.g. GPU throughput share) instead of equal -> balance COMPUTE across uneven devices, not node
    // count. Absent -> equal split (the original behaviour). A choice, not a replace.
    if(D>=1 && fread(&has_ratio,4,1,f)==1 && has_ratio){ ratio.resize(D);
        if(fread(ratio.data(),8,D,f)!=(size_t)D){ fprintf(stderr,"[ml] ratio read fail\n"); return 1; } }
    // OPTIONAL ATTRIBUTE weights (the piece the kernel was missing): vwgt[N] = per-node FEATURE BYTES
    // (F_v*4) so balance = FEATURE MEMORY, not node count; ewgt[M] = per-edge FEATURE BYTES (F_e) so the
    // min-cut minimizes the TRUE boundary feature COMM (cutting a heavy-F_e edge costs more) and HEM
    // coarsening keeps heavy-feature edges local. Absent -> unit weights (structural, original behaviour).
    i32 has_vwgt=0; vector<i64> vwin; i32 has_ewgt=0; vector<i64> ewin;
    if(fread(&has_vwgt,4,1,f)==1 && has_vwgt){ vwin.resize(N);
        if(fread(vwin.data(),8,N,f)!=(size_t)N){ fprintf(stderr,"[ml] vwgt read fail\n"); return 1; } }
    if(fread(&has_ewgt,4,1,f)==1 && has_ewgt){ ewin.resize(M);
        if(fread(ewin.data(),8,M,f)!=(size_t)M){ fprintf(stderr,"[ml] ewgt read fail\n"); return 1; } }
    // OPTIONAL WARM START (polish mode): an initial partition init[N] to REFINE directly on the
    // full graph (boundary FM + balance; no coarsening). Lets zord polish ANY baseline's partition
    // -- including METIS's -- under zord's byte objective; also the incremental/dynamic path.
    i32 has_init=0; vector<i32> initp;
    if(fread(&has_init,4,1,f)==1 && has_init){ initp.resize(N);
        if(fread(initp.data(),4,N,f)!=(size_t)N){ fprintf(stderr,"[ml] init read fail\n"); return 1; } }
    fclose(f);
    if(D<1) D=1;
    fprintf(stderr,"[ml] N=%lld M=%lld D=%d ub=%.3f ratio=%s attr=%s (read %.2fs)\n",(long long)N,(long long)M,D,ub,
            has_ratio?"custom":"equal",(has_vwgt||has_ewgt)?"feat-weighted":"structural",now_s()-t0);

    Graph g0=build_csr(N,src,dst, has_ewgt?ewin:vector<i64>{});   // ewgt=F_e -> feature-comm-weighted cut
    if(has_vwgt){ g0.vwgt=vwin; i64 tw=0; for(i64 v=0;v<N;v++) tw+=vwin[v]; g0.totalw=(tw>0?tw:N); } // vwgt=F_v -> feature-MEMORY balance
    vector<i32>().swap(src); vector<i32>().swap(dst);

    // ---- POLISH mode: refine the provided partition directly (no coarsening) ----
    if(has_init){
        vector<double> capP(D), tgtP(D);
        if(has_ratio){ double s=0; for(i32 d=0;d<D;d++) s+=(ratio[d]>0?ratio[d]:0.0);
            if(s<=0){ for(i32 d=0;d<D;d++) ratio[d]=1.0; s=D; }
            for(i32 d=0; d<D; d++){ tgtP[d]=(double)g0.totalw*(ratio[d]>0?ratio[d]:0.0)/s; capP[d]=ceil(tgtP[d])*ub; } }
        else { double t=ceil((double)g0.totalw/(double)D); for(i32 d=0;d<D;d++){ tgtP[d]=t; capP[d]=t*ub; } }
        vector<i32> part(initp.begin(), initp.end());
        refine_kway(g0, part, D, capP, 12);      // edge-cut descent
        balance_kway(g0, part, D, capP);
        refine_volume(g0, part, D, capP, 8);     // HALO-ROW descent (total volume)
        double mk_alpha=128.0;                   // rows-vs-edges cost ratio (see refine_makespan)
        if(const char* e=getenv("ZORD_MAKESPAN_ALPHA")) mk_alpha=atof(e);
        double mk_slack=0.0;                     // cumulative total-volume rise budget (fraction)
        if(const char* e=getenv("ZORD_MAKESPAN_SLACK")) mk_slack=atof(e);
        refine_makespan(g0, part, D, capP, 8*D, mk_alpha, mk_slack);  // STRAGGLER descent

        i64 cut=edgecut(g0,part);
        vector<i64> load(D,0); for(i64 v=0;v<N;v++) load[part[v]]+=g0.vwgt[v];
        double bal=0; for(i32 d=0;d<D;d++){ double rel=(tgtP[d]>0)?(double)load[d]/tgtP[d]:0.0; if(rel>bal) bal=rel; }
        fprintf(stderr,"[ml] POLISH edgecut=%lld balance=%.4f (%.2fs)\n",(long long)cut,bal,now_s()-t0);
        fprintf(stderr,"STAT edgecut=%lld balance=%.6f parts=%d\n",(long long)cut,bal,D);
        FILE* o=fopen(argv[2],"wb"); fwrite(&N,8,1,o); fwrite(part.data(),4,N,o); fclose(o);
        return 0;
    }

    // ---- coarsen down, remembering each level's cmap to uncoarsen ----
    vector<Graph> levels; levels.push_back(g0);
    vector<vector<i64>> cmaps;
    const i64 COARSEST = max((i64)30*D, (i64)200);
    i64 maxvw = max((i64)1, (i64)(1.5*(double)g0.totalw/(double)COARSEST));   // cap supernode weight (balanceable)
    while(levels.back().n > COARSEST){
        vector<i64> cmap; Graph c=coarsen(levels.back(), cmap, maxvw, has_vwgt||has_ewgt);
        if(c.n >= (i64)(levels.back().n*0.95)){ cmaps.push_back(move(cmap)); levels.push_back(move(c)); break; } // stalled
        cmaps.push_back(move(cmap)); levels.push_back(move(c));
        if(levels.size()>60) break;
    }
    fprintf(stderr,"[ml] coarsened %zu levels: %lld -> %lld vertices (%.2fs)\n",
            levels.size(),(long long)g0.n,(long long)levels.back().n,now_s()-t0);

    // per-part capacity: equal (default) or proportional to ratio[] (heterogeneity-aware)
    vector<double> cap(D), tgt(D);
    if(has_ratio){ double s=0; for(i32 d=0;d<D;d++) s+=(ratio[d]>0?ratio[d]:0.0);
        if(s<=0){ for(i32 d=0;d<D;d++) ratio[d]=1.0; s=D; }
        for(i32 d=0; d<D; d++){ tgt[d]=(double)g0.totalw*(ratio[d]>0?ratio[d]:0.0)/s; cap[d]=ceil(tgt[d])*ub; } }
    else { double t=ceil((double)g0.totalw/(double)D); for(i32 d=0;d<D;d++){ tgt[d]=t; cap[d]=t*ub; } }
    // ---- initial partition on coarsest: MULTI-START (coarsest graph is tiny -> nearly free).
    //      4 seeded greedy growths + refine each; keep the min-cut start. ----
    vector<i32> part; i64 bestcut=-1;
    for(u64 s=0; s<5; s++){
        // starts 0-3: seeded greedy growth; start 4: recursive bisection (METIS-style) -- the
        // multi-start portfolio picks whichever wins on THIS graph (more solutions, not replace).
        vector<i32> cand = (s<4) ? initial_kway(levels.back(), D, cap, s)
                                 : initial_rb(levels.back(), D, cap);
        refine_kway(levels.back(), cand, D, cap, 8);
        i64 c = edgecut(levels.back(), cand);
        if(bestcut<0 || c<bestcut){ bestcut=c; part.swap(cand); }
    }

    // ---- uncoarsen + refine: LOOSE caps (x1.08) on intermediate levels give FM room to chase the
    //      cut under skewed vwgt; the FINEST level refines under exact caps + an explicit balance
    //      pass + one re-refine (the METIS-style relax-then-restore schedule). ----
    vector<double> capL(D); for(i32 d=0;d<D;d++) capL[d]=cap[d]*1.08;
    for(int L=(int)levels.size()-2; L>=0; L--){
        const vector<i64>& cmap=cmaps[L];           // maps level L vertex -> level L+1 coarse vertex
        vector<i32> fpart(levels[L].n);
        for(i64 v=0; v<levels[L].n; v++) fpart[v]=part[cmap[v]];
        part.swap(fpart);
        int passes=(levels[L].n>5'000'000)?6:12;   // boundary-only passes are cheap -> afford more
        if(L>0) refine_kway(levels[L], part, D, capL, passes);
        else { refine_kway(levels[L], part, D, cap, passes);
               balance_kway(levels[L], part, D, cap);
               refine_kway(levels[L], part, D, cap, 2); }
    }

    i64 cut=edgecut(g0,part);
    vector<i64> load(D,0); for(i64 v=0;v<N;v++) load[part[v]]+=g0.vwgt[v];
    double bal=0; for(i32 d=0;d<D;d++){ double rel=(tgt[d]>0)?(double)load[d]/tgt[d]:0.0; if(rel>bal) bal=rel; }
    i64 mx=0; for(i32 d=0;d<D;d++) mx=max(mx,load[d]); (void)mx;
    fprintf(stderr,"[ml] edgecut=%lld balance=%.4f (%.2fs)\n",(long long)cut,bal,now_s()-t0);
    fprintf(stderr,"STAT edgecut=%lld balance=%.6f parts=%d\n",(long long)cut,bal,D);

    FILE* o=fopen(argv[2],"wb"); if(!o){ fprintf(stderr,"[ml] open out fail\n"); return 1; }
    fwrite(&N,8,1,o); fwrite(part.data(),4,N,o); fclose(o);
    return 0;
}
