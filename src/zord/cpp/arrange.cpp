// zord ARRANGE -- the adaptive-corner partitioner, in C++ (the performance core).
// =====================================================================================
// Build:  g++ -O3 -std=c++17 -fopenmp src/zord/cpp/arrange.cpp -o build/arrange
//
// WHY C++: partition/arrange.py builds every candidate's cut/incident/comm metrics and the
// 9-quantile dense-core sweep in numpy (np.unique/np.bincount over 2*E rows PER candidate).
// At 100M edges that is the bottleneck. This binary ports the EXACT semantics of arrange.py
// (predict_ms, edgecut_metrics, replicate_core_metrics, lpa_edgecut, temporal_partition, the
// feasibility test, the F_v feature-weighted folds, the balance gate, the adaptive pick) into
// C++. Graph kernels deg/lpa_rank/coreness are computed by graph_algos (C++) and passed IN.
//
// PROCESS-only: output is a vertex->device assignment (+ optional replicated core mask); a
// placement never changes WHAT is computed, only WHERE -- same data+model => same result.
//
// BINARY INPUT (LE): i64 N,E,D,S ; f64 link_gbps,F ; i32 has_fv,seed ;
//   i32 src[E],dst[E],snap[E] ; i64 deg[N],lpa_rank[N],core_val[N] ; f64 bw[D],cap_bytes[D] ;
//   f64 fv[N] (only if has_fv).
// BINARY OUTPUT (LE): i64 N ; i32 dev[N] ; i32 has_core ; i8 core[N] (if has_core) ;
//   i64 D ; f64 incident[D],comm_raw[D],counts[D],inc_fold[D],comm_fold[D],featb[D] ;
//   f64 makespan,cut ; i64 extra_core_rows.
//   (incident,comm_raw = RAW per-device work the planner folds by F for the scalar path;
//    inc_fold,comm_fold = feature-weighted folds for the F_v path; featb = actual feat bytes.)
// stderr: "STAT <name> cut=.. makespan=.. feasible=.. inc_imb=.." per candidate + "WINNER ..".
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <string>
#include <chrono>
using namespace std;
using i64 = int64_t; using i32 = int32_t; using u64 = uint64_t; using i8 = int8_t;

static const double BYTES_PER_EDGE_TRAVERSAL = 4.0, N_GATHERS = 2.0;
static const double FEATURE_ROW_BYTES = 4.0, BYTES_PER_EDGE_RESIDENT = 20.0;
static const i64 VERTEXCUT_FULL_SWEEP_MAX_EDGES = 5'000'000;
static const double Q_FULL[]   = {0.70,0.80,0.88,0.93,0.96,0.98,0.99,0.995,0.999};
static const double Q_COARSE[] = {0.80,0.93,0.98,0.99,0.999};
static double now_s(){ return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }

struct Cand {
    string name; vector<i32> dev; vector<i8> core; bool has_core=false;
    double cut=0, makespan=0, inc_imb=0; i64 extra_core_rows=0; bool feasible=false, is_floor=false;
    vector<double> incident, comm_raw, counts, inc_fold, comm_fold, featb;   // winner's complete plan
};

int main(int argc, char** argv){
    if(argc<3){ fprintf(stderr,"usage: %s in.bin out.bin\n",argv[0]); return 1; }
    double t0=now_s();
    FILE* f=fopen(argv[1],"rb"); if(!f){ fprintf(stderr,"[arrange] open in fail\n"); return 1; }
    i64 N=0,E=0,D=0,S=0; double link=0,Fd=0; i32 has_fv=0,seed=0;
    auto rd=[&](void*p,size_t sz,size_t n){ if(fread(p,sz,n,f)!=n){ fprintf(stderr,"[arrange] read fail\n"); exit(1);} };
    rd(&N,8,1);rd(&E,8,1);rd(&D,8,1);rd(&S,8,1); rd(&link,8,1);rd(&Fd,8,1); rd(&has_fv,4,1);rd(&seed,4,1);
    if(D<1)D=1; if(S<1)S=1;
    vector<i32> src(E),dst(E),snp(E); rd(src.data(),4,E);rd(dst.data(),4,E);rd(snp.data(),4,E);
    vector<i64> deg(N),rnk(N),core_val(N); rd(deg.data(),8,N);rd(rnk.data(),8,N);rd(core_val.data(),8,N);
    vector<double> bw(D),cap(D); rd(bw.data(),8,D);rd(cap.data(),8,D);
    vector<double> fv; if(has_fv){ fv.resize(N); rd(fv.data(),8,N);} fclose(f);
    const double F=Fd; const i64 E2=2*E;
    fprintf(stderr,"[arrange] N=%lld E=%lld D=%lld S=%lld link=%.4g F=%.0f fv=%d (%.2fs)\n",
            (long long)N,(long long)E,(long long)D,(long long)S,link,F,has_fv,now_s()-t0);
    auto A=[&](i64 i){ return i<E? src[i]:dst[i-E]; };
    auto B=[&](i64 i){ return i<E? dst[i]:src[i-E]; };

    auto makespan=[&](const vector<double>& inc,const vector<double>& comm)->double{
        double mx=0; for(i64 d=0;d<D;d++){
            double cp=inc[d]*BYTES_PER_EDGE_TRAVERSAL*N_GATHERS/(bw[d]*1e9)*1e3;
            double cm=comm[d]*FEATURE_ROW_BYTES*N_GATHERS/(max(link,1e-9)*1e9)*1e3;
            mx=max(mx,cp+cm);} return mx; };
    auto feasible=[&](const vector<double>& fb,const vector<double>& inc)->bool{
        for(i64 d=0;d<D;d++) if(fb[d]+inc[d]*BYTES_PER_EDGE_RESIDENT>cap[d]) return false; return true; };
    auto imb=[&](const vector<double>& inc)->double{
        double m=0; for(double x:inc)m+=x; m/=D; double mx=0; for(double x:inc)mx=max(mx,x); return mx/max(1e-9,m); };

    // EDGE-CUT candidate evaluation (single-home dev[v]).
    auto edgecut_eval=[&](const string& nm,const vector<i32>& dev,bool floor,Cand& o){
        o.name=nm; o.dev=dev; o.has_core=false; o.is_floor=floor; o.extra_core_rows=0;
        i64 cut=0;
        #pragma omp parallel for reduction(+:cut) schedule(static)   // OpenMP (selectable via OMP_NUM_THREADS; serial at 1)
        for(i64 i=0;i<E;i++) if(dev[src[i]]!=dev[dst[i]]) cut++;
        o.incident.assign(D,0.0); o.counts.assign(D,0.0);
        for(i64 v=0;v<N;v++){ o.incident[dev[v]]+=(double)deg[v]; o.counts[dev[v]]+=1.0; }
        // distinct (gathering-device, remote-neighbor) comm rows
        vector<u64> key; for(i64 i=0;i<E2;i++){ i32 a=A(i),b=B(i); if(dev[a]>=0&&dev[b]>=0&&dev[a]!=dev[b]) key.push_back((u64)dev[a]*(u64)N+(u64)b); }
        sort(key.begin(),key.end()); key.erase(unique(key.begin(),key.end()),key.end());
        o.comm_raw.assign(D,0.0); vector<double> comm_fw(D,0.0);
        for(u64 k:key){ i64 d=(i64)(k/(u64)N),b=(i64)(k%(u64)N); o.comm_raw[d]+=1.0; if(has_fv) comm_fw[d]+=fv[b]; }
        o.inc_fold.assign(D,0.0); o.featb.assign(D,0.0);
        if(!has_fv){ for(i64 d=0;d<D;d++){ o.inc_fold[d]=o.incident[d]*F; o.featb[d]=o.counts[d]*F*4.0; } }
        else{ for(i64 i=0;i<E2;i++) o.inc_fold[dev[A(i)]]+=fv[B(i)]; for(i64 v=0;v<N;v++) o.featb[dev[v]]+=fv[v]*4.0; }
        o.comm_fold.assign(D,0.0); for(i64 d=0;d<D;d++) o.comm_fold[d]= has_fv? comm_fw[d] : o.comm_raw[d]*F;
        o.cut=(double)cut; o.makespan=makespan(o.inc_fold,o.comm_fold); o.feasible=feasible(o.featb,o.incident); o.inc_imb=imb(o.incident);
    };

    std::mt19937_64 rng((uint64_t)seed+1);
    auto vertexcut_eval=[&](const vector<i8>& core,const vector<i32>& dev_p,Cand& o){
        o.name="vertex-cut(k-core)"; o.dev=dev_p; o.core=core; o.has_core=true;
        i64 core_size=0; for(i64 v=0;v<N;v++) core_size+=core[v];
        o.incident.assign(D,0.0); o.inc_fold.assign(D,0.0);
        vector<u64> key; double core_feat_total=0; if(has_fv){ for(i64 v=0;v<N;v++) if(core[v]) core_feat_total+=fv[v]; }
        for(i64 i=0;i<E2;i++){ i32 a=A(i),b=B(i); bool ac=core[a],bc=core[b]; i32 land;
            if(!ac) land=dev_p[a]; else if(ac&&!bc) land=dev_p[b]; else land=(i32)(rng()%(u64)D);
            o.incident[land]+=1.0; if(has_fv) o.inc_fold[land]+=fv[b];
            if(!ac&&!bc&&dev_p[a]>=0&&dev_p[b]>=0&&dev_p[a]!=dev_p[b]) key.push_back((u64)dev_p[a]*(u64)N+(u64)b); }
        if(!has_fv) for(i64 d=0;d<D;d++) o.inc_fold[d]=o.incident[d]*F;
        i64 cut=0;
        #pragma omp parallel for reduction(+:cut) schedule(static)
        for(i64 i=0;i<E;i++) if(!core[src[i]]&&!core[dst[i]]&&dev_p[src[i]]!=dev_p[dst[i]]) cut++;
        sort(key.begin(),key.end()); key.erase(unique(key.begin(),key.end()),key.end());
        double reduce_count=(double)core_size*(double)(D-1)/(double)D, reduce_feat=core_feat_total*(double)(D-1)/(double)D;
        vector<double> dcnt(D,0.0),dfw(D,0.0); for(u64 k:key){ i64 d=(i64)(k/(u64)N),b=(i64)(k%(u64)N); dcnt[d]+=1.0; if(has_fv) dfw[d]+=fv[b]; }
        o.comm_raw.assign(D,0.0); o.comm_fold.assign(D,0.0);
        for(i64 d=0;d<D;d++){ o.comm_raw[d]=dcnt[d]+reduce_count; o.comm_fold[d]= has_fv? (dfw[d]+reduce_feat) : o.comm_raw[d]*F; }
        o.counts.assign(D,0.0); o.featb.assign(D,0.0);
        for(i64 v=0;v<N;v++) if(!core[v]){ o.counts[dev_p[v]]+=1.0; if(has_fv) o.featb[dev_p[v]]+=fv[v]*4.0; }
        for(i64 d=0;d<D;d++) o.counts[d]+=(double)core_size;
        if(has_fv){ double cf=0; for(i64 v=0;v<N;v++) if(core[v]) cf+=fv[v]*4.0; for(i64 d=0;d<D;d++) o.featb[d]+=cf; }
        else for(i64 d=0;d<D;d++) o.featb[d]=o.counts[d]*F*4.0;
        o.cut=(double)cut; o.makespan=makespan(o.inc_fold,o.comm_fold); o.feasible=feasible(o.featb,o.incident);
        o.inc_imb=imb(o.incident); o.extra_core_rows=core_size*(D-1);
    };

    // lpa_edgecut: walk lpa-rank order, split into D segments balanced by weight (equal or cap-proportional)
    auto lpa_edgecut=[&](const vector<double>& weight,const double* caps)->vector<i32>{
        vector<i64> r2n(N); for(i64 v=0;v<N;v++) r2n[rnk[v]]=v;
        vector<double> cum(N); double s=0; for(i64 r=0;r<N;r++){ s+=weight[r2n[r]]; cum[r]=s; } double tot=cum[N-1];
        vector<double> tg(D>1?D-1:0);
        if(caps==nullptr){ for(i64 k=1;k<D;k++) tg[k-1]=(double)k*tot/(double)D; }
        else{ double cs=0; for(i64 d=0;d<D;d++) cs+=caps[d]; double acc=0; for(i64 k=0;k<D-1;k++){ acc+=caps[k]/cs; tg[k]=acc*tot; } }
        vector<i64> cuts(tg.size()); for(size_t k=0;k<tg.size();k++) cuts[k]=lower_bound(cum.begin(),cum.end(),tg[k])-cum.begin();
        vector<i32> dev(N); for(i64 r=0;r<N;r++){ i64 g=upper_bound(cuts.begin(),cuts.end(),r)-cuts.begin(); if(g>=D)g=D-1; dev[r2n[r]]=(i32)g; }
        return dev;
    };
    auto split_by_work=[&](const vector<i64>& seq,const vector<double>& w)->vector<i32>{
        i64 n=seq.size(); vector<double> cum(n); double s=0; for(i64 i=0;i<n;i++){ s+=w[i]; cum[i]=s; }
        vector<i32> seg(n); if(s<=0){ for(i64 i=0;i<n;i++) seg[i]=(i32)min((i64)D-1,i*D/max((i64)1,n)); return seg; }
        vector<double> tg(D>1?D-1:0); for(i64 k=1;k<D;k++) tg[k-1]=(double)k*s/(double)D;
        vector<i64> cuts(tg.size()); for(size_t k=0;k<tg.size();k++) cuts[k]=lower_bound(cum.begin(),cum.end(),tg[k])-cum.begin();
        for(i64 i=0;i<n;i++){ i64 g=upper_bound(cuts.begin(),cuts.end(),i)-cuts.begin(); seg[i]=(i32)min(g,(i64)D-1); } return seg;
    };

    vector<Cand> C;
    vector<double> dw(N); for(i64 v=0;v<N;v++) dw[v]=(double)deg[v];
    { Cand c; edgecut_eval("edge-cut(hetero)", lpa_edgecut(dw,bw.data()), false, c); C.push_back(move(c)); }
    if(has_fv){ Cand c; edgecut_eval("edge-cut(feat-aware)", lpa_edgecut(fv,cap.data()), false, c); C.push_back(move(c)); }
    {   // dense-core vertex-cut sweep
        const double* Q=(E<=VERTEXCUT_FULL_SWEEP_MAX_EDGES)?Q_FULL:Q_COARSE; int nQ=(E<=VERTEXCUT_FULL_SWEEP_MAX_EDGES)?9:5;
        vector<i64> scv(core_val); sort(scv.begin(),scv.end()); Cand best; bool have=false;
        for(int qi=0;qi<nQ;qi++){
            i64 tau=max((i64)2, scv[(i64)(Q[qi]*(double)(N-1))]);
            vector<i8> core(N); i64 cs=0; for(i64 v=0;v<N;v++){ core[v]=core_val[v]>=tau; cs+=core[v]; }
            if(!(cs>0&&cs<N)) continue;
            vector<i64> per; for(i64 v=0;v<N;v++) if(!core[v]) per.push_back(v);
            sort(per.begin(),per.end(),[&](i64 a,i64 b){ return rnk[a]<rnk[b]; });
            vector<double> pw(per.size()); for(size_t i=0;i<per.size();i++) pw[i]=(double)deg[per[i]];
            vector<i32> seg=split_by_work(per,pw); vector<i32> dev_p(N,-1); for(size_t i=0;i<per.size();i++) dev_p[per[i]]=seg[i];
            Cand c; vertexcut_eval(core,dev_p,c); if(!c.feasible) continue;
            if(!have||c.makespan<best.makespan){ best=move(c); have=true; } }
        if(have) C.push_back(move(best));
    }
    { Cand c; edgecut_eval("spatial(PSS)", lpa_edgecut(dw,nullptr), false, c); C.push_back(move(c)); }
    {   vector<i64> first(N,S+1); for(i64 i=0;i<E;i++){ i32 s=snp[i]; if(s<first[src[i]])first[src[i]]=s; if(s<first[dst[i]])first[dst[i]]=s; }
        vector<i64> seq(N); iota(seq.begin(),seq.end(),0); stable_sort(seq.begin(),seq.end(),[&](i64 a,i64 b){ return first[a]<first[b]; });
        vector<double> w(N); for(i64 i=0;i<N;i++) w[i]=(double)deg[seq[i]]; vector<i32> seg=split_by_work(seq,w);
        vector<i32> dev(N); for(i64 i=0;i<N;i++) dev[seq[i]]=seg[i]; Cand c; edgecut_eval("temporal(PTS)",dev,false,c); C.push_back(move(c)); }
    { Cand c; edgecut_eval("cheap-cut(lpa-proxy)", lpa_edgecut(dw,nullptr), true, c); C.push_back(move(c)); }

    double gate=0.5*(double)D+0.5; int best=-1; double bcost=1e300;
    for(int i=0;i<(int)C.size();i++){ Cand& c=C[i];
        fprintf(stderr,"STAT %-22s cut=%.0f makespan=%.6g feasible=%d inc_imb=%.3f%s\n",
                c.name.c_str(),c.cut,c.makespan,(int)c.feasible,c.inc_imb,c.is_floor?" FLOOR":"");
        if(c.feasible && (c.inc_imb<=gate||c.is_floor) && c.makespan<bcost){ bcost=c.makespan; best=i; } }
    if(best<0){ double bi=1e300; for(int i=0;i<(int)C.size();i++) if(C[i].inc_imb<bi){ bi=C[i].inc_imb; best=i; } }
    Cand& W=C[best];
    fprintf(stderr,"WINNER %s makespan=%.6g cut=%.0f repl=%lld (%.2fs)\n",W.name.c_str(),W.makespan,W.cut,(long long)W.extra_core_rows,now_s()-t0);

    FILE* o=fopen(argv[2],"wb"); if(!o){ fprintf(stderr,"[arrange] open out fail\n"); return 1; }
    fwrite(&N,8,1,o); fwrite(W.dev.data(),4,N,o);
    i32 hc=W.has_core?1:0; fwrite(&hc,4,1,o); if(hc) fwrite(W.core.data(),1,N,o);
    fwrite(&D,8,1,o);
    fwrite(W.incident.data(),8,D,o); fwrite(W.comm_raw.data(),8,D,o); fwrite(W.counts.data(),8,D,o);
    fwrite(W.inc_fold.data(),8,D,o); fwrite(W.comm_fold.data(),8,D,o); fwrite(W.featb.data(),8,D,o);
    double mk=W.makespan,ct=W.cut; i64 ecr=W.extra_core_rows; fwrite(&mk,8,1,o); fwrite(&ct,8,1,o); fwrite(&ecr,8,1,o);
    fclose(o); return 0;
}
