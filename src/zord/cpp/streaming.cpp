// zord STREAMING partitioners (C++17): Fennel + LDG (edge-cut) + HDRF (vertex-cut).
// =====================================================================================
// Build:  g++ -O3 -std=c++17 -fopenmp src/zord/cpp/streaming.cpp -o build/streaming
//
// WHY: the multilevel partitioner (build/multilevel) is high-quality but multi-pass. For a
// graph that arrives as a STREAM (or is too big to coarsen in memory), a SINGLE-PASS streaming
// partitioner is the tool -- O(M), bounded extra memory, and the basis of "streaming arrange"
// (cut-one-send-one) and online/dynamic re-partitioning. We implement the three canonical ones:
//   mode 0 FENNEL (Tsourakakis et al. WSDM'14): assign each vertex (in arrival order) to the
//          part maximising  |N(v) ∩ P_p| - alpha*gamma*load_p^(gamma-1)  (gamma=1.5 default).
//   mode 1 LDG    (Stanton-Kliot KDD'12): assign v to argmax  |N(v) ∩ P_p| * (1 - load_p/C).
//   mode 2 HDRF   (Petroni et al. CIKM'15): STREAM EDGES, vertex-cut/edge-partition; replicate
//          the HIGHER-DEGREE endpoint, balance edge load. theta(w)=deg(w)/(deg(u)+deg(v)).
// PROCESS-only: a partition/edge-assignment is a result-preserving placement.
//
// BINARY INPUT (LE): i64 N,M ; i32 src[M],dst[M] ; i32 D ; i32 mode ; f64 param
//   (param = gamma[fennel,def 1.5] / slack[ldg,def 1.0] / lambda[hdrf,def 1.0])
// BINARY OUTPUT (LE): i32 mode ; if mode<2: i64 N, i32 part[N] ; else: i64 M, i32 epart[M]
// stderr: STAT edgecut/balance (fennel/ldg) or replication_factor/edge_balance (hdrf).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
using namespace std;
using i64=int64_t; using i32=int32_t; using u64=uint64_t;
static double now_s(){ return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s in.bin out.bin\n",argv[0]); return 1; }
    double t0=now_s();
    FILE* f=fopen(argv[1],"rb"); if(!f){ fprintf(stderr,"[stream] open in fail\n"); return 1; }
    i64 N=0,M=0; auto rd=[&](void*p,size_t s,size_t k){ if(fread(p,s,k,f)!=k){ fprintf(stderr,"[stream] read fail\n"); exit(1);} };
    rd(&N,8,1); rd(&M,8,1);
    vector<i32> src(M),dst(M); rd(src.data(),4,M); rd(dst.data(),4,M);
    i32 D=0,mode=0; double param=0; i32 has_order=0; vector<i32> norder;
    rd(&D,4,1); rd(&mode,4,1);
    if(fread(&param,8,1,f)!=1) param=0;
    // OPTIONAL arrival order (additive): stream vertices in this order (e.g. lpa_rank) for a much
    // better cut than id-order. Absent -> id-order (the original behaviour). A choice, not a replace.
    if(fread(&has_order,4,1,f)==1 && has_order){ norder.resize(N);
        if(fread(norder.data(),4,N,f)!=(size_t)N){ fprintf(stderr,"[stream] order read fail\n"); return 1; } }
    // OPTIONAL per-node weight vwgt[N] = FEATURE BYTES F_v (attribute-aware): balance FEATURE MEMORY
    // not node count (Fennel/LDG). Absent -> unit (count). HDRF (edge-cut) is unaffected.
    i32 has_vw=0; vector<i64> vw;
    if(fread(&has_vw,4,1,f)==1 && has_vw){ vw.resize(N);
        if(fread(vw.data(),8,N,f)!=(size_t)N){ fprintf(stderr,"[stream] vwgt read fail\n"); return 1; } }
    fclose(f);
    if(D<1) D=1;
    fprintf(stderr,"[stream] N=%lld M=%lld D=%d mode=%d param=%.3g (%.2fs)\n",(long long)N,(long long)M,D,mode,param,now_s()-t0);

    // degree (both for HDRF theta and as a fallback)
    vector<i64> deg(N,0); for(i64 i=0;i<M;i++){ if(src[i]!=dst[i]){ deg[src[i]]++; deg[dst[i]]++; } }

    if(mode==2){
        // ---- HDRF vertex-cut / edge partitioning (stream edges) ----
        if(D>64){ fprintf(stderr,"[stream] HDRF needs D<=64 (bitmask)\n"); return 1; }
        double lambda = (param>0)? param : 1.0;
        vector<u64> pmask(N,0);                 // which parts each vertex is replicated into
        vector<i64> size(D,0);                  // edges per part
        vector<i32> epart(M);
        i64 maxs=0, mins=0;
        for(i64 i=0;i<M;i++){ i32 u=src[i],v=dst[i];
            double du=(double)deg[u], dv=(double)deg[v], s=du+dv; double thu=(s>0?du/s:0.5), thv=1.0-thu;
            i32 best=0; double bestsc=-1e300;
            for(i32 p=0;p<D;p++){
                double crep=0;
                if(pmask[u]&(1ull<<p)) crep += 1.0+(1.0-thu);
                if(pmask[v]&(1ull<<p)) crep += 1.0+(1.0-thv);
                double cbal = lambda * (double)(maxs - size[p]) / (1.0 + (double)(maxs - mins));
                double sc = crep + cbal;
                if(sc>bestsc){ bestsc=sc; best=p; }
            }
            epart[i]=best; pmask[u]|=(1ull<<best); pmask[v]|=(1ull<<best); size[best]++;
            maxs=0; mins=size[0]; for(i32 p=0;p<D;p++){ maxs=max(maxs,size[p]); mins=min(mins,size[p]); }
        }
        i64 repl=0; for(i64 v=0;v<N;v++) repl += __builtin_popcountll(pmask[v]);
        double rf=(double)repl/(double)max((i64)1,N);
        double ebal=(double)maxs/((double)M/(double)D);
        fprintf(stderr,"[stream] HDRF replication_factor=%.4f edge_balance=%.4f (%.2fs)\n",rf,ebal,now_s()-t0);
        fprintf(stderr,"STAT mode=hdrf replication_factor=%.6f edge_balance=%.6f parts=%d\n",rf,ebal,D);
        FILE* o=fopen(argv[2],"wb"); fwrite(&mode,4,1,o); fwrite(&M,8,1,o); fwrite(epart.data(),4,M,o); fclose(o);
        return 0;
    }

    // ---- Fennel / LDG: stream VERTICES (id = arrival order); need neighbour parts -> CSR ----
    vector<i64> xadj(N+1,0); for(i64 v=0;v<N;v++) xadj[v+1]=xadj[v]+deg[v];
    vector<i32> adj(xadj[N]); { vector<i64> pos(xadj.begin(),xadj.end());
        for(i64 i=0;i<M;i++){ if(src[i]==dst[i]) continue; adj[pos[src[i]]++]=dst[i]; adj[pos[dst[i]]++]=src[i]; } }

    double gamma = (mode==0)? ((param>0)?param:1.5) : 0.0;
    double slack = (mode==1)? ((param>0)?param:1.0) : 0.0;
    double alpha = (mode==0)? ((double)M * pow((double)D, gamma-1.0) / pow((double)max((i64)1,N), gamma)) : 0.0;
    i64 totalw=0; if(has_vw){ for(i64 i=0;i<N;i++) totalw+=vw[i]; } if(totalw<=0) totalw=N;
    double C = (double)totalw/(double)D * (slack>0?slack:1.0);   // LDG capacity (weight units)
    double capF = (double)totalw/(double)D * 1.10;               // Fennel soft cap (weight units)

    vector<i32> part(N,-1); vector<i64> load(D,0);
    vector<double> cnt(D,0.0); vector<i32> touched;
    for(i64 idx=0; idx<N; idx++){ i64 v = has_order ? (i64)norder[idx] : idx;
        for(i64 p=xadj[v]; p<xadj[v+1]; p++){ i32 u=adj[p]; if(part[u]>=0){ if(cnt[part[u]]==0.0) touched.push_back(part[u]); cnt[part[u]]+=1.0; } }
        i32 best=-1; double bestsc=-1e300;
        for(i32 p=0;p<D;p++){
            double sc;
            if(mode==0){ if(load[p]>=capF && best>=0) continue; sc = cnt[p] - alpha*gamma*pow((double)load[p], gamma-1.0); }
            else       { double room=1.0-(double)load[p]/C; if(room<0) room=0; sc = cnt[p]*room - 1e-9*(double)load[p]; }
            if(sc>bestsc){ bestsc=sc; best=p; }
        }
        if(best<0){ best=0; for(i32 p=1;p<D;p++) if(load[p]<load[best]) best=p; }
        part[v]=best; load[best] += has_vw ? vw[v] : 1;   // accumulate FEATURE bytes when attribute-aware
        for(i32 p:touched) cnt[p]=0.0; touched.clear();
    }
    i64 cut=0; for(i64 i=0;i<M;i++) if(part[src[i]]!=part[dst[i]]) cut++;
    i64 mx=0; for(i32 p=0;p<D;p++) mx=max(mx,load[p]); double bal=(double)mx/((double)totalw/(double)D);
    fprintf(stderr,"[stream] %s edgecut=%lld balance=%.4f (%.2fs)\n", mode==0?"FENNEL":"LDG",(long long)cut,bal,now_s()-t0);
    fprintf(stderr,"STAT mode=%s edgecut=%lld balance=%.6f parts=%d\n", mode==0?"fennel":"ldg",(long long)cut,bal,D);
    FILE* o=fopen(argv[2],"wb"); fwrite(&mode,4,1,o); fwrite(&N,8,1,o); fwrite(part.data(),4,N,o); fclose(o);
    return 0;
}
