// zord SPECTRAL: Fiedler vector + algebraic connectivity lambda2 (C++17).
// =====================================================================================
// Build:  g++ -O3 -std=c++17 -fopenmp src/zord/cpp/spectral.cpp -o build/spectral
//
// WHY: THEORY.md's space-time lower bound L uses the per-snapshot Cheeger constant h(G_t), and
// the easy spectral direction h >= lambda2/2 makes L computable -- but the bound estimate was
// done with an approximation in Python. This binary computes the REAL algebraic connectivity
// lambda2 (2nd-smallest eigenvalue of the NORMALIZED Laplacian L_norm = I - D^-1/2 A D^-1/2) via
// deflated power iteration on the normalized adjacency A_hat = D^-1/2 A D^-1/2 (whose largest
// eigenvalue is 1 with eigenvector v1 = D^1/2 . 1; the 2nd largest mu2 gives lambda2 = 1 - mu2).
// It also returns the Fiedler bipartition (sign of the eigenvector) -> a spectral partitioner,
// and the Cheeger LOWER bound h_lb = lambda2/2 the duality-bound checker needs.
//
// BINARY INPUT (LE):  i64 N, M ; i32 src[M], dst[M] ; i32 iters (0 -> default 200) ; i32 seed
// BINARY OUTPUT (LE): i64 N ; f64 lambda2 ; f64 cheeger_lb ; i32 part[N] (0/1 Fiedler sign)
// stderr: STAT lambda2 cheeger_lb conductance(bipartition) iters.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
using namespace std;
using i64=int64_t; using i32=int32_t;
static double now_s(){ return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s in.bin out.bin\n",argv[0]); return 1; }
    double t0=now_s();
    FILE* f=fopen(argv[1],"rb"); if(!f){ fprintf(stderr,"[spec] open in fail\n"); return 1; }
    i64 N=0,M=0; auto rd=[&](void*p,size_t s,size_t k){ if(fread(p,s,k,f)!=k){ fprintf(stderr,"[spec] read fail\n"); exit(1);} };
    rd(&N,8,1); rd(&M,8,1);
    vector<i32> src(M),dst(M); rd(src.data(),4,M); rd(dst.data(),4,M);
    i32 iters=0,seed=0; rd(&iters,4,1); if(fread(&seed,4,1,f)!=1) seed=0;
    // OPTIONAL ATTRIBUTE weights vwgt[N] (per-node feature bytes F_v*4): the Fiedler bipartition
    // splits at the WEIGHTED median of the eigenvector order -> the two sides hold equal FEATURE
    // MEMORY, not equal node counts. Absent -> sign split (structural, original behaviour).
    i32 has_vw=0; vector<i64> vw;
    if(fread(&has_vw,4,1,f)==1 && has_vw){ vw.resize(N);
        if(fread(vw.data(),8,N,f)!=(size_t)N){ fprintf(stderr,"[spec] vwgt read fail\n"); return 1; } }
    fclose(f);
    if(iters<=0) iters=200;

    // undirected CSR (drop self-loops)
    vector<i64> deg(N,0); for(i64 i=0;i<M;i++) if(src[i]!=dst[i]){ deg[src[i]]++; deg[dst[i]]++; }
    vector<i64> xadj(N+1,0); for(i64 v=0;v<N;v++) xadj[v+1]=xadj[v]+deg[v];
    vector<i32> adj(xadj[N]); { vector<i64> pos(xadj.begin(),xadj.end());
        for(i64 i=0;i<M;i++){ if(src[i]==dst[i]) continue; adj[pos[src[i]]++]=dst[i]; adj[pos[dst[i]]++]=src[i]; } }

    vector<double> dsi(N,0.0);                          // D^{-1/2}
    for(i64 v=0;v<N;v++) dsi[v] = deg[v]>0? 1.0/sqrt((double)deg[v]) : 0.0;
    // v1 = D^{1/2} . 1, normalized -> top eigenvector of A_hat (eigenvalue 1)
    vector<double> v1(N); double nrm=0; for(i64 v=0;v<N;v++){ v1[v]=sqrt((double)deg[v]); nrm+=v1[v]*v1[v]; }
    nrm=sqrt(nrm); if(nrm<=0) nrm=1; for(i64 v=0;v<N;v++) v1[v]/=nrm;

    // x0: random, deflate v1, normalize
    std::mt19937_64 rng((uint64_t)seed+12345);
    vector<double> x(N), y(N);
    { double s=0; for(i64 v=0;v<N;v++){ x[v]=(double)((int64_t)(rng()%20000)-10000)/10000.0; }
      double pr=0; for(i64 v=0;v<N;v++) pr+=x[v]*v1[v]; for(i64 v=0;v<N;v++) x[v]-=pr*v1[v];
      for(i64 v=0;v<N;v++) s+=x[v]*x[v]; s=sqrt(s); if(s<=0)s=1; for(i64 v=0;v<N;v++) x[v]/=s; }

    // A_hat x = D^{-1/2} A D^{-1/2} x ; deflated power iteration for mu2 (2nd-largest eigenvalue)
    // A_hat x : per-row independent (out[v] written only by iter v; reads are shared read-only)
    // -> embarrassingly parallel, the dominant power-iteration cost. OpenMP across rows.
    auto apply=[&](const vector<double>& in, vector<double>& out){
        #pragma omp parallel for schedule(static)
        for(i64 v=0;v<N;v++){ double acc=0; double sv=dsi[v];
            for(i64 p=xadj[v];p<xadj[v+1];p++){ i32 u=adj[p]; acc += dsi[u]*in[u]; }
            out[v]=sv*acc; }
    };
    double mu2=0;
    for(int it=0; it<iters; it++){
        apply(x,y);
        double pr=0;
        #pragma omp parallel for reduction(+:pr) schedule(static)
        for(i64 v=0;v<N;v++) pr+=y[v]*v1[v];
        double s=0;
        #pragma omp parallel for reduction(+:s) schedule(static)
        for(i64 v=0;v<N;v++){ y[v]-=pr*v1[v]; s+=y[v]*y[v]; }                 // deflate v1 + ||y||^2
        s=sqrt(s); if(s<=1e-300) break;
        #pragma omp parallel for schedule(static)
        for(i64 v=0;v<N;v++) y[v]/=s;
        x.swap(y);
    }
    // Rayleigh quotient mu2 = x . A_hat x  (with v1 deflated, this is the 2nd eigenvalue)
    apply(x,y);
    { double pr=0;
      #pragma omp parallel for reduction(+:pr) schedule(static)
      for(i64 v=0;v<N;v++) pr+=y[v]*v1[v];
      #pragma omp parallel for schedule(static)
      for(i64 v=0;v<N;v++) y[v]-=pr*v1[v]; }
    #pragma omp parallel for reduction(+:mu2) schedule(static)
    for(i64 v=0;v<N;v++) mu2 += x[v]*y[v];
    double lambda2 = 1.0 - mu2; if(lambda2<0) lambda2=0;
    double cheeger_lb = lambda2/2.0;

    // Fiedler bipartition: sign split (structural) OR, with vwgt, split the eigenvector ORDER at the
    // weighted median so both sides carry equal FEATURE BYTES (attribute-aware; cut stays spectral-
    // guided because the order is still the Fiedler embedding).
    vector<i32> part(N);
    if(has_vw){
        vector<i64> ord(N); for(i64 v=0;v<N;v++) ord[v]=v;
        stable_sort(ord.begin(),ord.end(),[&](i64 a,i64 b){ return x[a]<x[b]; });
        i64 total=0; for(i64 v=0;v<N;v++) total += (vw[v]>0?vw[v]:1);
        i64 acc=0;
        for(i64 k=0;k<N;k++){ i64 v=ord[k]; acc += (vw[v]>0?vw[v]:1); part[v] = (acc*2<=total)?0:1; }
    } else {
        for(i64 v=0;v<N;v++) part[v] = (x[v] >= 0.0) ? 0 : 1;
    }
    // conductance of the bipartition (sanity)
    i64 cut=0, vol0=0, vol1=0;
    for(i64 v=0;v<N;v++){ if(part[v]==0) vol0+=deg[v]; else vol1+=deg[v];
        for(i64 p=xadj[v];p<xadj[v+1];p++){ i32 u=adj[p]; if(u>v && part[u]!=part[v]) cut++; } }
    double cond = (double)cut / (double)max((i64)1, min(vol0,vol1));
    fprintf(stderr,"[spec] lambda2=%.6g cheeger_lb=%.6g bipart_cut=%lld conductance=%.6g (%.2fs, %d iters)\n",
            lambda2,cheeger_lb,(long long)cut,cond,now_s()-t0,iters);
    fprintf(stderr,"STAT lambda2=%.8g cheeger_lb=%.8g conductance=%.8g iters=%d\n",lambda2,cheeger_lb,cond,iters);

    FILE* o=fopen(argv[2],"wb"); fwrite(&N,8,1,o); fwrite(&lambda2,8,1,o); fwrite(&cheeger_lb,8,1,o);
    fwrite(part.data(),4,N,o); fclose(o);
    return 0;
}
