// zord RMAT graph generator (C++17, OpenMP) -- for BILLION-EDGE SCALABILITY tests only.
// =====================================================================================
// Build: g++ -O3 -std=c++17 -fopenmp src/zord/cpp/rmat.cpp -o build/rmat   (auto via Makefile glob)
//
// WHY: real staged temporal graphs top out at stackoverflow (63.5M edges). To test whether zord's
// OWN min-cut kernel can PARTITION a billion-edge graph in bounded time/memory (where pymetis OOMs),
// we need a billion-edge input. RMAT (Graph500 params a=.57,b=c=.19,d=.05) is the FIELD-STANDARD
// large-graph scalability benchmark -- power-law degree, realistic skew. THIS IS A DISCLOSED
// SYNTHETIC INPUT for FEASIBILITY/THROUGHPUT ONLY; cut-QUALITY claims use REAL data, never RMAT.
//
// argv: rmat <n_log2> <m_edges> <seed> <out.bin>
// OUT (LE, == the multilevel/streaming input prefix): i64 N(=2^n_log2), i64 M, i32 src[M], i32 dst[M]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
using i64=int64_t; using i32=int32_t;
static double now_s(){ return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s n_log2 m_edges seed out.bin\n",argv[0]); return 1; }
    int nlog2 = atoi(argv[1]);
    i64 M = atoll(argv[2]);
    unsigned long seed = strtoul(argv[3],nullptr,10);
    const char* out = argv[4];
    if(nlog2<1 || nlog2>31 || M<1){ fprintf(stderr,"[rmat] bad args\n"); return 1; }
    i64 N = (i64)1 << nlog2;
    double t0=now_s();
    fprintf(stderr,"[rmat] N=2^%d=%lld  M=%lld  (Graph500 a=.57 b=c=.19 d=.05)\n",nlog2,(long long)N,(long long)M);

    vector<i32> src, dst;
    src.resize(M); dst.resize(M);
    const double A=0.57, AB=0.57+0.19, ABC=0.57+0.19+0.19;   // quadrant thresholds

    #pragma omp parallel
    {
        int tid=0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        mt19937_64 rng(seed + 0x9e3779b97f4a7c15ULL * (unsigned long)(tid+1));
        uniform_real_distribution<double> U(0.0,1.0);
        #pragma omp for schedule(static)
        for(i64 e=0;e<M;e++){
            i64 u=0,v=0;
            for(int b=0;b<nlog2;b++){
                double r=U(rng);
                int sb = (r>AB)?1:0;                 // src bit set in quadrants (1,0) and (1,1)
                int db = (r>A && r<=AB) || (r>ABC);  // dst bit set in (0,1) and (1,1)
                u |= ((i64)sb)<<b; v |= ((i64)(db?1:0))<<b;
            }
            src[e]=(i32)u; dst[e]=(i32)v;
        }
    }
    fprintf(stderr,"[rmat] generated in %.2fs, writing...\n",now_s()-t0);
    FILE* o=fopen(out,"wb"); if(!o){ fprintf(stderr,"[rmat] open out fail\n"); return 1; }
    fwrite(&N,8,1,o); fwrite(&M,8,1,o);
    fwrite(src.data(),4,M,o); fwrite(dst.data(),4,M,o);
    fclose(o);
    fprintf(stderr,"[rmat] done %.2fs total\n",now_s()-t0);
    return 0;
}
