// Benchmark: sppark PTX fp_mul vs our __int128 fp_mul
#include <cstdint>
#include <cstdio>
#include <chrono>

#include "/workspace/sppark/ff/bls12-381.hpp"

// Our fp_mul for comparison
struct OurFp { uint64_t v[6]; };
__device__ __constant__ uint64_t FP_P[6]={0xb9feffffffffaaabULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};
#define M0 0x89f3fffcfffcfffdULL

__device__ __noinline__ OurFp our_fp_mul(const OurFp&a,const OurFp&b){
    uint64_t t[7]={0};
    #pragma unroll
    for(int i=0;i<6;i++){
        uint64_t carry=0;
        #pragma unroll
        for(int j=0;j<6;j++){unsigned __int128 p=(unsigned __int128)a.v[j]*b.v[i]+t[j]+carry;t[j]=(uint64_t)p;carry=(uint64_t)(p>>64);}
        t[6]=carry;uint64_t m=t[0]*M0;unsigned __int128 rd=(unsigned __int128)m*FP_P[0]+t[0];carry=(uint64_t)(rd>>64);
        #pragma unroll
        for(int j=1;j<6;j++){rd=(unsigned __int128)m*FP_P[j]+t[j]+carry;t[j-1]=(uint64_t)rd;carry=(uint64_t)(rd>>64);}
        t[5]=t[6]+carry;t[6]=(t[5]<carry)?1:0;
    }
    OurFp r;for(int i=0;i<6;i++)r.v[i]=t[i];
    OurFp s;unsigned __int128 bw=0;
    #pragma unroll
    for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw;s.v[i]=(uint64_t)d;bw=(d>>127)&1;}
    return(bw==0)?s:r;
}

// Benchmark kernels
__global__ void bench_our(OurFp* out, const OurFp* in, int N) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= N) return;
    OurFp a = in[idx], r = a;
    #pragma unroll 1
    for (int i = 0; i < 200; i++) r = our_fp_mul(r, a);
    out[idx] = r;
}

__global__ void bench_sppark(bls12_381::fp_t* out, const bls12_381::fp_t* in, int N) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= N) return;
    auto a = in[idx], r = a;
    #pragma unroll 1
    for (int i = 0; i < 200; i++) r = r * a;
    out[idx] = r;
}

int main() {
    printf("=== fp_mul Benchmark: sppark PTX vs our __int128 ===\n\n");

    int N = 10000;
    int thr = 256, blk = (N + thr - 1) / thr;

    // Our format
    OurFp h_val = {{0x5cb38790fd530c16ULL,0x7817fc679976fff5ULL,0x154f95c7143ba1c1ULL,
                     0xf0ae6acdf3d0e747ULL,0xedce6ecc21dbf440ULL,0x120177419e0bfb75ULL}};
    OurFp* dI_our; OurFp* dO_our;
    cudaMalloc(&dI_our, N*sizeof(OurFp)); cudaMalloc(&dO_our, N*sizeof(OurFp));
    OurFp* hI = new OurFp[N]; for(int i=0;i<N;i++) hI[i]=h_val;
    cudaMemcpy(dI_our, hI, N*sizeof(OurFp), cudaMemcpyHostToDevice);

    // sppark format (same data, different struct)
    bls12_381::fp_t* dI_sp; bls12_381::fp_t* dO_sp;
    cudaMalloc(&dI_sp, N*sizeof(bls12_381::fp_t)); cudaMalloc(&dO_sp, N*sizeof(bls12_381::fp_t));
    cudaMemcpy(dI_sp, dI_our, N*sizeof(OurFp), cudaMemcpyDeviceToDevice); // Same layout in memory

    // Warmup
    bench_our<<<blk,thr>>>(dO_our, dI_our, N); cudaDeviceSynchronize();
    bench_sppark<<<blk,thr>>>(dO_sp, dI_sp, N); cudaDeviceSynchronize();

    // Benchmark our
    int rounds = 5;
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int r=0;r<rounds;r++) bench_our<<<blk,thr>>>(dO_our, dI_our, N);
    cudaDeviceSynchronize();
    double ms_our = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-s1).count()/rounds;

    // Benchmark sppark
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int r=0;r<rounds;r++) bench_sppark<<<blk,thr>>>(dO_sp, dI_sp, N);
    cudaDeviceSynchronize();
    double ms_sp = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-s2).count()/rounds;

    double ops = (double)N * 200;
    printf("Our __int128 fp_mul:  %.2fms  (%.0fns per mul, %.1fM mul/sec)\n",
           ms_our, ms_our/ops*1e6, ops/ms_our/1e3);
    printf("sppark PTX fp_mul:    %.2fms  (%.0fns per mul, %.1fM mul/sec)\n",
           ms_sp, ms_sp/ops*1e6, ops/ms_sp/1e3);
    printf("Speedup:              %.2fx\n", ms_our/ms_sp);

    delete[] hI;
    cudaFree(dI_our); cudaFree(dO_our); cudaFree(dI_sp); cudaFree(dO_sp);
    printf("\n=== Done ===\n");
    return 0;
}
