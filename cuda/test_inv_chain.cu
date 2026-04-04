// Test: fp_inv addition chain vs naive Fermat
// Both must produce same result, chain should be faster

#include <cstdint>
#include <cstdio>
#include <chrono>

struct Fp { uint64_t v[6]; };

__device__ __constant__ uint64_t FP_P[6]={0xb9feffffffffaaabULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};
__device__ __constant__ uint64_t FP_ONE[6]={0x760900000002fffdULL,0xebf4000bc40c0002ULL,0x5f48985753c758baULL,0x77ce585370525745ULL,0x5c071a97a256ec6dULL,0x15f65ec3fa80e493ULL};
#define M0 0x89f3fffcfffcfffdULL

__device__ Fp fp_zero(){Fp r={};return r;}
__device__ Fp fp_one(){Fp r;for(int i=0;i<6;i++)r.v[i]=FP_ONE[i];return r;}
__device__ bool fp_eq(const Fp&a,const Fp&b){for(int i=0;i<6;i++)if(a.v[i]!=b.v[i])return false;return true;}
__device__ bool fp_is_zero(const Fp&a){uint64_t ac=0;for(int i=0;i<6;i++)ac|=a.v[i];return ac==0;}
__device__ Fp fp_add(const Fp&a,const Fp&b){Fp r;unsigned __int128 c=0;for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)a.v[i]+b.v[i]+c;r.v[i]=(uint64_t)s;c=s>>64;}Fp t;unsigned __int128 bw=0;for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw;t.v[i]=(uint64_t)d;bw=(d>>127)&1;}return(bw==0)?t:r;}
__device__ Fp fp_sub(const Fp&a,const Fp&b){Fp r;unsigned __int128 bw=0;for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)a.v[i]-b.v[i]-bw;r.v[i]=(uint64_t)d;bw=(d>>127)&1;}if(bw){unsigned __int128 c=0;for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)r.v[i]+FP_P[i]+c;r.v[i]=(uint64_t)s;c=s>>64;}}return r;}
__device__ __noinline__ Fp fp_mul(const Fp&a,const Fp&b){uint64_t t[7]={0};for(int i=0;i<6;i++){uint64_t carry=0;for(int j=0;j<6;j++){unsigned __int128 p=(unsigned __int128)a.v[j]*b.v[i]+t[j]+carry;t[j]=(uint64_t)p;carry=(uint64_t)(p>>64);}t[6]=carry;uint64_t m=t[0]*M0;unsigned __int128 rd=(unsigned __int128)m*FP_P[0]+t[0];carry=(uint64_t)(rd>>64);for(int j=1;j<6;j++){rd=(unsigned __int128)m*FP_P[j]+t[j]+carry;t[j-1]=(uint64_t)rd;carry=(uint64_t)(rd>>64);}t[5]=t[6]+carry;t[6]=(t[5]<carry)?1:0;}Fp r;for(int i=0;i<6;i++)r.v[i]=t[i];Fp s;unsigned __int128 bw=0;for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw;s.v[i]=(uint64_t)d;bw=(d>>127)&1;}return(bw==0)?s:r;}
__device__ Fp fp_sqr(const Fp&a){return fp_mul(a,a);}

// Naive Fermat (proven correct, slow)
__device__ __noinline__ Fp fp_inv_naive(const Fp&a){
    Fp r=fp_one(),base=a;
    uint64_t exp[6]={0xb9feffffffffaaa9ULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};
    for(int w=0;w<6;w++)for(int bit=0;bit<64;bit++){if(w==5&&bit>=61)break;if((exp[w]>>bit)&1)r=fp_mul(r,base);base=fp_sqr(base);}
    return r;
}

// Addition chain (from blst, should be faster)
#include "fp_inv_chain.cuh"

__global__ void test_correctness() {
    // Build mont(42)
    Fp v = fp_one();
    for(int i=1;i<42;i++) v=fp_add(v,fp_one());

    Fp inv_naive = fp_inv_naive(v);
    Fp inv_chain = fp_inv_chain(v);

    // Both should give same result
    bool match = fp_eq(inv_naive, inv_chain);
    printf("TEST inv(42) naive==chain: %s\n", match ? "PASS" : "FAIL");

    // Both should satisfy v * inv == 1
    Fp check_n = fp_mul(v, inv_naive);
    Fp check_c = fp_mul(v, inv_chain);
    printf("TEST 42*inv_naive(42)==1:  %s\n", fp_eq(check_n, fp_one()) ? "PASS" : "FAIL");
    printf("TEST 42*inv_chain(42)==1:  %s\n", fp_eq(check_c, fp_one()) ? "PASS" : "FAIL");

    // Test with a "random-looking" value (G1 generator x coordinate)
    Fp gx;
    gx.v[0]=0x5cb38790fd530c16ULL;gx.v[1]=0x7817fc679976fff5ULL;
    gx.v[2]=0x154f95c7143ba1c1ULL;gx.v[3]=0xf0ae6acdf3d0e747ULL;
    gx.v[4]=0xedce6ecc21dbf440ULL;gx.v[5]=0x120177419e0bfb75ULL;

    Fp inv_gx_n = fp_inv_naive(gx);
    Fp inv_gx_c = fp_inv_chain(gx);
    printf("TEST inv(G1.x) match:     %s\n", fp_eq(inv_gx_n, inv_gx_c) ? "PASS" : "FAIL");
    printf("TEST G1.x*inv_chain==1:   %s\n", fp_eq(fp_mul(gx, inv_gx_c), fp_one()) ? "PASS" : "FAIL");
}

// Benchmark kernel: N inversions
__global__ void bench_inv_naive(const Fp* inputs, Fp* outputs, int N) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=N)return;
    outputs[idx] = fp_inv_naive(inputs[idx]);
}
__global__ void bench_inv_chain(const Fp* inputs, Fp* outputs, int N) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=N)return;
    outputs[idx] = fp_inv_chain(inputs[idx]);
}

int main() {
    printf("=== Fp Inversion: Addition Chain vs Naive Fermat ===\n\n");
    test_correctness<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){printf("CUDA ERROR: %s\n",cudaGetErrorString(err));return 1;}

    // Benchmark
    printf("\n=== Benchmark ===\n");
    int N = 1000;
    Fp h_val;
    h_val.v[0]=0x5cb38790fd530c16ULL;h_val.v[1]=0x7817fc679976fff5ULL;
    h_val.v[2]=0x154f95c7143ba1c1ULL;h_val.v[3]=0xf0ae6acdf3d0e747ULL;
    h_val.v[4]=0xedce6ecc21dbf440ULL;h_val.v[5]=0x120177419e0bfb75ULL;

    Fp*dI,*dO;
    cudaMalloc(&dI,N*sizeof(Fp));cudaMalloc(&dO,N*sizeof(Fp));
    Fp*hI=new Fp[N]; for(int i=0;i<N;i++)hI[i]=h_val;
    cudaMemcpy(dI,hI,N*sizeof(Fp),cudaMemcpyHostToDevice);

    int thr=64,blk=(N+thr-1)/thr;

    // Warm up
    bench_inv_naive<<<blk,thr>>>(dI,dO,N);cudaDeviceSynchronize();
    bench_inv_chain<<<blk,thr>>>(dI,dO,N);cudaDeviceSynchronize();

    // Naive
    auto s1=std::chrono::high_resolution_clock::now();
    for(int r=0;r<5;r++) bench_inv_naive<<<blk,thr>>>(dI,dO,N);
    cudaDeviceSynchronize();
    double ms_naive=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-s1).count()/5;

    // Chain
    auto s2=std::chrono::high_resolution_clock::now();
    for(int r=0;r<5;r++) bench_inv_chain<<<blk,thr>>>(dI,dO,N);
    cudaDeviceSynchronize();
    double ms_chain=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-s2).count()/5;

    printf("  Naive Fermat:    %.2fms for %d inversions (%.1fµs each)\n", ms_naive, N, ms_naive/N*1000);
    printf("  Addition Chain:  %.2fms for %d inversions (%.1fµs each)\n", ms_chain, N, ms_chain/N*1000);
    printf("  Speedup:         %.2fx\n", ms_naive/ms_chain);

    delete[]hI;cudaFree(dI);cudaFree(dO);
    printf("\n=== Done ===\n");
    return 0;
}
