// Test: can we make fp_mul faster with PTX intrinsics?
// BLS12-381 Fp = 384-bit = 6 × 64-bit limbs
// GPU has 32-bit ALU. Current code uses unsigned __int128 which compiles
// to many 32-bit ops. Let's measure and try alternatives.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__device__ __constant__ uint64_t FP_P[6]={0xb9feffffffffaaabULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};
#define M0 0x89f3fffcfffcfffdULL

struct Fp{uint64_t v[6];};

// Version 1: Current (unsigned __int128)
__device__ __noinline__ Fp fp_mul_v1(const Fp&a,const Fp&b){
    uint64_t t[7]={0};
    for(int i=0;i<6;i++){
        uint64_t carry=0;
        for(int j=0;j<6;j++){
            unsigned __int128 p=(unsigned __int128)a.v[j]*b.v[i]+t[j]+carry;
            t[j]=(uint64_t)p;carry=(uint64_t)(p>>64);
        }
        t[6]=carry;
        uint64_t m=t[0]*M0;
        unsigned __int128 rd=(unsigned __int128)m*FP_P[0]+t[0];
        carry=(uint64_t)(rd>>64);
        for(int j=1;j<6;j++){
            rd=(unsigned __int128)m*FP_P[j]+t[j]+carry;
            t[j-1]=(uint64_t)rd;carry=(uint64_t)(rd>>64);
        }
        t[5]=t[6]+carry;t[6]=(t[5]<carry)?1:0;
    }
    Fp r;for(int i=0;i<6;i++)r.v[i]=t[i];
    Fp s;unsigned __int128 bw=0;
    for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw;s.v[i]=(uint64_t)d;bw=(d>>127)&1;}
    return(bw==0)?s:r;
}

// Version 2: Using PTX mul.hi / mul.lo for 64×64→128
__device__ __forceinline__ void mul64(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
    asm("mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;\n\t"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b));
}

__device__ __forceinline__ uint64_t add_cc(uint64_t a, uint64_t b, uint64_t& carry) {
    uint64_t r;
    asm("add.cc.u64 %0, %1, %2;\n\t"
        "addc.u64 %3, 0, 0;\n\t"
        : "=l"(r), "=l"(carry) : "l"(a), "l"(b));
    // Actually, this needs proper carry chain. Let's do it simply:
    r = a + b;
    carry = (r < a) ? 1 : 0;
    return r;
}

__device__ __noinline__ Fp fp_mul_v2(const Fp&a,const Fp&b){
    uint64_t t[7]={0};
    for(int i=0;i<6;i++){
        uint64_t carry=0;
        for(int j=0;j<6;j++){
            uint64_t hi, lo;
            mul64(a.v[j], b.v[i], hi, lo);
            // lo + t[j] + carry → t[j], carry_out
            uint64_t s = lo + t[j];
            uint64_t c1 = (s < lo) ? 1 : 0;
            s += carry;
            c1 += (s < carry) ? 1 : 0;
            t[j] = s;
            carry = hi + c1;
        }
        t[6]=carry;
        uint64_t m=t[0]*M0;
        // Montgomery reduction
        uint64_t hi0, lo0;
        mul64(m, FP_P[0], hi0, lo0);
        uint64_t s0 = lo0 + t[0]; // guaranteed to zero out bottom limb
        carry = hi0 + ((s0 < lo0) ? 1 : 0);
        for(int j=1;j<6;j++){
            uint64_t hi, lo;
            mul64(m, FP_P[j], hi, lo);
            s0 = lo + t[j];
            uint64_t c1 = (s0 < lo) ? 1 : 0;
            s0 += carry;
            c1 += (s0 < carry) ? 1 : 0;
            t[j-1] = s0;
            carry = hi + c1;
        }
        t[5]=t[6]+carry;t[6]=(t[5]<carry)?1:0;
    }
    Fp r;for(int i=0;i<6;i++)r.v[i]=t[i];
    // Conditional subtraction
    Fp s;uint64_t bw=0;
    for(int i=0;i<6;i++){
        uint64_t diff = r.v[i] - FP_P[i] - bw;
        bw = (r.v[i] < FP_P[i] + bw) ? 1 : 0;
        // More careful: if r.v[i] < FP_P[i] or (r.v[i]==FP_P[i] && bw)
        // Simplify:
        s.v[i] = diff;
    }
    // Redo borrow properly
    bw=0;
    for(int i=0;i<6;i++){
        unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw;
        s.v[i]=(uint64_t)d;bw=(d>>127)&1;
    }
    return(bw==0)?s:r;
}

// Benchmark kernel
__global__ void bench_fp_mul(uint64_t* out, int version, int iters) {
    Fp a, b;
    for(int i=0;i<6;i++) { a.v[i] = 0x760900000002fffdULL + i; b.v[i] = 0xebf4000bc40c0002ULL + i; }

    Fp r = a;
    if (version == 1) {
        for(int n=0;n<iters;n++) r = fp_mul_v1(r, b);
    } else {
        for(int n=0;n<iters;n++) r = fp_mul_v2(r, b);
    }
    for(int i=0;i<6;i++) out[i] = r.v[i];
}

int main() {
    printf("=== fp_mul Optimization Test ===\n\n");

    uint64_t *d_out;
    cudaMalloc(&d_out, 48);

    int ITERS = 10000;

    // Correctness: both versions should produce same result
    uint64_t h1[6], h2[6];
    bench_fp_mul<<<1,1>>>(d_out, 1, 1); cudaDeviceSynchronize();
    cudaMemcpy(h1, d_out, 48, cudaMemcpyDeviceToHost);
    bench_fp_mul<<<1,1>>>(d_out, 2, 1); cudaDeviceSynchronize();
    cudaMemcpy(h2, d_out, 48, cudaMemcpyDeviceToHost);
    bool match = true;
    for(int i=0;i<6;i++) if(h1[i]!=h2[i]) match=false;
    printf("Correctness: %s\n", match?"MATCH":"MISMATCH");
    if(!match) {
        printf("  V1: "); for(int i=0;i<6;i++) printf("%016llx ", (unsigned long long)h1[i]); printf("\n");
        printf("  V2: "); for(int i=0;i<6;i++) printf("%016llx ", (unsigned long long)h2[i]); printf("\n");
    }

    // Benchmark V1 (current __int128)
    bench_fp_mul<<<1,1>>>(d_out, 1, ITERS); cudaDeviceSynchronize(); // warmup
    cudaEvent_t s,e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    cudaEventRecord(s);
    bench_fp_mul<<<1,1>>>(d_out, 1, ITERS);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms1; cudaEventElapsedTime(&ms1, s, e);

    printf("\nV1 (__int128):  %d iters in %.2fms = %.0fns/fp_mul\n",
           ITERS, ms1, ms1*1e6/ITERS);

    // Benchmark V2 (PTX mul.hi/lo)
    bench_fp_mul<<<1,1>>>(d_out, 2, ITERS); cudaDeviceSynchronize(); // warmup
    cudaEventRecord(s);
    bench_fp_mul<<<1,1>>>(d_out, 2, ITERS);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms2; cudaEventElapsedTime(&ms2, s, e);

    printf("V2 (PTX mul):   %d iters in %.2fms = %.0fns/fp_mul\n",
           ITERS, ms2, ms2*1e6/ITERS);
    printf("Speedup: %.2fx\n", ms1/ms2);

    // Project to BLS verify
    printf("\nProjected BLS verify time:\n");
    printf("  V1: %.1fms (current)\n", 14000 * ms1/ITERS);
    printf("  V2: %.1fms (projected)\n", 14000 * ms2/ITERS);

    cudaFree(d_out);
    cudaEventDestroy(s); cudaEventDestroy(e);
    printf("\n=== Done ===\n");
    return 0;
}
