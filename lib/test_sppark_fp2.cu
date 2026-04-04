// Step 2: Fp2 built on sppark's fp_t — oracle test

#include <cstdint>
#include <cstdio>
#include "/workspace/sppark/ff/bls12-381.hpp"

using fp_t = bls12_381::fp_t;

struct Fp2 { fp_t c0, c1; };

__device__ Fp2 fp2_add(const Fp2& a, const Fp2& b) { return {a.c0+b.c0, a.c1+b.c1}; }
__device__ Fp2 fp2_sub(const Fp2& a, const Fp2& b) { return {a.c0-b.c0, a.c1-b.c1}; }
__device__ Fp2 fp2_neg(const Fp2& a) { return {-a.c0, -a.c1}; }

__device__ __noinline__ Fp2 fp2_mul(const Fp2& a, const Fp2& b) {
    fp_t t0 = a.c0*b.c0, t1 = a.c1*b.c1;
    return {t0-t1, (a.c0+a.c1)*(b.c0+b.c1)-t0-t1};
}

__device__ Fp2 fp2_sqr(const Fp2& a) {
    fp_t t = a.c0*a.c1;
    return {(a.c0+a.c1)*(a.c0-a.c1), t+t};
}

// mul by non-residue β=(1+u): (c0+c1*u)(1+u) = (c0-c1)+(c0+c1)*u
__device__ Fp2 fp2_mul_nr(const Fp2& a) { return {a.c0-a.c1, a.c0+a.c1}; }

// Helper: convert to u64 for printing/comparison
__device__ void to_u64(const fp_t& sp, uint64_t out[6]) {
    const uint32_t* w = (const uint32_t*)&sp;
    for (int i = 0; i < 6; i++) out[i] = ((uint64_t)w[2*i+1]<<32)|w[2*i];
}

__global__ void test() {
    fp_t one = fp_t::one();
    fp_t two = one + one;
    fp_t three = two + one;
    fp_t four = two + two;
    fp_t seven = four + three;

    // TEST 1: (1+u)^2 = 2u
    Fp2 a = {one, one};
    Fp2 asq = fp2_sqr(a);
    printf("TEST 1 ((1+u)^2 c0=0): %s\n", asq.c0.is_zero() ? "PASS" : "FAIL");
    uint64_t asq1[6]; to_u64(asq.c1, asq1);
    uint64_t two_u64[6]; to_u64(two, two_u64);
    bool t1 = true; for(int i=0;i<6;i++) if(asq1[i]!=two_u64[i]) t1=false;
    printf("TEST 2 ((1+u)^2 c1=2): %s\n", t1 ? "PASS" : "FAIL");

    // TEST 3: (2+3u)(4+5u) = -7+22u
    Fp2 b = {two, three};
    Fp2 c = {four, four+one}; // 4+5u
    Fp2 bc = fp2_mul(b, c);
    fp_t neg7 = -seven;
    fp_t twentytwo = four+four+four+four+four+two;
    uint64_t bc0[6],bc1[6],n7[6],t22[6];
    to_u64(bc.c0,bc0); to_u64(bc.c1,bc1);
    to_u64(neg7,n7); to_u64(twentytwo,t22);
    bool t3a=true,t3b=true;
    for(int i=0;i<6;i++){if(bc0[i]!=n7[i])t3a=false;if(bc1[i]!=t22[i])t3b=false;}
    printf("TEST 3 ((2+3u)(4+5u) c0=-7): %s\n", t3a ? "PASS" : "FAIL");
    printf("TEST 4 ((2+3u)(4+5u) c1=22): %s\n", t3b ? "PASS" : "FAIL");

    // Cross-check with Python oracle
    // Python: mont(-7) = 0x21baffffffe90017,...
    printf("TEST 5 (c0 matches Python): %s\n",
           bc0[0]==0x21baffffffe90017ULL ? "PASS" : "FAIL");
    printf("TEST 6 (c1 matches Python): %s\n",
           bc1[0]==0x10d800000047ffb8ULL ? "PASS" : "FAIL");

    // TEST 7: mul_nr: β*(3+7u) = (3-7)+(3+7)u = -4+10u
    Fp2 d = {three, seven};
    Fp2 nr = fp2_mul_nr(d);
    fp_t neg4 = -four;
    fp_t ten = four+four+two;
    uint64_t nr0[6],nr1[6],n4[6],t10[6];
    to_u64(nr.c0,nr0);to_u64(nr.c1,nr1);to_u64(neg4,n4);to_u64(ten,t10);
    bool t7a=true,t7b=true;
    for(int i=0;i<6;i++){if(nr0[i]!=n4[i])t7a=false;if(nr1[i]!=t10[i])t7b=false;}
    printf("TEST 7 (β*(3+7u) c0=-4): %s\n", t7a ? "PASS" : "FAIL");
    printf("TEST 8 (β*(3+7u) c1=10): %s\n", t7b ? "PASS" : "FAIL");

    // TEST 9: sqr == mul for same element
    Fp2 bsq_mul = fp2_mul(b, b);
    Fp2 bsq_sqr = fp2_sqr(b);
    uint64_t m0[6],s0[6];
    to_u64(bsq_mul.c0,m0);to_u64(bsq_sqr.c0,s0);
    bool t9=true;for(int i=0;i<6;i++)if(m0[i]!=s0[i])t9=false;
    printf("TEST 9 (sqr==mul): %s\n", t9 ? "PASS" : "FAIL");
}

int main() {
    printf("=== sppark Fp2 Oracle Test ===\n");
    test<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){printf("CUDA ERROR: %s\n",cudaGetErrorString(err));return 1;}
    printf("=== Done ===\n");
    return 0;
}
