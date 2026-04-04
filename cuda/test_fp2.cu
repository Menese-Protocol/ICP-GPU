// Step 2: Fp2 = Fp[u]/(u^2+1) on GPU
// Builds on proven Fp from step 1

#include <cstdint>
#include <cstdio>

struct Fp { uint64_t v[6]; };
struct Fp2 { Fp c0, c1; };

__device__ __constant__ uint64_t FP_P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};
__device__ __constant__ uint64_t FP_ONE[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL,
    0x5f48985753c758baULL, 0x77ce585370525745ULL,
    0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL
};
#define M0 0x89f3fffcfffcfffdULL

// --- Fp (proven) ---
__device__ Fp fp_zero() { Fp r = {}; return r; }
__device__ Fp fp_one() { Fp r; for(int i=0;i<6;i++) r.v[i]=FP_ONE[i]; return r; }
__device__ bool fp_is_zero(const Fp& a) { uint64_t acc=0; for(int i=0;i<6;i++) acc|=a.v[i]; return acc==0; }
__device__ bool fp_eq(const Fp& a, const Fp& b) { for(int i=0;i<6;i++) if(a.v[i]!=b.v[i]) return false; return true; }

__device__ Fp fp_add(const Fp& a, const Fp& b) {
    Fp r; unsigned __int128 carry=0;
    for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)a.v[i]+b.v[i]+carry; r.v[i]=(uint64_t)s; carry=s>>64;}
    Fp t; unsigned __int128 borrow=0;
    for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-borrow; t.v[i]=(uint64_t)d; borrow=(d>>127)&1;}
    return (borrow==0)?t:r;
}

__device__ Fp fp_sub(const Fp& a, const Fp& b) {
    Fp r; unsigned __int128 borrow=0;
    for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)a.v[i]-b.v[i]-borrow; r.v[i]=(uint64_t)d; borrow=(d>>127)&1;}
    if(borrow){unsigned __int128 carry=0; for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)r.v[i]+FP_P[i]+carry; r.v[i]=(uint64_t)s; carry=s>>64;}}
    return r;
}

__device__ Fp fp_neg(const Fp& a) { if(fp_is_zero(a)) return a; return fp_sub(fp_zero(), a); }

__device__ Fp fp_mul(const Fp& a, const Fp& b) {
    uint64_t t[7]={0};
    for(int i=0;i<6;i++){
        uint64_t carry=0;
        for(int j=0;j<6;j++){unsigned __int128 prod=(unsigned __int128)a.v[j]*b.v[i]+t[j]+carry; t[j]=(uint64_t)prod; carry=(uint64_t)(prod>>64);}
        t[6]=carry;
        uint64_t m=t[0]*M0;
        unsigned __int128 red=(unsigned __int128)m*FP_P[0]+t[0]; carry=(uint64_t)(red>>64);
        for(int j=1;j<6;j++){red=(unsigned __int128)m*FP_P[j]+t[j]+carry; t[j-1]=(uint64_t)red; carry=(uint64_t)(red>>64);}
        t[5]=t[6]+carry; t[6]=(t[5]<carry)?1:0;
    }
    Fp r; for(int i=0;i<6;i++) r.v[i]=t[i];
    Fp s; unsigned __int128 borrow=0;
    for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-borrow; s.v[i]=(uint64_t)d; borrow=(d>>127)&1;}
    return (borrow==0)?s:r;
}

// --- Fp2 ---
__device__ Fp2 fp2_zero() { return {fp_zero(), fp_zero()}; }
__device__ Fp2 fp2_one()  { return {fp_one(), fp_zero()}; }
__device__ bool fp2_eq(const Fp2& a, const Fp2& b) { return fp_eq(a.c0,b.c0) && fp_eq(a.c1,b.c1); }

__device__ Fp2 fp2_add(const Fp2& a, const Fp2& b) { return {fp_add(a.c0,b.c0), fp_add(a.c1,b.c1)}; }
__device__ Fp2 fp2_sub(const Fp2& a, const Fp2& b) { return {fp_sub(a.c0,b.c0), fp_sub(a.c1,b.c1)}; }
__device__ Fp2 fp2_neg(const Fp2& a) { return {fp_neg(a.c0), fp_neg(a.c1)}; }

__device__ Fp2 fp2_mul(const Fp2& a, const Fp2& b) {
    // (a0+a1*u)(b0+b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
    Fp t0 = fp_mul(a.c0, b.c0);
    Fp t1 = fp_mul(a.c1, b.c1);
    Fp c0 = fp_sub(t0, t1);
    Fp c1 = fp_sub(fp_sub(fp_mul(fp_add(a.c0,a.c1), fp_add(b.c0,b.c1)), t0), t1);
    return {c0, c1};
}

__device__ Fp2 fp2_sqr(const Fp2& a) {
    Fp t = fp_mul(a.c0, a.c1);
    return {fp_mul(fp_add(a.c0,a.c1), fp_sub(a.c0,a.c1)), fp_add(t,t)};
}

// mul by non-residue β = 1+u: (c0+c1*u)(1+u) = (c0-c1) + (c0+c1)*u
__device__ Fp2 fp2_mul_nr(const Fp2& a) { return {fp_sub(a.c0,a.c1), fp_add(a.c0,a.c1)}; }

// mul Fp2 by Fp scalar
__device__ Fp2 fp2_mul_fp(const Fp2& a, const Fp& s) { return {fp_mul(a.c0,s), fp_mul(a.c1,s)}; }

__global__ void test_fp2() {
    Fp one = fp_one();
    Fp2 one2 = fp2_one();
    Fp2 zero2 = fp2_zero();

    // Test 1: Fp2 one * one = one
    Fp2 r1 = fp2_mul(one2, one2);
    printf("TEST 1 (Fp2 1*1=1):  %s\n", fp2_eq(r1, one2) ? "PASS" : "FAIL");

    // Test 2: (1+u)^2 = 2u  (since u^2 = -1: (1+u)^2 = 1+2u-1 = 2u)
    Fp2 a = {one, one}; // 1 + u
    Fp2 a_sq = fp2_sqr(a);
    Fp two = fp_add(one, one);
    printf("TEST 2 ((1+u)^2 c0=0): %s\n", fp_is_zero(a_sq.c0) ? "PASS" : "FAIL");
    printf("TEST 3 ((1+u)^2 c1=2): %s\n", fp_eq(a_sq.c1, two) ? "PASS" : "FAIL");

    // Test 4: (2+3u)*(4+5u) = (8-15) + (10+12)u = -7 + 22u
    Fp two_f = fp_add(one, one);
    Fp three = fp_add(two_f, one);
    Fp four = fp_add(two_f, two_f);
    Fp five = fp_add(four, one);
    Fp2 b = {two_f, three};   // 2 + 3u
    Fp2 c = {four, five};     // 4 + 5u
    Fp2 bc = fp2_mul(b, c);
    // Expected: -7 + 22u
    Fp seven = fp_add(fp_add(four, two_f), one);
    Fp neg7 = fp_neg(seven);
    // 22 = 16 + 4 + 2
    Fp eight = fp_add(four, four);
    Fp sixteen = fp_add(eight, eight);
    Fp twentytwo = fp_add(fp_add(sixteen, four), two_f);
    printf("TEST 4 ((2+3u)(4+5u) c0=-7): %s\n", fp_eq(bc.c0, neg7) ? "PASS" : "FAIL");
    printf("TEST 5 ((2+3u)(4+5u) c1=22): %s\n", fp_eq(bc.c1, twentytwo) ? "PASS" : "FAIL");

    // Test 6: mul by non-residue: (1+u)*(3+7u) via mul vs via fp2_mul_nr
    // β*(3+7u) = (1+u)(3+7u) = (3-7) + (3+7)u = -4 + 10u
    Fp2 d = {three, seven}; // 3 + 7u
    Fp2 nr = fp2_mul_nr(d);
    Fp neg4 = fp_neg(four);
    Fp ten = fp_add(eight, two_f);
    printf("TEST 6 (β*(3+7u) c0=-4): %s\n", fp_eq(nr.c0, neg4) ? "PASS" : "FAIL");
    printf("TEST 7 (β*(3+7u) c1=10): %s\n", fp_eq(nr.c1, ten) ? "PASS" : "FAIL");

    // Test 8: a - a = 0
    Fp2 r8 = fp2_sub(b, b);
    printf("TEST 8 (a-a=0): %s\n", (fp_is_zero(r8.c0) && fp_is_zero(r8.c1)) ? "PASS" : "FAIL");

    // Test 9: cross-check mul vs sqr for same element
    Fp2 b_sq_mul = fp2_mul(b, b);
    Fp2 b_sq_sqr = fp2_sqr(b);
    printf("TEST 9 (mul==sqr): %s\n", fp2_eq(b_sq_mul, b_sq_sqr) ? "PASS" : "FAIL");

    // Print (2+3u)*(4+5u) for cross-check with Python
    printf("\nbc.c0 = ["); for(int i=0;i<6;i++) printf("0x%016llx%s",(unsigned long long)bc.c0.v[i],i<5?",":""); printf("]\n");
    printf("bc.c1 = ["); for(int i=0;i<6;i++) printf("0x%016llx%s",(unsigned long long)bc.c1.v[i],i<5?",":""); printf("]\n");
}

int main() {
    printf("=== Fp2 Arithmetic GPU Test ===\n");
    test_fp2<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }
    printf("=== Done ===\n");
    return 0;
}
