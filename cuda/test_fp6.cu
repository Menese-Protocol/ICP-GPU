// Step 3: Fp6 = Fp2[v]/(v^3 - β) on GPU
// β = 1+u (the quadratic non-residue in Fp2)

#include <cstdint>
#include <cstdio>

struct Fp { uint64_t v[6]; };
struct Fp2 { Fp c0, c1; };
struct Fp6 { Fp2 c0, c1, c2; };

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

// --- Fp (proven step 1) ---
__device__ Fp fp_zero() { Fp r={}; return r; }
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

// --- Fp2 (proven step 2) ---
__device__ Fp2 fp2_zero() { return {fp_zero(), fp_zero()}; }
__device__ Fp2 fp2_one()  { return {fp_one(), fp_zero()}; }
__device__ bool fp2_is_zero(const Fp2& a) { return fp_is_zero(a.c0) && fp_is_zero(a.c1); }
__device__ bool fp2_eq(const Fp2& a, const Fp2& b) { return fp_eq(a.c0,b.c0) && fp_eq(a.c1,b.c1); }
__device__ Fp2 fp2_add(const Fp2& a, const Fp2& b) { return {fp_add(a.c0,b.c0), fp_add(a.c1,b.c1)}; }
__device__ Fp2 fp2_sub(const Fp2& a, const Fp2& b) { return {fp_sub(a.c0,b.c0), fp_sub(a.c1,b.c1)}; }
__device__ Fp2 fp2_neg(const Fp2& a) { return {fp_neg(a.c0), fp_neg(a.c1)}; }
__device__ Fp2 fp2_mul(const Fp2& a, const Fp2& b) {
    Fp t0=fp_mul(a.c0,b.c0), t1=fp_mul(a.c1,b.c1);
    return {fp_sub(t0,t1), fp_sub(fp_sub(fp_mul(fp_add(a.c0,a.c1),fp_add(b.c0,b.c1)),t0),t1)};
}
__device__ Fp2 fp2_sqr(const Fp2& a) {
    Fp t=fp_mul(a.c0,a.c1);
    return {fp_mul(fp_add(a.c0,a.c1),fp_sub(a.c0,a.c1)), fp_add(t,t)};
}
__device__ Fp2 fp2_mul_nr(const Fp2& a) { return {fp_sub(a.c0,a.c1), fp_add(a.c0,a.c1)}; }

// --- Fp6 (new — testing now) ---
__device__ Fp6 fp6_zero() { return {fp2_zero(), fp2_zero(), fp2_zero()}; }
__device__ Fp6 fp6_one()  { return {fp2_one(), fp2_zero(), fp2_zero()}; }
__device__ bool fp6_is_one(const Fp6& a) { return fp2_eq(a.c0, fp2_one()) && fp2_is_zero(a.c1) && fp2_is_zero(a.c2); }
__device__ Fp6 fp6_add(const Fp6& a, const Fp6& b) { return {fp2_add(a.c0,b.c0), fp2_add(a.c1,b.c1), fp2_add(a.c2,b.c2)}; }
__device__ Fp6 fp6_sub(const Fp6& a, const Fp6& b) { return {fp2_sub(a.c0,b.c0), fp2_sub(a.c1,b.c1), fp2_sub(a.c2,b.c2)}; }
__device__ Fp6 fp6_neg(const Fp6& a) { return {fp2_neg(a.c0), fp2_neg(a.c1), fp2_neg(a.c2)}; }

__device__ Fp6 fp6_mul(const Fp6& a, const Fp6& b) {
    Fp2 a_a = fp2_mul(a.c0, b.c0);
    Fp2 b_b = fp2_mul(a.c1, b.c1);
    Fp2 c_c = fp2_mul(a.c2, b.c2);

    Fp2 t1 = fp2_add(a_a, fp2_mul_nr(
        fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c1,a.c2), fp2_add(b.c1,b.c2)), b_b), c_c)));
    Fp2 t2 = fp2_add(
        fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c1), fp2_add(b.c0,b.c1)), a_a), b_b),
        fp2_mul_nr(c_c));
    Fp2 t3 = fp2_add(
        fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c2), fp2_add(b.c0,b.c2)), a_a), c_c),
        b_b);

    return {t1, t2, t3};
}

// mul by v: v*(c0+c1*v+c2*v^2) = β*c2 + c0*v + c1*v^2
__device__ Fp6 fp6_mul_by_v(const Fp6& a) { return {fp2_mul_nr(a.c2), a.c0, a.c1}; }

// Helper to build Fp from small integer (in Montgomery form)
__device__ Fp fp_from_int(int n) {
    Fp r = fp_zero();
    if (n == 0) return r;
    Fp one = fp_one();
    r = one;
    for (int i = 1; i < n; i++) r = fp_add(r, one);
    return r;
}

__global__ void test_fp6() {
    // Build test elements matching Python oracle:
    // a = Fp6(Fp2(1,2), Fp2(3,4), Fp2(5,6))
    // b = Fp6(Fp2(7,8), Fp2(9,10), Fp2(11,12))
    Fp2 a0 = {fp_from_int(1), fp_from_int(2)};
    Fp2 a1 = {fp_from_int(3), fp_from_int(4)};
    Fp2 a2 = {fp_from_int(5), fp_from_int(6)};
    Fp6 a = {a0, a1, a2};

    Fp2 b0 = {fp_from_int(7), fp_from_int(8)};
    Fp2 b1 = {fp_from_int(9), fp_from_int(10)};
    Fp2 b2 = {fp_from_int(11), fp_from_int(12)};
    Fp6 b = {b0, b1, b2};

    // Test 1: Fp6 one * one = one
    Fp6 one6 = fp6_one();
    Fp6 r1 = fp6_mul(one6, one6);
    printf("TEST 1 (Fp6 1*1=1): %s\n", fp6_is_one(r1) ? "PASS" : "FAIL");

    // Test 2: a * 1 = a
    Fp6 r2 = fp6_mul(a, one6);
    bool t2 = fp2_eq(r2.c0, a.c0) && fp2_eq(r2.c1, a.c1) && fp2_eq(r2.c2, a.c2);
    printf("TEST 2 (a*1=a): %s\n", t2 ? "PASS" : "FAIL");

    // Test 3: a - a = 0
    Fp6 r3 = fp6_sub(a, a);
    printf("TEST 3 (a-a=0): %s\n",
           (fp2_is_zero(r3.c0) && fp2_is_zero(r3.c1) && fp2_is_zero(r3.c2)) ? "PASS" : "FAIL");

    // Test 4: a * b — cross-check with Python
    // Python says: c0 = Fp2(P-223, 176), c1 = Fp2(P-165, 189), c2 = Fp2(P-39, 182)
    // P - 223 means -223 mod P
    Fp6 ab = fp6_mul(a, b);

    // Check c0: should be Fp2(-223, 176)
    Fp neg223 = fp_neg(fp_from_int(223));
    Fp pos176 = fp_from_int(176);
    printf("TEST 4 (a*b c0.c0=-223): %s\n", fp_eq(ab.c0.c0, neg223) ? "PASS" : "FAIL");
    printf("TEST 5 (a*b c0.c1=176):  %s\n", fp_eq(ab.c0.c1, pos176) ? "PASS" : "FAIL");

    // Check c1: should be Fp2(-165, 189)
    Fp neg165 = fp_neg(fp_from_int(165));
    Fp pos189 = fp_from_int(189);
    printf("TEST 6 (a*b c1.c0=-165): %s\n", fp_eq(ab.c1.c0, neg165) ? "PASS" : "FAIL");
    printf("TEST 7 (a*b c1.c1=189):  %s\n", fp_eq(ab.c1.c1, pos189) ? "PASS" : "FAIL");

    // Check c2: should be Fp2(-39, 182)
    Fp neg39 = fp_neg(fp_from_int(39));
    Fp pos182 = fp_from_int(182);
    printf("TEST 8 (a*b c2.c0=-39):  %s\n", fp_eq(ab.c2.c0, neg39) ? "PASS" : "FAIL");
    printf("TEST 9 (a*b c2.c1=182):  %s\n", fp_eq(ab.c2.c1, pos182) ? "PASS" : "FAIL");

    // Test 10: mul_by_v
    // v * (1+2u, 3+4u, 5+6u) = (β*(5+6u), 1+2u, 3+4u)
    // β*(5+6u) = (5-6)+(5+6)u = (-1, 11)
    Fp6 va = fp6_mul_by_v(a);
    Fp neg1 = fp_neg(fp_one());
    Fp pos11 = fp_from_int(11);
    printf("TEST 10 (v*a c0=(-1,11)): %s\n",
           (fp_eq(va.c0.c0, neg1) && fp_eq(va.c0.c1, pos11)) ? "PASS" : "FAIL");
    printf("TEST 11 (v*a c1=(1,2)):   %s\n", fp2_eq(va.c1, a0) ? "PASS" : "FAIL");
    printf("TEST 12 (v*a c2=(3,4)):   %s\n", fp2_eq(va.c2, a1) ? "PASS" : "FAIL");
}

int main() {
    printf("=== Fp6 Arithmetic GPU Test ===\n");
    test_fp6<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }
    printf("=== Done ===\n");
    return 0;
}
