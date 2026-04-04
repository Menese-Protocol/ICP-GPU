// BLS12-381 Tower: Fp6 → Fp12 + Frobenius + Cyclotomic + Final Exp
// Built on 6×64-bit Fp arithmetic (fp64.cuh)
#pragma once
#include "fp64.cuh"

// ==================== Fp6 = Fp2[v]/(v³-β), β=1+u ====================
struct Fp6 { Fp2 c0, c1, c2; };

__device__ __forceinline__ Fp6 fp6_zero() { return {fp2_zero(),fp2_zero(),fp2_zero()}; }
__device__ __forceinline__ Fp6 fp6_one()  { return {fp2_one(),fp2_zero(),fp2_zero()}; }
__device__ __forceinline__ Fp6 fp6_add(const Fp6& a, const Fp6& b) { return {fp2_add(a.c0,b.c0),fp2_add(a.c1,b.c1),fp2_add(a.c2,b.c2)}; }
__device__ __forceinline__ Fp6 fp6_sub(const Fp6& a, const Fp6& b) { return {fp2_sub(a.c0,b.c0),fp2_sub(a.c1,b.c1),fp2_sub(a.c2,b.c2)}; }
__device__ __forceinline__ Fp6 fp6_neg(const Fp6& a) { return {fp2_neg(a.c0),fp2_neg(a.c1),fp2_neg(a.c2)}; }
__device__ __forceinline__ Fp6 fp6_mul_by_v(const Fp6& a) { return {fp2_mul_nr(a.c2),a.c0,a.c1}; }

__device__ __noinline__ Fp6 fp6_mul(const Fp6& a, const Fp6& b) {
    Fp2 aa=fp2_mul(a.c0,b.c0), bb=fp2_mul(a.c1,b.c1), cc=fp2_mul(a.c2,b.c2);
    return {
        fp2_add(aa, fp2_mul_nr(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c1,a.c2),fp2_add(b.c1,b.c2)),bb),cc))),
        fp2_add(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c1),fp2_add(b.c0,b.c1)),aa),bb), fp2_mul_nr(cc)),
        fp2_add(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c2),fp2_add(b.c0,b.c2)),aa),cc), bb)
    };
}

__device__ __noinline__ Fp6 fp6_inv(const Fp6& f) {
    Fp2 c0s=fp2_sqr(f.c0),c1s=fp2_sqr(f.c1),c2s=fp2_sqr(f.c2);
    Fp2 c01=fp2_mul(f.c0,f.c1),c02=fp2_mul(f.c0,f.c2),c12=fp2_mul(f.c1,f.c2);
    Fp2 t0=fp2_sub(c0s,fp2_mul_nr(c12)), t1=fp2_sub(fp2_mul_nr(c2s),c01), t2=fp2_sub(c1s,c02);
    Fp2 sc=fp2_add(fp2_mul(f.c0,t0),fp2_mul_nr(fp2_add(fp2_mul(f.c2,t1),fp2_mul(f.c1,t2))));
    Fp2 si=fp2_inv(sc);
    return {fp2_mul(t0,si),fp2_mul(t1,si),fp2_mul(t2,si)};
}

// Sparse Fp6 mul by (c0, c1, 0)
__device__ __noinline__ Fp6 fp6_mul_by_01(const Fp6& s, const Fp2& c0, const Fp2& c1) {
    Fp2 aa=fp2_mul(s.c0,c0), bb=fp2_mul(s.c1,c1);
    return {fp2_add(fp2_mul_nr(fp2_mul(s.c2,c1)),aa),
            fp2_sub(fp2_sub(fp2_mul(fp2_add(c0,c1),fp2_add(s.c0,s.c1)),aa),bb),
            fp2_add(fp2_mul(s.c2,c0),bb)};
}

__device__ __noinline__ Fp6 fp6_mul_by_1(const Fp6& s, const Fp2& c1) {
    return {fp2_mul_nr(fp2_mul(s.c2,c1)), fp2_mul(s.c0,c1), fp2_mul(s.c1,c1)};
}

// Frobenius coefficients (exact from ic_bls12_381, same as field_sppark.cuh)
__device__ Fp6 fp6_frob(const Fp6& f) {
    Fp2 c0=fp2_conj(f.c0), c1=fp2_conj(f.c1), c2=fp2_conj(f.c2);
    Fp fc1_c0 = fp_zero();
    Fp fc1_c1 = {{0xcd03c9e48671f071ULL,0x5dab22461fcda5d2ULL,0x587042afd3851b95ULL,
                   0x8eb60ebe01bacb9eULL,0x03f97d6e83d050d2ULL,0x18f0206554638741ULL}};
    Fp2 fc1={fc1_c0, fc1_c1};
    Fp fc2_c0 = {{0x890dc9e4867545c3ULL,0x2af322533285a5d5ULL,0x50880866309b7e2cULL,
                   0xa20d1b8c7e881024ULL,0x14e4f04fe2db9068ULL,0x14e56d3f1564853aULL}};
    Fp fc2_c1 = fp_zero();
    Fp2 fc2={fc2_c0, fc2_c1};
    return {c0, fp2_mul(c1,fc1), fp2_mul(c2,fc2)};
}

// ==================== Fp12 = Fp6[w]/(w²-v) ====================
struct Fp12 { Fp6 c0, c1; };

__device__ __forceinline__ Fp12 fp12_one() { return {fp6_one(),fp6_zero()}; }
__device__ __forceinline__ Fp12 fp12_conj(const Fp12& a) { return {a.c0,fp6_neg(a.c1)}; }

__device__ __noinline__ Fp12 fp12_mul(const Fp12& a, const Fp12& b) {
    Fp6 aa=fp6_mul(a.c0,b.c0), bb=fp6_mul(a.c1,b.c1);
    return {fp6_add(aa,fp6_mul_by_v(bb)),
            fp6_sub(fp6_sub(fp6_mul(fp6_add(a.c0,a.c1),fp6_add(b.c0,b.c1)),aa),bb)};
}

__device__ __noinline__ Fp12 fp12_sqr(const Fp12& a) {
    Fp6 ab=fp6_mul(a.c0,a.c1), s1=fp6_add(a.c0,a.c1), s2=fp6_add(a.c0,fp6_mul_by_v(a.c1));
    return {fp6_sub(fp6_sub(fp6_mul(s2,s1),ab),fp6_mul_by_v(ab)), fp6_add(ab,ab)};
}

__device__ __noinline__ Fp12 fp12_inv(const Fp12& f) {
    Fp6 t=fp6_sub(fp6_mul(f.c0,f.c0),fp6_mul_by_v(fp6_mul(f.c1,f.c1)));
    Fp6 ti=fp6_inv(t);
    return {fp6_mul(f.c0,ti), fp6_neg(fp6_mul(f.c1,ti))};
}

__device__ Fp12 fp12_mul_by_014(const Fp12& f, const Fp2& c0, const Fp2& c1, const Fp2& c4) {
    Fp6 aa=fp6_mul_by_01(f.c0,c0,c1), bb=fp6_mul_by_1(f.c1,c4);
    return {fp6_add(fp6_mul_by_v(bb),aa),
            fp6_sub(fp6_sub(fp6_mul_by_01(fp6_add(f.c1,f.c0),c0,fp2_add(c1,c4)),aa),bb)};
}

__device__ Fp12 fp12_frob(const Fp12& f) {
    Fp6 c0=fp6_frob(f.c0), c1=fp6_frob(f.c1);
    Fp co_c0 = {{0x07089552b319d465ULL,0xc6695f92b50a8313ULL,0x97e83cccd117228fULL,
                  0xa35baecab2dc29eeULL,0x1ce393ea5daace4dULL,0x08f2220fb0fb66ebULL}};
    Fp co_c1 = {{0xb2f66aad4ce5d646ULL,0x5842a06bfc497cecULL,0xcf4895d42599d394ULL,
                  0xc11b9cba40a8e8d0ULL,0x2e3813cbe5a0de89ULL,0x110eefda88847fafULL}};
    Fp2 co={co_c0,co_c1};
    return {c0, {fp2_mul(c1.c0,co),fp2_mul(c1.c1,co),fp2_mul(c1.c2,co)}};
}

// Cyclotomic square
__device__ void fp4_square(const Fp2& a, const Fp2& b, Fp2& out0, Fp2& out1) {
    Fp2 t0=fp2_sqr(a), t1=fp2_sqr(b);
    out0 = fp2_add(fp2_mul_nr(t1), t0);
    out1 = fp2_sub(fp2_sub(fp2_sqr(fp2_add(a,b)), t0), t1);
}

__device__ __noinline__ Fp12 cyclotomic_square(const Fp12& f) {
    Fp2 z0=f.c0.c0,z4=f.c0.c1,z3=f.c0.c2,z2=f.c1.c0,z1=f.c1.c1,z5=f.c1.c2;
    Fp2 A0,A1; fp4_square(z0,z1,A0,A1);
    Fp2 t;
    t=fp2_sub(A0,z0); z0=fp2_add(fp2_add(t,t),A0);
    t=fp2_add(A1,z1); z1=fp2_add(fp2_add(t,t),A1);
    Fp2 B0,B1; fp4_square(z2,z3,B0,B1);
    Fp2 C0,C1; fp4_square(z4,z5,C0,C1);
    t=fp2_sub(B0,z4); z4=fp2_add(fp2_add(t,t),B0);
    t=fp2_add(B1,z5); z5=fp2_add(fp2_add(t,t),B1);
    Fp2 nr_C1=fp2_mul_nr(C1);
    t=fp2_add(nr_C1,z2); z2=fp2_add(fp2_add(t,t),nr_C1);
    t=fp2_sub(C0,z3); z3=fp2_add(fp2_add(t,t),C0);
    return {{z0,z4,z3},{z2,z1,z5}};
}

// BLS parameter
#define BLS_X 0xd201000000010000ULL
#define BLS_X_IS_NEG true

__device__ __noinline__ Fp12 cyc_exp(const Fp12& f) {
    Fp12 t = fp12_one();
    bool found = false;
    for (int i = 63; i >= 0; i--) {
        if (found) t = cyclotomic_square(t);
        bool bit = ((BLS_X >> i) & 1) == 1;
        if (!found) { found = bit; if (!bit) continue; }
        if (bit) t = fp12_mul(t, f);
    }
    return fp12_conj(t);
}

__device__ __noinline__ Fp12 final_exp(const Fp12& f_in) {
    Fp12 f = f_in;
    Fp12 t0 = fp12_conj(f);
    Fp12 t1 = fp12_inv(f);
    Fp12 t2 = fp12_mul(t0, t1);
    t1 = t2;
    t2 = fp12_mul(fp12_frob(fp12_frob(t2)), t1);
    f = t2;
    t1 = fp12_conj(cyclotomic_square(t2));
    Fp12 t3 = cyc_exp(t2);
    Fp12 t4 = fp12_mul(t1, t3);
    t1 = cyc_exp(t4);
    t4 = fp12_conj(t4);
    f = fp12_mul(f, t4);
    t4 = cyclotomic_square(t3);
    t0 = cyc_exp(t1);
    t3 = fp12_mul(t3, t0);
    t3 = fp12_frob(fp12_frob(t3));
    f = fp12_mul(f, t3);
    t4 = fp12_mul(t4, cyc_exp(t0));
    f = fp12_mul(f, cyc_exp(t4));
    t4 = fp12_mul(t4, fp12_conj(t2));
    t2 = fp12_mul(t2, t1);
    t2 = fp12_frob(fp12_frob(fp12_frob(t2)));
    f = fp12_mul(f, t2);
    t4 = fp12_frob(t4);
    f = fp12_mul(f, t4);
    return f;
}

// G1 affine point
struct G1Affine { Fp x, y; };
