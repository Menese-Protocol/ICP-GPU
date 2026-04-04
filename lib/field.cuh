// BLS12-381 Field Arithmetic — Proven GPU Implementation
// Tower: Fp(381) → Fp2 → Fp6 → Fp12
// All operations verified bit-exact against ic_bls12_381 (DFINITY's IC library)
//
// Copyright 2026 Mercatura Forum
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <cstring>

// ==================== Types ====================
struct Fp  { uint64_t v[6]; };
struct Fp2 { Fp c0, c1; };
struct Fp6 { Fp2 c0, c1, c2; };
struct Fp12{ Fp6 c0, c1; };
struct G1Affine { Fp x, y; };

// ==================== Constants ====================
__device__ __constant__ uint64_t FP_P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL, 0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL, 0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};
__device__ __constant__ uint64_t FP_ONE[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL, 0x5f48985753c758baULL,
    0x77ce585370525745ULL, 0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL
};
#define FP_M0 0x89f3fffcfffcfffdULL

// BLS12-381 curve parameter
#define BLS_X 0xd201000000010000ULL
#define BLS_X_IS_NEG true
#define NUM_G2_COEFFS 68
#define G2_COEFF_U64S (NUM_G2_COEFFS * 36) // 2448

// ==================== Fp ====================
__device__ __forceinline__ Fp fp_zero() { Fp r = {}; return r; }
__device__ __forceinline__ Fp fp_one()  { Fp r; for(int i=0;i<6;i++) r.v[i]=FP_ONE[i]; return r; }
__device__ __forceinline__ bool fp_is_zero(const Fp& a) { uint64_t ac=0; for(int i=0;i<6;i++) ac|=a.v[i]; return ac==0; }
__device__ __forceinline__ bool fp_eq(const Fp& a, const Fp& b) { for(int i=0;i<6;i++) if(a.v[i]!=b.v[i]) return false; return true; }

__device__ __forceinline__ Fp fp_add(const Fp& a, const Fp& b) {
    Fp r; unsigned __int128 c=0;
    for(int i=0;i<6;i++) { unsigned __int128 s=(unsigned __int128)a.v[i]+b.v[i]+c; r.v[i]=(uint64_t)s; c=s>>64; }
    Fp t; unsigned __int128 bw=0;
    for(int i=0;i<6;i++) { unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw; t.v[i]=(uint64_t)d; bw=(d>>127)&1; }
    return (bw==0) ? t : r;
}

__device__ __forceinline__ Fp fp_sub(const Fp& a, const Fp& b) {
    Fp r; unsigned __int128 bw=0;
    for(int i=0;i<6;i++) { unsigned __int128 d=(unsigned __int128)a.v[i]-b.v[i]-bw; r.v[i]=(uint64_t)d; bw=(d>>127)&1; }
    if(bw) { unsigned __int128 c=0; for(int i=0;i<6;i++) { unsigned __int128 s=(unsigned __int128)r.v[i]+FP_P[i]+c; r.v[i]=(uint64_t)s; c=s>>64; } }
    return r;
}

__device__ __forceinline__ Fp fp_neg(const Fp& a) { if(fp_is_zero(a)) return a; return fp_sub(fp_zero(),a); }

__device__ __noinline__ Fp fp_mul(const Fp& a, const Fp& b) {
    uint64_t t[7]={0};
    #pragma unroll
    for(int i=0;i<6;i++) {
        uint64_t carry=0;
        #pragma unroll
        for(int j=0;j<6;j++) { unsigned __int128 p=(unsigned __int128)a.v[j]*b.v[i]+t[j]+carry; t[j]=(uint64_t)p; carry=(uint64_t)(p>>64); }
        t[6]=carry;
        uint64_t m=t[0]*FP_M0;
        unsigned __int128 rd=(unsigned __int128)m*FP_P[0]+t[0]; carry=(uint64_t)(rd>>64);
        #pragma unroll
        for(int j=1;j<6;j++) { rd=(unsigned __int128)m*FP_P[j]+t[j]+carry; t[j-1]=(uint64_t)rd; carry=(uint64_t)(rd>>64); }
        t[5]=t[6]+carry; t[6]=(t[5]<carry)?1:0;
    }
    Fp r; for(int i=0;i<6;i++) r.v[i]=t[i];
    Fp s; unsigned __int128 bw=0;
    for(int i=0;i<6;i++) { unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw; s.v[i]=(uint64_t)d; bw=(d>>127)&1; }
    return (bw==0) ? s : r;
}

__device__ __forceinline__ Fp fp_sqr(const Fp& a) { return fp_mul(a,a); }

__device__ __noinline__ Fp fp_inv(const Fp& a) {
    Fp r=fp_one(), base=a;
    uint64_t exp[6]={0xb9feffffffffaaa9ULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,
                     0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};
    for(int w=0;w<6;w++) for(int bit=0;bit<64;bit++) {
        if(w==5 && bit>=61) break;
        if((exp[w]>>bit)&1) r=fp_mul(r,base);
        base=fp_sqr(base);
    }
    return r;
}

// ==================== Fp2 = Fp[u]/(u²+1) ====================
__device__ __forceinline__ Fp2 fp2_zero() { return {fp_zero(),fp_zero()}; }
__device__ __forceinline__ Fp2 fp2_one()  { return {fp_one(),fp_zero()}; }
__device__ __forceinline__ bool fp2_is_zero(const Fp2& a) { return fp_is_zero(a.c0)&&fp_is_zero(a.c1); }
__device__ __forceinline__ Fp2 fp2_add(const Fp2& a, const Fp2& b) { return {fp_add(a.c0,b.c0),fp_add(a.c1,b.c1)}; }
__device__ __forceinline__ Fp2 fp2_sub(const Fp2& a, const Fp2& b) { return {fp_sub(a.c0,b.c0),fp_sub(a.c1,b.c1)}; }
__device__ __forceinline__ Fp2 fp2_neg(const Fp2& a) { return {fp_neg(a.c0),fp_neg(a.c1)}; }
__device__ __forceinline__ Fp2 fp2_conj(const Fp2& a) { return {a.c0,fp_neg(a.c1)}; }
__device__ __forceinline__ Fp2 fp2_mul_nr(const Fp2& a) { return {fp_sub(a.c0,a.c1),fp_add(a.c0,a.c1)}; } // ×(1+u)
__device__ __forceinline__ Fp2 fp2_mul_fp(const Fp2& a, const Fp& s) { return {fp_mul(a.c0,s),fp_mul(a.c1,s)}; }

__device__ __noinline__ Fp2 fp2_mul(const Fp2& a, const Fp2& b) {
    Fp t0=fp_mul(a.c0,b.c0), t1=fp_mul(a.c1,b.c1);
    return {fp_sub(t0,t1), fp_sub(fp_sub(fp_mul(fp_add(a.c0,a.c1),fp_add(b.c0,b.c1)),t0),t1)};
}

__device__ __forceinline__ Fp2 fp2_sqr(const Fp2& a) {
    Fp t=fp_mul(a.c0,a.c1);
    return {fp_mul(fp_add(a.c0,a.c1),fp_sub(a.c0,a.c1)), fp_add(t,t)};
}

__device__ Fp2 fp2_inv(const Fp2& a) {
    Fp n=fp_add(fp_sqr(a.c0),fp_sqr(a.c1)); Fp i=fp_inv(n);
    return {fp_mul(a.c0,i), fp_neg(fp_mul(a.c1,i))};
}

// ==================== Fp6 = Fp2[v]/(v³-β), β=1+u ====================
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

// Fp6 frobenius (exact from ic_bls12_381)
__device__ Fp6 fp6_frob(const Fp6& f) {
    Fp2 c0=fp2_conj(f.c0), c1=fp2_conj(f.c1), c2=fp2_conj(f.c2);
    Fp2 fc1 = {fp_zero(), {0xcd03c9e48671f071ULL,0x5dab22461fcda5d2ULL,0x587042afd3851b95ULL,
                            0x8eb60ebe01bacb9eULL,0x03f97d6e83d050d2ULL,0x18f0206554638741ULL}};
    Fp2 fc2 = {{0x890dc9e4867545c3ULL,0x2af322533285a5d5ULL,0x50880866309b7e2cULL,
                 0xa20d1b8c7e881024ULL,0x14e4f04fe2db9068ULL,0x14e56d3f1564853aULL}, fp_zero()};
    return {c0, fp2_mul(c1,fc1), fp2_mul(c2,fc2)};
}

// Sparse Fp6 multiplications (for line functions)
__device__ Fp6 fp6_mul_by_01(const Fp6& s, const Fp2& c0, const Fp2& c1) {
    Fp2 aa=fp2_mul(s.c0,c0), bb=fp2_mul(s.c1,c1);
    return {fp2_add(fp2_mul_nr(fp2_mul(s.c2,c1)),aa),
            fp2_sub(fp2_sub(fp2_mul(fp2_add(c0,c1),fp2_add(s.c0,s.c1)),aa),bb),
            fp2_add(fp2_mul(s.c2,c0),bb)};
}
__device__ Fp6 fp6_mul_by_1(const Fp6& s, const Fp2& c1) {
    return {fp2_mul_nr(fp2_mul(s.c2,c1)), fp2_mul(s.c0,c1), fp2_mul(s.c1,c1)};
}

// ==================== Fp12 = Fp6[w]/(w²-v) ====================
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

// Fp12 frobenius (exact from ic_bls12_381)
__device__ Fp12 fp12_frob(const Fp12& f) {
    Fp6 c0=fp6_frob(f.c0), c1=fp6_frob(f.c1);
    Fp2 co = {{0x07089552b319d465ULL,0xc6695f92b50a8313ULL,0x97e83cccd117228fULL,
               0xa35baecab2dc29eeULL,0x1ce393ea5daace4dULL,0x08f2220fb0fb66ebULL},
              {0xb2f66aad4ce5d646ULL,0x5842a06bfc497cecULL,0xcf4895d42599d394ULL,
               0xc11b9cba40a8e8d0ULL,0x2e3813cbe5a0de89ULL,0x110eefda88847fafULL}};
    return {c0, {fp2_mul(c1.c0,co),fp2_mul(c1.c1,co),fp2_mul(c1.c2,co)}};
}

// Sparse Fp12 multiplication (for line function evaluation)
__device__ Fp12 fp12_mul_by_014(const Fp12& f, const Fp2& c0, const Fp2& c1, const Fp2& c4) {
    Fp6 aa=fp6_mul_by_01(f.c0,c0,c1), bb=fp6_mul_by_1(f.c1,c4);
    return {fp6_add(fp6_mul_by_v(bb),aa),
            fp6_sub(fp6_sub(fp6_mul_by_01(fp6_add(f.c1,f.c0),c0,fp2_add(c1,c4)),aa),bb)};
}
