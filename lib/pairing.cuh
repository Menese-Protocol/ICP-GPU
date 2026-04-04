// BLS12-381 Pairing — GPU Implementation
// Miller loop + Final exponentiation + BLS signature verification
// Verified bit-exact against ic_bls12_381 v0.10.1 (DFINITY's IC library)
//
// Copyright 2026 Mercatura Forum
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "field.cuh"

// ==================== Coefficient loading ====================
__device__ __forceinline__ void load_coeff(const uint64_t* co, int idx,
                                            Fp2& c0, Fp2& c1, Fp2& c2) {
    int b = idx * 36;
    for(int i=0;i<6;i++) { c0.c0.v[i]=co[b+i]; c0.c1.v[i]=co[b+6+i]; }
    for(int i=0;i<6;i++) { c1.c0.v[i]=co[b+12+i]; c1.c1.v[i]=co[b+18+i]; }
    for(int i=0;i<6;i++) { c2.c0.v[i]=co[b+24+i]; c2.c1.v[i]=co[b+30+i]; }
}

// ==================== Line evaluation ====================
__device__ __forceinline__ Fp12 ell(const Fp12& f, const Fp2& c0, const Fp2& c1,
                                     const Fp2& c2, const G1Affine& p) {
    return fp12_mul_by_014(f, c2, fp2_mul_fp(c1, p.x), fp2_mul_fp(c0, p.y));
}

// ==================== Single Miller Loop (precomputed Q) ====================
__device__ __noinline__ Fp12 miller_loop(const G1Affine& p, const uint64_t* q_coeffs) {
    Fp12 f = fp12_one();
    int ci = 0; bool found = false;
    for (int b = 63; b >= 0; b--) {
        bool bit = (((BLS_X >> 1) >> b) & 1) == 1;
        if (!found) { found = bit; continue; }
        Fp2 c0,c1,c2;
        load_coeff(q_coeffs, ci, c0, c1, c2); f = ell(f, c0, c1, c2, p); ci++;
        if (bit) { load_coeff(q_coeffs, ci, c0, c1, c2); f = ell(f, c0, c1, c2, p); ci++; }
        f = fp12_sqr(f);
    }
    Fp2 c0,c1,c2;
    load_coeff(q_coeffs, ci, c0, c1, c2); f = ell(f, c0, c1, c2, p);
    if (BLS_X_IS_NEG) f = fp12_conj(f);
    return f;
}

// ==================== Multi-Miller Loop (2 pairs) ====================
__device__ __noinline__ Fp12 multi_miller_2(
    const G1Affine& p1, const uint64_t* q1_coeffs,
    const G1Affine& p2, const uint64_t* q2_coeffs
) {
    Fp12 f = fp12_one();
    int ci = 0; bool found = false;
    for (int b = 63; b >= 0; b--) {
        bool bit = (((BLS_X >> 1) >> b) & 1) == 1;
        if (!found) { found = bit; continue; }
        Fp2 c0,c1,c2;
        load_coeff(q1_coeffs, ci, c0, c1, c2); f = ell(f, c0, c1, c2, p1);
        load_coeff(q2_coeffs, ci, c0, c1, c2); f = ell(f, c0, c1, c2, p2);
        ci++;
        if (bit) {
            load_coeff(q1_coeffs, ci, c0, c1, c2); f = ell(f, c0, c1, c2, p1);
            load_coeff(q2_coeffs, ci, c0, c1, c2); f = ell(f, c0, c1, c2, p2);
            ci++;
        }
        f = fp12_sqr(f);
    }
    Fp2 c0,c1,c2;
    load_coeff(q1_coeffs, ci, c0, c1, c2); f = ell(f, c0, c1, c2, p1);
    load_coeff(q2_coeffs, ci, c0, c1, c2); f = ell(f, c0, c1, c2, p2);
    if (BLS_X_IS_NEG) f = fp12_conj(f);
    return f;
}

// ==================== Cyclotomic Square (optimized) ====================
// For elements in the cyclotomic subgroup (after easy part of final exp)
// Algorithm 5.5.4, Guide to Pairing-Based Cryptography

__device__ void fp4_square(const Fp2& a, const Fp2& b, Fp2& out0, Fp2& out1) {
    Fp2 t0=fp2_sqr(a), t1=fp2_sqr(b);
    out0 = fp2_add(fp2_mul_nr(t1), t0);
    Fp2 t2 = fp2_sqr(fp2_add(a,b));
    out1 = fp2_sub(fp2_sub(t2, t0), t1);
}

__device__ __noinline__ Fp12 cyclotomic_square(const Fp12& f) {
    Fp2 z0=f.c0.c0, z4=f.c0.c1, z3=f.c0.c2, z2=f.c1.c0, z1=f.c1.c1, z5=f.c1.c2;
    Fp2 A0,A1; fp4_square(z0,z1,A0,A1);
    z0=fp2_sub(A0,z0); z0=fp2_add(fp2_add(z0,z0),A0);
    z1=fp2_add(A1,z1); z1=fp2_add(fp2_add(z1,z1),A1);
    Fp2 B0,B1; fp4_square(z2,z3,B0,B1);
    Fp2 C0,C1; fp4_square(z4,z5,C0,C1);
    // Cross-application: C output uses B data, B output uses C data
    z4=fp2_sub(B0,z4); z4=fp2_add(fp2_add(z4,z4),B0);
    z5=fp2_add(B1,z5); z5=fp2_add(fp2_add(z5,z5),B1);
    Fp2 nr_C1=fp2_mul_nr(C1);
    z2=fp2_add(nr_C1,z2); z2=fp2_add(fp2_add(z2,z2),nr_C1);
    z3=fp2_sub(C0,z3); z3=fp2_add(fp2_add(z3,z3),C0);
    return {{z0,z4,z3},{z2,z1,z5}};
}

// ==================== Final Exponentiation ====================
// Exact port from ic_bls12_381/src/pairings.rs

__device__ __noinline__ Fp12 cyc_exp(const Fp12& f) {
    Fp12 t = fp12_one();
    bool found = false;
    for (int i = 63; i >= 0; i--) {
        if (found) t = cyclotomic_square(t);  // Use optimized cyclotomic square
        bool bit = ((BLS_X >> i) & 1) == 1;
        if (!found) { found = bit; if (!bit) continue; }
        if (bit) t = fp12_mul(t, f);
    }
    return fp12_conj(t); // x is negative
}

__device__ __noinline__ Fp12 final_exp(const Fp12& f_in) {
    // Easy part: f^((p^6-1)(p^2+1))
    Fp12 f = f_in;
    Fp12 t0 = fp12_conj(f);  // f^(p^6) = conjugate
    Fp12 t1 = fp12_inv(f);
    Fp12 t2 = fp12_mul(t0, t1); // f^(p^6-1)
    t1 = t2;
    t2 = fp12_mul(fp12_frob(fp12_frob(t2)), t1); // f^((p^6-1)(p^2+1))

    // Hard part (exact from ic_bls12_381)
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

// ==================== Full Pairing ====================
__device__ __forceinline__ Fp12 pairing(const G1Affine& p, const uint64_t* q_coeffs) {
    return final_exp(miller_loop(p, q_coeffs));
}

// ==================== BLS Signature Verification ====================
// Checks: e(sig, G2) * e(-H(m), pk) == Gt::identity
// Returns true if valid
__device__ bool bls_verify(
    const G1Affine& sig,        // Signature point
    const G1Affine& neg_hm,     // Negated hash-to-curve point
    const uint64_t* g2_coeffs,  // G2 generator prepared coefficients
    const uint64_t* pk_coeffs   // Public key prepared coefficients
) {
    Fp12 ml = multi_miller_2(sig, g2_coeffs, neg_hm, pk_coeffs);
    Fp12 result = final_exp(ml);

    // Check result == Gt::identity (c0.c0.c0 = ONE, everything else = 0)
    Fp* rp = (Fp*)&result;
    if (!fp_eq(rp[0], fp_one())) return false;
    for (int i = 1; i < 12; i++)
        for (int j = 0; j < 6; j++)
            if (rp[i].v[j] != 0) return false;
    return true;
}

// ==================== Batch Kernels ====================

// Batch BLS signature verification
__global__ void kernel_batch_bls_verify(
    const G1Affine* sigs,       // M signatures
    const G1Affine* neg_hms,    // M negated H(m) points
    const uint64_t* g2_coeffs,  // G2 generator prepared (shared)
    const uint64_t* pk_coeffs,  // M × G2_COEFF_U64S prepared pubkeys
    bool* results,              // M results
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    results[idx] = bls_verify(sigs[idx], neg_hms[idx], g2_coeffs,
                               pk_coeffs + (uint64_t)idx * G2_COEFF_U64S);
}

// Batch single pairings (for non-verify use cases)
__global__ void kernel_batch_pairing(
    const G1Affine* P,
    const uint64_t* q_coeffs,  // shared Q
    Fp12* results,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    results[idx] = pairing(P[idx], q_coeffs);
}

// Batch miller loop only
__global__ void kernel_batch_miller(
    const G1Affine* P,
    const uint64_t* q_coeffs,
    Fp12* results,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    results[idx] = miller_loop(P[idx], q_coeffs);
}
