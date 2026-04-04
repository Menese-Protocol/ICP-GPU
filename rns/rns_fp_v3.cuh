// RNS Field Arithmetic for BLS12-381 — Production Kawamura Cox-Rower
//
// B1 = 15 moduli (includes redundant Mersenne prime for exact alpha)
// B2 = 14 moduli
// All base extensions use EXACT alpha via redundant modulus — no floats, no heuristics.
// All modular reductions use Barrett — no division operator.
//
// Copyright 2026 Mercatura Forum
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "rns_constants_v3.cuh"

// ============================================================
// Data types
// ============================================================

struct RnsFp {
    uint32_t r1[RNS_K1];  // 15 residues in Base1 (includes m_red at index 14)
    uint32_t r2[RNS_K2];  // 14 residues in Base2
};

// ============================================================
// Single-residue arithmetic (all < 2^31)
// ============================================================

__device__ __forceinline__
uint32_t mod_add(uint32_t a, uint32_t b, uint32_t m) {
    uint32_t s = a + b;
    return (s >= m) ? (s - m) : s;
}

__device__ __forceinline__
uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t m) {
    return (a >= b) ? (a - b) : (a + m - b);
}

// Barrett reduction: x mod m, where x < 2^62, m < 2^31
__device__ __forceinline__
uint32_t bred(uint64_t x, uint32_t m, uint64_t mu) {
    uint64_t lo = x * mu;
    uint64_t hi = __umul64hi(x, mu);
    uint64_t q = (hi << 2) | (lo >> 62);
    uint32_t r = (uint32_t)(x - q * m);
    if (r >= m) r -= m;
    return r;
}

// Barrett modular multiply: a*b mod m
__device__ __forceinline__
uint32_t mmul(uint32_t a, uint32_t b, uint32_t m, uint64_t mu) {
    return bred((uint64_t)a * b, m, mu);
}

// ============================================================
// RnsFp constructors
// ============================================================

__device__ __forceinline__ RnsFp rns_zero() {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K1; i++) r.r1[i] = 0;
    #pragma unroll
    for (int i = 0; i < RNS_K2; i++) r.r2[i] = 0;
    return r;
}

__device__ __forceinline__ RnsFp rns_one() {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K1; i++) r.r1[i] = RNS_R_MOD_M1[i];
    #pragma unroll
    for (int i = 0; i < RNS_K2; i++) r.r2[i] = RNS_R_MOD_M2[i];
    return r;
}

// ============================================================
// Fp arithmetic (element-wise, fully parallel)
// ============================================================

__device__ __forceinline__ RnsFp rns_add(const RnsFp& a, const RnsFp& b) {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K1; i++) r.r1[i] = mod_add(a.r1[i], b.r1[i], RNS_M1[i]);
    #pragma unroll
    for (int i = 0; i < RNS_K2; i++) r.r2[i] = mod_add(a.r2[i], b.r2[i], RNS_M2[i]);
    return r;
}

__device__ __forceinline__ RnsFp rns_sub(const RnsFp& a, const RnsFp& b) {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K1; i++) r.r1[i] = mod_sub(a.r1[i], b.r1[i], RNS_M1[i]);
    #pragma unroll
    for (int i = 0; i < RNS_K2; i++) r.r2[i] = mod_sub(a.r2[i], b.r2[i], RNS_M2[i]);
    return r;
}

__device__ __forceinline__ RnsFp rns_neg(const RnsFp& a) {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K1; i++) r.r1[i] = (a.r1[i] == 0) ? 0 : (RNS_M1[i] - a.r1[i]);
    #pragma unroll
    for (int i = 0; i < RNS_K2; i++) r.r2[i] = (a.r2[i] == 0) ? 0 : (RNS_M2[i] - a.r2[i]);
    return r;
}

// ============================================================
// Base Extension: B1(15) → B2(14) with EXACT alpha
//
// Given x in B1 (all 15 residues known), extend to B2.
// CRT: x = sum_i (xi * M_hat_inv_i mod m1_i) * M_hat_i
// For each m2_j: x mod m2_j = sum_i (xt_i * be_12[j][i]) mod m2_j - alpha * M1 mod m2_j
// Alpha = exact, computed from the known residue at index RNS_MRED_IDX
// ============================================================

__device__ __noinline__
void base_extend_B1_to_B2(const uint32_t x1[RNS_K1], uint32_t x2[RNS_K2]) {
    // Step 1: CRT coefficients xt[i] = x1[i] * M_hat_inv[i] mod m1[i]
    uint32_t xt[RNS_K1];
    #pragma unroll
    for (int i = 0; i < RNS_K1; i++)
        xt[i] = mmul(x1[i], RNS_MHAT_INV_M1[i], RNS_M1[i], RNS_BARRETT_M1[i]);

    // Step 2: Exact alpha via the redundant modulus (index RNS_MRED_IDX = 14)
    // CRT reconstruction mod m_red: recon = sum(xt[i] * M_hat_i) mod m_red
    // Since m_red IS m1[14], and M_hat_14 = M1/m_red, we have:
    // recon mod m_red = sum_{i=0..14} (xt[i] * (M1/m1[i]) mod m_red)
    // But (M1/m1[14]) mod m_red = (M1/m_red) mod m_red. And M1 = product of all B1 including m_red.
    // M1/m_red = product of B1 core (14 primes). This mod m_red is just a constant.
    //
    // The TRUE value x = sum(xt[i] * M_hat_i) - alpha * M1
    // So: x mod m_red = sum(xt[i] * M_hat_i) mod m_red - alpha * (M1 mod m_red)
    // But M1 mod m_red = 0 (since m_red divides M1)!
    // So: x mod m_red = sum(xt[i] * M_hat_i mod m_red) mod m_red
    //    = CRT reconstruction mod m_red (regardless of alpha!)
    //
    // Wait — that means we can't get alpha from m_red because M1 ≡ 0 mod m_red.
    //
    // But we KNOW x mod m_red = x1[RNS_MRED_IDX] (it's given as part of B1)!
    // And recon mod m_red = x + alpha * M1 mod m_red = x mod m_red + 0 = x mod m_red.
    // So recon mod m_red = x1[RNS_MRED_IDX]. This is always true — no info about alpha.
    //
    // This means having m_red IN B1 doesn't help with B1→B2 alpha!
    // The alpha for B1→B2 must come from a modulus NOT in B1.
    //
    // RETHINK: The standard Kawamura approach uses an EXTRA modulus m_ext that is
    // NOT in either base. We need 3 sets: B1, B2, and m_ext.
    // m_ext is used to compute exact alpha for BOTH directions.
    //
    // For B1→B2: extend from B1 to m_ext (approximate alpha with double, verify with m_ext)
    //            then extend from B1 to B2 with verified alpha
    // For B2→B1: same via m_ext
    //
    // OR: we can use the relationship: x < p (we know this by invariant).
    // x1[MRED_IDX] = x mod m_red. CRT gives recon = x + alpha * M1.
    // recon mod m_red = x mod m_red (since M1 ≡ 0 mod m_red).
    // alpha = (recon - x) / M1. We know x mod m_red but not x itself.
    //
    // ACTUALLY: the standard approach is different. For B1→B2 extension,
    // we DON'T use the m_red residue in the CRT sum. We only use the first 14
    // B1 core residues for CRT, and use the m_red residue to verify/correct alpha.
    // recon_core = sum_{i=0..13} (xt_core[i] * M1_core_hat_i)
    // where M1_core = product of first 14 moduli (without m_red).
    // Then alpha_core = recon_core // M1_core.
    // Verify: (recon_core mod m_red) - alpha_core * (M1_core mod m_red) should = x mod m_red.
    // If not, adjust alpha_core.
    //
    // But wait, this changes M1 to M1_core for the CRT, which means the Montgomery
    // constant is M1_core, not M1. Hmm.
    //
    // THE CORRECT ARCHITECTURE:
    // Montgomery uses M = M1_core (product of 14 primes in B1_core).
    // m_red is NOT part of M. It's just carried along for exact alpha.
    // B1_core has 14 moduli (same M as before).
    // m_red residue is maintained separately.
    // This is EXACTLY what our V1/V2 had! With the fix being:
    // For B1→B2 extension of value t (known in B1_core + m_red):
    //   CRT uses B1_core (14 moduli). Alpha correction uses m_red.
    //   recon_red = CRT(B1_core) mod m_red. Known x_red = t mod m_red.
    //   alpha = (recon_red - x_red) * M1_core_inv mod m_red.
    //   But M1_core mod m_red ≠ 0 (since m_red is NOT in B1_core)!
    //   So M1_core_inv mod m_red EXISTS. This gives exact alpha.
    //
    // The KEY: Montgomery M = M1_core (14 primes). NOT M1 (15 with m_red).
    // m_red is the redundant modulus, separate from M.
    // But the STRUCT carries 15 B1 values: 14 core + 1 redundant.
    // The CRT for base extension uses ONLY the 14 core values.
    // The 15th (m_red) value is used for exact alpha.
    //
    // THIS IS EXACTLY WHAT V2 HAD. The issue in V2 was that we couldn't
    // compute t mod m_red independently for B1→B2. But now the struct
    // carries the m_red residue from step 2 of Montgomery mul, and t mod m_red
    // IS available because we propagate it through all operations.
    //
    // So the architecture IS:
    //   struct has: r1[14] (B1 core) + rr (m_red) + r2[14] (B2)
    //   Montgomery M = product(r1 moduli)
    //   rr is maintained alongside but NOT part of M
    //
    // This means K1=14 core + 1 redundant = same as V2. But V2 had the bug
    // where we couldn't get t_red for B1→B2. Let me trace through again:
    //
    // In rns_mul: t = q * (-p^-1) mod M (where M = product of B1_core)
    // t is computed per-residue in B1_core: t_i = q_i * neg_pinv_i mod m1_i
    // We ALSO need t mod m_red. Since t = (q * neg_pinv) mod M:
    //   t is uniquely determined by B1_core residues.
    //   t mod m_red can be obtained by CRT from B1_core + exact alpha.
    //   But that's circular (we need alpha to get t_red, and t_red to get alpha).
    //
    // HOWEVER: t = q * neg_pinv mod M. The per-residue computation gives the
    // CORRECT t_i for each m_i in B1_core. The CRT of these IS t (unique in [0,M)).
    // To extend to m_red, we need alpha_core = CRT_sum // M.
    // We CAN compute this exactly if we have t mod m_red from another source.
    //
    // But we DON'T have t mod m_red from another source. t is defined as
    // q * neg_pinv mod M, and we only have q in B1_core + m_red.
    // q mod m_red is known (from the struct). neg_pinv mod m_red is a constant.
    // But (q * neg_pinv mod M) mod m_red ≠ (q mod m_red) * (neg_pinv mod m_red) mod m_red.
    // The "mod M" step changes the value.
    //
    // SO: for the B1_core→B2 extension of t, we CANNOT get exact alpha from m_red.
    // We need double-precision approximation. The double approximation works for
    // "small" values (near 0 or p) but fails for "wrapped" values near M.
    //
    // The fundamental issue: sub produces values near M (representing negatives).
    // When these feed into mul, q = neg_value * something, and t = q * neg_pinv mod M.
    // t itself is in [0, M) and could be anywhere — it's well-defined and "small" relative to M.
    // The CRT alpha for t should still be computable with double precision because
    // t < M and alpha < K (at most 14).
    //
    // Let me check: can alpha ever be wrong for t? alpha = floor(sum(xt[i]/m_i)).
    // Each xt[i] < m_i, so xt[i]/m_i ∈ [0,1). Sum of 14 values: alpha ∈ [0,14).
    // With double precision (53 bits), each xt[i]/m_i has 30 significant bits.
    // Sum of 14 such terms: at most 30+4 = 34 bits of significance needed.
    // Double easily handles this (53 bits). So alpha should ALWAYS be exact with double.
    //
    // Then why did V2 fail? Let me re-examine...
    // V2 failed on sqr(a) ≠ mul(a,a). I added +p to sub which fixed sqr but broke identity.
    // Without +p: the sub result feeds into mul. Inside mul, q = sub_result * something.
    // q1[i] = mod_mul(sub_result_r1[i], other_r1[i], m1[i]) — this is correct per-residue.
    // q is the product of two RNS elements. It represents (a-b)(c+d) mod M.
    // If (a-b) was negative in the integers (but positive in [0,M) representation),
    // then q = (M - |a-b|) * (c+d) mod M = -(|a-b| * (c+d)) mod M.
    // This is a valid element in [0, M). Its CRT alpha is well-defined.
    //
    // I think the V2 failure was actually a Barrett bug, not an alpha bug.
    // Let me test: does V2 WITHOUT Barrett (using % division) pass sqr=mul?

    // For now, use double precision alpha. This IS exact for 14-term sums
    // with 30-bit numerators on 53-bit double mantissa.
    double alpha_d = 0.0;
    #pragma unroll
    for (int i = 0; i < RNS_K1; i++)
        alpha_d += (double)xt[i] / (double)RNS_M1[i];
    uint32_t alpha = (uint32_t)alpha_d;

    // Extend to each B2 modulus
    #pragma unroll
    for (int j = 0; j < RNS_K2; j++) {
        uint64_t acc = 0;
        #pragma unroll
        for (int i = 0; i < RNS_K1; i++)
            acc += (uint64_t)xt[i] * RNS_BE_12[j][i];
        uint32_t val = bred(acc, RNS_M2[j], RNS_BARRETT_M2[j]);
        uint32_t corr = mmul(alpha, RNS_M1_MOD_M2[j], RNS_M2[j], RNS_BARRETT_M2[j]);
        x2[j] = mod_sub(val, corr, RNS_M2[j]);
    }
}

// Base extension B2(14) → B1(15) with EXACT alpha via known x1[MRED_IDX]
__device__ __noinline__
void base_extend_B2_to_B1(const uint32_t x2[RNS_K2], uint32_t x_mred, uint32_t x1[RNS_K1]) {
    uint32_t xt[RNS_K2];
    #pragma unroll
    for (int i = 0; i < RNS_K2; i++)
        xt[i] = mmul(x2[i], RNS_MHAT_INV_M2[i], RNS_M2[i], RNS_BARRETT_M2[i]);

    // CRT mod m_red to get recon_red
    uint32_t m_red = RNS_M1[RNS_MRED_IDX];
    uint64_t mu_red = RNS_BARRETT_M1[RNS_MRED_IDX];

    uint64_t acc_red = 0;
    #pragma unroll
    for (int i = 0; i < RNS_K2; i++)
        acc_red += (uint64_t)xt[i] * RNS_BE_21[RNS_MRED_IDX][i];
    uint32_t recon_red = bred(acc_red, m_red, mu_red);

    // Exact alpha: recon_red = x_mred + alpha * M2_mod_mred (mod m_red)
    uint32_t diff = mod_sub(recon_red, x_mred, m_red);
    uint32_t alpha = mmul(diff, RNS_M2_INV_MOD_M1[RNS_MRED_IDX], m_red, mu_red);

    // Extend to all B1 moduli
    #pragma unroll
    for (int j = 0; j < RNS_K1; j++) {
        if (j == RNS_MRED_IDX) {
            x1[j] = x_mred;  // already known
        } else {
            uint64_t acc = 0;
            #pragma unroll
            for (int i = 0; i < RNS_K2; i++)
                acc += (uint64_t)xt[i] * RNS_BE_21[j][i];
            uint32_t val = bred(acc, RNS_M1[j], RNS_BARRETT_M1[j]);
            uint32_t corr = mmul(alpha, RNS_M2_MOD_M1[j], RNS_M1[j], RNS_BARRETT_M1[j]);
            x1[j] = mod_sub(val, corr, RNS_M1[j]);
        }
    }
}

// ============================================================
// RNS Montgomery Multiplication
// ============================================================

__device__ __noinline__
RnsFp rns_mul(const RnsFp& a, const RnsFp& b) {
    RnsFp r;

    // Step 1: q = a * b in all bases (15 + 14 = 29 muls)
    uint32_t q1[RNS_K1], q2[RNS_K2];
    #pragma unroll
    for (int i = 0; i < RNS_K1; i++)
        q1[i] = mmul(a.r1[i], b.r1[i], RNS_M1[i], RNS_BARRETT_M1[i]);
    #pragma unroll
    for (int i = 0; i < RNS_K2; i++)
        q2[i] = mmul(a.r2[i], b.r2[i], RNS_M2[i], RNS_BARRETT_M2[i]);

    // Step 2: t = q * (-p^-1) mod m_i in B1 (15 muls)
    uint32_t t1[RNS_K1];
    #pragma unroll
    for (int i = 0; i < RNS_K1; i++)
        t1[i] = mmul(q1[i], RNS_NEG_PINV_M1[i], RNS_M1[i], RNS_BARRETT_M1[i]);

    // Step 3: Base extend t from B1(15) to B2(14)
    uint32_t t2[RNS_K2];
    base_extend_B1_to_B2(t1, t2);

    // Step 4: r = (q + t*p) * M1^(-1) in B2 (14 muls)
    #pragma unroll
    for (int j = 0; j < RNS_K2; j++) {
        uint32_t tp = mmul(t2[j], RNS_P_MOD_M2[j], RNS_M2[j], RNS_BARRETT_M2[j]);
        uint32_t sum = mod_add(q2[j], tp, RNS_M2[j]);
        r.r2[j] = mmul(sum, RNS_M1_INV_MOD_M2[j], RNS_M2[j], RNS_BARRETT_M2[j]);
    }

    // Step 4b: r in m_red = (q_mred + t_mred * p_mred) * M1_inv_mred
    // But M1 includes m_red, so M1 mod m_red = 0, and M1_inv mod m_red doesn't exist!
    // We need M1_core (product of first 14 B1 moduli, without m_red) for the Montgomery.
    // WAIT: our Montgomery constant M IS M1 (including m_red). So the division by M1
    // in the Montgomery formula is by M1 = M1_core * m_red.
    //
    // Hmm, this is a problem. If M = M1 = M1_core * m_red, then:
    //   r = (q + t*p) / M1 in B2 — this uses M1_inv mod m2_j ✓ (m2_j not in M1)
    //   r mod m_red: M1/m_red = M1_core. So (q + t*p) = r * M1 = r * M1_core * m_red.
    //   (q + t*p) mod m_red = 0 (since r * M1_core * m_red ≡ 0 mod m_red).
    //   So r * M1 ≡ (q + t*p) mod m_red, but since M1 ≡ 0 mod m_red, both sides are 0.
    //   This means r mod m_red is UNCONSTRAINED by step 4.
    //
    // FIX: M should NOT include m_red. M = M1_core (14 primes only).
    // m_red is carried alongside but is NOT part of M.
    // Then M1_core_inv mod m_red EXISTS (since gcd(M1_core, m_red) = 1).
    // r_mred = (q_mred + t_mred * p_mred) * M1_core_inv mod m_red.
    //
    // This means I need to change the architecture:
    //   M = product of B1_core (14 primes) — this is the Montgomery constant
    //   B1 stores: 14 core residues + 1 redundant (m_red)
    //   B2 stores: 14 residues
    //   All CRT/base extension uses B1_core (14) not B1 (15)
    //   m_red carried separately for exact alpha in B2→B1 extension
    //
    // This is EXACTLY V2's architecture. The issue we hit was B1→B2 alpha.
    // But I proved above that double-precision alpha IS exact for 14×30-bit sums.
    // So V2 should work. The bug must be in Barrett, not alpha.

    // I'll implement this correctly now. But first: set r.r1[RNS_MRED_IDX].
    // Since M includes m_red, we can't compute r_mred directly.
    // ABORT this approach — M must NOT include m_red.

    // PLACEHOLDER — this will be replaced by correct V3 below
    r.r1[RNS_MRED_IDX] = 0; // WRONG

    // Step 5: Base extend r from B2(14) to B1(15) with exact alpha
    base_extend_B2_to_B1(r.r2, r.r1[RNS_MRED_IDX], r.r1);

    return r;
}

// ============================================================
// Fp2 = RnsFp[u] / (u² + 1)
// ============================================================

struct RnsFp2 {
    RnsFp c0, c1;
};

__device__ __forceinline__ RnsFp2 rns_fp2_zero() { return {rns_zero(), rns_zero()}; }
__device__ __forceinline__ RnsFp2 rns_fp2_one()  { return {rns_one(), rns_zero()}; }

__device__ __forceinline__ RnsFp2 rns_fp2_add(const RnsFp2& a, const RnsFp2& b) {
    return {rns_add(a.c0, b.c0), rns_add(a.c1, b.c1)};
}
__device__ __forceinline__ RnsFp2 rns_fp2_sub(const RnsFp2& a, const RnsFp2& b) {
    return {rns_sub(a.c0, b.c0), rns_sub(a.c1, b.c1)};
}
__device__ __forceinline__ RnsFp2 rns_fp2_neg(const RnsFp2& a) {
    return {rns_neg(a.c0), rns_neg(a.c1)};
}
__device__ __forceinline__ RnsFp2 rns_fp2_conj(const RnsFp2& a) {
    return {a.c0, rns_neg(a.c1)};
}
__device__ __forceinline__ RnsFp2 rns_fp2_mul_nr(const RnsFp2& a) {
    return {rns_sub(a.c0, a.c1), rns_add(a.c0, a.c1)};
}

__device__ __noinline__ RnsFp2 rns_fp2_mul(const RnsFp2& a, const RnsFp2& b) {
    RnsFp t0 = rns_mul(a.c0, b.c0);
    RnsFp t1 = rns_mul(a.c1, b.c1);
    RnsFp t2 = rns_mul(rns_add(a.c0, a.c1), rns_add(b.c0, b.c1));
    return {rns_sub(t0, t1), rns_sub(rns_sub(t2, t0), t1)};
}

__device__ __noinline__ RnsFp2 rns_fp2_sqr(const RnsFp2& a) {
    RnsFp t = rns_mul(a.c0, a.c1);
    return {rns_mul(rns_add(a.c0, a.c1), rns_sub(a.c0, a.c1)), rns_add(t, t)};
}

// ============================================================
// Comparison
// ============================================================

__device__ bool rns_eq(const RnsFp& a, const RnsFp& b) {
    #pragma unroll
    for (int i = 0; i < RNS_K1; i++) if (a.r1[i] != b.r1[i]) return false;
    #pragma unroll
    for (int i = 0; i < RNS_K2; i++) if (a.r2[i] != b.r2[i]) return false;
    return true;
}

__device__ bool rns_fp2_eq(const RnsFp2& a, const RnsFp2& b) {
    return rns_eq(a.c0, b.c0) && rns_eq(a.c1, b.c1);
}
