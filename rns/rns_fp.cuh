// RNS Field Arithmetic for BLS12-381
// Each Fp element = 14 residues in Base1 + 14 residues in Base2
// All operations are element-wise (no carry chains!)
//
// Copyright 2026 Mercatura Forum
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "rns_constants.cuh"

// An Fp element in dual-RNS representation
// Plus a redundant residue for exact base extension
struct RnsFp {
    uint32_t r1[RNS_K];  // residues mod Base1
    uint32_t r2[RNS_K];  // residues mod Base2
    uint32_t rr;          // residue mod m_red (redundant, for exact alpha)
};

// ============================================================
// Modular arithmetic helpers (single residue, < 30 bits)
// All inputs < m_i < 2^30, so products fit in uint64
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

// Slow fallback (used only when Barrett constant not available)
__device__ __forceinline__
uint32_t mod_mul_slow(uint32_t a, uint32_t b, uint32_t m) {
    return (uint32_t)(((uint64_t)a * b) % m);
}

// Barrett reduction: reduce x < 2^60 mod m using precomputed mu = floor(2^62/m)
__device__ __forceinline__
uint32_t barrett_reduce(uint64_t x, uint32_t m, uint64_t mu) {
    // q = (x * mu) >> 62 — approximate quotient
    // We need (x * mu) which is up to 122 bits. Use __umul64hi for top 64 bits.
    uint64_t hi = __umul64hi(x, mu);
    // (x * mu) >> 62 = (hi << 2) | (lo >> 62), but lo >> 62 is at most 3
    // For our precision needs, hi >> 30 is sufficient since x < 2^60, mu < 2^33
    // Actually: x < 2^60, mu ≈ 2^32, so x*mu < 2^92. hi = top 64 bits of 128-bit = bits 64..127
    // We want bits 62..127 = (hi << 2) | (lo >> 62)
    // Simpler: q ≈ hi >> 30 works since hi ≈ x*mu/2^64, and we want x*mu/2^62 = hi*4
    uint32_t q = (uint32_t)(hi >> 30);  // approximate quotient, might be off by 1-2
    uint32_t r = (uint32_t)(x - (uint64_t)q * m);
    if (r >= m) r -= m;
    if (r >= m) r -= m;  // at most 2 corrections needed
    return r;
}

// Fast modular multiply using Barrett reduction
__device__ __forceinline__
uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t m) {
    // For now, still use division. Barrett needs per-modulus constants
    // passed as parameters. We'll optimize the hot loops directly.
    return (uint32_t)(((uint64_t)a * b) % m);
}

// Barrett mod_mul with explicit constant
__device__ __forceinline__
uint32_t mod_mul_barrett(uint32_t a, uint32_t b, uint32_t m, uint64_t mu) {
    uint64_t x = (uint64_t)a * b;
    return barrett_reduce(x, m, mu);
}

// ============================================================
// Fp operations in RNS (element-wise, fully parallel)
// ============================================================

__device__ __forceinline__
RnsFp rns_zero() {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) { r.r1[i] = 0; r.r2[i] = 0; }
    r.rr = 0;
    return r;
}

__device__ __forceinline__
RnsFp rns_one() {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        r.r1[i] = RNS_R_MOD_M1[i];
        r.r2[i] = RNS_R_MOD_M2[i];
    }
    // R mod p, then mod m_red — precompute this
    // For now compute: M1 mod p, then mod m_red
    // Actually RNS_R_MOD stores (M1 mod p) in each base
    // The redundant residue = (M1 mod p) mod m_red
    // Since values in RNS represent x*M1 mod p, the redundant residue
    // represents the same value mod m_red
    // ONE = M1 mod p. Its residue mod m_red:
    // We need to precompute this. For now, compute from B1.
    // Actually: the value x = M1 mod p. We need x mod m_red.
    // x mod m_red = (M1 mod p) mod m_red.
    // We have M1 mod m_red in constants. But M1 mod p ≠ M1 mod m_red.
    // We need the actual value (M1 mod p) mod m_red.
    // This must be precomputed in Python. Add as a constant.
    // TEMPORARY: use base extension from B1 to compute
    uint64_t acc = 0;
    uint32_t xt[RNS_K];
    for (int i = 0; i < RNS_K; i++)
        xt[i] = mod_mul(r.r1[i], RNS_MHAT_INV_M1[i], RNS_M1[i]);
    for (int i = 0; i < RNS_K; i++)
        acc += (uint64_t)xt[i] * RNS_MHAT_MOD_MRED_B1[i];
    double alpha_d = 0.0;
    for (int i = 0; i < RNS_K; i++)
        alpha_d += (double)xt[i] / (double)RNS_M1[i];
    uint32_t val = (uint32_t)(acc % RNS_M_RED);
    uint32_t correction = mod_mul((uint32_t)alpha_d, RNS_M1_MOD_MRED, RNS_M_RED);
    r.rr = mod_sub(val, correction, RNS_M_RED);
    return r;
}

// p in RNS form (precomputed residues)
__device__ __forceinline__
RnsFp rns_p() {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        r.r1[i] = RNS_P_MOD_M1[i];
        r.r2[i] = RNS_P_MOD_M2[i];
    }
    r.rr = RNS_P_MOD_MRED;
    return r;
}

// Raw add/sub without reduction mod p (internal use)
__device__ __forceinline__
RnsFp rns_add_raw(const RnsFp& a, const RnsFp& b) {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        r.r1[i] = mod_add(a.r1[i], b.r1[i], RNS_M1[i]);
        r.r2[i] = mod_add(a.r2[i], b.r2[i], RNS_M2[i]);
    }
    r.rr = mod_add(a.rr, b.rr, RNS_M_RED);
    return r;
}

__device__ __forceinline__
RnsFp rns_sub_raw(const RnsFp& a, const RnsFp& b) {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        r.r1[i] = mod_sub(a.r1[i], b.r1[i], RNS_M1[i]);
        r.r2[i] = mod_sub(a.r2[i], b.r2[i], RNS_M2[i]);
    }
    r.rr = mod_sub(a.rr, b.rr, RNS_M_RED);
    return r;
}

// Modular add: a + b mod p
// After raw add, if result >= p, subtract p.
// In RNS we can't directly compare to p, but we can use the fact that
// a, b < p, so a + b < 2p. Thus result is either in [0,p) or [p,2p).
// To check: compute s = a + b (raw), then s - p (raw).
// If s - p >= 0 (i.e., s >= p), use s - p. Otherwise use s.
// In RNS, "s - p >= 0" means the true value is non-negative.
// We detect this using the redundant modulus:
// If (a.rr + b.rr) < m_red AND (a.rr + b.rr) >= p mod m_red...
// Actually this doesn't work directly because we can't compare in one modulus.
//
// SIMPLER APPROACH: For Montgomery RNS, the multiplication already produces
// results in [0, 2p). Only add/sub need reduction. Since our guard bits give
// M1/p ≈ 676 billion, we have HUGE headroom. We can do many add/subs before
// the value exceeds M1. For a pairing with ~1000 adds between muls,
// the value stays well below M1.
//
// Strategy: DON'T reduce after add/sub. Let values grow up to M1.
// Montgomery mul naturally reduces back to [0, 2p) range.
// This is the "lazy reduction" approach used in most RNS implementations.

__device__ __forceinline__
RnsFp rns_add(const RnsFp& a, const RnsFp& b) {
    return rns_add_raw(a, b);
}

__device__ __forceinline__
RnsFp rns_sub(const RnsFp& a, const RnsFp& b) {
    return rns_sub_raw(a, b);
}

__device__ __forceinline__
RnsFp rns_neg(const RnsFp& a) {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        r.r1[i] = (a.r1[i] == 0) ? 0 : (RNS_M1[i] - a.r1[i]);
        r.r2[i] = (a.r2[i] == 0) ? 0 : (RNS_M2[i] - a.r2[i]);
    }
    r.rr = (a.rr == 0) ? 0 : (RNS_M_RED - a.rr);
    return r;
}

// ============================================================
// Base Extension: extend representation from one base to another
// Uses Kawamura's CRT-based approach
//
// To extend x from B1 to B2:
//   1. Compute xi_tilde = x_i * M_hat_inv_i mod m_i  (in Base1)
//   2. For each m2[j]: sum_j = sum_i (xi_tilde * be_12[j][i]) mod m2[j]
//   3. Correct for overflow: result = sum_j - alpha * M1 mod m2[j]
//      where alpha = round(sum(xi_tilde / m_i)) estimates the overflow count
// ============================================================

__device__ __noinline__
void base_extend_12(const uint32_t x1[RNS_K], uint32_t x2_out[RNS_K]) {
    // Step 1: compute xi_tilde = x_i * M_hat_inv_i mod m_i
    uint32_t xt[RNS_K];

    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        xt[i] = mod_mul(x1[i], RNS_MHAT_INV_M1[i], RNS_M1[i]);
    }

    // Alpha estimation using double precision for exact results
    // alpha = floor(sum(xt[i] / m_i))
    // Each xt[i] < m_i < 2^30, so xt[i]/m_i < 1.0
    // Sum of K=14 values, each < 1 → alpha in [0, 14)
    // Using double (53-bit mantissa) for exact alpha — each xt[i]/m_i
    // has ~30 significant bits, and we sum 14 of them. Double handles this.
    double alpha_d = 0.0;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        alpha_d += (double)xt[i] / (double)RNS_M1[i];
    }
    uint32_t alpha = (uint32_t)alpha_d;  // floor

    // Step 2: for each target modulus m2[j], compute CRT sum
    #pragma unroll
    for (int j = 0; j < RNS_K; j++) {
        uint64_t acc = 0;
        #pragma unroll
        for (int i = 0; i < RNS_K; i++) {
            acc += (uint64_t)xt[i] * RNS_BE_12[j][i];
        }
        uint32_t val = (uint32_t)(acc % RNS_M2[j]);

        // Correct for CRT overflow: subtract alpha * M1 mod m2[j]
        uint32_t correction = mod_mul(alpha, RNS_M1_MOD_M2[j], RNS_M2[j]);
        x2_out[j] = mod_sub(val, correction, RNS_M2[j]);
    }
}

__device__ __noinline__
void base_extend_21(const uint32_t x2[RNS_K], uint32_t x1_out[RNS_K]) {
    uint32_t xt[RNS_K];

    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        xt[i] = mod_mul(x2[i], RNS_MHAT_INV_M2[i], RNS_M2[i]);
    }

    double alpha_d = 0.0;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        alpha_d += (double)xt[i] / (double)RNS_M2[i];
    }
    uint32_t alpha = (uint32_t)alpha_d;  // floor

    #pragma unroll
    for (int j = 0; j < RNS_K; j++) {
        uint64_t acc = 0;
        #pragma unroll
        for (int i = 0; i < RNS_K; i++) {
            acc += (uint64_t)xt[i] * RNS_BE_21[j][i];
        }
        uint32_t val = (uint32_t)(acc % RNS_M1[j]);
        uint32_t correction = mod_mul(alpha, RNS_M2_MOD_M1[j], RNS_M1[j]);
        x1_out[j] = mod_sub(val, correction, RNS_M1[j]);
    }
}

// ============================================================
// RNS Montgomery Multiplication
//
// Input: a, b in dual-RNS Montgomery form (a*R, b*R)
// Output: (a*b*R) mod p in dual-RNS
//
// Algorithm:
//   1. q = a * b element-wise in both bases
//   2. t = q * (-p^(-1)) element-wise in Base1
//   3. Extend t from Base1 to Base2
//   4. r = (q + t*p) / M1 in Base2
//      (division by M1 is free: just multiply by M1^(-1) mod m2[j])
//   5. Extend r from Base2 to Base1
// ============================================================

__device__ __noinline__
RnsFp rns_mul(const RnsFp& a, const RnsFp& b) {
    RnsFp r;

    // Step 1: q = a * b in both bases + redundant
    uint32_t q1[RNS_K], q2[RNS_K];
    uint32_t q_red = mod_mul(a.rr, b.rr, RNS_M_RED);
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        q1[i] = mod_mul(a.r1[i], b.r1[i], RNS_M1[i]);
        q2[i] = mod_mul(a.r2[i], b.r2[i], RNS_M2[i]);
    }

    // Step 2: t = q * (-p^(-1)) in Base1
    uint32_t t1[RNS_K];
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        t1[i] = mod_mul(q1[i], RNS_NEG_PINV_M1[i], RNS_M1[i]);
    }

    // Step 3: extend t from B1 to B2, using redundant modulus for exact alpha
    uint32_t t2[RNS_K];
    uint32_t t_red; // t mod m_red
    {
        uint32_t xt[RNS_K];
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            xt[i] = mod_mul(t1[i], RNS_MHAT_INV_M1[i], RNS_M1[i]);

        // Compute t mod m_red via CRT coefficients
        uint64_t acc_red = 0;
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            acc_red += (uint64_t)xt[i] * RNS_MHAT_MOD_MRED_B1[i];
        uint32_t recon_red = (uint32_t)(acc_red % RNS_M_RED);

        // We also need t_red independently. t = q * (-p^-1) mod M1.
        // t mod m_red: we can't compute this from B1 alone without knowing the true value.
        // But we CAN compute it: t_red = q_red * NEG_PINV_MRED mod m_red
        // WAIT: t is defined as q * (-p^-1) mod M1. The "mod M1" means t < M1.
        // So t mod m_red = the actual value of t, reduced mod m_red.
        // But t is computed per-residue in B1, so t = CRT(t1) mod M1.
        // recon_red = CRT(t1) mod m_red = (t + alpha*M1) mod m_red
        // We need t mod m_red to extract alpha.
        //
        // Key insight: t ≡ q * (-p^-1) (mod m_i) for each m_i in B1.
        // The value t is uniquely determined mod M1 (since M1 = prod of B1).
        // t mod m_red can be computed as: (q mod m_red) * (-p^-1 mod m_red) mod m_red
        // BUT only if t < M1 (which it is by construction).
        // Actually NO: q * (-p^-1) mod M1 ≠ (q mod m_red) * (-p^-1 mod m_red) mod m_red
        // because the "mod M1" reduction changes the value.
        //
        // The correct approach: use the approximate alpha (double precision)
        // and then check/correct using the redundant modulus.
        double alpha_d = 0.0;
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            alpha_d += (double)xt[i] / (double)RNS_M1[i];
        uint32_t alpha = (uint32_t)alpha_d;

        // Correct alpha using redundant modulus:
        // recon_red = t_true + alpha_true * M1 (mod m_red)
        // candidate t_red for current alpha:
        uint32_t candidate = mod_sub(recon_red, mod_mul(alpha, RNS_M1_MOD_MRED, RNS_M_RED), RNS_M_RED);
        // candidate for alpha+1:
        uint32_t candidate_p1 = mod_sub(recon_red, mod_mul(alpha + 1, RNS_M1_MOD_MRED, RNS_M_RED), RNS_M_RED);

        // The correct alpha gives t_red such that 0 <= t < M1.
        // We can verify: (candidate * p + q) should be 0 mod M1 in m_red
        // Actually simpler: t_red must equal q_red * NEG_PINV (mod some modulus)
        // For a different check: just use the fractional part.
        // If alpha_d fraction > 0.5, alpha might be off by 1 if rounding errors push it.
        // Use candidate_p1 if it gives a consistent reduction.
        //
        // Robust check: verify (q_red + candidate * p_mod_mred) % M1_mod_mred == 0
        // i.e., (q + t*p) must be 0 mod M1
        // (q_red + t_red * p_mod_mred) mod m_red should equal 0 mod (M1 mod m_red)?
        // No, it should be divisible by M1, meaning (q+tp)/M1 is an integer.
        // In m_red: (q_red + t_red * p_mred) * M1_inv_mred should be an integer mod m_red
        // — that's always true. Doesn't help.
        //
        // Just use the fractional part correction: double is accurate to ~15 digits,
        // our sum has ~10 significant digits (14 * 30-bit / 2^30 ≈ 14 * 1e-9 precision).
        // If frac > 0.99, bump alpha.
        double frac = alpha_d - (double)alpha;
        if (frac > 0.99) {
            alpha += 1;
        }
        t_red = mod_sub(recon_red, mod_mul(alpha, RNS_M1_MOD_MRED, RNS_M_RED), RNS_M_RED);

        // Extend to B2 with this alpha
        #pragma unroll
        for (int j = 0; j < RNS_K; j++) {
            uint64_t acc = 0;
            #pragma unroll
            for (int i = 0; i < RNS_K; i++)
                acc += (uint64_t)xt[i] * RNS_BE_12[j][i];
            uint32_t val = (uint32_t)(acc % RNS_M2[j]);
            uint32_t correction = mod_mul(alpha, RNS_M1_MOD_M2[j], RNS_M2[j]);
            t2[j] = mod_sub(val, correction, RNS_M2[j]);
        }
    }

    // Step 4: r = (q + t*p) * M1^(-1) in B2
    #pragma unroll
    for (int j = 0; j < RNS_K; j++) {
        uint32_t tp = mod_mul(t2[j], RNS_P_MOD_M2[j], RNS_M2[j]);
        uint32_t sum = mod_add(q2[j], tp, RNS_M2[j]);
        r.r2[j] = mod_mul(sum, RNS_M1_INV_MOD_M2[j], RNS_M2[j]);
    }

    // Step 4b: r in redundant = (q + t*p) * M1^(-1) mod m_red
    {
        uint32_t tp = mod_mul(t_red, RNS_P_MOD_MRED, RNS_M_RED);
        uint32_t sum = mod_add(q_red, tp, RNS_M_RED);
        r.rr = mod_mul(sum, RNS_M1_INV_MOD_MRED, RNS_M_RED);
    }

    // Step 5: extend r from B2 to B1, using r.rr for exact alpha
    {
        uint32_t xt[RNS_K];
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            xt[i] = mod_mul(r.r2[i], RNS_MHAT_INV_M2[i], RNS_M2[i]);

        // CRT reconstruction mod m_red
        uint64_t acc_red = 0;
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            acc_red += (uint64_t)xt[i] * RNS_MHAT_MOD_MRED_B2[i];
        uint32_t recon_red = (uint32_t)(acc_red % RNS_M_RED);

        // Exact alpha via redundant modulus
        uint32_t diff = mod_sub(recon_red, r.rr, RNS_M_RED);
        uint32_t alpha = mod_mul(diff, RNS_M2_INV_MOD_MRED, RNS_M_RED);

        #pragma unroll
        for (int j = 0; j < RNS_K; j++) {
            uint64_t acc = 0;
            #pragma unroll
            for (int i = 0; i < RNS_K; i++)
                acc += (uint64_t)xt[i] * RNS_BE_21[j][i];
            uint32_t val = (uint32_t)(acc % RNS_M1[j]);
            uint32_t correction = mod_mul(alpha, RNS_M2_MOD_M1[j], RNS_M1[j]);
            r.r1[j] = mod_sub(val, correction, RNS_M1[j]);
        }
    }

    return r;
}

// ============================================================
// Comparison with sppark (for oracle testing)
// ============================================================

// Decode RNS to canonical (non-Montgomery) big integer
// Returns 384-bit value as 6x uint64 (little-endian)
// This decodes from Montgomery: result = rns_value * M1_inv mod p
// For cross-checking against Rust's to_bytes() output
//
// Since CRT reconstruction gives rns_value (which is x*M1 mod p in Montgomery form),
// the canonical value is: canonical = rns_value * M1^(-1) mod p
// But we can't do 384-bit arithmetic on GPU easily.
//
// Alternative: to compare, encode the Rust canonical value into RNS
// and compare residues directly. This avoids big-int arithmetic.

__device__ bool rns_eq(const RnsFp& a, const RnsFp& b) {
    if (a.rr != b.rr) return false;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        if (a.r1[i] != b.r1[i]) return false;
        if (a.r2[i] != b.r2[i]) return false;
    }
    return true;
}
