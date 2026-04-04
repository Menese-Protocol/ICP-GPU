// RNS Field Arithmetic for BLS12-381 — V2 with Barrett reduction
// Each Fp element = 14 residues in Base1 + 14 residues in Base2 + 1 redundant
// All mul/reduce operations use Barrett (no % division operator)
//
// Copyright 2026 Mercatura Forum
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "rns_constants.cuh"

struct RnsFp {
    uint32_t r1[RNS_K];  // residues mod Base1
    uint32_t r2[RNS_K];  // residues mod Base2
    uint32_t rr;          // residue mod m_red (redundant)
};

// ============================================================
// Single-residue arithmetic
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
// mu = floor(2^62 / m), precomputed
__device__ __forceinline__
uint32_t fast_red(uint64_t x, uint32_t m, uint64_t mu) {
    uint64_t lo = x * mu;
    uint64_t hi = __umul64hi(x, mu);
    uint64_t q = (hi << 2) | (lo >> 62);
    uint32_t r = (uint32_t)(x - q * m);
    if (r >= m) r -= m;
    return r;
}

// Fast modular multiply: a*b mod m
__device__ __forceinline__
uint32_t mmul(uint32_t a, uint32_t b, uint32_t m, uint64_t mu) {
    return fast_red((uint64_t)a * b, m, mu);
}

// Alias
__device__ __forceinline__
uint32_t bred(uint64_t x, uint32_t m, uint64_t mu) {
    return fast_red(x, m, mu);
}

// Legacy alias
__device__ __forceinline__
uint32_t bred64(uint64_t x, uint32_t m, uint64_t mu) {
    return fast_red(x, m, mu);
}

// ============================================================
// RnsFp operations
// ============================================================

__device__ __forceinline__ RnsFp rns_zero() {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) { r.r1[i] = 0; r.r2[i] = 0; }
    r.rr = 0;
    return r;
}

__device__ __forceinline__ RnsFp rns_one() {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        r.r1[i] = RNS_R_MOD_M1[i];
        r.r2[i] = RNS_R_MOD_M2[i];
    }
    // Compute rr via base extension from B1
    uint32_t xt[RNS_K];
    #pragma unroll
    for (int i = 0; i < RNS_K; i++)
        xt[i] = mmul(r.r1[i], RNS_MHAT_INV_M1[i], RNS_M1[i], RNS_BARRETT_M1[i]);
    uint64_t acc = 0;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++)
        acc += (uint64_t)xt[i] * RNS_MHAT_MOD_MRED_B1[i];
    double alpha_d = 0.0;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        double recip;
        asm("mov.b64 %0, %1;" : "=d"(recip) : "l"(RNS_RECIP_M1[i]));
        alpha_d += (double)xt[i] * recip;
    }
    uint32_t alpha_one = (uint32_t)alpha_d;
    if (alpha_d - (double)alpha_one > 0.9999) alpha_one++;
    uint32_t val = bred64(acc, RNS_M_RED, RNS_BARRETT_MRED);
    uint32_t corr = mmul(alpha_one, RNS_M1_MOD_MRED, RNS_M_RED, RNS_BARRETT_MRED);
    r.rr = mod_sub(val, corr, RNS_M_RED);
    return r;
}

// Raw add without reduction mod p
__device__ __forceinline__ RnsFp rns_add_raw(const RnsFp& a, const RnsFp& b) {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        r.r1[i] = mod_add(a.r1[i], b.r1[i], RNS_M1[i]);
        r.r2[i] = mod_add(a.r2[i], b.r2[i], RNS_M2[i]);
    }
    r.rr = mod_add(a.rr, b.rr, RNS_M_RED);
    return r;
}

// Conditional reduction: if x >= p, return x - p, else return x.
// In RNS, we detect x >= p using the redundant modulus heuristic:
// After add, if result.rr >= p_mod_mred AND result came from two values < p,
// then result might be >= p. Subtract p and check.
// For a,b < p: a+b < 2p. So (a+b) - p is in [0,p).
// We compute both s = a+b and s-p, and select based on whether s >= p.
// Detection: compute s-p in redundant. If the "borrow" from the full subtraction
// would make s-p negative, then s < p and we keep s. Otherwise keep s-p.
//
// Simpler approach for lazy reduction in RNS:
// Just subtract p unconditionally from the raw sum. Then add p back if
// the result went negative. We detect negativity via the redundant modulus.
//
// EVEN SIMPLER: a, b < p implies a+b < 2p. 2p < M1 (since M1/p > 676 billion).
// So a+b fits in RNS. The rns_mul already handles inputs < M1 correctly.
// The issue is that Fp2 mul uses (a0+a1) which gives a sum ≥ p.
// rns_mul of (a0+a1) works correctly because M1 > 2p, so the sum is
// a valid RNS element. The issue is that the RESULT of rns_mul may not
// equal the result of multiplying reduced operands.
//
// Actually: Montgomery mul satisfies mont_mul(a, b) = a*b*M1^(-1) mod p
// If a = a_true (reduced mod p) then mont_mul(a, b) = mont_mul(a mod p, b) only
// if a < M1 AND the algorithm reduces correctly. Since a+b < 2p < M1,
// the rns_mul should work... unless the lazy reduction causes the intermediate
// values to exceed M1 after multiple add-before-mul chains.
//
// The real fix: reduce mod p after add. Use Montgomery mul by 1 (rns_one)
// to force reduction. This is expensive but correct.
//
// OPTIMAL: conditionally subtract p. Use the redundant modulus to decide.

__device__ __forceinline__ RnsFp rns_cond_sub_p(const RnsFp& x) {
    // Compute x - p in all bases
    RnsFp xmp;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        xmp.r1[i] = mod_sub(x.r1[i], RNS_P_MOD_M1[i], RNS_M1[i]);
        xmp.r2[i] = mod_sub(x.r2[i], RNS_P_MOD_M2[i], RNS_M2[i]);
    }
    xmp.rr = mod_sub(x.rr, RNS_P_MOD_MRED, RNS_M_RED);

    // Detect if x >= p: if x < p, then x - p wrapped around (is very large in true value).
    // If x >= p, then x - p is the correct small value.
    // We can't compare directly in RNS, but we know:
    //   - If x came from a+b where a,b < p, then x < 2p
    //   - If x >= p: x-p is in [0, p)
    //   - If x < p: x-p would be negative, wrapping to M1-p+x which is huge
    //
    // Heuristic using redundant modulus:
    // For values < p: x.rr < p mod m_red (not always, but statistically likely)
    // This isn't reliable. Better: use the sign of (x - p) in the integer sense.
    //
    // Reliable method: carry a "reduced" flag or always subtract when we know
    // the operands came from an add of two reduced values.
    //
    // For now: ALWAYS subtract p after add (caller knows a+b < 2p, so x-p is in [-p, p))
    // If x.rr (before sub) was >= p_mod_mred, use x-p. Else use x.
    // This works because for x in [0, 2p), with high probability the redundant
    // residue distinguishes [0,p) from [p,2p). Not 100% reliable but good enough
    // for most values.
    //
    // ACTUALLY: the most reliable RNS comparison uses the Bajard sign detection.
    // But that requires another base extension. Too expensive.
    //
    // PRAGMATIC FIX: Since we know a, b < p for field elements, and a+b < 2p,
    // we just need to distinguish x in [0,p) from x in [p, 2p).
    // x mod m_red can be in any range for both cases, so single-modulus detection fails.
    //
    // CORRECT FIX: Don't use conditional subtraction. Instead, accept lazy reduction
    // and fix the Fp2 operations to not depend on fully reduced inputs.
    // Specifically: Karatsuba uses (a0+a1) which can be up to 2p.
    // The product (a0+a1)(b0+b1) via rns_mul handles inputs < M1 correctly
    // (since M1/p > 676 billion, even 100p fits). So the algebra works.
    // The issue is that rns_eq compares residues, which differ between
    // x and x+p (they represent different values mod M_i, even though
    // they're the same mod p).
    //
    // THE REAL FIX: Tests should not compare rns_eq directly for lazy-reduced values.
    // Instead, verify algebraic properties via mul (which forces reduction).
    // OR: implement proper comparison via rns_mul(a - b, one) == zero.

    // For now, just return the subtracted version (always subtract)
    // Caller should only use this when they know x ∈ [p, 2p)
    return xmp;
}

__device__ __forceinline__ RnsFp rns_add(const RnsFp& a, const RnsFp& b) {
    // For field elements a,b < p: a+b ∈ [0, 2p)
    // We need result mod p. Since we can't detect x >= p reliably in RNS,
    // we use the fact that Montgomery mul normalizes. For the Fp2 Karatsuba
    // to work, we need the add result to feed into mul correctly.
    //
    // Key insight: rns_mul(x, y) computes x*y*M1^(-1) mod p regardless of
    // whether x is reduced or not, AS LONG AS x < M1. Since a+b < 2p << M1,
    // the mul will produce the correct reduced result.
    //
    // So add/sub DON'T need to reduce. The Fp2 test failures must be something else.
    // Let me re-examine...
    return rns_add_raw(a, b);
}

__device__ __forceinline__ RnsFp rns_sub(const RnsFp& a, const RnsFp& b) {
    // Raw sub everywhere: all residues wrap consistently around their modulus.
    // CRT(B1) = CRT(B2) = value mod M. rr = value mod m_red. All consistent.
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        r.r1[i] = mod_sub(a.r1[i], b.r1[i], RNS_M1[i]);
        r.r2[i] = mod_sub(a.r2[i], b.r2[i], RNS_M2[i]);
    }
    r.rr = mod_sub(a.rr, b.rr, RNS_M_RED);
    return r;
}

// Forward declaration for normalize
__device__ __noinline__ RnsFp rns_mul(const RnsFp& a, const RnsFp& b);

// Normalize: reduce to canonical [0, p) via mul by ONE
// rns_mul(x, ONE) = x * R * M^(-1) mod p = x mod p (since R = M mod p)
__device__ __noinline__ RnsFp rns_normalize(const RnsFp& a) {
    return rns_mul(a, rns_one());
}

__device__ __forceinline__ RnsFp rns_neg(const RnsFp& a) {
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
// RNS Montgomery Multiplication — Barrett optimized
// ============================================================

__device__ __noinline__
RnsFp rns_mul(const RnsFp& a, const RnsFp& b) {
    RnsFp r;

    // Step 1: q = a * b in all bases (29 independent muls)
    uint32_t q1[RNS_K], q2[RNS_K];
    uint32_t q_red = mmul(a.rr, b.rr, RNS_M_RED, RNS_BARRETT_MRED);
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        q1[i] = mmul(a.r1[i], b.r1[i], RNS_M1[i], RNS_BARRETT_M1[i]);
        q2[i] = mmul(a.r2[i], b.r2[i], RNS_M2[i], RNS_BARRETT_M2[i]);
    }

    // Step 2: t = q * (-p^-1) in B1 (14 muls)
    uint32_t t1[RNS_K];
    #pragma unroll
    for (int i = 0; i < RNS_K; i++)
        t1[i] = mmul(q1[i], RNS_NEG_PINV_M1[i], RNS_M1[i], RNS_BARRETT_M1[i]);

    // Step 3: Base extend t from B1 to B2
    uint32_t t2[RNS_K];
    uint32_t t_red;
    {
        // CRT coefficients
        uint32_t xt[RNS_K];
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            xt[i] = mmul(t1[i], RNS_MHAT_INV_M1[i], RNS_M1[i], RNS_BARRETT_M1[i]);

        // Alpha via precomputed reciprocals (multiply instead of divide)
        double alpha_d = 0.0;
        #pragma unroll
        for (int i = 0; i < RNS_K; i++) {
            double recip;
            asm("mov.b64 %0, %1;" : "=d"(recip) : "l"(RNS_RECIP_M1[i]));
            alpha_d += (double)xt[i] * recip;
        }
        uint32_t alpha = (uint32_t)alpha_d;
        // Boundary correction: if fractional part is very close to 1,
        // double precision may have rounded down when the true value is integer
        if (alpha_d - (double)alpha > 0.9999) alpha++;

        // CRT mod m_red for t_red
        uint64_t acc_red = 0;
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            acc_red += (uint64_t)xt[i] * RNS_MHAT_MOD_MRED_B1[i];
        uint32_t recon_red = bred64(acc_red, RNS_M_RED, RNS_BARRETT_MRED);
        t_red = mod_sub(recon_red, mmul(alpha, RNS_M1_MOD_MRED, RNS_M_RED, RNS_BARRETT_MRED), RNS_M_RED);

        // Extend to B2 — transposed loop for better constant cache locality
        uint64_t be_acc[RNS_K];
        #pragma unroll
        for (int j = 0; j < RNS_K; j++) be_acc[j] = 0;
        #pragma unroll
        for (int i = 0; i < RNS_K; i++) {
            uint32_t xi = xt[i];
            #pragma unroll
            for (int j = 0; j < RNS_K; j++)
                be_acc[j] += (uint64_t)xi * RNS_BE_12[j][i];
        }
        #pragma unroll
        for (int j = 0; j < RNS_K; j++) {
            uint32_t val = bred64(be_acc[j], RNS_M2[j], RNS_BARRETT_M2[j]);
            uint32_t corr = mmul(alpha, RNS_M1_MOD_M2[j], RNS_M2[j], RNS_BARRETT_M2[j]);
            t2[j] = mod_sub(val, corr, RNS_M2[j]);
        }
    }

    // Step 4: r = (q + t*p) * M1^(-1) in B2 + m_red (14+1 muls)
    #pragma unroll
    for (int j = 0; j < RNS_K; j++) {
        uint32_t tp = mmul(t2[j], RNS_P_MOD_M2[j], RNS_M2[j], RNS_BARRETT_M2[j]);
        uint32_t sum = mod_add(q2[j], tp, RNS_M2[j]);
        r.r2[j] = mmul(sum, RNS_M1_INV_MOD_M2[j], RNS_M2[j], RNS_BARRETT_M2[j]);
    }
    {
        uint32_t tp = mmul(t_red, RNS_P_MOD_MRED, RNS_M_RED, RNS_BARRETT_MRED);
        uint32_t sum = mod_add(q_red, tp, RNS_M_RED);
        r.rr = mmul(sum, RNS_M1_INV_MOD_MRED, RNS_M_RED, RNS_BARRETT_MRED);
    }

    // Step 5: Base extend r from B2 to B1
    // Use double alpha + rr verification for exact result
    {
        uint32_t xt[RNS_K];
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            xt[i] = mmul(r.r2[i], RNS_MHAT_INV_M2[i], RNS_M2[i], RNS_BARRETT_M2[i]);

        // CRT reconstruction mod m_red
        uint64_t acc_red = 0;
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            acc_red += (uint64_t)xt[i] * RNS_MHAT_MOD_MRED_B2[i];
        uint32_t recon_red = bred64(acc_red, RNS_M_RED, RNS_BARRETT_MRED);

        // Exact alpha: recon_red = r.rr + alpha * M2 mod m_red
        // alpha = (recon_red - r.rr) * M2_inv mod m_red
        uint32_t diff = mod_sub(recon_red, r.rr, RNS_M_RED);
        uint32_t alpha = mmul(diff, RNS_M2_INV_MOD_MRED, RNS_M_RED, RNS_BARRETT_MRED);

        // Extend to B1 — transposed loop
        uint64_t be_acc2[RNS_K];
        #pragma unroll
        for (int j = 0; j < RNS_K; j++) be_acc2[j] = 0;
        #pragma unroll
        for (int i = 0; i < RNS_K; i++) {
            uint32_t xi = xt[i];
            #pragma unroll
            for (int j = 0; j < RNS_K; j++)
                be_acc2[j] += (uint64_t)xi * RNS_BE_21[j][i];
        }
        #pragma unroll
        for (int j = 0; j < RNS_K; j++) {
            uint32_t val = bred64(be_acc2[j], RNS_M1[j], RNS_BARRETT_M1[j]);
            uint32_t corr = mmul(alpha, RNS_M2_MOD_M1[j], RNS_M1[j], RNS_BARRETT_M1[j]);
            r.r1[j] = mod_sub(val, corr, RNS_M1[j]);
        }
    }

    return r;
}

// ============================================================
// Fp2 = RnsFp[u] / (u² + 1) — Quadratic extension
// ============================================================

struct RnsFp2 {
    RnsFp c0, c1;  // c0 + c1 * u, where u² = -1
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

// Conjugate: (c0 + c1*u)* = c0 - c1*u
__device__ __forceinline__ RnsFp2 rns_fp2_conj(const RnsFp2& a) {
    return {a.c0, rns_neg(a.c1)};
}

// Multiply by non-residue β = (1+u): (c0+c1*u)(1+u) = (c0-c1) + (c0+c1)*u
__device__ __forceinline__ RnsFp2 rns_fp2_mul_nr(const RnsFp2& a) {
    return {rns_sub(a.c0, a.c1), rns_add(a.c0, a.c1)};
}

// Scalar multiply: (c0+c1*u) * s = c0*s + c1*s*u
__device__ __forceinline__ RnsFp2 rns_fp2_mul_fp(const RnsFp2& a, const RnsFp& s) {
    return {rns_mul(a.c0, s), rns_mul(a.c1, s)};
}

// Fp2 multiplication: Karatsuba (3 Fp muls)
// (a0+a1*u)(b0+b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
// = (a0*b0 - a1*b1) + ((a0+a1)(b0+b1) - a0*b0 - a1*b1)*u
__device__ __noinline__ RnsFp2 rns_fp2_mul(const RnsFp2& a, const RnsFp2& b) {
    RnsFp t0 = rns_mul(a.c0, b.c0);  // a0*b0
    RnsFp t1 = rns_mul(a.c1, b.c1);  // a1*b1
    RnsFp t2 = rns_mul(rns_add(a.c0, a.c1), rns_add(b.c0, b.c1));  // (a0+a1)(b0+b1)
    return {rns_sub(t0, t1), rns_sub(rns_sub(t2, t0), t1)};
}

// Fp2 squaring: optimized (2 Fp muls + adds)
// (a0+a1*u)^2 = (a0+a1)(a0-a1) + 2*a0*a1*u
__device__ __noinline__ RnsFp2 rns_fp2_sqr(const RnsFp2& a) {
    RnsFp t = rns_mul(a.c0, a.c1);  // a0*a1
    RnsFp sum = rns_add(a.c0, a.c1);
    RnsFp diff = rns_sub(a.c0, a.c1);
    return {rns_mul(sum, diff), rns_add(t, t)};  // (a0+a1)(a0-a1), 2*a0*a1
}

// ============================================================
// Comparison
// ============================================================

// Raw comparison (requires both operands already normalized)
__device__ bool rns_eq_raw(const RnsFp& a, const RnsFp& b) {
    if (a.rr != b.rr) return false;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        if (a.r1[i] != b.r1[i]) return false;
        if (a.r2[i] != b.r2[i]) return false;
    }
    return true;
}

// Normalized comparison: reduces both to [0,p) then raw compare.
// With raw sub (no +p), normalize produces canonical [0,p) values
// because rns_mul(x, ONE) = x mod p for any x in [0, M).
__device__ bool rns_eq(const RnsFp& a, const RnsFp& b) {
    RnsFp na = rns_normalize(a);
    RnsFp nb = rns_normalize(b);
    return rns_eq_raw(na, nb);
}

__device__ bool rns_fp2_eq(const RnsFp2& a, const RnsFp2& b) {
    return rns_eq(a.c0, b.c0) && rns_eq(a.c1, b.c1);
}

// p in RNS form
__device__ __forceinline__ RnsFp rns_p() {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        r.r1[i] = RNS_P_MOD_M1[i];
        r.r2[i] = RNS_P_MOD_M2[i];
    }
    r.rr = RNS_P_MOD_MRED;
    return r;
}

// ============================================================
// Fp6 = Fp2[v] / (v³ - β), β = (1+u)
// ============================================================

struct RnsFp6 { RnsFp2 c0, c1, c2; };

__device__ __forceinline__ RnsFp6 rns_fp6_zero() { return {rns_fp2_zero(), rns_fp2_zero(), rns_fp2_zero()}; }
__device__ __forceinline__ RnsFp6 rns_fp6_one()  { return {rns_fp2_one(), rns_fp2_zero(), rns_fp2_zero()}; }

__device__ __forceinline__ RnsFp6 rns_fp6_add(const RnsFp6& a, const RnsFp6& b) {
    return {rns_fp2_add(a.c0, b.c0), rns_fp2_add(a.c1, b.c1), rns_fp2_add(a.c2, b.c2)};
}
__device__ __forceinline__ RnsFp6 rns_fp6_sub(const RnsFp6& a, const RnsFp6& b) {
    return {rns_fp2_sub(a.c0, b.c0), rns_fp2_sub(a.c1, b.c1), rns_fp2_sub(a.c2, b.c2)};
}
__device__ __forceinline__ RnsFp6 rns_fp6_neg(const RnsFp6& a) {
    return {rns_fp2_neg(a.c0), rns_fp2_neg(a.c1), rns_fp2_neg(a.c2)};
}

// Multiply by v: {c0, c1, c2} → {β*c2, c0, c1} where β = (1+u)
__device__ __forceinline__ RnsFp6 rns_fp6_mul_by_v(const RnsFp6& a) {
    return {rns_fp2_mul_nr(a.c2), a.c0, a.c1};
}

// Fp6 mul: Karatsuba (6 Fp2 muls = 18 Fp muls)
__device__ __noinline__ RnsFp6 rns_fp6_mul(const RnsFp6& a, const RnsFp6& b) {
    RnsFp2 aa = rns_fp2_mul(a.c0, b.c0);
    RnsFp2 bb = rns_fp2_mul(a.c1, b.c1);
    RnsFp2 cc = rns_fp2_mul(a.c2, b.c2);
    return {
        rns_fp2_add(aa, rns_fp2_mul_nr(rns_fp2_sub(rns_fp2_sub(
            rns_fp2_mul(rns_fp2_add(a.c1, a.c2), rns_fp2_add(b.c1, b.c2)), bb), cc))),
        rns_fp2_add(rns_fp2_sub(rns_fp2_sub(
            rns_fp2_mul(rns_fp2_add(a.c0, a.c1), rns_fp2_add(b.c0, b.c1)), aa), bb),
            rns_fp2_mul_nr(cc)),
        rns_fp2_add(rns_fp2_sub(rns_fp2_sub(
            rns_fp2_mul(rns_fp2_add(a.c0, a.c2), rns_fp2_add(b.c0, b.c2)), aa), cc), bb)
    };
}

// Sparse Fp6 mul by (c0, c1, 0)
__device__ __noinline__ RnsFp6 rns_fp6_mul_by_01(const RnsFp6& s, const RnsFp2& c0, const RnsFp2& c1) {
    RnsFp2 aa = rns_fp2_mul(s.c0, c0);
    RnsFp2 bb = rns_fp2_mul(s.c1, c1);
    return {
        rns_fp2_add(rns_fp2_mul_nr(rns_fp2_mul(s.c2, c1)), aa),
        rns_fp2_sub(rns_fp2_sub(rns_fp2_mul(rns_fp2_add(c0, c1), rns_fp2_add(s.c0, s.c1)), aa), bb),
        rns_fp2_add(rns_fp2_mul(s.c2, c0), bb)
    };
}

// Sparse Fp6 mul by (0, c1, 0)
__device__ __noinline__ RnsFp6 rns_fp6_mul_by_1(const RnsFp6& s, const RnsFp2& c1) {
    return {rns_fp2_mul_nr(rns_fp2_mul(s.c2, c1)), rns_fp2_mul(s.c0, c1), rns_fp2_mul(s.c1, c1)};
}

// ============================================================
// Fp12 = Fp6[w] / (w² - v)
// ============================================================

struct RnsFp12 { RnsFp6 c0, c1; };

__device__ __forceinline__ RnsFp12 rns_fp12_one()  { return {rns_fp6_one(), rns_fp6_zero()}; }
__device__ __forceinline__ RnsFp12 rns_fp12_conj(const RnsFp12& a) { return {a.c0, rns_fp6_neg(a.c1)}; }

// Fp12 mul: Karatsuba (3 Fp6 muls = 54 Fp muls)
__device__ __noinline__ RnsFp12 rns_fp12_mul(const RnsFp12& a, const RnsFp12& b) {
    RnsFp6 aa = rns_fp6_mul(a.c0, b.c0);
    RnsFp6 bb = rns_fp6_mul(a.c1, b.c1);
    return {
        rns_fp6_add(aa, rns_fp6_mul_by_v(bb)),
        rns_fp6_sub(rns_fp6_sub(rns_fp6_mul(rns_fp6_add(a.c0, a.c1), rns_fp6_add(b.c0, b.c1)), aa), bb)
    };
}

// Fp12 sqr: (2 Fp6 muls = 36 Fp muls)
__device__ __noinline__ RnsFp12 rns_fp12_sqr(const RnsFp12& a) {
    RnsFp6 ab = rns_fp6_mul(a.c0, a.c1);
    RnsFp6 s1 = rns_fp6_add(a.c0, a.c1);
    RnsFp6 s2 = rns_fp6_add(a.c0, rns_fp6_mul_by_v(a.c1));
    return {
        rns_fp6_sub(rns_fp6_sub(rns_fp6_mul(s2, s1), ab), rns_fp6_mul_by_v(ab)),
        rns_fp6_add(ab, ab)
    };
}

// Cyclotomic square: specialized for elements in cyclotomic subgroup (12 Fp muls via Fp2)
__device__ void rns_fp4_square(const RnsFp2& a, const RnsFp2& b, RnsFp2& out0, RnsFp2& out1) {
    RnsFp2 t0 = rns_fp2_sqr(a), t1 = rns_fp2_sqr(b);
    out0 = rns_fp2_add(rns_fp2_mul_nr(t1), t0);
    out1 = rns_fp2_sub(rns_fp2_sub(rns_fp2_sqr(rns_fp2_add(a, b)), t0), t1);
}

__device__ __noinline__ RnsFp12 rns_cyclotomic_square(const RnsFp12& f) {
    RnsFp2 z0=f.c0.c0, z4=f.c0.c1, z3=f.c0.c2, z2=f.c1.c0, z1=f.c1.c1, z5=f.c1.c2;
    RnsFp2 A0, A1; rns_fp4_square(z0, z1, A0, A1);
    z0 = rns_fp2_add(rns_fp2_add(rns_fp2_sub(A0, z0), rns_fp2_sub(A0, z0)), A0); // 3*A0 - 2*z0
    // Simpler: z0 = (A0 - z0); z0 = z0 + z0 + A0;
    // Let me match the exact sppark formula:
    z0 = f.c0.c0; z4 = f.c0.c1; z3 = f.c0.c2; z2 = f.c1.c0; z1 = f.c1.c1; z5 = f.c1.c2;
    rns_fp4_square(z0, z1, A0, A1);
    RnsFp2 t;
    t = rns_fp2_sub(A0, z0); z0 = rns_fp2_add(rns_fp2_add(t, t), A0);
    t = rns_fp2_add(A1, z1); z1 = rns_fp2_add(rns_fp2_add(t, t), A1);

    RnsFp2 B0, B1; rns_fp4_square(z2, z3, B0, B1);
    RnsFp2 C0, C1; rns_fp4_square(z4, z5, C0, C1);

    t = rns_fp2_sub(B0, z4); z4 = rns_fp2_add(rns_fp2_add(t, t), B0);
    t = rns_fp2_add(B1, z5); z5 = rns_fp2_add(rns_fp2_add(t, t), B1);

    RnsFp2 nr_C1 = rns_fp2_mul_nr(C1);
    t = rns_fp2_add(nr_C1, z2); z2 = rns_fp2_add(rns_fp2_add(t, t), nr_C1);
    t = rns_fp2_sub(C0, z3); z3 = rns_fp2_add(rns_fp2_add(t, t), C0);

    return {{z0, z4, z3}, {z2, z1, z5}};
}

// Forward declarations for inversion chain
__device__ __noinline__ RnsFp rns_fp_inv(const RnsFp& a);
__device__ __noinline__ RnsFp2 rns_fp2_inv(const RnsFp2& a);

// Helper: load an Fp constant from M1/M2/RED arrays
__device__ __forceinline__ RnsFp rns_load_fp(const uint32_t m1[], const uint32_t m2[], uint32_t red) {
    RnsFp r;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) { r.r1[i] = m1[i]; r.r2[i] = m2[i]; }
    r.rr = red;
    return r;
}

// Fp6 inversion
__device__ __noinline__ RnsFp6 rns_fp6_inv(const RnsFp6& f) {
    RnsFp2 c0s=rns_fp2_sqr(f.c0), c1s=rns_fp2_sqr(f.c1), c2s=rns_fp2_sqr(f.c2);
    RnsFp2 c01=rns_fp2_mul(f.c0,f.c1), c02=rns_fp2_mul(f.c0,f.c2), c12=rns_fp2_mul(f.c1,f.c2);
    RnsFp2 t0=rns_fp2_sub(c0s,rns_fp2_mul_nr(c12));
    RnsFp2 t1=rns_fp2_sub(rns_fp2_mul_nr(c2s),c01);
    RnsFp2 t2=rns_fp2_sub(c1s,c02);
    RnsFp2 sc=rns_fp2_add(rns_fp2_mul(f.c0,t0),
        rns_fp2_mul_nr(rns_fp2_add(rns_fp2_mul(f.c2,t1),rns_fp2_mul(f.c1,t2))));
    RnsFp2 si=rns_fp2_inv(sc);
    return {rns_fp2_mul(t0,si), rns_fp2_mul(t1,si), rns_fp2_mul(t2,si)};
}

// Fp2 inversion
__device__ __noinline__ RnsFp2 rns_fp2_inv(const RnsFp2& a) {
    RnsFp n = rns_add(rns_mul(a.c0, a.c0), rns_mul(a.c1, a.c1));
    RnsFp ni = rns_fp_inv(n);
    return {rns_mul(a.c0, ni), rns_neg(rns_mul(a.c1, ni))};
}

// Fp inversion via Fermat: a^(p-2) mod p
__device__ __noinline__ RnsFp rns_fp_inv(const RnsFp& a) {
    // p-2 in 64-bit words (little-endian)
    uint64_t exp[6] = {0xb9feffffffffaaa9ull, 0x1eabfffeb153ffffull,
                       0x6730d2a0f6b0f624ull, 0x64774b84f38512bfull,
                       0x4b1ba7b6434bacd7ull, 0x1a0111ea397fe69aull};
    RnsFp r = rns_one();
    RnsFp base = a;
    for (int w = 0; w < 6; w++) {
        for (int bit = 0; bit < 64; bit++) {
            if (w == 5 && bit >= 61) break;
            if ((exp[w] >> bit) & 1) r = rns_mul(r, base);
            base = rns_mul(base, base);
        }
    }
    return r;
}

// Fp6 Frobenius endomorphism
__device__ __noinline__ RnsFp6 rns_fp6_frob(const RnsFp6& f) {
    RnsFp2 c0 = rns_fp2_conj(f.c0);
    RnsFp2 c1 = rns_fp2_conj(f.c1);
    RnsFp2 c2 = rns_fp2_conj(f.c2);
    // Multiply by frobenius coefficients
    RnsFp2 fc1 = {rns_zero(), rns_load_fp(RNS_FROB6_C1_C1_M1, RNS_FROB6_C1_C1_M2, RNS_FROB6_C1_C1_RED)};
    RnsFp2 fc2 = {rns_load_fp(RNS_FROB6_C2_C0_M1, RNS_FROB6_C2_C0_M2, RNS_FROB6_C2_C0_RED), rns_zero()};

    return {c0, rns_fp2_mul(c1, fc1), rns_fp2_mul(c2, fc2)};
}

// Fp12 Frobenius
__device__ __noinline__ RnsFp12 rns_fp12_frob(const RnsFp12& f) {
    RnsFp6 c0 = rns_fp6_frob(f.c0);
    RnsFp6 c1 = rns_fp6_frob(f.c1);
    // Multiply c1 by frobenius Fp12 coefficient
    RnsFp2 co = {rns_load_fp(RNS_FROB12_C0_M1, RNS_FROB12_C0_M2, RNS_FROB12_C0_RED),
                  rns_load_fp(RNS_FROB12_C1_M1, RNS_FROB12_C1_M2, RNS_FROB12_C1_RED)};
    return {c0, {rns_fp2_mul(c1.c0, co), rns_fp2_mul(c1.c1, co), rns_fp2_mul(c1.c2, co)}};
}

// Fp12 inversion
__device__ __noinline__ RnsFp12 rns_fp12_inv(const RnsFp12& f) {
    RnsFp6 t = rns_fp6_sub(rns_fp6_mul(f.c0, f.c0), rns_fp6_mul_by_v(rns_fp6_mul(f.c1, f.c1)));
    RnsFp6 ti = rns_fp6_inv(t);
    return {rns_fp6_mul(f.c0, ti), rns_fp6_neg(rns_fp6_mul(f.c1, ti))};
}

// Fp12 sparse multiply by (c0, c1, 0, 0, c4, 0)
__device__ __noinline__ RnsFp12 rns_fp12_mul_by_014(const RnsFp12& f,
    const RnsFp2& c0, const RnsFp2& c1, const RnsFp2& c4) {
    RnsFp6 aa = rns_fp6_mul_by_01(f.c0, c0, c1);
    RnsFp6 bb = rns_fp6_mul_by_1(f.c1, c4);
    return {
        rns_fp6_add(rns_fp6_mul_by_v(bb), aa),
        rns_fp6_sub(rns_fp6_sub(rns_fp6_mul_by_01(rns_fp6_add(f.c1, f.c0), c0, rns_fp2_add(c1, c4)), aa), bb)
    };
}

// Cyclotomic exponentiation by BLS_X
// Normalize all Fp elements in an Fp12 to [0,p) — prevents +p drift accumulation
__device__ __noinline__ RnsFp12 rns_fp12_normalize(const RnsFp12& a) {
    return {{
        {rns_normalize(a.c0.c0.c0), rns_normalize(a.c0.c0.c1)},
        {rns_normalize(a.c0.c1.c0), rns_normalize(a.c0.c1.c1)},
        {rns_normalize(a.c0.c2.c0), rns_normalize(a.c0.c2.c1)}
    }, {
        {rns_normalize(a.c1.c0.c0), rns_normalize(a.c1.c0.c1)},
        {rns_normalize(a.c1.c1.c0), rns_normalize(a.c1.c1.c1)},
        {rns_normalize(a.c1.c2.c0), rns_normalize(a.c1.c2.c1)}
    }};
}

__device__ __noinline__ RnsFp12 rns_cyc_exp(const RnsFp12& f_in) {
    // Normalize input to prevent +p drift from accumulating over 64 iterations
    RnsFp12 f = rns_fp12_normalize(f_in);
    RnsFp12 t = rns_fp12_one();
    bool found = false;
    int count = 0;
    for (int i = 63; i >= 0; i--) {
        if (found) t = rns_cyclotomic_square(t);
        bool bit = ((BLS_X >> i) & 1) == 1;
        if (!found) { found = bit; if (!bit) continue; }
        if (bit) t = rns_fp12_mul(t, f);
        // Periodically normalize to prevent drift (every 16 iterations)
        count++;
        if (count % 16 == 0) t = rns_fp12_normalize(t);
    }
    return rns_fp12_conj(t); // BLS_X is negative
}

// Final exponentiation (exact port from ic_bls12_381)
// With periodic normalization to prevent +p drift
__device__ __noinline__ RnsFp12 rns_final_exp(const RnsFp12& f_in) {
    // Easy part: f^((p^6-1)(p^2+1))
    RnsFp12 f = rns_fp12_normalize(f_in);
    RnsFp12 t0 = rns_fp12_conj(f);
    RnsFp12 t1 = rns_fp12_inv(f);
    RnsFp12 t2 = rns_fp12_mul(t0, t1);
    t1 = t2;
    t2 = rns_fp12_normalize(t2); // normalize before frob
    t2 = rns_fp12_mul(rns_fp12_frob(rns_fp12_frob(t2)), t1);

    // Hard part — normalize key intermediates
    f = rns_fp12_normalize(t2);
    t2 = f; // keep normalized copy
    t1 = rns_fp12_conj(rns_cyclotomic_square(t2));
    RnsFp12 t3 = rns_cyc_exp(t2);
    RnsFp12 t4 = rns_fp12_mul(t1, t3);
    t4 = rns_fp12_normalize(t4); // normalize before cyc_exp
    t1 = rns_cyc_exp(t4);
    t4 = rns_fp12_conj(t4);
    f = rns_fp12_mul(f, t4);
    t3 = rns_fp12_normalize(t3);
    t4 = rns_cyclotomic_square(t3);
    t1 = rns_fp12_normalize(t1);
    t0 = rns_cyc_exp(t1);
    t3 = rns_fp12_mul(t3, t0);
    t3 = rns_fp12_normalize(t3); // normalize before frob
    t3 = rns_fp12_frob(rns_fp12_frob(t3));
    f = rns_fp12_mul(f, t3);
    t0 = rns_fp12_normalize(t0);
    t4 = rns_fp12_normalize(t4);
    t4 = rns_fp12_mul(t4, rns_cyc_exp(t0));
    t4 = rns_fp12_normalize(t4);
    f = rns_fp12_mul(f, rns_cyc_exp(t4));
    t4 = rns_fp12_mul(t4, rns_fp12_conj(t2));
    t2 = rns_fp12_mul(t2, t1);
    t2 = rns_fp12_normalize(t2); // normalize before frob
    t2 = rns_fp12_frob(rns_fp12_frob(rns_fp12_frob(t2)));
    f = rns_fp12_mul(f, t2);
    t4 = rns_fp12_normalize(t4); // normalize before frob
    t4 = rns_fp12_frob(t4);
    f = rns_fp12_mul(f, t4);
    return f;
}

// Fp6/Fp12 comparison (normalizes internally)
__device__ bool rns_fp6_eq(const RnsFp6& a, const RnsFp6& b) {
    return rns_fp2_eq(a.c0, b.c0) && rns_fp2_eq(a.c1, b.c1) && rns_fp2_eq(a.c2, b.c2);
}
__device__ bool rns_fp12_eq(const RnsFp12& a, const RnsFp12& b) {
    return rns_fp6_eq(a.c0, b.c0) && rns_fp6_eq(a.c1, b.c1);
}
