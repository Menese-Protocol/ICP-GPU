// RNS BLS12-381 Miller Loop + BLS Verify
// Uses precomputed G2 coefficients in RNS form
#pragma once
#include "rns_fp_v2.cuh"
#include "rns_pairing_constants.cuh"

// Load one Fp from the G2 coefficient arrays at given flat index
__device__ __forceinline__
RnsFp load_g2_fp(int flat_idx) {
    RnsFp r;
    int base = flat_idx * RNS_K;
    #pragma unroll
    for (int i = 0; i < RNS_K; i++) {
        r.r1[i] = G2_COEFFS_R1[base + i];
        r.r2[i] = G2_COEFFS_R2[base + i];
    }
    r.rr = G2_COEFFS_RR[flat_idx];
    return r;
}

// Load coefficient set (c0, c1, c2) where each is Fp2
// Coefficient ci has 6 Fp values at flat index ci*6 + {0..5}
__device__ __forceinline__
void load_coeff(int ci, RnsFp2& c0, RnsFp2& c1, RnsFp2& c2) {
    int base = ci * 6;
    c0 = {load_g2_fp(base + 0), load_g2_fp(base + 1)};
    c1 = {load_g2_fp(base + 2), load_g2_fp(base + 3)};
    c2 = {load_g2_fp(base + 4), load_g2_fp(base + 5)};
}

// Line evaluation: f = f * line(P)
// line uses sparse Fp12 multiply by (c2, c1*P.x, c0*P.y, 0, 0, 0)
__device__ __forceinline__
RnsFp12 ell(const RnsFp12& f, const RnsFp2& c0, const RnsFp2& c1,
            const RnsFp2& c2, const RnsFp& px, const RnsFp& py) {
    return rns_fp12_mul_by_014(f, c2, rns_fp2_mul_fp(c1, px), rns_fp2_mul_fp(c0, py));
}

// rns_fp2_mul_fp already defined in rns_fp_v2.cuh

// Miller loop with precomputed G2 coefficients
__device__ __noinline__
RnsFp12 rns_miller_loop(const RnsFp& px, const RnsFp& py) {
    RnsFp12 f = rns_fp12_one();
    int ci = 0;
    bool found = false;

    for (int b = 63; b >= 0; b--) {
        bool bit = (((BLS_X >> 1) >> b) & 1) == 1;
        if (!found) { found = bit; continue; }

        RnsFp2 c0, c1, c2;
        load_coeff(ci, c0, c1, c2);
        f = ell(f, c0, c1, c2, px, py);
        ci++;

        if (bit) {
            load_coeff(ci, c0, c1, c2);
            f = ell(f, c0, c1, c2, px, py);
            ci++;
        }

        f = rns_fp12_sqr(f);

        // Normalize periodically to prevent +p drift
        if (ci % 10 == 0) f = rns_fp12_normalize(f);
    }

    // Final coefficient
    RnsFp2 c0, c1, c2;
    load_coeff(ci, c0, c1, c2);
    f = ell(f, c0, c1, c2, px, py);

    if (BLS_X_IS_NEG) f = rns_fp12_conj(f);
    return f;
}

// Full pairing: e(P, Q) = final_exp(miller_loop(P, Q_prepared))
__device__ __noinline__
RnsFp12 rns_pairing(const RnsFp& px, const RnsFp& py) {
    RnsFp12 ml = rns_miller_loop(px, py);
    ml = rns_fp12_normalize(ml);
    return rns_final_exp(ml);
}
