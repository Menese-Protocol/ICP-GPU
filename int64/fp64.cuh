// BLS12-381 Fp arithmetic — 6×64-bit limb Montgomery multiplication
// Designed for Blackwell (sm_120) native INT64 support
// Oracle-verified against ic_bls12_381 v0.10.1
//
// Copyright 2026 Mercatura Forum
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

// BLS12-381 modulus p in 6×64-bit LE limbs
__device__ __constant__ uint64_t FP_P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};

// Montgomery R = 2^384 mod p
__device__ __constant__ uint64_t FP_ONE[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL,
    0x5f48985753c758baULL, 0x77ce585370525745ULL,
    0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL
};

// R^2 mod p (for converting to Montgomery form)
__device__ __constant__ uint64_t FP_R2[6] = {
    0xf4df1f341c341746ULL, 0x0a76e6a609d104f1ULL,
    0x8de5476c4c95b6d5ULL, 0x67eb88a939d83c08ULL,
    0x9a793e85b519952dULL, 0x11988fe592cae3aaULL
};

// M0 = -p^(-1) mod 2^64 (Montgomery constant)
#define FP_M0 0x89f3fffcfffcfffdULL

// ============================================================
// Core 384-bit arithmetic with 6×64-bit limbs
// Uses native INT64 multiply (mul.lo.u64, mul.hi.u64)
// ============================================================

struct Fp {
    uint64_t v[6];
};

// Multiply-accumulate: acc += a * b, return carry
__device__ __forceinline__
uint64_t mac(uint64_t a, uint64_t b, uint64_t acc, uint64_t* carry) {
    uint64_t lo, hi;
    asm("mul.lo.u64 %0, %2, %3;\n"
        "mul.hi.u64 %1, %2, %3;\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b));
    // acc + lo + carry_in
    uint64_t sum = acc + lo;
    uint64_t c1 = (sum < acc) ? 1ULL : 0ULL;
    uint64_t sum2 = sum + *carry;
    uint64_t c2 = (sum2 < sum) ? 1ULL : 0ULL;
    *carry = hi + c1 + c2;
    return sum2;
}

// Add with carry
__device__ __forceinline__
uint64_t adc(uint64_t a, uint64_t b, uint64_t* carry) {
    uint64_t sum = a + b + *carry;
    *carry = ((sum < a) || (*carry && sum == a)) ? 1ULL : 0ULL;
    return sum;
}

// Subtract with borrow
__device__ __forceinline__
uint64_t sbb(uint64_t a, uint64_t b, uint64_t* borrow) {
    uint64_t diff = a - b - *borrow;
    *borrow = ((a < b + *borrow) || (*borrow && b == 0xFFFFFFFFFFFFFFFFULL)) ? 1ULL : 0ULL;
    return diff;
}

// ==================== Fp operations ====================

__device__ __forceinline__ Fp fp_zero() {
    Fp r = {{0,0,0,0,0,0}};
    return r;
}

__device__ __forceinline__ Fp fp_one() {
    Fp r;
    for (int i = 0; i < 6; i++) r.v[i] = FP_ONE[i];
    return r;
}

__device__ __forceinline__ bool fp_is_zero(const Fp& a) {
    uint64_t acc = 0;
    for (int i = 0; i < 6; i++) acc |= a.v[i];
    return acc == 0;
}

__device__ __forceinline__ bool fp_eq(const Fp& a, const Fp& b) {
    for (int i = 0; i < 6; i++) if (a.v[i] != b.v[i]) return false;
    return true;
}

// a + b mod p
__device__ __forceinline__ Fp fp_add(const Fp& a, const Fp& b) {
    Fp r;
    uint64_t carry = 0;
    for (int i = 0; i < 6; i++) {
        uint64_t sum = a.v[i] + b.v[i] + carry;
        carry = (sum < a.v[i] || (carry && sum == a.v[i])) ? 1ULL : 0ULL;
        r.v[i] = sum;
    }
    // Conditional subtract p
    Fp t;
    uint64_t borrow = 0;
    for (int i = 0; i < 6; i++) t.v[i] = sbb(r.v[i], FP_P[i], &borrow);
    // If no borrow, use t (r >= p). If borrow, keep r (r < p).
    bool use_t = (borrow == 0);
    for (int i = 0; i < 6; i++) r.v[i] = use_t ? t.v[i] : r.v[i];
    return r;
}

// a - b mod p
__device__ __forceinline__ Fp fp_sub(const Fp& a, const Fp& b) {
    Fp r;
    uint64_t borrow = 0;
    for (int i = 0; i < 6; i++) r.v[i] = sbb(a.v[i], b.v[i], &borrow);
    // If borrow, add p
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 6; i++) r.v[i] = adc(r.v[i], FP_P[i], &carry);
    }
    return r;
}

// Negation: -a mod p
__device__ __forceinline__ Fp fp_neg(const Fp& a) {
    if (fp_is_zero(a)) return a;
    return fp_sub(fp_zero(), a);
}

// ==================== Montgomery Multiplication ====================
// CIOS (Coarsely Integrated Operand Scanning) method
// Same algorithm as ic_bls12_381, adapted for GPU INT64

__device__ __noinline__ Fp fp_mul(const Fp& a, const Fp& b) {
    // t is the intermediate accumulator (7 limbs to handle overflow)
    uint64_t t[7] = {0,0,0,0,0,0,0};

    for (int i = 0; i < 6; i++) {
        // Step 1: t += a * b[i]
        uint64_t carry = 0;
        for (int j = 0; j < 6; j++) {
            t[j] = mac(a.v[j], b.v[i], t[j], &carry);
        }
        t[6] += carry;

        // Step 2: Montgomery reduction
        // m = t[0] * M0 mod 2^64
        uint64_t m = t[0] * FP_M0;

        // t += m * p (and shift right by one limb)
        carry = 0;
        uint64_t discard = mac(m, FP_P[0], t[0], &carry); // low word discarded
        (void)discard;
        for (int j = 1; j < 6; j++) {
            t[j-1] = mac(m, FP_P[j], t[j], &carry);
        }
        t[5] = t[6] + carry;
        t[6] = 0;
    }

    // Final conditional subtraction
    Fp r;
    for (int i = 0; i < 6; i++) r.v[i] = t[i];

    // If r >= p, subtract p
    uint64_t borrow = 0;
    Fp s;
    for (int i = 0; i < 6; i++) s.v[i] = sbb(r.v[i], FP_P[i], &borrow);
    bool use_s = (borrow == 0);
    for (int i = 0; i < 6; i++) r.v[i] = use_s ? s.v[i] : r.v[i];

    return r;
}

// Squaring (same as mul for now — can optimize later)
__device__ __forceinline__ Fp fp_sqr(const Fp& a) {
    return fp_mul(a, a);
}

// Inversion via Fermat: a^(p-2) mod p
__device__ __noinline__ Fp fp_inv(const Fp& a) {
    uint64_t exp[6] = {
        0xb9feffffffffaaa9ULL, 0x1eabfffeb153ffffULL,
        0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
        0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
    };
    Fp r = fp_one();
    Fp base = a;
    for (int w = 0; w < 6; w++) {
        for (int bit = 0; bit < 64; bit++) {
            if (w == 5 && bit >= 61) break;
            if ((exp[w] >> bit) & 1) r = fp_mul(r, base);
            base = fp_sqr(base);
        }
    }
    return r;
}

// ==================== Fp2 = Fp[u]/(u²+1) ====================
struct Fp2 { Fp c0, c1; };

__device__ __forceinline__ Fp2 fp2_zero() { return {fp_zero(), fp_zero()}; }
__device__ __forceinline__ Fp2 fp2_one()  { return {fp_one(), fp_zero()}; }
__device__ __forceinline__ Fp2 fp2_add(const Fp2& a, const Fp2& b) { return {fp_add(a.c0,b.c0), fp_add(a.c1,b.c1)}; }
__device__ __forceinline__ Fp2 fp2_sub(const Fp2& a, const Fp2& b) { return {fp_sub(a.c0,b.c0), fp_sub(a.c1,b.c1)}; }
__device__ __forceinline__ Fp2 fp2_neg(const Fp2& a) { return {fp_neg(a.c0), fp_neg(a.c1)}; }
__device__ __forceinline__ Fp2 fp2_conj(const Fp2& a) { return {a.c0, fp_neg(a.c1)}; }
__device__ __forceinline__ Fp2 fp2_mul_nr(const Fp2& a) { return {fp_sub(a.c0,a.c1), fp_add(a.c0,a.c1)}; }
__device__ __forceinline__ Fp2 fp2_mul_fp(const Fp2& a, const Fp& s) { return {fp_mul(a.c0,s), fp_mul(a.c1,s)}; }

__device__ __noinline__ Fp2 fp2_mul(const Fp2& a, const Fp2& b) {
    Fp t0 = fp_mul(a.c0, b.c0);
    Fp t1 = fp_mul(a.c1, b.c1);
    Fp t2 = fp_mul(fp_add(a.c0, a.c1), fp_add(b.c0, b.c1));
    return {fp_sub(t0, t1), fp_sub(fp_sub(t2, t0), t1)};
}

__device__ __noinline__ Fp2 fp2_sqr(const Fp2& a) {
    Fp t = fp_mul(a.c0, a.c1);
    return {fp_mul(fp_add(a.c0, a.c1), fp_sub(a.c0, a.c1)), fp_add(t, t)};
}

__device__ __noinline__ Fp2 fp2_inv(const Fp2& a) {
    Fp n = fp_add(fp_sqr(a.c0), fp_sqr(a.c1));
    Fp ni = fp_inv(n);
    return {fp_mul(a.c0, ni), fp_neg(fp_mul(a.c1, ni))};
}
