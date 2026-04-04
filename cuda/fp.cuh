// BLS12-381 Base Field (Fp) — 381-bit Montgomery form
// GPU implementation using uint64_t[6] limbs
// Reference: blst and sppark for constants, ICICLE for algorithm

#pragma once
#include <cstdint>
#include <cstdio>

// BLS12-381 prime: p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
// Montgomery R = 2^384 mod p
// M0 = -p^{-1} mod 2^64 = 0x89f3fffcfffcfffd

namespace gpu_pairing {

// ============================================================
// Fp: 381-bit prime field element in Montgomery form
// 6 x uint64_t limbs (little-endian)
// ============================================================

struct Fp {
    uint64_t limbs[6];

    // BLS12-381 modulus
    static __device__ __constant__ const uint64_t P[6];
    static __device__ __constant__ const uint64_t P2[6];    // 2*P
    static __device__ __constant__ const uint64_t R_SQR[6]; // R^2 mod P
    static __device__ __constant__ const uint64_t ONE[6];   // R mod P (Montgomery 1)
    static constexpr uint64_t M0 = 0x89f3fffcfffcfffdULL;

    // ---------- Core arithmetic ----------

    // a + b mod p
    __device__ __forceinline__ static Fp add(const Fp& a, const Fp& b) {
        Fp r;
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            __uint128_t s = (__uint128_t)a.limbs[i] + b.limbs[i] + carry;
            r.limbs[i] = (uint64_t)s;
            carry = (uint64_t)(s >> 64);
        }
        // Conditional subtract p
        uint64_t borrow = 0;
        Fp t;
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            __uint128_t d = (__uint128_t)r.limbs[i] - P[i] - borrow;
            t.limbs[i] = (uint64_t)d;
            borrow = (d >> 64) & 1;
        }
        // If borrow, r < p, keep r. Else keep t = r - p.
        if (borrow == 0) return t;
        return r;
    }

    // a - b mod p
    __device__ __forceinline__ static Fp sub(const Fp& a, const Fp& b) {
        Fp r;
        uint64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            __uint128_t d = (__uint128_t)a.limbs[i] - b.limbs[i] - borrow;
            r.limbs[i] = (uint64_t)d;
            borrow = (d >> 64) & 1;
        }
        // If borrow, add p back
        if (borrow) {
            uint64_t carry = 0;
            #pragma unroll
            for (int i = 0; i < 6; i++) {
                __uint128_t s = (__uint128_t)r.limbs[i] + P[i] + carry;
                r.limbs[i] = (uint64_t)s;
                carry = (uint64_t)(s >> 64);
            }
        }
        return r;
    }

    // -a mod p
    __device__ __forceinline__ static Fp neg(const Fp& a) {
        Fp zero = {};
        return sub(zero, a);
    }

    // Montgomery multiplication: a * b * R^{-1} mod p
    __device__ __forceinline__ static Fp mul(const Fp& a, const Fp& b) {
        // CIOS (Coarsely Integrated Operand Scanning) Montgomery multiplication
        uint64_t t[7] = {0};

        #pragma unroll
        for (int i = 0; i < 6; i++) {
            // Multiply-accumulate: t += a * b[i]
            uint64_t carry = 0;
            #pragma unroll
            for (int j = 0; j < 6; j++) {
                __uint128_t prod = (__uint128_t)a.limbs[j] * b.limbs[i] + t[j] + carry;
                t[j] = (uint64_t)prod;
                carry = (uint64_t)(prod >> 64);
            }
            t[6] = carry;

            // Montgomery reduction step
            uint64_t m = t[0] * M0;
            __uint128_t red = (__uint128_t)m * P[0] + t[0];
            carry = (uint64_t)(red >> 64);

            #pragma unroll
            for (int j = 1; j < 6; j++) {
                red = (__uint128_t)m * P[j] + t[j] + carry;
                t[j - 1] = (uint64_t)red;
                carry = (uint64_t)(red >> 64);
            }
            t[5] = t[6] + carry;
            t[6] = (t[5] < carry) ? 1 : 0;
        }

        // Final conditional subtraction
        Fp r;
        #pragma unroll
        for (int i = 0; i < 6; i++) r.limbs[i] = t[i];

        uint64_t borrow = 0;
        Fp s;
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            __uint128_t d = (__uint128_t)r.limbs[i] - P[i] - borrow;
            s.limbs[i] = (uint64_t)d;
            borrow = (d >> 64) & 1;
        }
        if (borrow == 0) return s;
        return r;
    }

    // a^2 mod p (same as mul but can be slightly optimized later)
    __device__ __forceinline__ static Fp sqr(const Fp& a) {
        return mul(a, a);
    }

    __device__ __forceinline__ static Fp zero_val() {
        Fp r = {};
        return r;
    }

    __device__ __forceinline__ static Fp one_val() {
        Fp r;
        #pragma unroll
        for (int i = 0; i < 6; i++) r.limbs[i] = ONE[i];
        return r;
    }

    __device__ __forceinline__ bool is_zero() const {
        uint64_t acc = 0;
        #pragma unroll
        for (int i = 0; i < 6; i++) acc |= limbs[i];
        return acc == 0;
    }

    // Operators for convenience
    __device__ __forceinline__ Fp operator+(const Fp& o) const { return add(*this, o); }
    __device__ __forceinline__ Fp operator-(const Fp& o) const { return sub(*this, o); }
    __device__ __forceinline__ Fp operator*(const Fp& o) const { return mul(*this, o); }
    __device__ __forceinline__ Fp operator-() const { return neg(*this); }
    __device__ __forceinline__ Fp& operator+=(const Fp& o) { *this = *this + o; return *this; }
    __device__ __forceinline__ Fp& operator-=(const Fp& o) { *this = *this - o; return *this; }
    __device__ __forceinline__ Fp& operator*=(const Fp& o) { *this = *this * o; return *this; }
};

// Device constants (defined in fp_constants.cu)
__device__ __constant__ const uint64_t Fp::P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};

__device__ __constant__ const uint64_t Fp::P2[6] = {
    0x73fdffffffff5556ULL, 0x3d57fffd62a7ffffULL,
    0xce61a541ed61ec48ULL, 0xc8ee9709e70a257eULL,
    0x96374f6c869759aeULL, 0x340223d472ffcd34ULL
};

__device__ __constant__ const uint64_t Fp::R_SQR[6] = {
    0xf4df1f341c341746ULL, 0x0a76e6a609d104f1ULL,
    0x8de5476c4c95b6d5ULL, 0x67eb88a9939d83c0ULL,
    0x9a793e85b519952dULL, 0x11988fe592cae3aaULL
};

__device__ __constant__ const uint64_t Fp::ONE[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL,
    0x5f48985753c758baULL, 0x77ce585370525745ULL,
    0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL
};

} // namespace gpu_pairing
