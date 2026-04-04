// Step 1: MINIMAL test — just Fp Montgomery arithmetic on GPU
// Compile: nvcc -o test_fp test_fp_only.cu -arch=sm_120 -O2
// Should take <30 seconds

#include <cstdint>
#include <cstdio>

struct Fp {
    uint64_t v[6];
};

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

__device__ Fp fp_zero() {
    Fp r = {}; return r;
}

__device__ Fp fp_one() {
    Fp r;
    for (int i = 0; i < 6; i++) r.v[i] = FP_ONE[i];
    return r;
}

__device__ bool fp_eq(const Fp& a, const Fp& b) {
    for (int i = 0; i < 6; i++) if (a.v[i] != b.v[i]) return false;
    return true;
}

__device__ bool fp_is_zero(const Fp& a) {
    uint64_t acc = 0;
    for (int i = 0; i < 6; i++) acc |= a.v[i];
    return acc == 0;
}

__device__ Fp fp_add(const Fp& a, const Fp& b) {
    Fp r;
    unsigned __int128 carry = 0;
    for (int i = 0; i < 6; i++) {
        unsigned __int128 s = (unsigned __int128)a.v[i] + b.v[i] + carry;
        r.v[i] = (uint64_t)s;
        carry = s >> 64;
    }
    // Conditional subtract p
    Fp t;
    unsigned __int128 borrow = 0;
    for (int i = 0; i < 6; i++) {
        unsigned __int128 d = (unsigned __int128)r.v[i] - FP_P[i] - borrow;
        t.v[i] = (uint64_t)d;
        borrow = (d >> 127) & 1; // MSB of 128-bit = borrow
    }
    return (borrow == 0) ? t : r;
}

__device__ Fp fp_sub(const Fp& a, const Fp& b) {
    Fp r;
    unsigned __int128 borrow = 0;
    for (int i = 0; i < 6; i++) {
        unsigned __int128 d = (unsigned __int128)a.v[i] - b.v[i] - borrow;
        r.v[i] = (uint64_t)d;
        borrow = (d >> 127) & 1;
    }
    if (borrow) {
        unsigned __int128 carry = 0;
        for (int i = 0; i < 6; i++) {
            unsigned __int128 s = (unsigned __int128)r.v[i] + FP_P[i] + carry;
            r.v[i] = (uint64_t)s;
            carry = s >> 64;
        }
    }
    return r;
}

__device__ Fp fp_neg(const Fp& a) {
    if (fp_is_zero(a)) return a;
    return fp_sub(fp_zero(), a);
}

__device__ Fp fp_mul(const Fp& a, const Fp& b) {
    uint64_t t[7] = {0};

    for (int i = 0; i < 6; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 6; j++) {
            unsigned __int128 prod = (unsigned __int128)a.v[j] * b.v[i] + t[j] + carry;
            t[j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[6] = carry;

        uint64_t m = t[0] * M0;
        unsigned __int128 red = (unsigned __int128)m * FP_P[0] + t[0];
        carry = (uint64_t)(red >> 64);

        for (int j = 1; j < 6; j++) {
            red = (unsigned __int128)m * FP_P[j] + t[j] + carry;
            t[j - 1] = (uint64_t)red;
            carry = (uint64_t)(red >> 64);
        }
        t[5] = t[6] + carry;
        t[6] = (t[5] < carry) ? 1 : 0;
    }

    Fp r;
    for (int i = 0; i < 6; i++) r.v[i] = t[i];

    // Conditional subtraction
    Fp s;
    unsigned __int128 borrow = 0;
    for (int i = 0; i < 6; i++) {
        unsigned __int128 d = (unsigned __int128)r.v[i] - FP_P[i] - borrow;
        s.v[i] = (uint64_t)d;
        borrow = (d >> 127) & 1;
    }
    return (borrow == 0) ? s : r;
}

__global__ void test_fp() {
    Fp one = fp_one();
    Fp zero = fp_zero();

    // Test 1: one * one == one
    Fp r1 = fp_mul(one, one);
    printf("TEST 1 (1*1=1): %s\n", fp_eq(r1, one) ? "PASS" : "FAIL");

    // Test 2: one + zero == one
    Fp r2 = fp_add(one, zero);
    printf("TEST 2 (1+0=1): %s\n", fp_eq(r2, one) ? "PASS" : "FAIL");

    // Test 3: one - one == zero
    Fp r3 = fp_sub(one, one);
    printf("TEST 3 (1-1=0): %s\n", fp_is_zero(r3) ? "PASS" : "FAIL");

    // Test 4: neg(one) + one == zero
    Fp neg1 = fp_neg(one);
    Fp r4 = fp_add(neg1, one);
    printf("TEST 4 (-1+1=0): %s\n", fp_is_zero(r4) ? "PASS" : "FAIL");

    // Test 5: two = one + one, two * two should give mont(4)
    Fp two = fp_add(one, one);
    Fp four = fp_mul(two, two);
    Fp four_check = fp_add(fp_add(one, one), fp_add(one, one));
    printf("TEST 5 (2*2=4): %s\n", fp_eq(four, four_check) ? "PASS" : "FAIL");

    // Test 6: Compare with Python oracle
    // Python says: mont(42) = 0xef9d00000089aa21, 0x8484021beb7c0070, ...
    // We compute 42 = 42 * R mod p. We can build 42 from 1s:
    // 42 = 32 + 8 + 2 = (((1+1)*(1+1))*(1+1) + 1+1)*(1+1) + ... too complex
    // Instead let's verify: mont(2) * mont(3) should give mont(6)
    Fp three = fp_add(two, one);
    Fp six_mul = fp_mul(two, three);
    Fp six_add = fp_add(three, three);
    printf("TEST 6 (2*3=6): %s\n", fp_eq(six_mul, six_add) ? "PASS" : "FAIL");

    // Test 7: (a-b) + b == a
    Fp five = fp_add(two, three);
    Fp diff = fp_sub(five, three);
    printf("TEST 7 (5-3+3=5): %s\n", fp_eq(fp_add(diff, three), five) ? "PASS" : "FAIL");

    // Print mont(2) for cross-check with Python
    printf("\nmont(2) = [");
    for (int i = 0; i < 6; i++) printf("0x%016llx%s", (unsigned long long)two.v[i], i<5?", ":"");
    printf("]\n");

    printf("mont(6) = [");
    for (int i = 0; i < 6; i++) printf("0x%016llx%s", (unsigned long long)six_mul.v[i], i<5?", ":"");
    printf("]\n");
}

int main() {
    printf("=== Fp Montgomery Arithmetic GPU Test ===\n");
    test_fp<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("=== Done ===\n");
    return 0;
}
