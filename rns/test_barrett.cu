// Test Barrett reduction correctness
#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"

__global__ void test_barrett() {
    if (threadIdx.x != 0) return;

    int fail = 0;

    // Test barrett_red against % for all moduli
    uint32_t test_a[] = {0, 1, 1000, 999999999, 1073741788};
    uint32_t test_b[] = {1, 2, 12345, 777777777, 1073741788};

    for (int ti = 0; ti < 5; ti++) {
        for (int tj = 0; tj < 5; tj++) {
            uint32_t a = test_a[ti], b = test_b[tj];
            // Test against M1[0]
            uint64_t prod = (uint64_t)a * b;
            uint32_t expected = (uint32_t)(prod % RNS_M1[0]);
            uint32_t got = barrett_red(prod, RNS_M1[0], RNS_BARRETT_M1[0]);
            if (got != expected) {
                printf("BARRETT FAIL: %u * %u mod %u: got %u expected %u\n",
                       a, b, RNS_M1[0], got, expected);
                fail++;
            }
        }
    }

    // Test mmul specifically
    for (int i = 0; i < RNS_K; i++) {
        uint32_t a = 999999999, b = 888888888;
        uint32_t expected = (uint32_t)(((uint64_t)a * b) % RNS_M1[i]);
        uint32_t got = mmul(a, b, RNS_M1[i], RNS_BARRETT_M1[i]);
        if (got != expected) {
            printf("MMUL FAIL M1[%d]: %u * %u = got %u expected %u\n", i, a, b, got, expected);
            fail++;
        }
    }
    for (int i = 0; i < RNS_K; i++) {
        uint32_t a = 999999999, b = 888888888;
        uint32_t expected = (uint32_t)(((uint64_t)a * b) % RNS_M2[i]);
        uint32_t got = mmul(a, b, RNS_M2[i], RNS_BARRETT_M2[i]);
        if (got != expected) {
            printf("MMUL FAIL M2[%d]: %u * %u = got %u expected %u\n", i, a, b, got, expected);
            fail++;
        }
    }

    // Test bred64 for large accumulator values (simulating base extension)
    {
        uint64_t acc = 0;
        for (int i = 0; i < RNS_K; i++)
            acc += (uint64_t)999999999 * 999999999;  // 14 * ~10^18 ≈ 1.4 * 10^19 < 2^64
        uint32_t expected = (uint32_t)(acc % RNS_M2[0]);
        uint32_t got = bred64(acc, RNS_M2[0], RNS_BARRETT_M2[0]);
        if (got != expected) {
            printf("BRED64 FAIL: acc=%lu mod %u: got %u expected %u\n", acc, RNS_M2[0], got, expected);
            fail++;
        } else {
            printf("BRED64 large acc: PASS (acc=%lu)\n", acc);
        }
    }

    if (fail == 0)
        printf("All Barrett tests PASS\n");
    else
        printf("%d Barrett tests FAIL\n", fail);
}

int main() {
    test_barrett<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
