// Debug: find which iteration of 2^10 chain breaks
#include <cstdio>
#include <cstdint>
#include "rns_fp.cuh"

// Expose alpha for debugging
__device__ uint32_t debug_last_alpha_b12 = 0;
__device__ double debug_last_alpha_d_b12 = 0.0;

__global__ void debug_chain() {
    if (threadIdx.x != 0) return;

    RnsFp one = rns_one();
    printf("ONE.rr = %u\n", one.rr);
    RnsFp two = rns_add(one, one);
    printf("TWO.rr = %u, TWO.r1[0]=%u\n", two.rr, two.r1[0]);
    RnsFp acc = two;

    // Build expected values by addition: 2, 4, 8, 16, ...
    RnsFp expected = two;

    for (int i = 1; i < 10; i++) {
        acc = rns_mul(acc, two);  // acc = 2^(i+1)
        expected = rns_add(expected, expected);  // double via addition

        bool ok = rns_eq(acc, expected);
        printf("2^%2d: mul %s add\n", i+1, ok ? "==" : "!=");
        if (!ok) {
            printf("  mul.r1[0]=%u  add.r1[0]=%u\n", acc.r1[0], expected.r1[0]);
            printf("  mul.rr=%u  add.rr=%u\n", acc.rr, expected.rr);
            break;
        }
    }
}

int main() {
    debug_chain<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
