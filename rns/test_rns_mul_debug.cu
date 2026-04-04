// Debug: trace each step of rns_mul to find the divergence
#include <cstdio>
#include <cstdint>
#include "rns_fp.cuh"

__global__ void debug_kernel() {
    if (threadIdx.x != 0) return;

    // mont(7) from oracle
    RnsFp a;
    for (int i = 0; i < RNS_K; i++) {
        a.r1[i] = RNS_ORACLE_MONT7_M1[i];
        a.r2[i] = RNS_ORACLE_MONT7_M2[i];
    }

    printf("=== RNS mul debug: mont(7) * mont(7) ===\n\n");

    // Step 1: q = a * a
    uint32_t q1[RNS_K], q2[RNS_K];
    for (int i = 0; i < RNS_K; i++) {
        q1[i] = mod_mul(a.r1[i], a.r1[i], RNS_M1[i]);
        q2[i] = mod_mul(a.r2[i], a.r2[i], RNS_M2[i]);
    }
    printf("Step 1 - q = a*a in B1:\n  ");
    for (int i = 0; i < RNS_K; i++) printf("%u ", q1[i]);

    // Python expected q1: 802282646 183549360 424177400 147040499 488550639 807375552 905114669 953067030 261195697 925881429 233416539 743728373 362441457 192863711
    printf("\n  Expected: 802282646 183549360 424177400 147040499 488550639 807375552 905114669 953067030 261195697 925881429 233416539 743728373 362441457 192863711\n");

    // Step 2: t = q * (-p^-1) in B1
    uint32_t t1[RNS_K];
    for (int i = 0; i < RNS_K; i++) {
        t1[i] = mod_mul(q1[i], RNS_NEG_PINV_M1[i], RNS_M1[i]);
    }
    printf("\nStep 2 - t = q*(-p^-1) in B1:\n  ");
    for (int i = 0; i < RNS_K; i++) printf("%u ", t1[i]);
    printf("\n  Expected: 97649642 242164948 211185841 1067946485 1021763924 702423266 165058719 210617947 33464787 84608615 93319377 612166296 667961219 277774837\n");

    // Step 3: base extend t from B1 to B2
    uint32_t t2[RNS_K];
    base_extend_12(t1, t2);
    printf("\nStep 3 - t extended to B2:\n  ");
    for (int i = 0; i < RNS_K; i++) printf("%u ", t2[i]);
    printf("\n  Expected: 615773812 914326032 826202594 264772346 491868855 923034952 333746865 259479712 140307168 241929495 135845717 267197777 56330955 952574388\n");

    // Step 4: r = (q + t*p) * M1_inv in B2
    printf("\nStep 4 - r in B2:\n  ");
    for (int j = 0; j < RNS_K; j++) {
        uint32_t tp = mod_mul(t2[j], RNS_P_MOD_M2[j], RNS_M2[j]);
        uint32_t sum = mod_add(q2[j], tp, RNS_M2[j]);
        uint32_t r_j = mod_mul(sum, RNS_M1_INV_MOD_M2[j], RNS_M2[j]);
        printf("%u ", r_j);
    }

    // What should r be? Python: mont(49) in B2
    printf("\n  Expected mont(49) B2: ");
    for (int i = 0; i < RNS_K; i++) printf("%u ", RNS_ORACLE_MONT49_M2[i]);
    printf("\n");

    // Also check: what does the full rns_mul produce?
    RnsFp result = rns_mul(a, a);
    printf("\nFull rns_mul B1: ");
    for (int i = 0; i < RNS_K; i++) printf("%u ", result.r1[i]);
    printf("\nExpected    B1:  ");
    for (int i = 0; i < RNS_K; i++) printf("%u ", RNS_ORACLE_MONT49_M1[i]);
    printf("\n");
}

int main() {
    debug_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
