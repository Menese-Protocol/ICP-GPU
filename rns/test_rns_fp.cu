// Oracle test for RNS Fp arithmetic
// Tests: encode/decode, add, sub, mul against precomputed values
// Compile: nvcc -o test_rns_fp test_rns_fp.cu -arch=sm_120 -O1
#include <cstdio>
#include <cstdint>
#include "rns_fp.cuh"

__global__ void test_kernel(int* results) {
    if (threadIdx.x != 0) return;
    int pass = 0, fail = 0;

    // ---- Test 1: ONE is correctly encoded ----
    {
        RnsFp one = rns_one();
        bool ok = true;
        for (int i = 0; i < RNS_K; i++) {
            if (one.r1[i] != RNS_R_MOD_M1[i]) ok = false;
            if (one.r2[i] != RNS_R_MOD_M2[i]) ok = false;
        }
        printf("Test 1: ONE encoding: %s\n", ok?"PASS":"FAIL");
        if (ok) pass++; else fail++;
    }

    // ---- Test 2: ONE + ONE ----
    {
        RnsFp one = rns_one();
        RnsFp two = rns_add(one, one);
        // two should equal R2_MOD = (2 * R mod p) in RNS
        // We can verify by checking 2*R_MOD_M1[i] mod M1[i]
        bool ok = true;
        for (int i = 0; i < RNS_K; i++) {
            uint32_t expected = mod_add(RNS_R_MOD_M1[i], RNS_R_MOD_M1[i], RNS_M1[i]);
            if (two.r1[i] != expected) { ok = false; printf("  B1[%d]: got %u exp %u\n", i, two.r1[i], expected); }
        }
        for (int i = 0; i < RNS_K; i++) {
            uint32_t expected = mod_add(RNS_R_MOD_M2[i], RNS_R_MOD_M2[i], RNS_M2[i]);
            if (two.r2[i] != expected) { ok = false; printf("  B2[%d]: got %u exp %u\n", i, two.r2[i], expected); }
        }
        printf("Test 2: ONE + ONE = 2*ONE: %s\n", ok?"PASS":"FAIL");
        if (ok) pass++; else fail++;
    }

    // ---- Test 3: ONE - ONE = ZERO ----
    {
        RnsFp one = rns_one();
        RnsFp z = rns_sub(one, one);
        RnsFp zero = rns_zero();
        bool ok = rns_eq(z, zero);
        printf("Test 3: ONE - ONE = ZERO: %s\n", ok?"PASS":"FAIL");
        if (ok) pass++; else fail++;
    }

    // ---- Test 4: NEG(ONE) + ONE = ZERO ----
    {
        RnsFp one = rns_one();
        RnsFp neg_one = rns_neg(one);
        RnsFp z = rns_add(neg_one, one);
        RnsFp zero = rns_zero();
        bool ok = rns_eq(z, zero);
        printf("Test 4: NEG(ONE) + ONE = ZERO: %s\n", ok?"PASS":"FAIL");
        if (ok) pass++; else fail++;
    }

    // ---- Test 5: ONE * ONE = ONE (Montgomery mul identity) ----
    {
        RnsFp one = rns_one();
        RnsFp r = rns_mul(one, one);
        bool ok = rns_eq(r, one);
        printf("Test 5: ONE * ONE = ONE: %s\n", ok?"PASS":"FAIL");
        if (!ok) {
            printf("  Expected B1: ");
            for (int i = 0; i < RNS_K; i++) printf("%u ", one.r1[i]);
            printf("\n  Got      B1: ");
            for (int i = 0; i < RNS_K; i++) printf("%u ", r.r1[i]);
            printf("\n  Expected B2: ");
            for (int i = 0; i < RNS_K; i++) printf("%u ", one.r2[i]);
            printf("\n  Got      B2: ");
            for (int i = 0; i < RNS_K; i++) printf("%u ", r.r2[i]);
            printf("\n");
        }
        if (ok) pass++; else fail++;
    }

    // ---- Test 6: mont(7) * mont(7) = mont(49) ----
    {
        // mont(7) and mont(49) precomputed by Python oracle
        RnsFp m7;
        for (int i = 0; i < RNS_K; i++) {
            m7.r1[i] = RNS_ORACLE_MONT7_M1[i];
            m7.r2[i] = RNS_ORACLE_MONT7_M2[i];
        }
        m7.rr = RNS_ORACLE_MONT7_RED;
        RnsFp m49_expected;
        for (int i = 0; i < RNS_K; i++) {
            m49_expected.r1[i] = RNS_ORACLE_MONT49_M1[i];
            m49_expected.r2[i] = RNS_ORACLE_MONT49_M2[i];
        }
        m49_expected.rr = RNS_ORACLE_MONT49_RED;

        RnsFp m49_got = rns_mul(m7, m7);
        bool ok = rns_eq(m49_got, m49_expected);
        printf("Test 6: mont(7)^2 = mont(49): %s\n", ok?"PASS":"FAIL");
        if (!ok) {
            printf("  Diff B1: ");
            for (int i = 0; i < RNS_K; i++) {
                if (m49_got.r1[i] != m49_expected.r1[i])
                    printf("[%d]got=%u exp=%u ", i, m49_got.r1[i], m49_expected.r1[i]);
            }
            printf("\n  Diff B2: ");
            for (int i = 0; i < RNS_K; i++) {
                if (m49_got.r2[i] != m49_expected.r2[i])
                    printf("[%d]got=%u exp=%u ", i, m49_got.r2[i], m49_expected.r2[i]);
            }
            printf("\n");
        }
        if (ok) pass++; else fail++;
    }

    // ---- Test 7: Base extension round-trip ----
    // Encode in B1, extend to B2, extend back to B1
    {
        RnsFp one = rns_one();
        uint32_t b2_ext[RNS_K];
        base_extend_12(one.r1, b2_ext);

        // Check if extended B2 matches the precomputed B2 of ONE
        bool ok = true;
        for (int i = 0; i < RNS_K; i++) {
            if (b2_ext[i] != one.r2[i]) {
                ok = false;
                printf("  BE12 mismatch [%d]: got %u exp %u\n", i, b2_ext[i], one.r2[i]);
            }
        }
        printf("Test 7: Base extend B1->B2 (ONE): %s\n", ok?"PASS":"FAIL");
        if (ok) pass++; else fail++;
    }

    // ---- Test 8: Base extension B2->B1 round-trip ----
    {
        RnsFp one = rns_one();
        uint32_t b1_ext[RNS_K];
        base_extend_21(one.r2, b1_ext);

        bool ok = true;
        for (int i = 0; i < RNS_K; i++) {
            if (b1_ext[i] != one.r1[i]) {
                ok = false;
                printf("  BE21 mismatch [%d]: got %u exp %u\n", i, b1_ext[i], one.r1[i]);
            }
        }
        printf("Test 8: Base extend B2->B1 (ONE): %s\n", ok?"PASS":"FAIL");
        if (ok) pass++; else fail++;
    }

    printf("\n=== RESULTS: %d PASS, %d FAIL ===\n", pass, fail);
    results[0] = pass;
    results[1] = fail;
}

// Benchmark kernel
__global__ void bench_kernel(uint64_t* out_cycles, int n_iters) {
    if (threadIdx.x != 0) return;

    RnsFp a;
    for (int i = 0; i < RNS_K; i++) {
        a.r1[i] = RNS_ORACLE_MONT7_M1[i];
        a.r2[i] = RNS_ORACLE_MONT7_M2[i];
    }

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    RnsFp acc = a;
    for (int i = 0; i < n_iters; i++) {
        acc = rns_mul(acc, a);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    out_cycles[0] = t1 - t0;
    out_cycles[1] = acc.r1[0]; // anti-DCE
}

int main() {
    printf("=== RNS Fp Oracle Test (BLS12-381) ===\n\n");

    int *d_results;
    cudaMalloc(&d_results, 2 * sizeof(int));
    test_kernel<<<1, 1>>>(d_results);
    cudaDeviceSynchronize();

    int results[2];
    cudaMemcpy(results, d_results, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    if (results[1] > 0) {
        printf("\n*** FAILURES DETECTED — investigate before proceeding ***\n");
    }

    // Benchmark
    printf("\n=== Benchmark: 10000 RNS fp_mul ===\n");
    int n_iters = 10000;
    uint64_t *d_cycles;
    cudaMalloc(&d_cycles, 2 * sizeof(uint64_t));
    bench_kernel<<<1, 1>>>(d_cycles, n_iters);
    cudaDeviceSynchronize();

    uint64_t h_cycles[2];
    cudaMemcpy(h_cycles, d_cycles, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_cycles);

    int device;
    cudaGetDevice(&device);
    int clock_khz;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
    double clock_ghz = clock_khz / 1e6;

    double ns = (double)h_cycles[0] / (clock_ghz * n_iters);
    printf("  RNS fp_mul: %.0f ns/op (%lu cycles total)\n", ns, h_cycles[0]);
    printf("  GPU clock: %.2f GHz\n", clock_ghz);
    printf("  Compare: sppark fp_mul was 328 ns/op\n");

    return results[1] > 0 ? 1 : 0;
}
