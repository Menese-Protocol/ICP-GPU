// Oracle test: RNS pairing e(G1, G2) vs Rust ic_bls12_381
#include <cstdio>
#include <cstdint>
#include "rns_pairing.cuh"

// Load oracle Fp12 result
__device__ RnsFp12 load_oracle_gt() {
    RnsFp12 gt;
    gt.c0.c0 = {rns_load_fp(ORACLE_GT_C0C0C0_R1, ORACLE_GT_C0C0C0_R2, ORACLE_GT_C0C0C0_RR),
                 rns_load_fp(ORACLE_GT_C0C0C1_R1, ORACLE_GT_C0C0C1_R2, ORACLE_GT_C0C0C1_RR)};
    gt.c0.c1 = {rns_load_fp(ORACLE_GT_C0C1C0_R1, ORACLE_GT_C0C1C0_R2, ORACLE_GT_C0C1C0_RR),
                 rns_load_fp(ORACLE_GT_C0C1C1_R1, ORACLE_GT_C0C1C1_R2, ORACLE_GT_C0C1C1_RR)};
    gt.c0.c2 = {rns_load_fp(ORACLE_GT_C0C2C0_R1, ORACLE_GT_C0C2C0_R2, ORACLE_GT_C0C2C0_RR),
                 rns_load_fp(ORACLE_GT_C0C2C1_R1, ORACLE_GT_C0C2C1_R2, ORACLE_GT_C0C2C1_RR)};
    gt.c1.c0 = {rns_load_fp(ORACLE_GT_C1C0C0_R1, ORACLE_GT_C1C0C0_R2, ORACLE_GT_C1C0C0_RR),
                 rns_load_fp(ORACLE_GT_C1C0C1_R1, ORACLE_GT_C1C0C1_R2, ORACLE_GT_C1C0C1_RR)};
    gt.c1.c1 = {rns_load_fp(ORACLE_GT_C1C1C0_R1, ORACLE_GT_C1C1C0_R2, ORACLE_GT_C1C1C0_RR),
                 rns_load_fp(ORACLE_GT_C1C1C1_R1, ORACLE_GT_C1C1C1_R2, ORACLE_GT_C1C1C1_RR)};
    gt.c1.c2 = {rns_load_fp(ORACLE_GT_C1C2C0_R1, ORACLE_GT_C1C2C0_R2, ORACLE_GT_C1C2C0_RR),
                 rns_load_fp(ORACLE_GT_C1C2C1_R1, ORACLE_GT_C1C2C1_R2, ORACLE_GT_C1C2C1_RR)};
    return gt;
}

__global__ void test_kernel(int* results) {
    if (threadIdx.x != 0) return;
    int pass = 0, fail = 0;

    printf("=== RNS BLS12-381 Pairing Oracle Test ===\n\n");

    // Load G1 generator
    RnsFp px = rns_load_fp(G1_GEN_X_R1, G1_GEN_X_R2, G1_GEN_X_RR);
    RnsFp py = rns_load_fp(G1_GEN_Y_R1, G1_GEN_Y_R2, G1_GEN_Y_RR);

    // Test 1: Miller loop only (compare intermediate)
    printf("Computing Miller loop...\n");
    RnsFp12 ml = rns_miller_loop(px, py);
    printf("Miller loop done. Normalizing...\n");
    ml = rns_fp12_normalize(ml);
    printf("ML c0.c0.c0.rr = %u\n", ml.c0.c0.c0.rr);

    // Test 2: Full pairing e(G1, G2)
    printf("\nComputing final exponentiation...\n");
    RnsFp12 gt = rns_final_exp(ml);
    printf("Final exp done.\n");

    // Load oracle value
    RnsFp12 oracle_gt = load_oracle_gt();

    // Compare
    bool ok = rns_fp12_eq(gt, oracle_gt);
    printf("\nTest 1: e(G1,G2) matches Rust oracle: %s\n", ok?"PASS":"FAIL");
    if (ok) pass++; else {
        fail++;
        // Print first diverging component
        RnsFp12 gt_n = gt; // already from final_exp which normalizes
        printf("  GPU c0.c0.c0.rr = %u\n", rns_normalize(gt.c0.c0.c0).rr);
        printf("  Oracle c0.c0.c0.rr = %u\n", oracle_gt.c0.c0.c0.rr);
    }

    // Test 3: e(G1,G2)^p = e(G1,G2) (Frobenius fixes Gt elements... actually no)
    // Test 3: e(G1,G2) is not ONE (sanity)
    {
        bool not_one = !rns_fp12_eq(gt, rns_fp12_one());
        printf("Test 2: e(G1,G2) != ONE (sanity): %s\n", not_one?"PASS":"FAIL");
        if (not_one) pass++; else fail++;
    }

    printf("\n=== RESULTS: %d PASS, %d FAIL ===\n", pass, fail);
    results[0] = pass;
    results[1] = fail;
}

__global__ void bench_kernel(uint64_t* out) {
    if (threadIdx.x != 0) return;
    RnsFp px = rns_load_fp(G1_GEN_X_R1, G1_GEN_X_R2, G1_GEN_X_RR);
    RnsFp py = rns_load_fp(G1_GEN_Y_R1, G1_GEN_Y_R2, G1_GEN_Y_RR);

    unsigned long long t0, t1;

    // Miller loop timing
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    RnsFp12 ml = rns_miller_loop(px, py);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;

    // Final exp timing
    ml = rns_fp12_normalize(ml);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    RnsFp12 gt = rns_final_exp(ml);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[1] = t1 - t0;

    out[2] = gt.c0.c0.c0.r1[0]; // anti-DCE
}

int main() {
    int *d;
    cudaMalloc(&d, 2*sizeof(int));
    test_kernel<<<1,1>>>(d);
    cudaDeviceSynchronize();
    int r[2];
    cudaMemcpy(r, d, 2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);

    if (r[0] > 0) {
        printf("\n=== Benchmark ===\n");
        uint64_t *d_out;
        cudaMalloc(&d_out, 3*sizeof(uint64_t));
        bench_kernel<<<1,1>>>(d_out);
        cudaDeviceSynchronize();
        uint64_t h[3];
        cudaMemcpy(h, d_out, 3*sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaFree(d_out);

        int dev; cudaGetDevice(&dev);
        int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
        double ghz = clk / 1e6;

        printf("  Miller loop:   %.2f ms\n", h[0] / (ghz * 1e6));
        printf("  Final exp:     %.2f ms\n", h[1] / (ghz * 1e6));
        printf("  Total pairing: %.2f ms\n", (h[0]+h[1]) / (ghz * 1e6));
        printf("  Pairings/sec:  %.0f (single thread)\n", 1e9 / ((h[0]+h[1]) / ghz));
        printf("  GPU clock: %.2f GHz\n", ghz);
    }

    return r[1] > 0 ? 1 : 0;
}
