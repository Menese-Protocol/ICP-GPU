// Profile where time goes in rns_mul
#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"

__global__ void profile(uint64_t* out) {
    if (threadIdx.x != 0) return;

    RnsFp a; for(int i=0;i<RNS_K;i++){a.r1[i]=RNS_ORACLE_MONT7_M1[i];a.r2[i]=RNS_ORACLE_MONT7_M2[i];} a.rr=RNS_ORACLE_MONT7_RED;
    RnsFp b = rns_add(a, rns_one());
    int N = 1000;
    unsigned long long t0, t1;

    // 1: Just mmul (no base extension)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    uint32_t acc = a.r1[0];
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 29; i++) // same count as step1+step2
            acc = mmul(acc, b.r1[i%RNS_K], RNS_M1[i%RNS_K], RNS_BARRETT_M1[i%RNS_K]);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
    out[6] = acc; // anti-DCE

    // 2: Just bred on 64-bit accumulator
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    uint64_t acc64 = 12345678ULL;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 28; i++) // same as 2 base extensions
            acc = (uint32_t)((acc64 + i*1000000ULL, RNS_M2[i%RNS_K]) % RNS_M2[i%RNS_K]);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[1] = t1 - t0;

    // 3: __umul64hi alone
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    uint64_t h = 0;
    for (int n = 0; n < N*29; n++)
        h = __umul64hi(h + 1, 4294967291ULL);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[2] = t1 - t0;
    out[7] = h;

    // 4: Full rns_mul
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    RnsFp r = a;
    for (int n = 0; n < N; n++) r = rns_mul(r, b);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[3] = t1 - t0;
    out[8] = r.r1[0];

    // 5: Double-precision division (alpha computation)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    double dv = 0.0;
    for (int n = 0; n < N*2; n++) {
        dv = 0.0;
        for (int i = 0; i < RNS_K; i++)
            dv += (double)a.r1[i] / (double)RNS_M1[i];
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[4] = t1 - t0;
    out[9] = (uint64_t)dv;

    // 6: Simple shift reduction (x >> 30, no __umul64hi)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    uint32_t sr = 0;
    for (int n = 0; n < N*29; n++) {
        uint64_t x = (uint64_t)(sr + n) * 999999937ULL;
        uint32_t q = (uint32_t)(x >> 30);
        sr = (uint32_t)(x - (uint64_t)q * 1073741789u);
        if (sr >= 1073741789u) sr -= 1073741789u;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[5] = t1 - t0;
    out[10] = sr;
}

int main() {
    uint64_t *d; cudaMalloc(&d, 11*sizeof(uint64_t));
    profile<<<1,1>>>(d);
    cudaDeviceSynchronize();
    uint64_t h[11]; cudaMemcpy(h, d, 11*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    int dev; cudaGetDevice(&dev);
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    double ghz = clk / 1e6;
    int N = 1000;

    printf("=== RNS Fp mul profiling (1000 iterations) ===\n");
    printf("  29x mmul (step1+2):      %.0f ns/iter  (%.0f ns/mmul)\n", h[0]/(ghz*N), h[0]/(ghz*N*29));
    printf("  28x bred (base ext):     %.0f ns/iter  (%.0f ns/bred)\n", h[1]/(ghz*N), h[1]/(ghz*N*28));
    printf("  29x __umul64hi:          %.0f ns/iter  (%.0f ns/op)\n", h[2]/(ghz*N), h[2]/(ghz*N*29));
    printf("  Full rns_mul:            %.0f ns/iter\n", h[3]/(ghz*N));
    printf("  2x alpha (14 div each):  %.0f ns/iter  (%.0f ns/alpha)\n", h[4]/(ghz*N), h[4]/(ghz*N*2));
    printf("  29x shift-reduce:        %.0f ns/iter  (%.0f ns/op)\n", h[5]/(ghz*N), h[5]/(ghz*N*29));
    printf("  GPU clock: %.2f GHz\n", ghz);
}
