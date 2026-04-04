// Benchmark: INT32 vs INT64 multiply throughput on Blackwell (sm_120)
// The answer to: can we use 6×64-bit limbs instead of 12×32-bit?
#include <cstdio>
#include <cstdint>

__global__ void bench(uint64_t* out) {
    if (threadIdx.x != 0) return;
    int N = 1000000;
    unsigned long long t0, t1;

    // 1: 32-bit multiply (mad.lo.u32 + mad.hi.u32) — current approach
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    uint32_t a32 = 0xDEADBEEF, b32 = 0xCAFEBABE, r32 = 0;
    for (int i = 0; i < N; i++) {
        asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(r32) : "r"(a32), "r"(b32));
        asm volatile("mad.hi.u32 %0, %1, %2, %0;" : "+r"(r32) : "r"(a32), "r"(b32));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
    out[6] = r32;

    // 2: 64-bit multiply (mul.lo.u64 + mul.hi.u64) — what we need for 6-limb
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    uint64_t a64 = 0xDEADBEEFCAFEBABEull, b64 = 0x1234567890ABCDEFull, r64 = 0;
    for (int i = 0; i < N; i++) {
        uint64_t lo, hi;
        asm volatile("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a64), "l"(b64));
        asm volatile("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a64), "l"(b64));
        r64 += lo + hi;  // prevent DCE
        a64 ^= lo;       // data dependency
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[1] = t1 - t0;
    out[7] = r64;

    // 3: 64-bit mad (mad.lo.u64 + mad.hi.u64) — multiply-add, the CIOS inner op
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    r64 = 0;
    for (int i = 0; i < N; i++) {
        asm volatile("mad.lo.u64 %0, %1, %2, %0;" : "+l"(r64) : "l"(a64), "l"(b64));
        asm volatile("mad.hi.u64 %0, %1, %2, %0;" : "+l"(r64) : "l"(a64), "l"(b64));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[2] = t1 - t0;
    out[8] = r64;

    // 4: __umul64hi intrinsic (what Barrett uses)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    r64 = 0;
    for (int i = 0; i < N; i++) {
        r64 += __umul64hi(a64 + i, b64);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[3] = t1 - t0;
    out[9] = r64;

    // 5: 64-bit add (the fast one — 64/SM on Blackwell)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    r64 = 0;
    for (int i = 0; i < N; i++) {
        asm volatile("add.u64 %0, %0, %1;" : "+l"(r64) : "l"(a64));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[4] = t1 - t0;
    out[10] = r64;

    // 6: Full 6-limb fp_mul simulation (6x6 = 36 mul.lo + mul.hi pairs + reduction)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    uint64_t limbs[6] = {a64, b64, a64^b64, a64+1, b64+1, a64^1};
    uint64_t acc = 0;
    for (int n = 0; n < N/36; n++) {  // N/36 full fp_muls
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                uint64_t lo, hi;
                asm volatile("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(limbs[i]), "l"(limbs[j]));
                asm volatile("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(limbs[i]), "l"(limbs[j]));
                acc += lo + hi;
            }
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[5] = t1 - t0;
    out[11] = acc;
}

int main() {
    uint64_t *d; cudaMalloc(&d, 12*sizeof(uint64_t));
    bench<<<1,1>>>(d);
    cudaDeviceSynchronize();
    uint64_t h[12]; cudaMemcpy(h, d, 12*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    int dev; cudaGetDevice(&dev);
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    double ghz = clk / 1e6;
    int N = 1000000;

    printf("=== Blackwell INT64 Benchmark (sm_120) ===\n\n");
    printf("  32-bit mad pair (lo+hi):  %.1f cycles/pair  (%.1f ns)\n", (double)h[0]/N, h[0]/(ghz*N));
    printf("  64-bit mul pair (lo+hi):  %.1f cycles/pair  (%.1f ns)\n", (double)h[1]/N, h[1]/(ghz*N));
    printf("  64-bit mad pair (lo+hi):  %.1f cycles/pair  (%.1f ns)\n", (double)h[2]/N, h[2]/(ghz*N));
    printf("  __umul64hi:               %.1f cycles       (%.1f ns)\n", (double)h[3]/N, h[3]/(ghz*N));
    printf("  64-bit add:               %.1f cycles       (%.1f ns)\n", (double)h[4]/N, h[4]/(ghz*N));
    printf("  6-limb fp_mul (36 pairs): %.1f cycles/mul   (%.1f ns)\n", (double)h[5]/(N/36), h[5]/(ghz*N/36));
    printf("\n  GPU clock: %.2f GHz\n", ghz);
    printf("\n  Key question: if 64-bit mul pair < 4× 32-bit mad pair,\n");
    printf("  then 6-limb (64-bit) Montgomery beats 12-limb (32-bit).\n");
    printf("  Ratio: 64-bit/32-bit = %.2fx\n", (double)h[1]/h[0]);
}
