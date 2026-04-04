#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"
__global__ void bench(uint64_t* out) {
    if (threadIdx.x != 0) return;
    RnsFp a; for(int i=0;i<RNS_K;i++){a.r1[i]=RNS_ORACLE_MONT7_M1[i];a.r2[i]=RNS_ORACLE_MONT7_M2[i];} a.rr=RNS_ORACLE_MONT7_RED;
    int N = 10000;
    unsigned long long t0, t1;
    
    // Measure just mmul operations (no base extension)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    RnsFp acc = a;
    for (int n = 0; n < N; n++) {
        // Simulate step1+2+4: ~60 mmul calls
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            acc.r1[i] = mmul(acc.r1[i], a.r1[i], RNS_M1[i], RNS_BARRETT_M1[i]);
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            acc.r2[i] = mmul(acc.r2[i], a.r2[i], RNS_M2[i], RNS_BARRETT_M2[i]);
        #pragma unroll
        for (int i = 0; i < RNS_K; i++)
            acc.r1[i] = mmul(acc.r1[i], RNS_NEG_PINV_M1[i], RNS_M1[i], RNS_BARRETT_M1[i]);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
    out[2] = acc.r1[0];

    // Full mul for comparison
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    acc = a;
    for (int n = 0; n < N; n++) acc = rns_mul(acc, a);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[1] = t1 - t0;
    out[3] = acc.r1[0];
}
int main() {
    uint64_t *d; cudaMalloc(&d, 4*sizeof(uint64_t));
    bench<<<1,1>>>(d);
    cudaDeviceSynchronize();
    uint64_t h[4]; cudaMemcpy(h, d, 4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    int dev; cudaGetDevice(&dev); int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    double ghz = clk/1e6;
    printf("42x mmul only:  %.0f ns/iter\n", h[0]/(ghz*10000));
    printf("full rns_mul:   %.0f ns/iter\n", h[1]/(ghz*10000));
    printf("base_ext cost:  %.0f ns/iter (%.0f%%)\n", (h[1]-h[0])/(ghz*10000), 100.0*(h[1]-h[0])/h[1]);
}
