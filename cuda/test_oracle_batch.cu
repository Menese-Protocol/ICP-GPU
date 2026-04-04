// GPU Batch SHA-256 Benchmark — Oracle-verified, IC domain separator
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include "/workspace/ic/rs/crypto/internal/crypto_lib/gpu_crypto/cuda/sha256.cuh"

int main() {
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    printf("=== GPU Batch SHA-256 (IC ic-state-chunk domain) ===\n");
    printf("Oracle-verified bit-exact against Rust sha2 crate\n\n");

    const int RAW = 1024*1024;
    const int PREFIX = 15;
    const int FULL = PREFIX + RAW;

    // Build one reference chunk with domain prefix
    uint8_t* one_chunk = (uint8_t*)malloc(FULL);
    one_chunk[0] = 14;
    memcpy(one_chunk+1, "ic-state-chunk", 14);
    for(int i=0;i<RAW;i+=8) {
        uint64_t val = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
        memcpy(one_chunk+PREFIX+i, &val, 8);
    }

    int bench_sizes[] = {10, 100, 500, 1000, 4000};
    printf("%-30s  %10s  %10s  %8s\n", "Workload", "GPU(ms)", "GB/s(raw)", "vs 16T");
    printf("%-30s  %10s  %10s  %8s\n", "---", "---", "---", "---");

    for(int bi=0; bi<5; bi++) {
        int M = bench_sizes[bi];
        size_t total = (size_t)M * FULL;

        // Allocate and fill on host
        uint8_t* h_data = (uint8_t*)malloc(total);
        if(!h_data) { printf("SKIP %d (OOM)\n", M); continue; }
        for(int c=0; c<M; c++) memcpy(h_data + (size_t)c*FULL, one_chunk, FULL);

        uint8_t *d_data, *d_hashes;
        cudaMalloc(&d_data, total);
        cudaMalloc(&d_hashes, M * 32);
        cudaMemcpy(d_data, h_data, total, cudaMemcpyHostToDevice);

        int thr = 32;
        int blk = (M + thr - 1) / thr;

        // Warmup
        kernel_batch_sha256<<<blk, thr>>>(d_data, d_hashes, M, FULL);
        cudaDeviceSynchronize();

        // Correctness check on first run
        if(bi == 0) {
            uint8_t h_hash[32];
            cudaMemcpy(h_hash, d_hashes, 32, cudaMemcpyDeviceToHost);
            uint8_t exp[] = {0xb0,0x3d,0x7a,0xeb,0xe8,0xb9,0x6f,0xa7,0x1e,0xde,0x8e,0x16,0x90,0xf3,0xf5,0xf5,
                             0x34,0xab,0x35,0x84,0x62,0xfd,0x3b,0x4d,0x43,0x42,0xf5,0x88,0xdb,0xa6,0xd1,0x67};
            printf("  Oracle check: %s\n\n", memcmp(h_hash,exp,32)==0 ? "PASS" : "FAIL");
        }

        // Timed (compute only — data on GPU)
        cudaEvent_t gs, ge;
        cudaEventCreate(&gs); cudaEventCreate(&ge);
        int rounds = (M <= 100) ? 10 : 3;
        cudaEventRecord(gs);
        for(int r=0; r<rounds; r++)
            kernel_batch_sha256<<<blk, thr>>>(d_data, d_hashes, M, FULL);
        cudaEventRecord(ge);
        cudaEventSynchronize(ge);

        float tot_ms; cudaEventElapsedTime(&tot_ms, gs, ge);
        float gpu_ms = tot_ms / rounds;
        double raw_bytes = (double)M * RAW;
        double gbps = (raw_bytes / 1e9) / (gpu_ms / 1e3);
        double cpu16_ms = (raw_bytes / 1e9) / 33.9 * 1000.0;
        double speedup = cpu16_ms / gpu_ms;

        char desc[64];
        snprintf(desc, sizeof(desc), "%d x 1MiB (%.1f GiB)", M, raw_bytes/(1024.0*1024.0*1024.0));
        printf("%-30s  %10.1f  %10.1f  %7.1fx\n", desc, gpu_ms, gbps, speedup);

        free(h_data);
        cudaFree(d_data); cudaFree(d_hashes);
        cudaEventDestroy(gs); cudaEventDestroy(ge);
    }

    // End-to-end with H2D + D2H
    printf("\n=== End-to-End (H2D + compute + D2H) ===\n");
    {
        int M = 1000;
        size_t total = (size_t)M * FULL;
        uint8_t* h_data = (uint8_t*)malloc(total);
        for(int c=0;c<M;c++) memcpy(h_data+(size_t)c*FULL, one_chunk, FULL);

        uint8_t *d_data, *d_hashes;
        cudaMalloc(&d_data, total);
        cudaMalloc(&d_hashes, M*32);
        uint8_t* h_out = (uint8_t*)malloc(M*32);

        // Warmup
        cudaMemcpy(d_data, h_data, total, cudaMemcpyHostToDevice);
        kernel_batch_sha256<<<(M+31)/32,32>>>(d_data, d_hashes, M, FULL);
        cudaDeviceSynchronize();

        cudaEvent_t gs,ge;
        cudaEventCreate(&gs); cudaEventCreate(&ge);
        cudaEventRecord(gs);
        cudaMemcpy(d_data, h_data, total, cudaMemcpyHostToDevice);
        kernel_batch_sha256<<<(M+31)/32,32>>>(d_data, d_hashes, M, FULL);
        cudaMemcpy(h_out, d_hashes, M*32, cudaMemcpyDeviceToHost);
        cudaEventRecord(ge);
        cudaEventSynchronize(ge);

        float e2e_ms; cudaEventElapsedTime(&e2e_ms, gs, ge);
        double raw_gb = (double)M * RAW / 1e9;
        printf("  1000 x 1MiB: %.1fms (%.1f GB/s)\n", e2e_ms, raw_gb/(e2e_ms/1e3));
        printf("  vs 16-thread CPU (31.0ms): %.1fx\n", 31.0/e2e_ms);

        free(h_data); free(h_out);
        cudaFree(d_data); cudaFree(d_hashes);
        cudaEventDestroy(gs); cudaEventDestroy(ge);
    }

    free(one_chunk);
    printf("\n=== Done ===\n");
    return 0;
}
