// GPU SHA-256 Oracle Test — Verify GPU matches Rust oracle (IC domain separator)
// Uses kernel_batch_sha256 from sha256.cuh with domain prefix prepended on host.
// Stack size set to 8192 for 1MiB+ per-thread processing.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include "/workspace/ic/rs/crypto/internal/crypto_lib/gpu_crypto/cuda/sha256.cuh"

static void fill_deterministic(uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; i += 8) {
        uint64_t val = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
        size_t remaining = len - i;
        if (remaining >= 8) memcpy(data + i, &val, 8);
        else memcpy(data + i, &val, remaining);
    }
}

int main() {
    // Critical: GPU SHA-256 on 1MiB chunks needs larger stack
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    printf("=== GPU SHA-256 Oracle Test (IC domain separator) ===\n");
    printf("Oracle expected: b03d7aebe8b96fa71ede8e1690f3f5f534ab358462fd3b4d4342f588dba6d167\n\n");

    const int RAW_CHUNK = 1024 * 1024;
    const uint8_t DOMAIN[] = "ic-state-chunk"; // 14 bytes
    const int PREFIX_LEN = 15; // 1 byte len + 14 bytes domain
    const int FULL_CHUNK = PREFIX_LEN + RAW_CHUNK;
    const int N = 10;

    // Build host data with IC domain prefix
    uint8_t* raw = (uint8_t*)malloc(RAW_CHUNK);
    fill_deterministic(raw, RAW_CHUNK);

    uint8_t* h_data = (uint8_t*)malloc((size_t)N * FULL_CHUNK);
    for (int c = 0; c < N; c++) {
        uint8_t* chunk = h_data + (size_t)c * FULL_CHUNK;
        chunk[0] = 14;
        memcpy(chunk + 1, DOMAIN, 14);
        memcpy(chunk + PREFIX_LEN, raw, RAW_CHUNK);
    }

    uint8_t* h_hashes = (uint8_t*)calloc(N, 32);
    uint8_t *d_data, *d_hashes;
    cudaMalloc(&d_data, (size_t)N * FULL_CHUNK);
    cudaMalloc(&d_hashes, N * 32);
    cudaMemcpy(d_data, h_data, (size_t)N * FULL_CHUNK, cudaMemcpyHostToDevice);

    kernel_batch_sha256<<<1, N>>>(d_data, d_hashes, N, FULL_CHUNK);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_hashes, d_hashes, N * 32, cudaMemcpyDeviceToHost);

    const uint8_t expected[32] = {
        0xb0,0x3d,0x7a,0xeb,0xe8,0xb9,0x6f,0xa7,0x1e,0xde,0x8e,0x16,0x90,0xf3,0xf5,0xf5,
        0x34,0xab,0x35,0x84,0x62,0xfd,0x3b,0x4d,0x43,0x42,0xf5,0x88,0xdb,0xa6,0xd1,0x67
    };

    bool all_pass = true;
    for (int c = 0; c < N; c++) {
        bool match = (memcmp(h_hashes + c * 32, expected, 32) == 0);
        if (!match) all_pass = false;
        printf("  chunk[%d]: ", c);
        for (int j = 0; j < 32; j++) printf("%02x", h_hashes[c * 32 + j]);
        printf(" %s\n", match ? "PASS" : "FAIL");
    }

    printf("\n  Oracle match: %s\n", all_pass ? "ALL 10 PASS" : "FAILED");

    if (!all_pass) {
        free(h_data); free(h_hashes); free(raw);
        cudaFree(d_data); cudaFree(d_hashes);
        return 1;
    }

    // ==================== Benchmark ====================
    printf("\n=== Benchmark: GPU IC Domain-Separated Chunk Hash ===\n");
    printf("%-30s  %10s  %10s  %8s\n", "Workload", "GPU(ms)", "GB/s", "vs 16T");

    int bench_sizes[] = {10, 100, 500, 1000, 4000};
    for (int bi = 0; bi < 5; bi++) {
        int M = bench_sizes[bi];
        size_t total_raw = (size_t)M * RAW_CHUNK;
        size_t total_full = (size_t)M * FULL_CHUNK;

        uint8_t *d2, *dh2;
        cudaMalloc(&d2, total_full);
        cudaMalloc(&dh2, M * 32);

        // Fill GPU memory
        for (size_t off = 0; off < total_full; off += (size_t)N * FULL_CHUNK) {
            size_t copy = total_full - off;
            if (copy > (size_t)N * FULL_CHUNK) copy = (size_t)N * FULL_CHUNK;
            cudaMemcpy(d2 + off, d_data, copy, cudaMemcpyDeviceToDevice);
        }

        int thr = 32;
        int blk = (M + thr - 1) / thr;

        // Warmup
        kernel_batch_sha256<<<blk, thr>>>(d2, dh2, M, FULL_CHUNK);
        cudaDeviceSynchronize();

        // Timed (compute only)
        cudaEvent_t gs, ge;
        cudaEventCreate(&gs); cudaEventCreate(&ge);

        int rounds = (M <= 100) ? 5 : 2;
        cudaEventRecord(gs);
        for (int r = 0; r < rounds; r++)
            kernel_batch_sha256<<<blk, thr>>>(d2, dh2, M, FULL_CHUNK);
        cudaEventRecord(ge);
        cudaEventSynchronize(ge);

        float tot_ms;
        cudaEventElapsedTime(&tot_ms, gs, ge);
        float gpu_ms = tot_ms / rounds;

        double gbps = (total_raw / 1e9) / (gpu_ms / 1e3);
        double cpu_16t_ms = (total_raw / 1e9) / 33.9 * 1000.0;
        double speedup = cpu_16t_ms / gpu_ms;

        char desc[64];
        snprintf(desc, sizeof(desc), "%d x 1MiB (%.1f GiB)", M, total_raw / (1024.0*1024.0*1024.0));
        printf("%-30s  %10.1f  %10.1f  %7.1fx\n", desc, gpu_ms, gbps, speedup);

        cudaFree(d2); cudaFree(dh2);
        cudaEventDestroy(gs); cudaEventDestroy(ge);
    }

    // End-to-end (H2D + compute + D2H)
    printf("\n=== End-to-End (H2D + compute + D2H) ===\n");
    {
        int M = 1000;
        size_t total_full = (size_t)M * FULL_CHUNK;
        uint8_t* h_big = (uint8_t*)malloc(total_full);
        if (h_big) {
            for (size_t off = 0; off < total_full; off += (size_t)N * FULL_CHUNK) {
                size_t copy = total_full - off;
                if (copy > (size_t)N * FULL_CHUNK) copy = (size_t)N * FULL_CHUNK;
                memcpy(h_big + off, h_data, copy);
            }

            uint8_t *d3, *dh3;
            cudaMalloc(&d3, total_full);
            cudaMalloc(&dh3, M * 32);
            uint8_t* h_out = (uint8_t*)malloc(M * 32);

            // Warmup
            cudaMemcpy(d3, h_big, total_full, cudaMemcpyHostToDevice);
            kernel_batch_sha256<<<(M+31)/32, 32>>>(d3, dh3, M, FULL_CHUNK);
            cudaDeviceSynchronize();

            cudaEvent_t gs, ge;
            cudaEventCreate(&gs); cudaEventCreate(&ge);
            cudaEventRecord(gs);
            cudaMemcpy(d3, h_big, total_full, cudaMemcpyHostToDevice);
            kernel_batch_sha256<<<(M+31)/32, 32>>>(d3, dh3, M, FULL_CHUNK);
            cudaMemcpy(h_out, dh3, M * 32, cudaMemcpyDeviceToHost);
            cudaEventRecord(ge);
            cudaEventSynchronize(ge);

            float e2e_ms;
            cudaEventElapsedTime(&e2e_ms, gs, ge);
            printf("  1000 x 1MiB E2E: %.1fms (%.1f GB/s raw)\n",
                   e2e_ms, ((double)M * RAW_CHUNK / 1e9) / (e2e_ms / 1e3));
            printf("  vs 16-thread CPU (31.0ms): %.1fx\n", 31.0 / e2e_ms);

            free(h_big); free(h_out);
            cudaFree(d3); cudaFree(dh3);
            cudaEventDestroy(gs); cudaEventDestroy(ge);
        }
    }

    free(h_data); free(h_hashes); free(raw);
    cudaFree(d_data); cudaFree(d_hashes);
    printf("\n=== Done ===\n");
    return 0;
}
