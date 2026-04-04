// IC STATE CHECKPOINT SHA-256 BENCHMARK
// Oracle methodology: simulate real IC manifest hashing workload
//
// IC state_manager hashes state in 1 MiB chunks:
//   - Small canister: ~10 chunks (10 MiB state)
//   - Medium canister: ~100 chunks (100 MiB state)
//   - Large canister: ~1000 chunks (1 GiB state)
//   - Bitcoin canister: ~30000 chunks (30 GiB state)
//
// Currently done via Rayon (CPU parallel). We measure GPU vs single-thread CPU.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <openssl/sha.h>  // CPU reference

#include "sha256.cuh"

// ==================== CPU SHA-256 (single-thread, for baseline) ====================

void cpu_sha256(const uint8_t* msg, size_t len, uint8_t* hash) {
    SHA256(msg, len, hash);
}

// ==================== Benchmark ====================

int main() {
    printf("=== IC STATE CHECKPOINT SHA-256 BENCHMARK ===\n");
    printf("GPU: RTX PRO 6000 Blackwell | Chunk size: 1 MiB (IC default)\n\n");

    // IC uses 1 MiB chunks
    const int CHUNK_SIZE = 1024 * 1024;  // 1 MiB

    // Test sizes simulating different canister state sizes
    struct TestCase {
        int num_chunks;
        const char* description;
    };
    TestCase tests[] = {
        {10,    "Small canister (10 MiB)"},
        {100,   "Medium canister (100 MiB)"},
        {500,   "Large canister (500 MiB)"},
        {1000,  "Very large canister (1 GiB)"},
        {4000,  "Bitcoin canister (4 GiB)"},
    };
    int num_tests = sizeof(tests) / sizeof(tests[0]);

    printf("%-35s  %10s  %10s  %10s  %8s  %s\n",
           "Workload", "CPU 1T(ms)", "GPU(ms)", "Speedup", "GB/s", "Correct");
    printf("%-35s  %10s  %10s  %10s  %8s  %s\n",
           "---", "---", "---", "---", "---", "---");

    for (int ti = 0; ti < num_tests; ti++) {
        int N = tests[ti].num_chunks;
        size_t total_bytes = (size_t)N * CHUNK_SIZE;

        // Allocate and fill with pseudo-random data
        uint8_t* h_data = (uint8_t*)malloc(total_bytes);
        if (!h_data) {
            printf("  [SKIP] Cannot allocate %.1f GiB\n", total_bytes / 1e9);
            continue;
        }
        // Fill with deterministic pattern (fast, not random)
        for (size_t i = 0; i < total_bytes; i += 8) {
            uint64_t val = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
            size_t remaining = total_bytes - i;
            if (remaining >= 8) memcpy(h_data + i, &val, 8);
            else memcpy(h_data + i, &val, remaining);
        }

        uint8_t* h_hashes_cpu = (uint8_t*)calloc(N, 32);
        uint8_t* h_hashes_gpu = (uint8_t*)calloc(N, 32);

        // ==================== CPU single-thread ====================
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            cpu_sha256(h_data + (size_t)i * CHUNK_SIZE, CHUNK_SIZE, h_hashes_cpu + i * 32);
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // ==================== GPU batch ====================
        uint8_t *d_data, *d_hashes;
        cudaMalloc(&d_data, total_bytes);
        cudaMalloc(&d_hashes, N * 32);

        // Include H2D transfer in timing (realistic — data comes from host)
        cudaEvent_t g_start, g_end;
        cudaEventCreate(&g_start);
        cudaEventCreate(&g_end);

        // Warmup
        cudaMemcpy(d_data, h_data, total_bytes, cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        kernel_batch_sha256<<<blocks, threads>>>(d_data, d_hashes, N, CHUNK_SIZE);
        cudaDeviceSynchronize();

        // Timed run (include H2D + compute + D2H)
        cudaEventRecord(g_start);
        cudaMemcpy(d_data, h_data, total_bytes, cudaMemcpyHostToDevice);
        kernel_batch_sha256<<<blocks, threads>>>(d_data, d_hashes, N, CHUNK_SIZE);
        cudaMemcpy(h_hashes_gpu, d_hashes, N * 32, cudaMemcpyDeviceToHost);
        cudaEventRecord(g_end);
        cudaEventSynchronize(g_end);

        float gpu_ms;
        cudaEventElapsedTime(&gpu_ms, g_start, g_end);

        // ==================== Correctness ====================
        bool correct = (memcmp(h_hashes_cpu, h_hashes_gpu, N * 32) == 0);

        // Stats
        double speedup = cpu_ms / gpu_ms;
        double gpu_gbps = (total_bytes / 1e9) / (gpu_ms / 1e3);

        printf("%-35s  %10.1f  %10.1f  %9.1fx  %7.1f  %s\n",
               tests[ti].description, cpu_ms, gpu_ms, speedup, gpu_gbps,
               correct ? "PASS" : "FAIL");

        // Cleanup
        free(h_data); free(h_hashes_cpu); free(h_hashes_gpu);
        cudaFree(d_data); cudaFree(d_hashes);
        cudaEventDestroy(g_start); cudaEventDestroy(g_end);
    }

    printf("\n=== ANALYSIS ===\n");
    printf("GPU timing INCLUDES H2D + compute + D2H (realistic end-to-end).\n");
    printf("CPU timing is single-thread. IC uses Rayon (12 threads) = CPU/12 roughly.\n");
    printf("GPU must beat Rayon-parallel CPU to be worth integrating.\n");
    printf("Breakeven vs 12-core Rayon: GPU speedup > 12x over single-thread CPU.\n");

    return 0;
}
