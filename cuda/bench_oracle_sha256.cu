// GPU Batch SHA-256 Benchmark — Oracle-verified, fits in 3.5 GiB free VRAM
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include "/workspace/ic/rs/crypto/internal/crypto_lib/gpu_crypto/cuda/sha256.cuh"

int main() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("=== GPU Batch SHA-256 (IC Domain) — Oracle Verified ===\n");
    printf("GPU VRAM: %.1f GiB free / %.1f GiB total\n\n", free_mem/1e9, total_mem/1e9);

    const int RAW = 1024*1024;
    const int PREFIX = 15;
    const int FULL = PREFIX + RAW;

    // Build one reference chunk
    uint8_t* one_chunk = (uint8_t*)malloc(FULL);
    one_chunk[0] = 14;
    memcpy(one_chunk+1, "ic-state-chunk", 14);
    for(int i=0;i<RAW;i+=8) {
        uint64_t val = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
        memcpy(one_chunk+PREFIX+i, &val, 8);
    }

    // Oracle correctness check first (1 chunk)
    {
        uint8_t *d_data, *d_hash;
        cudaMalloc(&d_data, FULL);
        cudaMalloc(&d_hash, 32);
        cudaMemcpy(d_data, one_chunk, FULL, cudaMemcpyHostToDevice);
        kernel_batch_sha256<<<1,1>>>(d_data, d_hash, 1, FULL);
        cudaDeviceSynchronize();
        uint8_t h[32];
        cudaMemcpy(h, d_hash, 32, cudaMemcpyDeviceToHost);
        uint8_t exp[] = {0xb0,0x3d,0x7a,0xeb,0xe8,0xb9,0x6f,0xa7,0x1e,0xde,0x8e,0x16,0x90,0xf3,0xf5,0xf5,
                         0x34,0xab,0x35,0x84,0x62,0xfd,0x3b,0x4d,0x43,0x42,0xf5,0x88,0xdb,0xa6,0xd1,0x67};
        bool ok = memcmp(h,exp,32)==0;
        printf("Oracle check (1 chunk): %s\n", ok?"PASS":"FAIL");
        if(!ok) { printf("  Got: "); for(int j=0;j<32;j++) printf("%02x",h[j]); printf("\n"); return 1; }
        cudaFree(d_data); cudaFree(d_hash);
    }

    // Batch sizes that fit in available VRAM (each chunk ~1MiB)
    // Max chunks = free_mem / FULL - margin
    int max_chunks = (int)((free_mem - 256*1024*1024) / FULL); // leave 256MB margin
    printf("Max chunks in VRAM: %d (%.1f GiB)\n\n", max_chunks, (double)max_chunks*FULL/1e9);

    int bench_sizes[] = {10, 50, 100, 500, 1000, 2000, 3000};
    int num_bench = 7;

    printf("%-30s  %10s  %10s  %8s\n", "Workload", "GPU(ms)", "GB/s(raw)", "vs 16T");
    printf("%-30s  %10s  %10s  %8s\n", "---", "---", "---", "---");

    for(int bi=0; bi<num_bench; bi++) {
        int M = bench_sizes[bi];
        if(M > max_chunks) { printf("%-30s  SKIP (OOM)\n", ""); continue; }

        size_t total = (size_t)M * FULL;
        uint8_t* h_data = (uint8_t*)malloc(total);
        if(!h_data) { printf("host OOM at %d chunks\n", M); continue; }
        for(int c=0;c<M;c++) memcpy(h_data+(size_t)c*FULL, one_chunk, FULL);

        uint8_t *d_data, *d_hashes;
        if(cudaMalloc(&d_data, total) != cudaSuccess) { free(h_data); continue; }
        cudaMalloc(&d_hashes, M*32);
        cudaMemcpy(d_data, h_data, total, cudaMemcpyHostToDevice);

        int thr = 32, blk = (M+thr-1)/thr;

        // Warmup
        kernel_batch_sha256<<<blk,thr>>>(d_data, d_hashes, M, FULL);
        cudaDeviceSynchronize();

        // Timed
        cudaEvent_t gs,ge;
        cudaEventCreate(&gs); cudaEventCreate(&ge);
        int rounds = (M<=100)?10:3;
        cudaEventRecord(gs);
        for(int r=0;r<rounds;r++)
            kernel_batch_sha256<<<blk,thr>>>(d_data, d_hashes, M, FULL);
        cudaEventRecord(ge);
        cudaEventSynchronize(ge);

        float tot_ms; cudaEventElapsedTime(&tot_ms, gs, ge);
        float gpu_ms = tot_ms/rounds;
        double raw_bytes = (double)M*RAW;
        double gbps = (raw_bytes/1e9)/(gpu_ms/1e3);
        double cpu16_ms = (raw_bytes/1e9)/33.9*1000.0;

        char desc[64];
        snprintf(desc, sizeof(desc), "%d x 1MiB (%.1f GiB)", M, raw_bytes/(1024.0*1024.0*1024.0));
        printf("%-30s  %10.1f  %10.1f  %7.1fx\n", desc, gpu_ms, gbps, cpu16_ms/gpu_ms);

        free(h_data);
        cudaFree(d_data); cudaFree(d_hashes);
        cudaEventDestroy(gs); cudaEventDestroy(ge);
    }

    // End-to-end with max feasible batch
    int e2e_M = (max_chunks < 1000) ? max_chunks : 1000;
    printf("\n=== End-to-End (H2D + compute + D2H) — %d chunks ===\n", e2e_M);
    {
        size_t total = (size_t)e2e_M * FULL;
        uint8_t* h_data = (uint8_t*)malloc(total);
        for(int c=0;c<e2e_M;c++) memcpy(h_data+(size_t)c*FULL, one_chunk, FULL);

        uint8_t *d_data, *d_hashes;
        cudaMalloc(&d_data, total);
        cudaMalloc(&d_hashes, e2e_M*32);
        uint8_t* h_out = (uint8_t*)malloc(e2e_M*32);

        // Warmup
        cudaMemcpy(d_data, h_data, total, cudaMemcpyHostToDevice);
        kernel_batch_sha256<<<(e2e_M+31)/32,32>>>(d_data, d_hashes, e2e_M, FULL);
        cudaDeviceSynchronize();

        cudaEvent_t gs,ge;
        cudaEventCreate(&gs); cudaEventCreate(&ge);
        cudaEventRecord(gs);
        cudaMemcpy(d_data, h_data, total, cudaMemcpyHostToDevice);
        kernel_batch_sha256<<<(e2e_M+31)/32,32>>>(d_data, d_hashes, e2e_M, FULL);
        cudaMemcpy(h_out, d_hashes, e2e_M*32, cudaMemcpyDeviceToHost);
        cudaEventRecord(ge);
        cudaEventSynchronize(ge);

        float e2e_ms; cudaEventElapsedTime(&e2e_ms, gs, ge);
        double raw_gb = (double)e2e_M*RAW/1e9;
        printf("  %d x 1MiB: %.1fms (%.1f GB/s raw)\n", e2e_M, e2e_ms, raw_gb/(e2e_ms/1e3));
        // Scale to 1000 chunks for comparison
        double scaled_ms = e2e_ms * 1000.0 / e2e_M;
        printf("  Scaled to 1000 chunks: ~%.1fms vs CPU 16T (31.0ms): %.1fx\n",
               scaled_ms, 31.0/scaled_ms);

        free(h_data); free(h_out);
        cudaFree(d_data); cudaFree(d_hashes);
        cudaEventDestroy(gs); cudaEventDestroy(ge);
    }

    free(one_chunk);
    printf("\n=== Done ===\n");
    return 0;
}
