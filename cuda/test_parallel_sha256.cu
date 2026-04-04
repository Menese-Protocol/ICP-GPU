// PARALLEL SHA-256: Split 1MiB into sub-chunks, hash in parallel, Merkle combine
// This is NOT compatible with IC's current hash format but shows what's possible.
//
// Approach: Split 1MiB into 256 × 4KiB sub-chunks
// Phase 1: 256 threads hash 4KiB each in parallel
// Phase 2: Build Merkle tree (log2(256)=8 rounds of 128→64→32→...→1 hashes)
// Total: 256 + 128+64+32+16+8+4+2+1 = 511 hashes, but all parallel within phases

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include "/workspace/ic/rs/crypto/internal/crypto_lib/gpu_crypto/cuda/sha256.cuh"

// Phase 1: hash sub-chunks (many threads per 1MiB chunk)
__global__ void kernel_subchunk_hash(
    const uint8_t* data,  // N × 1MiB
    uint8_t* leaf_hashes, // N × SUB_CHUNKS × 32
    int N, int chunk_size, int sub_chunk_size
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int chunks_per_mib = chunk_size / sub_chunk_size;
    int chunk_idx = global_id / chunks_per_mib;
    int sub_idx = global_id % chunks_per_mib;

    if (chunk_idx >= N) return;

    const uint8_t* src = data + (uint64_t)chunk_idx * chunk_size + (uint64_t)sub_idx * sub_chunk_size;
    uint8_t* dst = leaf_hashes + ((uint64_t)chunk_idx * chunks_per_mib + sub_idx) * 32;
    sha256(src, sub_chunk_size, dst);
}

// Phase 2: Merkle combine (each thread hashes two 32-byte child hashes)
__global__ void kernel_merkle_combine(
    const uint8_t* in_hashes,  // N × count × 32
    uint8_t* out_hashes,       // N × (count/2) × 32
    int N, int count
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int half = count / 2;
    int chunk_idx = global_id / half;
    int pair_idx = global_id % half;

    if (chunk_idx >= N) return;

    const uint8_t* left = in_hashes + ((uint64_t)chunk_idx * count + 2*pair_idx) * 32;
    const uint8_t* right = in_hashes + ((uint64_t)chunk_idx * count + 2*pair_idx + 1) * 32;

    // Combine: SHA256(left || right)
    uint8_t combined[64];
    for(int i=0;i<32;i++) { combined[i]=left[i]; combined[32+i]=right[i]; }
    uint8_t* dst = out_hashes + ((uint64_t)chunk_idx * half + pair_idx) * 32;
    sha256(combined, 64, dst);
}

int main() {
    printf("=== Parallel SHA-256 (Merkle Tree Approach) ===\n");
    printf("Split 1MiB → sub-chunks → parallel hash → Merkle combine\n\n");

    const int CHUNK = 1024 * 1024;

    // Generate test data
    uint8_t* h_data = (uint8_t*)malloc(CHUNK);
    for(int i=0; i<CHUNK; i+=8) {
        uint64_t val = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
        memcpy(h_data+i, &val, 8);
    }

    // Test different sub-chunk sizes
    int sub_sizes[] = {4096, 8192, 16384, 32768, 65536};
    char* descs[] = {"4 KiB (256 sub)", "8 KiB (128 sub)", "16 KiB (64 sub)",
                     "32 KiB (32 sub)", "64 KiB (16 sub)"};

    printf("%-20s  %10s  %10s  %10s  %10s\n",
           "Sub-chunk", "Phase1(ms)", "Merkle(ms)", "Total(ms)", "vs Serial");
    printf("%-20s  %10s  %10s  %10s  %10s\n", "---","---","---","---","---");

    // First measure serial (1 thread, full 1MiB)
    uint8_t *d_data, *d_hash;
    cudaMalloc(&d_data, CHUNK);
    cudaMalloc(&d_hash, 32);
    cudaMemcpy(d_data, h_data, CHUNK, cudaMemcpyHostToDevice);

    // Warmup
    kernel_batch_sha256<<<1,1>>>(d_data, d_hash, 1, CHUNK);
    cudaDeviceSynchronize();

    cudaEvent_t s,e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    cudaEventRecord(s);
    for(int r=0;r<10;r++) kernel_batch_sha256<<<1,1>>>(d_data, d_hash, 1, CHUNK);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float serial_ms; cudaEventElapsedTime(&serial_ms, s, e); serial_ms /= 10;
    printf("%-20s  %10s  %10s  %10.2f  %10s\n", "Serial (1MiB)", "-", "-", serial_ms, "1.0x");

    // Now test parallel with N=100 chunks to be realistic
    int N = 100;
    uint8_t* h_big = (uint8_t*)malloc((size_t)N * CHUNK);
    for(int c=0;c<N;c++) memcpy(h_big+(size_t)c*CHUNK, h_data, CHUNK);

    uint8_t *d_big;
    cudaMalloc(&d_big, (size_t)N * CHUNK);
    cudaMemcpy(d_big, h_big, (size_t)N * CHUNK, cudaMemcpyHostToDevice);

    // Serial baseline for N chunks
    uint8_t* d_hashes_serial;
    cudaMalloc(&d_hashes_serial, N*32);
    kernel_batch_sha256<<<(N+31)/32,32>>>(d_big, d_hashes_serial, N, CHUNK);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    kernel_batch_sha256<<<(N+31)/32,32>>>(d_big, d_hashes_serial, N, CHUNK);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float serial_batch_ms; cudaEventElapsedTime(&serial_batch_ms, s, e);
    printf("%-20s  %10s  %10s  %10.2f  %10s  (N=%d)\n",
           "Serial batch", "-", "-", serial_batch_ms, "1.0x", N);
    printf("\n");

    for(int si=0; si<5; si++) {
        int SUB = sub_sizes[si];
        int SUBS = CHUNK / SUB;
        int total_subs = N * SUBS;

        // Allocate leaf hashes and temp buffers
        uint8_t *d_leaves, *d_temp1, *d_temp2;
        cudaMalloc(&d_leaves, (size_t)total_subs * 32);
        cudaMalloc(&d_temp1, (size_t)total_subs * 32);
        cudaMalloc(&d_temp2, (size_t)total_subs * 32);

        int thr = 256;
        int blk = (total_subs + thr - 1) / thr;

        // Warmup
        kernel_subchunk_hash<<<blk,thr>>>(d_big, d_leaves, N, CHUNK, SUB);
        cudaDeviceSynchronize();

        // Phase 1: parallel sub-chunk hashing
        cudaEventRecord(s);
        kernel_subchunk_hash<<<blk,thr>>>(d_big, d_leaves, N, CHUNK, SUB);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float phase1_ms; cudaEventElapsedTime(&phase1_ms, s, e);

        // Phase 2: Merkle tree combine
        float phase2_ms = 0;
        uint8_t *src = d_leaves, *dst = d_temp1;
        int count = SUBS;
        while(count > 1) {
            int half = count / 2;
            int total_pairs = N * half;
            int blk2 = (total_pairs + 255) / 256;
            cudaEventRecord(s);
            kernel_merkle_combine<<<blk2, 256>>>(src, dst, N, count);
            cudaEventRecord(e); cudaEventSynchronize(e);
            float ms; cudaEventElapsedTime(&ms, s, e);
            phase2_ms += ms;
            count = half;
            // Swap buffers
            uint8_t* tmp = src; src = dst; dst = tmp;
            if(dst == d_leaves) dst = d_temp2; // don't overwrite leaves
        }

        printf("%-20s  %10.2f  %10.2f  %10.2f  %9.1fx\n",
               descs[si], phase1_ms, phase2_ms, phase1_ms+phase2_ms,
               serial_batch_ms / (phase1_ms+phase2_ms));

        cudaFree(d_leaves); cudaFree(d_temp1); cudaFree(d_temp2);
    }

    free(h_data); free(h_big);
    cudaFree(d_data); cudaFree(d_hash); cudaFree(d_big); cudaFree(d_hashes_serial);
    cudaEventDestroy(s); cudaEventDestroy(e);

    printf("\nNOTE: Merkle approach produces DIFFERENT hashes than flat SHA-256.\n");
    printf("IC would need a protocol change (StateSyncV5) to adopt this.\n");
    printf("But it shows the theoretical GPU speedup if SHA-256 is parallelized.\n");
    printf("\n=== Done ===\n");
    return 0;
}
