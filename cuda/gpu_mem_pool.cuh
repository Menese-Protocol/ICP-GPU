// GPU Memory Pool for IC Consensus Crypto
// Pre-allocates buffers for DKG operations to avoid cudaMalloc/Free overhead
//
// Proven savings: 41μs per malloc+free → ~30ms per DKG round (122 dealings)
//
// Usage:
//   gpu_pool_init(max_points)     — call once at replica startup
//   gpu_pool_get_*()              — get pre-allocated device pointers
//   gpu_pool_destroy()            — cleanup at shutdown

#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Pool state
struct GpuMemPool {
    // Decompression buffers
    uint8_t* d_compressed;      // N × 48 bytes input
    uint8_t* d_affine_out;      // N × 96 bytes output (also used as MSM input)

    // MSM buffers
    uint8_t* d_scalars;         // N × 32 bytes
    uint8_t* d_msm_result;      // 144 bytes (Jacobian point)

    // BLS batch verify buffers
    uint64_t* d_sigs;           // M × 12 u64
    uint64_t* d_neg_hms;        // M × 12 u64
    uint64_t* d_pk_coeffs;      // M × 2448 u64
    int32_t*  d_results;        // M × 1

    // SHA-256 buffers
    uint8_t* d_sha_chunks;      // for batch hashing
    uint8_t* d_sha_hashes;      // N × 32 bytes

    // Pinned host memory for faster transfers
    uint8_t* h_pinned_in;       // pinned input buffer
    uint8_t* h_pinned_out;      // pinned output buffer

    // Capacity
    int max_points;             // max N for decompress/MSM
    int max_bls_batch;          // max M for BLS verify
    size_t pinned_size;         // size of pinned buffers

    // Stream (persistent, avoids create/destroy overhead)
    cudaStream_t stream;

    bool initialized;
};

// Global pool (singleton)
static GpuMemPool g_pool = {0};

// Initialize pool for given capacity
// max_points: max points for decompress/MSM (IC DKG: ~448 for 28 nodes, ~1000 for 40+ nodes)
// max_bls_batch: max signatures for batch BLS verify (IC: ~500 for large subnets)
inline int gpu_pool_init(int max_points = 1024, int max_bls_batch = 512) {
    if (g_pool.initialized) return 0;  // already initialized

    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;

    g_pool.max_points = max_points;
    g_pool.max_bls_batch = max_bls_batch;

    // Decompression + MSM buffers
    if (cudaMalloc(&g_pool.d_compressed, max_points * 48) != cudaSuccess) return -1;
    if (cudaMalloc(&g_pool.d_affine_out, max_points * 96) != cudaSuccess) return -1;
    if (cudaMalloc(&g_pool.d_scalars, max_points * 32) != cudaSuccess) return -1;
    if (cudaMalloc(&g_pool.d_msm_result, 144) != cudaSuccess) return -1;

    // BLS batch verify buffers
    size_t coeff_u64s = 68 * 36; // 2448 u64 per G2Prepared
    if (cudaMalloc(&g_pool.d_sigs, max_bls_batch * 12 * sizeof(uint64_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_pool.d_neg_hms, max_bls_batch * 12 * sizeof(uint64_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_pool.d_pk_coeffs, max_bls_batch * coeff_u64s * sizeof(uint64_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_pool.d_results, max_bls_batch * sizeof(int32_t)) != cudaSuccess) return -1;

    // Pinned host memory (largest expected transfer: max_points × 96 bytes)
    g_pool.pinned_size = max_points * 96;
    if (g_pool.pinned_size < max_bls_batch * coeff_u64s * sizeof(uint64_t)) {
        g_pool.pinned_size = max_bls_batch * coeff_u64s * sizeof(uint64_t);
    }
    if (cudaMallocHost(&g_pool.h_pinned_in, g_pool.pinned_size) != cudaSuccess) return -1;
    if (cudaMallocHost(&g_pool.h_pinned_out, g_pool.pinned_size) != cudaSuccess) return -1;

    // Persistent stream
    if (cudaStreamCreate(&g_pool.stream) != cudaSuccess) return -1;

    g_pool.initialized = true;

    size_t total_gpu = max_points * (48 + 96 + 32) + 144 +
                       max_bls_batch * (12 + 12 + coeff_u64s + 1) * sizeof(uint64_t);
    size_t total_pinned = g_pool.pinned_size * 2;

    fprintf(stderr, "[GPU-POOL] Initialized: %d points, %d BLS batch, "
            "%.1f KB GPU, %.1f KB pinned host\n",
            max_points, max_bls_batch,
            total_gpu / 1024.0, total_pinned / 1024.0);

    return 0;
}

// Destroy pool
inline void gpu_pool_destroy() {
    if (!g_pool.initialized) return;

    cudaFree(g_pool.d_compressed);
    cudaFree(g_pool.d_affine_out);
    cudaFree(g_pool.d_scalars);
    cudaFree(g_pool.d_msm_result);
    cudaFree(g_pool.d_sigs);
    cudaFree(g_pool.d_neg_hms);
    cudaFree(g_pool.d_pk_coeffs);
    cudaFree(g_pool.d_results);
    cudaFreeHost(g_pool.h_pinned_in);
    cudaFreeHost(g_pool.h_pinned_out);
    cudaStreamDestroy(g_pool.stream);

    g_pool = {0};
    fprintf(stderr, "[GPU-POOL] Destroyed\n");
}

// Accessors
inline GpuMemPool* gpu_pool_get() {
    if (!g_pool.initialized) return nullptr;
    return &g_pool;
}

inline bool gpu_pool_is_initialized() { return g_pool.initialized; }
inline cudaStream_t gpu_pool_stream() { return g_pool.stream; }
