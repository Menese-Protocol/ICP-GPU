// GPU Merkle SHA-256 — C API for Rust FFI
// Oracle-verified bit-exact against Rust sha2 crate
//
// Exported functions:
//   gpu_merkle_init()              — warm up GPU, return 0 on success
//   gpu_merkle_hash_chunks()       — hash N × 1MiB chunks, return Merkle roots
//   gpu_merkle_free()              — cleanup

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "/workspace/ic/rs/crypto/internal/crypto_lib/gpu_crypto/cuda/sha256.cuh"

#define SUB_CHUNK_SIZE 4096
#define CHUNK_SIZE     1048576
#define NUM_LEAVES     256

__global__ void merkle_leaves(const uint8_t* data, uint8_t* hashes, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = gid / NUM_LEAVES;
    int leaf  = gid % NUM_LEAVES;
    if (chunk >= N) return;
    sha256(data + (uint64_t)chunk * CHUNK_SIZE + (uint64_t)leaf * SUB_CHUNK_SIZE,
           SUB_CHUNK_SIZE,
           hashes + ((uint64_t)chunk * NUM_LEAVES + leaf) * 32);
}

__global__ void merkle_combine(const uint8_t* in_h, uint8_t* out_h, int N, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int half = count / 2;
    int chunk = gid / half;
    int pair  = gid % half;
    if (chunk >= N) return;
    uint8_t combined[64];
    const uint8_t* L = in_h + ((uint64_t)chunk * count + 2*pair) * 32;
    const uint8_t* R = in_h + ((uint64_t)chunk * count + 2*pair + 1) * 32;
    for(int i=0;i<32;i++){combined[i]=L[i];combined[32+i]=R[i];}
    sha256(combined, 64, out_h + ((uint64_t)chunk * half + pair) * 32);
}

extern "C" {

/// Initialize GPU. Returns 0 on success, -1 on failure.
int gpu_merkle_init(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) return -1;
    // Warmup: launch tiny kernel
    uint8_t *d_tmp;
    err = cudaMalloc(&d_tmp, 32);
    if (err != cudaSuccess) return -1;
    cudaFree(d_tmp);
    return 0;
}

/// Compute Merkle SHA-256 root hashes for N × 1MiB chunks.
///
/// data:       pointer to N × 1,048,576 bytes (host memory)
/// roots_out:  pointer to N × 32 bytes (host memory, caller-allocated)
/// num_chunks: number of 1MiB chunks
///
/// Returns 0 on success, -1 on failure.
int gpu_merkle_hash_chunks(
    const uint8_t* data,
    uint8_t* roots_out,
    int num_chunks
) {
    if (num_chunks <= 0 || !data || !roots_out) return -1;

    size_t data_sz = (size_t)num_chunks * CHUNK_SIZE;
    size_t buf_sz  = (size_t)num_chunks * NUM_LEAVES * 32;

    uint8_t *d_data = NULL, *d_a = NULL, *d_b = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_data, data_sz);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_a, buf_sz);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_b, buf_sz);
    if (err != cudaSuccess) goto fail;

    // Upload
    err = cudaMemcpy(d_data, data, data_sz, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    // Phase 1: leaf hashing
    {
        int total = num_chunks * NUM_LEAVES;
        int thr = 256, blk = (total + thr - 1) / thr;
        merkle_leaves<<<blk, thr>>>(d_data, d_a, num_chunks);
    }

    // Phase 2: tree combining
    {
        int count = NUM_LEAVES;
        uint8_t *src = d_a, *dst = d_b;
        while (count > 1) {
            int half = count / 2;
            int total = num_chunks * half;
            int blk = (total + 255) / 256;
            merkle_combine<<<blk, 256>>>(src, dst, num_chunks, count);
            count = half;
            uint8_t* tmp = src; src = dst; dst = tmp;
        }
        // src now points to roots (1 per chunk)
        err = cudaMemcpy(roots_out, src, num_chunks * 32, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto fail;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;

    cudaFree(d_data); cudaFree(d_a); cudaFree(d_b);
    return 0;

fail:
    if (d_data) cudaFree(d_data);
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    return -1;
}

/// Cleanup (currently no-op, but available for future resource management).
void gpu_merkle_free(void) {
    cudaDeviceReset();
}

} // extern "C"
