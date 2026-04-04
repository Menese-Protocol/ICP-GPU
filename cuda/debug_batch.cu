#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "/workspace/ic/rs/crypto/internal/crypto_lib/gpu_crypto/cuda/sha256.cuh"

int main() {
    // Step 1: Check current stack limit
    size_t stack_limit;
    cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize);
    printf("Default stack: %zu bytes\n", stack_limit);

    // Step 2: Set larger stack
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize);
    printf("New stack: %zu bytes\n", stack_limit);

    const int FULL = 15 + 1024*1024;

    // Step 3: Test with increasing thread counts
    int tests[] = {1, 2, 4, 8, 16, 32};
    for(int ti=0; ti<6; ti++) {
        int N = tests[ti];
        size_t total = (size_t)N * FULL;

        uint8_t* h_data = (uint8_t*)calloc(N, FULL);
        for(int c=0; c<N; c++) {
            h_data[(size_t)c*FULL] = 14;
            memcpy(h_data+(size_t)c*FULL+1, "ic-state-chunk", 14);
            for(int i=0; i<1024*1024; i+=8) {
                uint64_t val = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
                memcpy(h_data+(size_t)c*FULL+15+i, &val, 8);
            }
        }

        uint8_t *d_data, *d_hashes;
        cudaMalloc(&d_data, total);
        cudaMalloc(&d_hashes, N*32);
        cudaMemcpy(d_data, h_data, total, cudaMemcpyHostToDevice);

        // Use 1 block, N threads (all in same warp/block)
        kernel_batch_sha256<<<1, N>>>(d_data, d_hashes, N, FULL);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();

        uint8_t h_hash[32];
        cudaMemcpy(h_hash, d_hashes, 32, cudaMemcpyDeviceToHost);

        printf("N=%2d: %s  hash=", N, err==cudaSuccess ? "OK  " : cudaGetErrorString(err));
        if(err==cudaSuccess) {
            for(int j=0;j<8;j++) printf("%02x",h_hash[j]);
            printf("...\n");
        } else {
            printf("(error)\n");
        }

        free(h_data);
        cudaFree(d_data); cudaFree(d_hashes);
    }
    return 0;
}
