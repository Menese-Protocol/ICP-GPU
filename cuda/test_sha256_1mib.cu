#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "/workspace/ic/rs/crypto/internal/crypto_lib/gpu_crypto/cuda/sha256.cuh"

__global__ void hash_1mib(const uint8_t* data, uint8_t* hash, int chunk_size) {
    sha256(data, chunk_size, hash);
}

int main() {
    // Set large stack size for SHA-256 on big data
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    int CHUNK = 1024*1024;
    uint8_t* h_data = (uint8_t*)malloc(CHUNK);
    for(int i=0;i<CHUNK;i+=8) {
        uint64_t val = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
        memcpy(h_data+i, &val, 8);
    }

    uint8_t *d_data, *d_hash;
    uint8_t h_hash[32];
    cudaMalloc(&d_data, CHUNK);
    cudaMalloc(&d_hash, 32);
    cudaMemcpy(d_data, h_data, CHUNK, cudaMemcpyHostToDevice);

    hash_1mib<<<1,1>>>(d_data, d_hash, CHUNK);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_hash, d_hash, 32, cudaMemcpyDeviceToHost);
    printf("SHA256(1MiB raw)=");
    for(int i=0;i<32;i++) printf("%02x",h_hash[i]);
    printf("\n");
    printf("Oracle expected: fb1591cf79df72016ab0dffe3ef6a84dab5ccd6473bd94255331eaac8367e903\n");

    free(h_data);
    cudaFree(d_data);
    cudaFree(d_hash);
    return 0;
}
