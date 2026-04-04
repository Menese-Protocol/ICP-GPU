#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "/workspace/ic/rs/crypto/internal/crypto_lib/gpu_crypto/cuda/sha256.cuh"

int main() {
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    // Test: hash domain_prefix + 1MiB data as single message
    const int RAW = 1024*1024;
    const int PREFIX = 15; // 1 + 14
    const int FULL = PREFIX + RAW;

    uint8_t* h_data = (uint8_t*)malloc(FULL);
    h_data[0] = 14;
    memcpy(h_data+1, "ic-state-chunk", 14);
    for(int i=0;i<RAW;i+=8) {
        uint64_t val = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
        memcpy(h_data+PREFIX+i, &val, 8);
    }

    uint8_t *d_data, *d_hash;
    uint8_t h_hash[32];
    cudaMalloc(&d_data, FULL);
    cudaMalloc(&d_hash, 32);
    cudaMemcpy(d_data, h_data, FULL, cudaMemcpyHostToDevice);

    // Single thread, single chunk
    kernel_batch_sha256<<<1,1>>>(d_data, d_hash, 1, FULL);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(h_hash, d_hash, 32, cudaMemcpyDeviceToHost);
    printf("GPU:      ");
    for(int i=0;i<32;i++) printf("%02x",h_hash[i]);
    printf("\nExpected: b03d7aebe8b96fa71ede8e1690f3f5f534ab358462fd3b4d4342f588dba6d167\n");

    uint8_t expected[] = {0xb0,0x3d,0x7a,0xeb,0xe8,0xb9,0x6f,0xa7,0x1e,0xde,0x8e,0x16,0x90,0xf3,0xf5,0xf5,
                          0x34,0xab,0x35,0x84,0x62,0xfd,0x3b,0x4d,0x43,0x42,0xf5,0x88,0xdb,0xa6,0xd1,0x67};
    printf("Match: %s\n", memcmp(h_hash, expected, 32)==0 ? "YES" : "NO");

    free(h_data);
    cudaFree(d_data);
    cudaFree(d_hash);
    return 0;
}
