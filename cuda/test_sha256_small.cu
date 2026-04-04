#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include "/workspace/ic/rs/crypto/internal/crypto_lib/gpu_crypto/cuda/sha256.cuh"

__global__ void test_small() {
    uint8_t msg[] = "hello";
    uint8_t hash[32];
    sha256(msg, 5, hash);
    printf("SHA256(hello)=");
    for(int i=0;i<32;i++) printf("%02x",hash[i]);
    printf("\n");
}

int main() {
    test_small<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
    return 0;
}
