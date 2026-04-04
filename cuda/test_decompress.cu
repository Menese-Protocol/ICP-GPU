// GPU BLS12-381 Fp Square Root — Step 1 of point decompression
// Oracle test: compute sqrt on GPU, compare vs CPU (ic_bls12_381)

#include <cuda.h>
#include <cstdio>
#include <cstring>

#include <ff/bls12-381.hpp>

using namespace bls12_381;

// ==================== GPU Kernel ====================

// Fp sqrt: a^((p+1)/4) using square-and-multiply
// (p+1)/4 = 0x0680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaab
__device__ fp_t fp_sqrt_dev(const fp_t& a) {
    // Exponent (p+1)/4 in 6 x 64-bit limbs (little-endian)
    const uint64_t exp[6] = {
        0xe7fbfffffffeaabULL,
        0x07aaffffac54ffffULL,
        0x9cc34a83dac3d890ULL,
        0x91dd2e13ce144afdULL,
        0x2c6e9ed90d2eb35dULL,
        0x0680447a8e5ff9a6ULL,
    };

    fp_t result = fp_t::one();
    fp_t base = a;

    for (int limb = 0; limb < 6; limb++) {
        uint64_t bits = exp[limb];
        for (int bit = 0; bit < 64; bit++) {
            if (bits & 1ULL) {
                result *= base;
            }
            base *= base;
            bits >>= 1;
        }
    }
    return result;
}

__global__ void fp_sqrt_kernel(const uint32_t* input, uint32_t* output) {
    fp_t a;
    memcpy(&a, input, sizeof(fp_t));
    fp_t result = fp_sqrt_dev(a);
    memcpy(output, &result, sizeof(fp_t));
}

// ==================== Host API ====================

#ifndef __CUDA_ARCH__
extern "C" {

// Compute Fp sqrt on GPU
// Input/output: 48 bytes, Montgomery form (same as ic_bls12_381 Fp internal repr)
int gpu_fp_sqrt(const unsigned char* input, unsigned char* output) {
    uint32_t *d_in, *d_out;
    cudaMalloc(&d_in, 48);
    cudaMalloc(&d_out, 48);
    cudaMemcpy(d_in, input, 48, cudaMemcpyHostToDevice);

    fp_sqrt_kernel<<<1, 1>>>(d_in, d_out);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[GPU] sqrt kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_in); cudaFree(d_out);
        return -1;
    }

    cudaMemcpy(output, d_out, 48, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

} // extern "C"
#endif
