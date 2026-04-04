// GPU BLS12-381 G1 Batch Point Decompression
// Takes N compressed points (48 bytes each), outputs N affine points (96 bytes each)
// Oracle-tested against ic_bls12_381::G1Affine::from_compressed

#include <cuda.h>
#include <cstdio>
#include <cstring>

#include <ff/bls12-381.hpp>

using namespace bls12_381;

// ==================== Device Functions ====================

// Fp sqrt: a^((p+1)/4)
__device__ fp_t fp_sqrt_dev(const fp_t& a) {
    const uint64_t exp[6] = {
        0xee7fbfffffffeaabULL,
        0x07aaffffac54ffffULL,
        0xd9cc34a83dac3d89ULL,
        0xd91dd2e13ce144afULL,
        0x92c6e9ed90d2eb35ULL,
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

// Convert 48 big-endian bytes to fp_t in Montgomery form
// Steps: read big-endian → to 6x64-bit LE limbs → multiply by R² → reduce
__device__ fp_t fp_from_be_bytes(const uint8_t bytes[48]) {
    // Read as 6 x 64-bit big-endian limbs
    uint64_t limbs[6];
    for (int i = 0; i < 6; i++) {
        uint64_t val = 0;
        for (int j = 0; j < 8; j++) {
            val = (val << 8) | bytes[i * 8 + j];
        }
        limbs[5 - i] = val;  // store little-endian
    }

    // Now limbs[0..6] is the value in normal form (not Montgomery)
    // To convert to Montgomery: multiply by R² mod p, then reduce
    // sppark's fp_t has a `to()` method that converts FROM Montgomery to normal
    // We need the reverse: `to()` undoes Montgomery by multiplying by R^-1
    // So we want: result = val * R mod p = val * R² * R^-1 mod p
    // Which is: construct fp_t with the limbs, then DON'T call to()

    // Actually, fp_t stores values in Montgomery form: val_mont = val * R mod p
    // If we put raw limbs into fp_t, it interprets them AS Montgomery
    // So we'd have: fp_t contains L, meaning the actual value is L * R^-1 mod p
    // We want: actual value = limbs, Montgomery form = limbs * R mod p

    // Method: create fp_t from limbs (treated as Montgomery = limbs * R^-1)
    // Then multiply by R² to get: (limbs * R^-1) * R² = limbs * R = what we want

    // RR (R² mod p) is available as device constant BLS12_381_RR
    fp_t raw;
    uint32_t* raw32 = (uint32_t*)&raw;
    for (int i = 0; i < 6; i++) {
        raw32[2*i] = (uint32_t)limbs[i];
        raw32[2*i+1] = (uint32_t)(limbs[i] >> 32);
    }

    // raw now contains the value interpreted as Montgomery = raw * R^-1
    // Multiply by RR to get: raw * R^-1 * R² = raw * R = Montgomery(value)
    fp_t rr;
    uint32_t* rr32 = (uint32_t*)&rr;
    rr32[0]  = 0x1c341746; rr32[1]  = 0xf4df1f34;
    rr32[2]  = 0x09d104f1; rr32[3]  = 0x0a76e6a6;
    rr32[4]  = 0x4c95b6d5; rr32[5]  = 0x8de5476c;
    rr32[6]  = 0x939d83c0; rr32[7]  = 0x67eb88a9;
    rr32[8]  = 0xb519952d; rr32[9]  = 0x9a793e85;
    rr32[10] = 0x92cae3aa; rr32[11] = 0x11988fe5;

    return raw * rr;  // Montgomery multiplication: (raw * R^-1) * (R² * R^-1) = raw
    // Wait, that gives raw (normal form) stored in fp_t (interpreted as Montgomery)
    // So the "actual value" would be raw * R^-1, which is NOT what we want.

    // Let me think again:
    // fp_t internally stores: m = value * R mod p
    // Montgomery mul(a, b) computes: a * b * R^-1 mod p
    // If a = raw limbs (treated as Montgomery = raw * R^-1 actual)
    // And b = RR (treated as Montgomery = RR * R^-1 = R actual)
    // Then mul(a, b) = a * b * R^-1 = (raw limbs) * RR * R^-1 mod p
    // Stored value = raw * R mod p ← this IS the Montgomery encoding of 'raw'!

    // So raw * rr IS correct. The stored value is raw * R mod p = Montgomery(raw).
    // ✓
}

// Get the "sign" of an Fp element (for choosing y vs -y)
// BLS12-381 convention: sign bit = whether the element is lexicographically larger than (p-1)/2
__device__ bool fp_sign(const fp_t& a) {
    // Convert from Montgomery to get actual value
    fp_t normal = a;
    normal.from();  // now contains the actual value

    // Compare with (p-1)/2
    // (p-1)/2 = 0x0d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd555
    const uint64_t half_p[6] = {
        0xdcff7fffffffd555ULL,
        0x0f55ffff58a9ffffULL,
        0xb39869507b587b12ULL,
        0xb23ba5c279c2895fULL,
        0x258dd3db21a5d66bULL,
        0x0d0088f51cbff34dULL,
    };

    // Compare normal > half_p (little-endian limb comparison)
    uint32_t* n32 = (uint32_t*)&normal;
    for (int i = 11; i >= 0; i--) {
        uint32_t hp;
        int limb64 = i / 2;
        if (i & 1) hp = (uint32_t)(half_p[limb64] >> 32);
        else hp = (uint32_t)(half_p[limb64]);

        if (n32[i] > hp) return true;
        if (n32[i] < hp) return false;
    }
    return false;
}

// Full G1 decompression kernel
// compressed: N × 48 bytes (BLS12-381 compressed format, big-endian)
// out_points: N × 96 bytes (x, y in Montgomery fp_t format = ic_bls12_381 internal)
__global__ void batch_g1_decompress(
    const uint8_t* compressed,
    uint8_t* out_points,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint8_t* in = compressed + idx * 48;
    uint8_t* out = out_points + idx * 96;

    // Read flags
    uint8_t top = in[0];
    // bool is_compressed = (top >> 7) & 1;  // should always be 1
    bool is_infinity = (top >> 6) & 1;
    bool y_flag = (top >> 5) & 1;

    if (is_infinity) {
        memset(out, 0, 96);
        return;
    }

    // Extract x bytes (mask flag bits)
    uint8_t x_be[48];
    memcpy(x_be, in, 48);
    x_be[0] &= 0x1f;

    // Convert to Montgomery Fp
    fp_t x = fp_from_be_bytes(x_be);

    // Compute y² = x³ + 4
    fp_t x2 = x * x;      // x²
    fp_t x3 = x2 * x;     // x³
    fp_t b = fp_t::one();
    b += b; b += b;        // b = 4 (in Montgomery form: 4R mod p)
    fp_t y2 = x3 + b;     // y² = x³ + 4

    // Compute y = sqrt(y²)
    fp_t y = fp_sqrt_dev(y2);

    // Choose sign: if y_flag != fp_sign(y), negate y
    if (y_flag != fp_sign(y)) {
        y = -y;  // fp_t supports negation (p - y)
    }

    // Output: x (48 bytes) + y (48 bytes) in Montgomery form
    memcpy(out, &x, 48);
    memcpy(out + 48, &y, 48);
}

// ==================== Host API ====================
#ifndef __CUDA_ARCH__
extern "C" {

// Batch decompress N compressed G1 points on GPU
// compressed: N × 48 bytes
// out_points: N × 96 bytes (x, y Montgomery Fp = ic_bls12_381 internal format)
// Returns 0 on success
int gpu_g1_decompress_batch(
    const unsigned char* compressed,
    unsigned char* out_points,
    int n
) {
    if (n <= 0) return -1;

    uint8_t *d_in, *d_out;
    cudaMalloc(&d_in, n * 48);
    cudaMalloc(&d_out, n * 96);
    cudaMemcpy(d_in, compressed, n * 48, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    batch_g1_decompress<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[GPU] decompress error: %s\n", cudaGetErrorString(err));
        cudaFree(d_in); cudaFree(d_out);
        return -1;
    }

    cudaMemcpy(out_points, d_out, n * 96, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

// Decompress single point (convenience)
int gpu_g1_decompress(const unsigned char* compressed, unsigned char* out_point) {
    return gpu_g1_decompress_batch(compressed, out_point, 1);
}

} // extern "C"
#endif
