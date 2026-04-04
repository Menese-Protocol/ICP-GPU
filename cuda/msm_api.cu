// GPU Multi-Scalar Multiplication for BLS12-381 G1
// Uses sppark's Pippenger implementation
// Oracle-tested against ic_bls12_381::G1Projective::muln_affine_vartime

#include <cuda.h>

// sppark includes
#include <ff/bls12-381.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

#ifndef __CUDA_ARCH__
// Host-side type definitions
using namespace bls12_381;
#endif

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__

// C API for Rust FFI
extern "C" {

// Initialize GPU for MSM
int gpu_msm_init() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) return -1;
    cudaSetDevice(0);
    return 0;
}

// GPU MSM: result = sum(points[i] * scalars[i]) for i in 0..npoints
// points: array of affine G1 points (each 96 bytes: x,y in Montgomery form)
// scalars: array of 256-bit scalars (each 32 bytes, little-endian)
// result: output jacobian G1 point (144 bytes: x,y,z in Montgomery form)
// Returns 0 on success, -1 on error
int gpu_msm_g1(
    const unsigned char* points_bytes,
    const unsigned char* scalars_bytes,
    size_t npoints,
    unsigned char* result_bytes
) {
    if (npoints == 0) return -1;

    const affine_t* points = reinterpret_cast<const affine_t*>(points_bytes);
    const scalar_t* scalars = reinterpret_cast<const scalar_t*>(scalars_bytes);
    point_t result;

    RustError err = mult_pippenger<bucket_t>(&result, points, npoints, scalars, true);
    if (err.code != 0) return -1;

    // Copy result back
    memcpy(result_bytes, &result, sizeof(point_t));
    return 0;
}

// Batch MSM: perform multiple independent MSMs
// Each MSM has the same points but different scalars
// Useful for DKG verify_chunking where we do 32 MSMs on the same ciphertext points
int gpu_msm_g1_batch(
    const unsigned char* points_bytes,
    const unsigned char* scalars_batch_bytes,  // batch_size * npoints * 32 bytes
    size_t npoints,
    size_t batch_size,
    unsigned char* results_bytes  // batch_size * sizeof(point_t) bytes
) {
    if (npoints == 0 || batch_size == 0) return -1;

    const affine_t* points = reinterpret_cast<const affine_t*>(points_bytes);

    for (size_t b = 0; b < batch_size; b++) {
        const scalar_t* scalars = reinterpret_cast<const scalar_t*>(
            scalars_batch_bytes + b * npoints * sizeof(scalar_t)
        );
        point_t* result = reinterpret_cast<point_t*>(
            results_bytes + b * sizeof(point_t)
        );

        RustError err = mult_pippenger<bucket_t>(result, points, npoints, scalars, true);
        if (err.code != 0) return -1;
    }
    return 0;
}

} // extern "C"

#endif // !__CUDA_ARCH__
