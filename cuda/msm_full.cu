// Complete GPU MSM for BLS12-381 G1
// Includes all sppark dependencies

#include <cuda.h>

// Include sppark GPU utilities (gpu_props etc)
#include <util/all_gpus.cpp>

// sppark BLS12-381 types
#include <ff/bls12-381.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

#ifndef __CUDA_ARCH__
using namespace bls12_381;
#endif

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__

extern "C" {

int gpu_msm_init() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) return -1;
    cudaSetDevice(0);
    return 0;
}

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

    memcpy(result_bytes, &result, sizeof(point_t));
    return 0;
}

// Version that takes canonical (non-Montgomery) scalars
int gpu_msm_g1_canonical(
    const unsigned char* points_bytes,
    const unsigned char* scalars_bytes,
    size_t npoints,
    unsigned char* result_bytes
) {
    if (npoints == 0) return -1;

    const affine_t* points = reinterpret_cast<const affine_t*>(points_bytes);
    const scalar_t* scalars = reinterpret_cast<const scalar_t*>(scalars_bytes);
    point_t result;

    RustError err = mult_pippenger<bucket_t>(&result, points, npoints, scalars, false);
    if (err.code != 0) return -1;

    memcpy(result_bytes, &result, sizeof(point_t));
    return 0;
}

} // extern "C"

#endif
