// GPU Multi-Scalar Multiplication for BLS12-381 G1
// Pure C API — no C++ standard library dependency
// Uses CUDA API directly instead of sppark's C++ host utilities

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Only device-side sppark types needed for the kernel
#include <ff/bls12-381.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

// Include Pippenger — this pulls in GPU kernels + host-side orchestration
// The host-side code uses std::thread etc. We wrap it in a separate
// compilation unit that accepts the C++ dependency.
#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__
// Host-side: needs all_gpus.cpp for gpu_props()
#include <util/all_gpus.cpp>

extern "C" {

int gpu_msm_g1(
    const unsigned char* points_bytes,
    const unsigned char* scalars_bytes,
    size_t npoints,
    unsigned char* result_jac_bytes
) {
    if (npoints == 0) return -1;

    const affine_t* points = reinterpret_cast<const affine_t*>(points_bytes);
    const scalar_t* scalars = reinterpret_cast<const scalar_t*>(scalars_bytes);
    point_t result;

    RustError err = mult_pippenger<bucket_t>(&result, points, npoints, scalars, true);
    if (err.code != 0) return -1;

    memcpy(result_jac_bytes, &result, sizeof(point_t));
    return 0;
}

int gpu_jacobian_to_affine(const unsigned char* in_jac, unsigned char* out_aff) {
    const point_t* jp = reinterpret_cast<const point_t*>(in_jac);
    affine_t result = *jp;
    memcpy(out_aff, &result, sizeof(affine_t));
    return 0;
}

} // extern "C"

#endif
