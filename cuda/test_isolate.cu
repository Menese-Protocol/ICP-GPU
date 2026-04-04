// Isolation tests for MSM debugging
// Test each layer independently: host arith, GPU arith, data transfer

#include <cstdio>
#include <cstring>

#include <ff/bls12-381.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

#ifndef __CUDA_ARCH__
using namespace bls12_381;

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

extern "C" {

// Test A: Host-only point doubling using sppark types
// Input: affine point (96 bytes)
// Output: jacobian 2*P (144 bytes)
int test_host_double(const unsigned char* in_affine, unsigned char* out_jacobian) {
    const affine_t* p = reinterpret_cast<const affine_t*>(in_affine);

    // Convert affine to jacobian
    point_t jp = *p;

    // Double it on HOST using sppark's blst_384_t arithmetic
    point_t doubled;
    doubled = jp;
    doubled.dbl();

    memcpy(out_jacobian, &doubled, sizeof(point_t));
    return 0;
}

// Test B: Host-only point addition P + P using sppark types
int test_host_add(const unsigned char* in_affine, unsigned char* out_jacobian) {
    const affine_t* p = reinterpret_cast<const affine_t*>(in_affine);

    point_t jp = *p;
    point_t result = jp;
    result.add(jp);

    memcpy(out_jacobian, &result, sizeof(point_t));
    return 0;
}

// Test C: Host P + Q (two different affine points)
int test_host_add_two(const unsigned char* in_p, const unsigned char* in_q,
                       unsigned char* out_jacobian) {
    const affine_t* p = reinterpret_cast<const affine_t*>(in_p);
    const affine_t* q = reinterpret_cast<const affine_t*>(in_q);

    point_t result;
    point_t::dadd(result, point_t(*p), point_t(*q));

    memcpy(out_jacobian, &result, sizeof(point_t));
    return 0;
}

// Test D: Convert affine → jacobian → affine roundtrip
int test_roundtrip(const unsigned char* in_affine, unsigned char* out_affine) {
    const affine_t* p = reinterpret_cast<const affine_t*>(in_affine);
    point_t jp = *p;

    // Convert back to affine
    affine_t result = jp;
    memcpy(out_affine, &result, sizeof(affine_t));
    return 0;
}

// Convert Jacobian (144 bytes) to Affine (96 bytes) using sppark's types
int jacobian_to_affine_c(const unsigned char* in_jac, unsigned char* out_aff) {
    const point_t* jp = reinterpret_cast<const point_t*>(in_jac);
    affine_t result = *jp;  // jacobian_t has implicit conversion to affine_t
    memcpy(out_aff, &result, sizeof(affine_t));
    return 0;
}

} // extern "C"
#endif
