// Minimal format test: dump what GPU sees for points and scalars
// No MSM — just read and echo back the raw bytes

#include <cstdio>
#include <cstring>

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

extern "C" {

// Echo back what the GPU code sees as the first affine point
int test_affine_echo(const unsigned char* in_bytes, unsigned char* out_bytes) {
    const affine_t* p = reinterpret_cast<const affine_t*>(in_bytes);
    // Just copy the affine point back
    memcpy(out_bytes, p, sizeof(affine_t));
    printf("[C] sizeof(affine_t) = %zu\n", sizeof(affine_t));
    printf("[C] sizeof(fp_t) = %zu\n", sizeof(fp_t));
    printf("[C] sizeof(scalar_t) = %zu\n", sizeof(scalar_t));
    return 0;
}

// Echo scalar bytes and show what from() does
int test_scalar_echo(const unsigned char* in_bytes, unsigned char* out_mont, unsigned char* out_canonical) {
    scalar_t s;
    memcpy(&s, in_bytes, sizeof(scalar_t));
    // Copy Montgomery form back
    memcpy(out_mont, &s, sizeof(scalar_t));
    // Convert from Montgomery
    scalar_t s_canon = s;
    s_canon.from();
    memcpy(out_canonical, &s_canon, sizeof(scalar_t));
    return 0;
}

// Return sizes for verification
int test_sizes(size_t* affine_sz, size_t* scalar_sz, size_t* point_sz) {
    *affine_sz = sizeof(affine_t);
    *scalar_sz = sizeof(scalar_t);
    *point_sz = sizeof(point_t);
    return 0;
}

} // extern "C"
