// Step 1: Verify sppark fp_t matches our proven fp_mul
// sppark uses 12×uint32 limbs, we use 6×uint64
// Same Montgomery form, same constants — just different word size

#include <cstdint>
#include <cstdio>

// Include sppark's BLS12-381 field type
#include "/workspace/sppark/ff/bls12-381.hpp"

// Our proven constants for comparison
__device__ __constant__ uint64_t OUR_FP_ONE[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL, 0x5f48985753c758baULL,
    0x77ce585370525745ULL, 0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL
};

// Convert sppark fp_t (12×u32) to our format (6×u64)
__device__ void sp_to_u64(const bls12_381::fp_t& sp, uint64_t out[6]) {
    const uint32_t* w = (const uint32_t*)&sp;
    for (int i = 0; i < 6; i++) {
        out[i] = ((uint64_t)w[2*i+1] << 32) | w[2*i];
    }
}

// Convert our format (6×u64) to sppark fp_t (12×u32)
__device__ bls12_381::fp_t u64_to_sp(const uint64_t in[6]) {
    bls12_381::fp_t r;
    uint32_t* w = (uint32_t*)&r;
    for (int i = 0; i < 6; i++) {
        w[2*i] = (uint32_t)(in[i]);
        w[2*i+1] = (uint32_t)(in[i] >> 32);
    }
    return r;
}

__global__ void test_sppark_fp() {
    using fp_t = bls12_381::fp_t;

    // Test 1: sppark ONE should match our ONE
    fp_t one = fp_t::one();
    uint64_t one_u64[6];
    sp_to_u64(one, one_u64);

    bool one_match = true;
    for (int i = 0; i < 6; i++) if (one_u64[i] != OUR_FP_ONE[i]) one_match = false;
    printf("TEST 1 (sppark ONE == our ONE): %s\n", one_match ? "PASS" : "FAIL");
    if (!one_match) {
        printf("  sppark: [%016llx,%016llx,%016llx,%016llx,%016llx,%016llx]\n",
               (unsigned long long)one_u64[0],(unsigned long long)one_u64[1],
               (unsigned long long)one_u64[2],(unsigned long long)one_u64[3],
               (unsigned long long)one_u64[4],(unsigned long long)one_u64[5]);
        printf("  ours:   [760900000002fffd,ebf4000bc40c0002,5f48985753c758ba,77ce585370525745,5c071a97a256ec6d,15f65ec3fa80e493]\n");
    }

    // Test 2: sppark mul(2,3) should give 6
    // Build mont(2) = ONE + ONE
    fp_t two = one + one;
    fp_t three = two + one;
    fp_t six_mul = two * three;
    fp_t six_add = three + three;

    uint64_t six_m[6], six_a[6];
    sp_to_u64(six_mul, six_m);
    sp_to_u64(six_add, six_a);

    bool six_match = true;
    for (int i = 0; i < 6; i++) if (six_m[i] != six_a[i]) six_match = false;
    printf("TEST 2 (sppark 2*3 == 3+3): %s\n", six_match ? "PASS" : "FAIL");

    // Test 3: Cross-check with our proven mont(6) value from Python oracle
    // Python: mont(6) = 0x223b00000013aa97, 0xee5c004d21a40010, ...
    uint64_t expected_six[6] = {0x223b00000013aa97ULL, 0xee5c004d21a40010ULL,
        0x37bf74e7253745acULL, 0xd881985be054ade3ULL, 0xb0a058fe7d8f2a5bULL, 0x01c0df04bf85da70ULL};
    bool oracle_match = true;
    for (int i = 0; i < 6; i++) if (six_m[i] != expected_six[i]) oracle_match = false;
    printf("TEST 3 (sppark mont(6) == Python oracle): %s\n", oracle_match ? "PASS" : "FAIL");

    // Test 4: Multiply G1.x coordinate
    uint64_t gx[6] = {0x5cb38790fd530c16ULL,0x7817fc679976fff5ULL,0x154f95c7143ba1c1ULL,
                       0xf0ae6acdf3d0e747ULL,0xedce6ecc21dbf440ULL,0x120177419e0bfb75ULL};
    fp_t sp_gx = u64_to_sp(gx);
    fp_t sp_gx_sq = sp_gx * sp_gx;

    // Compare with our proven fp_mul result
    // We need to compute gx*gx with our code to get the expected value
    // For now, just verify sp*sp == sp^2 (self-consistency)
    uint64_t gx_sq[6];
    sp_to_u64(sp_gx_sq, gx_sq);
    printf("TEST 4 (sppark G1.x² first limb): %016llx\n", (unsigned long long)gx_sq[0]);

    // Test 5: Negation roundtrip
    fp_t neg_one = -one;
    fp_t zero = one + neg_one;
    printf("TEST 5 (sppark 1 + (-1) == 0): %s\n", zero.is_zero() ? "PASS" : "FAIL");

    // Test 6: Subtraction
    fp_t diff = three - two;
    uint64_t diff_u64[6], one_u64b[6];
    sp_to_u64(diff, diff_u64);
    sp_to_u64(one, one_u64b);
    bool sub_match = true;
    for (int i = 0; i < 6; i++) if (diff_u64[i] != one_u64b[i]) sub_match = false;
    printf("TEST 6 (sppark 3-2 == 1): %s\n", sub_match ? "PASS" : "FAIL");
}

int main() {
    printf("=== sppark Fp vs Our Proven Fp ===\n");
    test_sppark_fp<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }
    printf("=== Done ===\n");
    return 0;
}
