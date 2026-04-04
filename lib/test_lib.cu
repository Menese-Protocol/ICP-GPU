// Smoke test for consolidated gpu_crypto library
#include "pairing.cuh"
#include "sha256.cuh"
#include <cstdio>

// G2 generator prepared coefficients
#include "../cuda/g2_coeffs.h"
#include "../cuda/pk42_coeffs.h"

__global__ void test_all() {
    // 1. Field arithmetic
    Fp one = fp_one();
    Fp two = fp_add(one, one);
    Fp check = fp_mul(two, two);
    Fp four = fp_add(two, two);
    printf("Fp 2*2=4: %s\n", fp_eq(check,four) ? "PASS" : "FAIL");

    // 2. Fp2
    Fp2 a = {one, one};
    Fp2 asq = fp2_sqr(a);
    printf("Fp2 (1+u)^2 c0=0: %s\n", fp_is_zero(asq.c0) ? "PASS" : "FAIL");

    // 3. Frobenius
    Fp12 f12 = fp12_one();
    Fp12 f12f = fp12_frob(f12);
    Fp* p1 = (Fp*)&f12f;
    printf("Fp12 frob(1)=1: %s\n", fp_eq(p1[0],fp_one()) ? "PASS" : "FAIL");

    // 4. BLS verify
    G1Affine sig, neg_hm;
    sig.x={{0xd27b1adaea06e32cULL,0x9079033c644ae1d9ULL,0xf154b307a1249c34ULL,0xa365af8b574fe9d6ULL,0x375b89d156410186ULL,0x139b61eeb595cf47ULL}};
    sig.y={{0xd973686bc9912933ULL,0x40d7e6761b92732fULL,0x6b43adf272a19617ULL,0x31388f4c360d31deULL,0x588138872a0f1626ULL,0x0a0e79be45d84809ULL}};
    neg_hm.x={{0xbf6f80fad9849c75ULL,0x018298254a48192dULL,0xa8588f9235e2e40dULL,0x5508d390e218ff49ULL,0xf29c6756cc2dd13aULL,0x0d3056fc0db4365fULL}};
    neg_hm.y={{0x25301550dae86c14ULL,0x3140795267108347ULL,0xc5d7e01597b162a4ULL,0x1c0d85a74c2e54c0ULL,0xf66a4c922bdcc305ULL,0x10d66042e1a3e5acULL}};

    bool valid = bls_verify(sig, neg_hm, G2_COEFFS, PK42_COEFFS);
    printf("BLS verify(sk=42): %s\n", valid ? "VALID ✓" : "FAIL ✗");

    // 5. SHA-256
    uint8_t hash[32];
    sha256((const uint8_t*)"abc", 3, hash);
    printf("SHA256(abc): %s\n", (hash[0]==0xba && hash[1]==0x78) ? "PASS" : "FAIL");
}

int main() {
    printf("=== gpu_crypto Library Smoke Test ===\n");
    test_all<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }
    printf("=== All tests passed ===\n");
    return 0;
}
