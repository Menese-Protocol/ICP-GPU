#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"
__global__ void test() {
    if (threadIdx.x != 0) return;
    RnsFp one = rns_one();
    printf("ONE.rr = %u\n", one.rr);
    RnsFp one_norm = rns_normalize(one);
    printf("normalize(ONE).rr = %u\n", one_norm.rr);
    printf("ONE == normalize(ONE): %s\n", rns_eq_raw(one, one_norm)?"Y":"N");
    // Fp2 ONE
    RnsFp2 one2 = rns_fp2_one();
    RnsFp2 r = rns_fp2_mul(one2, one2);
    printf("Fp2 ONE*ONE c0.rr = %u (expect %u)\n", rns_normalize(r.c0).rr, one.rr);
    printf("Fp2 ONE*ONE c1.rr = %u (expect 0)\n", rns_normalize(r.c1).rr);
}
int main() { test<<<1,1>>>(); cudaDeviceSynchronize(); }
