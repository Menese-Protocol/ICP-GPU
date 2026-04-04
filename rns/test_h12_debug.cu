#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"
__global__ void test() {
    if (threadIdx.x != 0) return;
    RnsFp12 one12 = rns_fp12_one();

    // Reproduce exact path to H12
    RnsFp12 t2 = one12; // after easy part, t2 = ONE for input ONE
    RnsFp12 f = t2;
    RnsFp12 t1 = rns_fp12_conj(rns_cyclotomic_square(t2));
    RnsFp12 t3 = rns_cyc_exp(t2);
    RnsFp12 t4 = rns_fp12_mul(t1, t3);
    t1 = rns_cyc_exp(t4);
    t4 = rns_fp12_conj(t4);
    f = rns_fp12_mul(f, t4);
    t4 = rns_cyclotomic_square(t3);  // H7
    RnsFp12 t0 = rns_cyc_exp(t1);   // H8

    printf("At H12:\n");
    printf("  t4.c0.c0.c0.rr (raw) = %u\n", t4.c0.c0.c0.rr);
    printf("  t4.c0.c0.c0.rr (norm) = %u\n", rns_normalize(t4.c0.c0.c0).rr);
    printf("  t0.c0.c0.c0.rr (raw) = %u\n", t0.c0.c0.c0.rr);
    printf("  t0.c0.c0.c0.rr (norm) = %u\n", rns_normalize(t0.c0.c0.c0).rr);

    RnsFp12 ce_t0 = rns_cyc_exp(t0);
    printf("  cyc_exp(t0).rr (raw) = %u\n", ce_t0.c0.c0.c0.rr);
    printf("  cyc_exp(t0).rr (norm) = %u\n", rns_normalize(ce_t0.c0.c0.c0).rr);

    RnsFp12 h12 = rns_fp12_mul(t4, ce_t0);
    printf("  t4*ce(t0).rr (raw) = %u\n", h12.c0.c0.c0.rr);
    printf("  t4*ce(t0).rr (norm) = %u\n", rns_normalize(h12.c0.c0.c0).rr);
    printf("  expected ONE rr = %u\n", rns_one().rr);

    // Also check: is t4 == ONE?
    printf("\n  t4 == ONE: %s\n", rns_fp12_eq(t4, one12)?"Y":"N");
    printf("  ce(t0) == ONE: %s\n", rns_fp12_eq(ce_t0, one12)?"Y":"N");
    printf("  t4*ce(t0) == ONE: %s\n", rns_fp12_eq(h12, one12)?"Y":"N");

    // Direct: ONE * ONE
    RnsFp12 oo = rns_fp12_mul(one12, one12);
    printf("\n  ONE*ONE == ONE: %s\n", rns_fp12_eq(oo, one12)?"Y":"N");
    printf("  ONE*ONE rr (norm) = %u\n", rns_normalize(oo.c0.c0.c0).rr);
}
int main() { test<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
