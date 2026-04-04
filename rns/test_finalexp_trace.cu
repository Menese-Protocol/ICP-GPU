// Trace final_exp step by step — print normalized rr at each intermediate
#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"

// Inline the final_exp with trace points
__global__ void trace_final_exp() {
    if (threadIdx.x != 0) return;

    RnsFp12 f_in = rns_fp12_one();

    // Easy part
    RnsFp12 f = f_in;
    RnsFp12 t0 = rns_fp12_conj(f);
    printf("E1 conj:     %u\n", rns_normalize(t0.c0.c0.c0).rr);

    RnsFp12 t1 = rns_fp12_inv(f);
    printf("E2 inv:      %u\n", rns_normalize(t1.c0.c0.c0).rr);

    RnsFp12 t2 = rns_fp12_mul(t0, t1);
    printf("E3 t0*t1:    %u\n", rns_normalize(t2.c0.c0.c0).rr);

    t1 = t2;
    t2 = rns_fp12_mul(rns_fp12_frob(rns_fp12_frob(t2)), t1);
    printf("E4 frob2*t1: %u\n", rns_normalize(t2.c0.c0.c0).rr);

    // Hard part
    f = t2;
    printf("\nH0 f=t2:     %u\n", rns_normalize(f.c0.c0.c0).rr);

    t1 = rns_fp12_conj(rns_cyclotomic_square(t2));
    printf("H1 conj(csq): %u\n", rns_normalize(t1.c0.c0.c0).rr);

    RnsFp12 t3 = rns_cyc_exp(t2);
    printf("H2 cyc_exp:  %u\n", rns_normalize(t3.c0.c0.c0).rr);

    RnsFp12 t4 = rns_fp12_mul(t1, t3);
    printf("H3 t1*t3:    %u\n", rns_normalize(t4.c0.c0.c0).rr);

    t1 = rns_cyc_exp(t4);
    printf("H4 cyc_exp:  %u\n", rns_normalize(t1.c0.c0.c0).rr);

    t4 = rns_fp12_conj(t4);
    printf("H5 conj(t4): %u\n", rns_normalize(t4.c0.c0.c0).rr);

    f = rns_fp12_mul(f, t4);
    printf("H6 f*t4:     %u\n", rns_normalize(f.c0.c0.c0).rr);

    t4 = rns_cyclotomic_square(t3);
    printf("H7 csq(t3):  %u\n", rns_normalize(t4.c0.c0.c0).rr);

    t0 = rns_cyc_exp(t1);
    printf("H8 cyc_exp:  %u\n", rns_normalize(t0.c0.c0.c0).rr);

    t3 = rns_fp12_mul(t3, t0);
    printf("H9 t3*t0:    %u\n", rns_normalize(t3.c0.c0.c0).rr);

    t3 = rns_fp12_frob(rns_fp12_frob(t3));
    printf("H10 frob2:   %u\n", rns_normalize(t3.c0.c0.c0).rr);

    f = rns_fp12_mul(f, t3);
    printf("H11 f*t3:    %u\n", rns_normalize(f.c0.c0.c0).rr);

    t4 = rns_fp12_mul(t4, rns_cyc_exp(t0));
    printf("H12 t4*ce:   %u\n", rns_normalize(t4.c0.c0.c0).rr);

    f = rns_fp12_mul(f, rns_cyc_exp(t4));
    printf("H13 f*ce:    %u\n", rns_normalize(f.c0.c0.c0).rr);

    t4 = rns_fp12_mul(t4, rns_fp12_conj(t2));
    printf("H14 t4*cj:   %u\n", rns_normalize(t4.c0.c0.c0).rr);

    t2 = rns_fp12_mul(t2, t1);
    printf("H15 t2*t1:   %u\n", rns_normalize(t2.c0.c0.c0).rr);

    t2 = rns_fp12_frob(rns_fp12_frob(rns_fp12_frob(t2)));
    printf("H16 frob3:   %u\n", rns_normalize(t2.c0.c0.c0).rr);

    f = rns_fp12_mul(f, t2);
    printf("H17 f*t2:    %u\n", rns_normalize(f.c0.c0.c0).rr);

    t4 = rns_fp12_frob(t4);
    printf("H18 frob:    %u\n", rns_normalize(t4.c0.c0.c0).rr);

    f = rns_fp12_mul(f, t4);
    printf("H19 final:   %u\n", rns_normalize(f.c0.c0.c0).rr);

    printf("\nExpected ONE: %u\n", rns_one().rr);
}

int main() {
    trace_final_exp<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
