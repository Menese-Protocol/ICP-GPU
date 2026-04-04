#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"
__global__ void test() {
    if (threadIdx.x != 0) return;
    RnsFp12 one12 = rns_fp12_one();
    
    // Easy part step by step
    RnsFp12 t0 = rns_fp12_conj(one12);
    printf("conj(ONE)==ONE: %s\n", rns_fp12_eq(t0, one12)?"Y":"N");
    
    RnsFp12 t1 = rns_fp12_inv(one12);
    printf("inv(ONE)==ONE: %s\n", rns_fp12_eq(t1, one12)?"Y":"N");
    
    RnsFp12 t2 = rns_fp12_mul(t0, t1);  // conj(1) * inv(1) = 1*1 = 1
    printf("conj*inv==ONE: %s\n", rns_fp12_eq(t2, one12)?"Y":"N");
    
    // t2 = frob(frob(t2)) * t1
    RnsFp12 t2ff = rns_fp12_mul(rns_fp12_frob(rns_fp12_frob(t2)), t2);
    printf("frob^2(1)*1==ONE: %s\n", rns_fp12_eq(t2ff, one12)?"Y":"N");
    
    // Hard part starts with cyclotomic_square
    RnsFp12 cs = rns_cyclotomic_square(t2ff);
    printf("cyc_sqr(ONE)==ONE: %s\n", rns_fp12_eq(cs, one12)?"Y":"N");
    
    RnsFp12 cs_conj = rns_fp12_conj(cs);
    printf("conj(cyc_sqr(ONE))==ONE: %s\n", rns_fp12_eq(cs_conj, one12)?"Y":"N");
    
    RnsFp12 ce = rns_cyc_exp(t2ff);
    printf("cyc_exp(ONE)==ONE: %s\n", rns_fp12_eq(ce, one12)?"Y":"N");
    
    // Full final_exp
    RnsFp12 fe = rns_final_exp(one12);
    printf("final_exp(ONE)==ONE: %s\n", rns_fp12_eq(fe, one12)?"Y":"N");
    printf("final_exp(ONE).c0.c0.c0.rr = %u (ONE.rr = %u)\n", 
           rns_normalize(fe.c0.c0.c0).rr, rns_one().rr);
}
int main() { test<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
