// Test V2 Fp2 with % operator (no Barrett) to isolate the bug
#include <cstdio>
#include <cstdint>

// Temporarily redefine bred and mmul to use % 
#define USE_DIVISION_NOT_BARRETT
#include "rns_fp_v2.cuh"

__global__ void test() {
    if (threadIdx.x != 0) return;
    RnsFp two = rns_add(rns_one(), rns_one());
    RnsFp seven; for(int i=0;i<RNS_K;i++){seven.r1[i]=RNS_ORACLE_MONT7_M1[i];seven.r2[i]=RNS_ORACLE_MONT7_M2[i];} seven.rr=RNS_ORACLE_MONT7_RED;
    RnsFp2 a = {two, seven};
    RnsFp2 sq = rns_fp2_sqr(a);
    RnsFp2 mm = rns_fp2_mul(a, a);
    printf("sqr.c0.rr=%u mul.c0.rr=%u match=%s\n", sq.c0.rr, mm.c0.rr, rns_fp2_eq(sq,mm)?"YES":"NO");
    
    // Test distributivity
    RnsFp three = rns_add(two, rns_one());
    RnsFp five = rns_add(three, two);
    RnsFp2 aa={two,three}, bb={five,rns_one()}, cc={three,two};
    RnsFp2 lhs = rns_fp2_mul(rns_fp2_add(aa,bb),cc);
    RnsFp2 rhs = rns_fp2_add(rns_fp2_mul(aa,cc),rns_fp2_mul(bb,cc));
    printf("distrib: %s\n", rns_fp2_eq(lhs,rhs)?"PASS":"FAIL");
}
int main() { test<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
