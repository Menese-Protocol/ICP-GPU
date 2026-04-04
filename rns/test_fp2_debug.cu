// Debug Fp2 sqr vs mul
#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"

__global__ void debug_fp2() {
    if (threadIdx.x != 0) return;

    RnsFp two = rns_add(rns_one(), rns_one());
    RnsFp seven; for(int i=0;i<RNS_K;i++){seven.r1[i]=RNS_ORACLE_MONT7_M1[i];seven.r2[i]=RNS_ORACLE_MONT7_M2[i];} seven.rr=RNS_ORACLE_MONT7_RED;

    RnsFp2 a = {two, seven};  // 2 + 7u

    // sqr: (2+7u)^2 = (2+7)(2-7) + 2*2*7*u = (4-49) + 28u = -45 + 28u
    // mul: (2+7u)(2+7u) same

    RnsFp2 sq = rns_fp2_sqr(a);
    RnsFp2 mm = rns_fp2_mul(a, a);

    printf("sqr.c0.rr = %u\n", sq.c0.rr);
    printf("mul.c0.rr = %u\n", mm.c0.rr);
    printf("sqr.c1.rr = %u\n", sq.c1.rr);
    printf("mul.c1.rr = %u\n", mm.c1.rr);
    printf("c0 match: %s\n", rns_eq(sq.c0, mm.c0) ? "YES" : "NO");
    printf("c1 match: %s\n", rns_eq(sq.c1, mm.c1) ? "YES" : "NO");

    // Check each residue
    for (int i = 0; i < RNS_K; i++) {
        if (sq.c0.r1[i] != mm.c0.r1[i])
            printf("  c0.r1[%d]: sqr=%u mul=%u\n", i, sq.c0.r1[i], mm.c0.r1[i]);
    }
    for (int i = 0; i < RNS_K; i++) {
        if (sq.c1.r1[i] != mm.c1.r1[i])
            printf("  c1.r1[%d]: sqr=%u mul=%u\n", i, sq.c1.r1[i], mm.c1.r1[i]);
    }

    // The issue might be that `two` from add is unreduced.
    // Let's try with values from mul only:
    RnsFp one = rns_one();
    RnsFp m2 = rns_mul(one, rns_add(one, one));  // mul forces reduction
    RnsFp m7_red = rns_mul(seven, one);  // force reduction

    printf("\ntwo from add:  rr=%u r1[0]=%u\n", two.rr, two.r1[0]);
    printf("two from mul:  rr=%u r1[0]=%u\n", m2.rr, m2.r1[0]);
    printf("match: %s\n", rns_eq(two, m2) ? "YES" : "NO");
}

int main() {
    debug_fp2<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
