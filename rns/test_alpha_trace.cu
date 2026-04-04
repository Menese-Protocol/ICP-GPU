#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"

// Modified rns_mul that prints alpha values
__device__ __noinline__
RnsFp rns_mul_debug(const RnsFp& a, const RnsFp& b, const char* label) {
    // Just trace the B1→B2 alpha
    uint32_t q1[RNS_K];
    for (int i = 0; i < RNS_K; i++)
        q1[i] = mmul(a.r1[i], b.r1[i], RNS_M1[i], RNS_BARRETT_M1[i]);
    
    uint32_t t1[RNS_K];
    for (int i = 0; i < RNS_K; i++)
        t1[i] = mmul(q1[i], RNS_NEG_PINV_M1[i], RNS_M1[i], RNS_BARRETT_M1[i]);
    
    // CRT coefficients for base extension
    uint32_t xt[RNS_K];
    for (int i = 0; i < RNS_K; i++)
        xt[i] = mmul(t1[i], RNS_MHAT_INV_M1[i], RNS_M1[i], RNS_BARRETT_M1[i]);
    
    double alpha_d = 0.0;
    for (int i = 0; i < RNS_K; i++)
        alpha_d += (double)xt[i] / (double)RNS_M1[i];
    
    printf("%s: alpha_d=%.6f floor=%d frac=%.6f\n", 
           label, alpha_d, (int)alpha_d, alpha_d - (int)alpha_d);
    
    return rns_mul(a, b);
}

__global__ void test() {
    if (threadIdx.x != 0) return;
    
    RnsFp two = rns_add(rns_one(), rns_one());
    RnsFp seven; for(int i=0;i<RNS_K;i++){seven.r1[i]=RNS_ORACLE_MONT7_M1[i];seven.r2[i]=RNS_ORACLE_MONT7_M2[i];} seven.rr=RNS_ORACLE_MONT7_RED;
    
    // Fp2 sqr: needs rns_mul(sum, diff) where diff = c0 - c1
    RnsFp2 a = {two, seven};
    RnsFp sum = rns_add(a.c0, a.c1);   // 2 + 7 = 9 (unreduced, ok)
    RnsFp diff = rns_sub(a.c0, a.c1);  // 2 - 7 = -5 (wraps to M-5)
    
    printf("sum.rr = %u (expect 9*R mod mred)\n", sum.rr);
    printf("diff.rr = %u\n", diff.rr);
    printf("diff.r1[0] = %u, M1[0] = %u\n", diff.r1[0], RNS_M1[0]);
    
    // This is the mul inside fp2_sqr
    RnsFp sqr_c0 = rns_mul_debug(sum, diff, "sqr(sum*diff)");
    
    // For mul(a,a): t0 = c0*c0, t1 = c1*c1
    RnsFp t0 = rns_mul_debug(a.c0, a.c0, "mul(c0*c0)");
    RnsFp t1 = rns_mul_debug(a.c1, a.c1, "mul(c1*c1)");
    RnsFp mul_c0 = rns_sub(t0, t1);
    
    printf("\nsqr c0 rr=%u  mul c0 rr=%u  match=%s\n", 
           sqr_c0.rr, mul_c0.rr, rns_eq(sqr_c0, mul_c0)?"YES":"NO");
}
int main() { test<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
