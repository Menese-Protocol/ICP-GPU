#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"
__global__ void test() {
    if (threadIdx.x != 0) return;
    RnsFp two = rns_add(rns_one(), rns_one());
    RnsFp seven; for(int i=0;i<RNS_K;i++){seven.r1[i]=RNS_ORACLE_MONT7_M1[i];seven.r2[i]=RNS_ORACLE_MONT7_M2[i];} seven.rr=RNS_ORACLE_MONT7_RED;
    
    // sum*diff (from sqr path)
    RnsFp sum = rns_add(two, seven);
    RnsFp diff = rns_sub(two, seven);
    
    // Step 1: q = sum * diff in B1
    printf("=== sum * diff (sqr path) ===\n");
    uint32_t q1[RNS_K];
    for (int i = 0; i < RNS_K; i++)
        q1[i] = mmul(sum.r1[i], diff.r1[i], RNS_M1[i], RNS_BARRETT_M1[i]);
    printf("q1[:3]: %u %u %u\n", q1[0], q1[1], q1[2]);
    
    // Step 2: t = q * neg_pinv
    uint32_t t1[RNS_K];
    for (int i = 0; i < RNS_K; i++)
        t1[i] = mmul(q1[i], RNS_NEG_PINV_M1[i], RNS_M1[i], RNS_BARRETT_M1[i]);
    printf("t1[:3]: %u %u %u\n", t1[0], t1[1], t1[2]);
    
    // Step 3: base extend t to B2
    // Manual: CRT coefficients
    uint32_t xt[RNS_K];
    for (int i = 0; i < RNS_K; i++)
        xt[i] = mmul(t1[i], RNS_MHAT_INV_M1[i], RNS_M1[i], RNS_BARRETT_M1[i]);
    printf("xt[:3]: %u %u %u\n", xt[0], xt[1], xt[2]);
    
    // Verify against division
    uint32_t q1_div = (uint32_t)(((uint64_t)sum.r1[0] * diff.r1[0]) % RNS_M1[0]);
    printf("\nq1[0] Barrett=%u, division=%u, match=%s\n", q1[0], q1_div, q1[0]==q1_div?"YES":"NO");
    
    uint32_t t1_div = (uint32_t)(((uint64_t)q1_div * RNS_NEG_PINV_M1[0]) % RNS_M1[0]);
    printf("t1[0] Barrett=%u, division=%u, match=%s\n", t1[0], t1_div, t1[0]==t1_div?"YES":"NO");
    
    uint32_t xt_div = (uint32_t)(((uint64_t)t1_div * RNS_MHAT_INV_M1[0]) % RNS_M1[0]);
    printf("xt[0] Barrett=%u, division=%u, match=%s\n", xt[0], xt_div, xt[0]==xt_div?"YES":"NO");
    
    // Full rns_mul result
    RnsFp r_sqr = rns_mul(sum, diff);
    printf("\nresult.rr=%u  result.r1[0]=%u\n", r_sqr.rr, r_sqr.r1[0]);
    
    // Now c0*c0
    printf("\n=== c0*c0 (mul path) ===\n");
    RnsFp r_c0c0 = rns_mul(two, two);
    printf("c0*c0.rr=%u  c0*c0.r1[0]=%u\n", r_c0c0.rr, r_c0c0.r1[0]);
    
    RnsFp r_c1c1 = rns_mul(seven, seven);
    printf("c1*c1.rr=%u  c1*c1.r1[0]=%u\n", r_c1c1.rr, r_c1c1.r1[0]);
    
    RnsFp r_mul_c0 = rns_sub(r_c0c0, r_c1c1);
    printf("c0c0-c1c1.rr=%u  .r1[0]=%u\n", r_mul_c0.rr, r_mul_c0.r1[0]);
    
    printf("\nsqr c0: rr=%u r1[0]=%u\n", r_sqr.rr, r_sqr.r1[0]);
    printf("mul c0: rr=%u r1[0]=%u\n", r_mul_c0.rr, r_mul_c0.r1[0]);
}
int main() { test<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
