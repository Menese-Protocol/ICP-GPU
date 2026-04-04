#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"
__global__ void test() {
    if (threadIdx.x != 0) return;
    RnsFp two = rns_add(rns_one(), rns_one());
    RnsFp seven; for(int i=0;i<RNS_K;i++){seven.r1[i]=RNS_ORACLE_MONT7_M1[i];seven.r2[i]=RNS_ORACLE_MONT7_M2[i];} seven.rr=RNS_ORACLE_MONT7_RED;
    RnsFp sum = rns_add(two, seven);
    RnsFp diff = rns_sub(two, seven);
    printf("sum  B1[:3]: %u %u %u\n", sum.r1[0], sum.r1[1], sum.r1[2]);
    printf("diff B1[:3]: %u %u %u\n", diff.r1[0], diff.r1[1], diff.r1[2]);
    printf("Python sum:  3202603 658325242 904902748\n");
    printf("Python diff: 21254410 251165773 167000367\n");
}
int main() { test<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
