#include <cstdio>
#include <cstdint>
#include "rns_fp_v2_div.cuh"
__global__ void test() {
    if (threadIdx.x != 0) return;
    RnsFp two = rns_add(rns_one(), rns_one());
    RnsFp three = rns_add(two, rns_one());
    RnsFp five = rns_add(three, two);
    RnsFp seven; for(int i=0;i<RNS_K;i++){seven.r1[i]=RNS_ORACLE_MONT7_M1[i];seven.r2[i]=RNS_ORACLE_MONT7_M2[i];} seven.rr=RNS_ORACLE_MONT7_RED;
    
    RnsFp2 a={two,seven};
    printf("sqr=mul: %s\n", rns_fp2_eq(rns_fp2_sqr(a),rns_fp2_mul(a,a))?"PASS":"FAIL");
    
    RnsFp2 aa={two,three}, bb={five,rns_one()}, cc={three,two};
    RnsFp2 lhs=rns_fp2_mul(rns_fp2_add(aa,bb),cc);
    RnsFp2 rhs=rns_fp2_add(rns_fp2_mul(aa,cc),rns_fp2_mul(bb,cc));
    printf("distrib: %s\n", rns_fp2_eq(lhs,rhs)?"PASS":"FAIL");
    
    RnsFp2 one=rns_fp2_one();
    printf("ONE*ONE=ONE: %s\n", rns_fp2_eq(rns_fp2_mul(one,one),one)?"PASS":"FAIL");
    
    RnsFp2 nr={rns_one(),rns_one()};
    printf("mul_nr: %s\n", rns_fp2_eq(rns_fp2_mul(a,nr),rns_fp2_mul_nr(a))?"PASS":"FAIL");
    
    RnsFp2 ab=rns_fp2_mul(aa,bb);
    printf("(ab)^2=a^2*b^2: %s\n", rns_fp2_eq(rns_fp2_mul(ab,ab),rns_fp2_mul(rns_fp2_mul(aa,aa),rns_fp2_mul(bb,bb)))?"PASS":"FAIL");
}
int main() { test<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
