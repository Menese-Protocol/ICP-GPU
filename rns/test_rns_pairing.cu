// Test RNS pairing building blocks: Frobenius, inversion, cyc_exp, final_exp
#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"

__global__ void test_kernel(int* results) {
    if (threadIdx.x != 0) return;
    int pass = 0, fail = 0;

    printf("=== Pairing Building Blocks ===\n\n");

    // Build test Fp12 value
    RnsFp one = rns_one();
    RnsFp two = rns_add(one, one);
    RnsFp three = rns_add(two, one);
    RnsFp five = rns_add(three, two);
    RnsFp seven; for(int i=0;i<RNS_K;i++){seven.r1[i]=RNS_ORACLE_MONT7_M1[i];seven.r2[i]=RNS_ORACLE_MONT7_M2[i];} seven.rr=RNS_ORACLE_MONT7_RED;
    RnsFp2 f2a={two,three}, f2b={five,one}, f2c={seven,two};
    RnsFp6 f6a={f2a,f2b,f2c}, f6b={f2c,f2a,f2b};

    // 1: Fp2 inv: a * a^(-1) = 1
    { RnsFp2 a = {two, seven};
      RnsFp2 ai = rns_fp2_inv(a);
      RnsFp2 r = rns_fp2_mul(a, ai);
      bool ok = rns_fp2_eq(r, rns_fp2_one());
      printf("1: Fp2 a * a^(-1) = 1          : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 2: Fp12 frob(one) = one (frobenius fixes 1)
    { RnsFp12 one12 = rns_fp12_one();
      RnsFp12 fr = rns_fp12_frob(one12);
      bool ok = rns_fp12_eq(fr, one12);
      printf("2: frob(ONE) = ONE             : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 3: frob^12(a) = a (frobenius has order 12... actually order dividing 12)
    // For Fp12 over Fp: frob^1 has order 12. But let's test frob^6 = conjugate
    { RnsFp12 a = {f6a, f6b};
      RnsFp12 f6 = a;
      for (int i = 0; i < 6; i++) f6 = rns_fp12_frob(f6);
      bool ok = rns_fp12_eq(f6, rns_fp12_conj(a));
      printf("3: frob^6(a) = conj(a)         : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 4: Fp12 inv: a * a^(-1) = 1
    { RnsFp12 a = {f6a, f6b};
      RnsFp12 ai = rns_fp12_inv(a);
      RnsFp12 r = rns_fp12_mul(a, ai);
      bool ok = rns_fp12_eq(r, rns_fp12_one());
      printf("4: Fp12 a * a^(-1) = 1         : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 5: final_exp(ONE) = ONE (e(identity) = identity)
    { RnsFp12 one12 = rns_fp12_one();
      RnsFp12 r = rns_final_exp(one12);
      bool ok = rns_fp12_eq(r, one12);
      printf("5: final_exp(ONE) = ONE        : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 6: cyc_exp(ONE) = ONE
    { RnsFp12 one12 = rns_fp12_one();
      RnsFp12 r = rns_cyc_exp(one12);
      bool ok = rns_fp12_eq(r, one12);
      printf("6: cyc_exp(ONE) = ONE          : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    printf("\n=== RESULTS: %d PASS, %d FAIL ===\n", pass, fail);
    results[0] = pass;
    results[1] = fail;
}

int main() {
    int *d;
    cudaMalloc(&d, 2*sizeof(int));
    test_kernel<<<1,1>>>(d);
    cudaDeviceSynchronize();
    int r[2];
    cudaMemcpy(r, d, 2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return r[1] > 0 ? 1 : 0;
}
