// Test RNS Fp6 + Fp12 tower — algebraic properties
#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"

__global__ void test_kernel(int* results) {
    if (threadIdx.x != 0) return;
    int pass = 0, fail = 0;

    // Build some Fp2 test values
    RnsFp one = rns_one();
    RnsFp two = rns_add(one, one);
    RnsFp three = rns_add(two, one);
    RnsFp five = rns_add(three, two);
    RnsFp seven; for(int i=0;i<RNS_K;i++){seven.r1[i]=RNS_ORACLE_MONT7_M1[i];seven.r2[i]=RNS_ORACLE_MONT7_M2[i];} seven.rr=RNS_ORACLE_MONT7_RED;

    RnsFp2 f2a = {two, three};
    RnsFp2 f2b = {five, one};
    RnsFp2 f2c = {seven, two};

    printf("=== Fp6 Tests ===\n");

    // Fp6 1: ONE * ONE = ONE
    { RnsFp6 one6 = rns_fp6_one();
      bool ok = rns_fp6_eq(rns_fp6_mul(one6, one6), one6);
      printf("Fp6  1: ONE*ONE=ONE            : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Fp6 2: a*b = b*a (commutativity)
    { RnsFp6 a = {f2a, f2b, f2c};
      RnsFp6 b = {f2c, f2a, f2b};
      bool ok = rns_fp6_eq(rns_fp6_mul(a, b), rns_fp6_mul(b, a));
      printf("Fp6  2: a*b = b*a              : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Fp6 3: (a*b)*c = a*(b*c) (associativity)
    { RnsFp6 a = {f2a, f2b, f2c};
      RnsFp6 b = {f2c, f2a, f2b};
      RnsFp6 c = {f2b, f2c, f2a};
      bool ok = rns_fp6_eq(rns_fp6_mul(rns_fp6_mul(a,b),c), rns_fp6_mul(a,rns_fp6_mul(b,c)));
      printf("Fp6  3: (a*b)*c = a*(b*c)      : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Fp6 4: (a+b)*c = a*c + b*c (distributivity)
    { RnsFp6 a = {f2a, f2b, f2c};
      RnsFp6 b = {f2c, f2a, f2b};
      RnsFp6 c = {f2b, f2c, f2a};
      RnsFp6 lhs = rns_fp6_mul(rns_fp6_add(a,b), c);
      RnsFp6 rhs = rns_fp6_add(rns_fp6_mul(a,c), rns_fp6_mul(b,c));
      bool ok = rns_fp6_eq(lhs, rhs);
      printf("Fp6  4: (a+b)*c = a*c+b*c      : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    printf("\n=== Fp12 Tests ===\n");

    // Build Fp6 values
    RnsFp6 f6a = {f2a, f2b, f2c};
    RnsFp6 f6b = {f2c, f2a, f2b};

    // Fp12 1: ONE * ONE = ONE
    { RnsFp12 one12 = rns_fp12_one();
      bool ok = rns_fp12_eq(rns_fp12_mul(one12, one12), one12);
      printf("Fp12 1: ONE*ONE=ONE            : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Fp12 2: a*b = b*a
    { RnsFp12 a = {f6a, f6b};
      RnsFp12 b = {f6b, f6a};
      bool ok = rns_fp12_eq(rns_fp12_mul(a,b), rns_fp12_mul(b,a));
      printf("Fp12 2: a*b = b*a              : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Fp12 3: sqr(a) = a*a
    { RnsFp12 a = {f6a, f6b};
      bool ok = rns_fp12_eq(rns_fp12_sqr(a), rns_fp12_mul(a,a));
      printf("Fp12 3: sqr(a) = a*a           : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Fp12 4: cyclotomic_sqr(a) = a*a (for cyclotomic subgroup elements)
    // For non-cyclotomic elements, cyclotomic_sqr gives wrong results.
    // Test: cyc_sqr(ONE) should equal ONE (ONE is in cyclotomic subgroup)
    { RnsFp12 one12 = rns_fp12_one();
      RnsFp12 cyc = rns_cyclotomic_square(one12);
      bool ok = rns_fp12_eq(cyc, one12);
      printf("Fp12 4: cyc_sqr(ONE) = ONE     : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Fp12 5: conj(a) * a should have c1 = 0 for norm elements
    // Actually: conj(a)*a = |a|^2 which is in Fp6 (c1=0 only for specific cases)
    // Better test: a + conj(a) has c1 = 0
    { RnsFp12 a = {f6a, f6b};
      RnsFp12 ac = rns_fp12_conj(a);
      RnsFp12 sum = {rns_fp6_add(a.c0, ac.c0), rns_fp6_add(a.c1, ac.c1)};
      // c1 should be f6b + (-f6b) = 0
      bool ok = rns_fp6_eq(sum.c1, rns_fp6_zero());
      printf("Fp12 5: a+conj(a) has c1=0     : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Fp12 6: (a*b)^2 = a^2 * b^2
    { RnsFp12 a = {f6a, f6b};
      RnsFp12 b = {f6b, f6a};
      RnsFp12 ab = rns_fp12_mul(a, b);
      bool ok = rns_fp12_eq(rns_fp12_sqr(ab), rns_fp12_mul(rns_fp12_sqr(a), rns_fp12_sqr(b)));
      printf("Fp12 6: (a*b)^2 = a^2*b^2      : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    printf("\n=== RESULTS: %d PASS, %d FAIL ===\n", pass, fail);
    results[0] = pass;
    results[1] = fail;
}

__global__ void bench_kernel(uint64_t* out, int n) {
    if (threadIdx.x != 0) return;
    RnsFp one = rns_one();
    RnsFp two = rns_add(one, one);
    RnsFp three = rns_add(two, one);
    RnsFp five = rns_add(three, two);
    RnsFp seven; for(int i=0;i<RNS_K;i++){seven.r1[i]=RNS_ORACLE_MONT7_M1[i];seven.r2[i]=RNS_ORACLE_MONT7_M2[i];} seven.rr=RNS_ORACLE_MONT7_RED;

    RnsFp2 f2a={two,three}, f2b={five,one}, f2c={seven,two};
    RnsFp6 f6a={f2a,f2b,f2c}, f6b={f2c,f2a,f2b};
    RnsFp12 a12={f6a, f6b};

    unsigned long long t0, t1;

    // Fp12 mul
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    RnsFp12 acc = a12;
    for (int i = 0; i < n; i++) acc = rns_fp12_mul(acc, a12);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
    out[4] = acc.c0.c0.c0.r1[0];

    // Fp12 sqr
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    acc = a12;
    for (int i = 0; i < n; i++) acc = rns_fp12_sqr(acc);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[1] = t1 - t0;
    out[5] = acc.c0.c0.c0.r1[0];

    // Cyclotomic sqr
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    acc = a12;
    for (int i = 0; i < n; i++) acc = rns_cyclotomic_square(acc);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[2] = t1 - t0;
    out[6] = acc.c0.c0.c0.r1[0];

    // Fp6 mul
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    RnsFp6 acc6 = f6a;
    for (int i = 0; i < n; i++) acc6 = rns_fp6_mul(acc6, f6a);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[3] = t1 - t0;
    out[7] = acc6.c0.c0.r1[0];
}

int main() {
    int *d_results;
    cudaMalloc(&d_results, 2*sizeof(int));
    test_kernel<<<1,1>>>(d_results);
    cudaDeviceSynchronize();
    int results[2];
    cudaMemcpy(results, d_results, 2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    printf("\n=== Benchmark: 1000 iterations ===\n");
    int N = 1000;
    uint64_t *d_out;
    cudaMalloc(&d_out, 8*sizeof(uint64_t));
    bench_kernel<<<1,1>>>(d_out, N);
    cudaDeviceSynchronize();
    uint64_t h[8];
    cudaMemcpy(h, d_out, 8*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    int dev; cudaGetDevice(&dev);
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    double ghz = clk / 1e6;

    printf("  Fp6  mul:        %7.0f ns/op (18 Fp muls)\n", (double)h[3]/(ghz*N));
    printf("  Fp12 mul:        %7.0f ns/op (54 Fp muls)\n", (double)h[0]/(ghz*N));
    printf("  Fp12 sqr:        %7.0f ns/op (36 Fp muls)\n", (double)h[1]/(ghz*N));
    printf("  Fp12 cyc_sqr:    %7.0f ns/op (12 Fp muls)\n", (double)h[2]/(ghz*N));
    printf("  GPU clock: %.2f GHz\n", ghz);

    return results[1] > 0 ? 1 : 0;
}
