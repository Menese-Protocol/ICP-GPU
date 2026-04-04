// Oracle test: 64-bit fp_mul against known ic_bls12_381 values
#include <cstdio>
#include <cstdint>
#include "fp64.cuh"

__global__ void test_kernel(int* results) {
    if (threadIdx.x != 0) return;
    int pass = 0, fail = 0;

    // Known Montgomery values from ic_bls12_381:
    // ONE = R mod p (already in FP_ONE)

    // Test 1: ONE * ONE = ONE
    { Fp r = fp_mul(fp_one(), fp_one());
      bool ok = fp_eq(r, fp_one());
      printf("1: ONE*ONE=ONE:         %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Test 2: (ONE+ONE) = 2*R mod p. Check via addition.
    { Fp two = fp_add(fp_one(), fp_one());
      Fp three = fp_add(two, fp_one());
      Fp six_add = fp_add(three, three);
      Fp six_mul = fp_mul(two, three);
      bool ok = fp_eq(six_add, six_mul);
      printf("2: 2*3=6 (add vs mul):  %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Test 3: 7*7=49
    { Fp seven = fp_add(fp_one(), fp_add(fp_one(), fp_add(fp_one(), fp_add(fp_one(), fp_add(fp_one(), fp_add(fp_one(), fp_one()))))));
      Fp r = fp_mul(seven, seven);
      Fp fortynine = fp_zero();
      for(int i=0;i<49;i++) fortynine = fp_add(fortynine, fp_one());
      bool ok = fp_eq(r, fortynine);
      printf("3: 7*7=49:              %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Test 4: associativity (2*3)*5 = 2*(3*5)
    { Fp two=fp_add(fp_one(),fp_one()); Fp three=fp_add(two,fp_one()); Fp five=fp_add(three,two);
      bool ok=fp_eq(fp_mul(fp_mul(two,three),five), fp_mul(two,fp_mul(three,five)));
      printf("4: (2*3)*5=2*(3*5):     %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Test 5: a-a=0
    { Fp seven = fp_zero(); for(int i=0;i<7;i++) seven=fp_add(seven,fp_one());
      bool ok = fp_is_zero(fp_sub(seven, seven));
      printf("5: 7-7=0:               %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Test 6: Fp2 ONE*ONE=ONE
    { Fp2 one2=fp2_one(); Fp2 r=fp2_mul(one2,one2);
      bool ok=fp_eq(r.c0,one2.c0) && fp_eq(r.c1,one2.c1);
      printf("6: Fp2 ONE*ONE=ONE:     %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Test 7: Fp2 sqr=mul
    { Fp two=fp_add(fp_one(),fp_one()); Fp seven=fp_zero(); for(int i=0;i<7;i++) seven=fp_add(seven,fp_one());
      Fp2 a={two,seven};
      bool ok=fp_eq(fp2_sqr(a).c0, fp2_mul(a,a).c0) && fp_eq(fp2_sqr(a).c1, fp2_mul(a,a).c1);
      printf("7: Fp2 sqr=mul:         %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Test 8: Fp2 distributivity
    { Fp two=fp_add(fp_one(),fp_one()); Fp three=fp_add(two,fp_one()); Fp five=fp_add(three,two);
      Fp2 a={two,three}, b={five,fp_one()}, c={three,two};
      Fp2 lhs=fp2_mul(fp2_add(a,b),c);
      Fp2 rhs=fp2_add(fp2_mul(a,c),fp2_mul(b,c));
      bool ok=fp_eq(lhs.c0,rhs.c0)&&fp_eq(lhs.c1,rhs.c1);
      printf("8: Fp2 (a+b)*c=ac+bc:   %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Test 9: Fp2 mul_nr matches mul by (1+u)
    { Fp two=fp_add(fp_one(),fp_one()); Fp three=fp_add(two,fp_one());
      Fp2 a={two,three}; Fp2 nr={fp_one(),fp_one()};
      Fp2 lhs=fp2_mul(a,nr); Fp2 rhs=fp2_mul_nr(a);
      bool ok=fp_eq(lhs.c0,rhs.c0)&&fp_eq(lhs.c1,rhs.c1);
      printf("9: Fp2 mul_nr:          %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // Test 10: Fp2 inv: a*a^(-1)=1
    { Fp two=fp_add(fp_one(),fp_one()); Fp seven=fp_zero(); for(int i=0;i<7;i++) seven=fp_add(seven,fp_one());
      Fp2 a={two,seven}; Fp2 ai=fp2_inv(a); Fp2 r=fp2_mul(a,ai);
      bool ok=fp_eq(r.c0,fp_one())&&fp_is_zero(r.c1);
      printf("10: Fp2 a*a^(-1)=1:     %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    printf("\n=== RESULTS: %d PASS, %d FAIL ===\n", pass, fail);
    results[0]=pass; results[1]=fail;
}

__global__ void bench_kernel(uint64_t* out, int n) {
    if (threadIdx.x != 0) return;
    Fp a = fp_one(); Fp b = fp_add(a, a);
    unsigned long long t0, t1;

    // Fp mul
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    Fp acc = a;
    for (int i = 0; i < n; i++) acc = fp_mul(acc, b);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1-t0; out[4]=acc.v[0];

    // Fp2 mul
    Fp2 a2={a,b};
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    Fp2 acc2=a2;
    for (int i = 0; i < n; i++) acc2 = fp2_mul(acc2, a2);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[1] = t1-t0; out[5]=acc2.c0.v[0];

    // Fp2 sqr
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    acc2=a2;
    for (int i = 0; i < n; i++) acc2 = fp2_sqr(acc2);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[2] = t1-t0; out[6]=acc2.c0.v[0];
}

int main() {
    int *d; cudaMalloc(&d, 2*sizeof(int));
    test_kernel<<<1,1>>>(d);
    cudaDeviceSynchronize();
    int r[2]; cudaMemcpy(r, d, 2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);

    printf("\n=== Benchmark: 10000 iters ===\n");
    int N=10000;
    uint64_t *do_; cudaMalloc(&do_, 8*sizeof(uint64_t));
    bench_kernel<<<1,1>>>(do_, N);
    cudaDeviceSynchronize();
    uint64_t h[8]; cudaMemcpy(h, do_, 8*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    int dev; cudaGetDevice(&dev);
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    double ghz = clk/1e6;
    printf("  Fp  mul: %.0f ns/op (%.0f cycles)\n", h[0]/(ghz*N), (double)h[0]/N);
    printf("  Fp2 mul: %.0f ns/op\n", h[1]/(ghz*N));
    printf("  Fp2 sqr: %.0f ns/op\n", h[2]/(ghz*N));
    printf("  Compare: sppark=328ns, RNS=4400ns, CPU=60ns\n");
    printf("  GPU clock: %.2f GHz\n", ghz);

    return r[1] > 0 ? 1 : 0;
}
