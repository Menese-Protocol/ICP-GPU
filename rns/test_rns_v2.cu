// Test RNS V2: Barrett reduction + Fp2 tower
#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"

__device__ RnsFp rns_one_fn() { return rns_one(); }

__global__ void test_kernel(int* results) {
    if (threadIdx.x != 0) return;
    int pass = 0, fail = 0;

    printf("=== RNS V2: Barrett + Fp2 Oracle Test ===\n\n");

    // --- Fp tests (same as before, now with Barrett) ---

    // 1: ONE * ONE = ONE
    { RnsFp one = rns_one(); RnsFp r = rns_mul(one, one);
      bool ok = rns_eq(r, one);
      printf("Fp  1: ONE*ONE=ONE             : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 2: mont(7)^2 = mont(49)
    { RnsFp m7; for(int i=0;i<RNS_K;i++){m7.r1[i]=RNS_ORACLE_MONT7_M1[i];m7.r2[i]=RNS_ORACLE_MONT7_M2[i];} m7.rr=RNS_ORACLE_MONT7_RED;
      RnsFp m49; for(int i=0;i<RNS_K;i++){m49.r1[i]=RNS_ORACLE_MONT49_M1[i];m49.r2[i]=RNS_ORACLE_MONT49_M2[i];} m49.rr=RNS_ORACLE_MONT49_RED;
      RnsFp r = rns_mul(m7, m7);
      bool ok = rns_eq(r, m49);
      printf("Fp  2: mont(7)^2=mont(49)      : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 3: associativity (2*3)*5 = 2*(3*5)
    { RnsFp two=rns_add(rns_one(),rns_one());
      RnsFp three=rns_add(two,rns_one());
      RnsFp five=rns_add(three,two);
      RnsFp lhs=rns_mul(rns_mul(two,three),five);
      RnsFp rhs=rns_mul(two,rns_mul(three,five));
      bool ok = rns_eq(lhs,rhs);
      printf("Fp  3: (2*3)*5=2*(3*5)         : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 4: 6*6 = 4*9 (chained)
    { RnsFp two=rns_add(rns_one(),rns_one());
      RnsFp three=rns_add(two,rns_one());
      RnsFp six=rns_mul(two,three);
      RnsFp four=rns_mul(two,two);
      RnsFp nine=rns_mul(three,three);
      bool ok = rns_eq(rns_mul(six,six), rns_mul(four,nine));
      printf("Fp  4: 6*6=4*9=36              : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 5: ((2*3)*5)*7 = (2*(3*5))*7
    { RnsFp two=rns_add(rns_one(),rns_one());
      RnsFp three=rns_add(two,rns_one());
      RnsFp five=rns_add(three,two);
      RnsFp seven; for(int i=0;i<RNS_K;i++){seven.r1[i]=RNS_ORACLE_MONT7_M1[i];seven.r2[i]=RNS_ORACLE_MONT7_M2[i];} seven.rr=RNS_ORACLE_MONT7_RED;
      RnsFp lhs=rns_mul(rns_mul(rns_mul(two,three),five),seven);
      RnsFp rhs=rns_mul(rns_mul(two,rns_mul(three,five)),seven);
      bool ok=rns_eq(lhs,rhs);
      printf("Fp  5: deep chain assoc         : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // --- Fp2 tests ---
    printf("\n");

    // 6: Fp2 ONE * ONE = ONE
    { RnsFp2 one=rns_fp2_one(); RnsFp2 r=rns_fp2_mul(one,one);
      bool ok=rns_fp2_eq(r,one);
      printf("Fp2 6: ONE*ONE=ONE             : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 7: Fp2 (a+b)*c = a*c + b*c (distributivity)
    { RnsFp two=rns_add(rns_one(),rns_one());
      RnsFp three=rns_add(two,rns_one());
      RnsFp five=rns_add(three,two);
      RnsFp2 a={two, three};     // 2 + 3u
      RnsFp2 b={five, rns_one()}; // 5 + u
      RnsFp2 c={three, two};      // 3 + 2u
      RnsFp2 lhs=rns_fp2_mul(rns_fp2_add(a,b),c);
      RnsFp2 rhs=rns_fp2_add(rns_fp2_mul(a,c),rns_fp2_mul(b,c));
      bool ok=rns_fp2_eq(lhs,rhs);
      printf("Fp2 7: (a+b)*c = a*c+b*c       : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 8: Fp2 a*b = b*a (commutativity)
    { RnsFp two=rns_add(rns_one(),rns_one());
      RnsFp three=rns_add(two,rns_one());
      RnsFp2 a={two,three}; RnsFp2 b={three,two};
      bool ok=rns_fp2_eq(rns_fp2_mul(a,b),rns_fp2_mul(b,a));
      printf("Fp2 8: a*b = b*a               : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 9: Fp2 sqr(a) = a*a
    { RnsFp two=rns_add(rns_one(),rns_one());
      RnsFp seven; for(int i=0;i<RNS_K;i++){seven.r1[i]=RNS_ORACLE_MONT7_M1[i];seven.r2[i]=RNS_ORACLE_MONT7_M2[i];} seven.rr=RNS_ORACLE_MONT7_RED;
      RnsFp2 a={two,seven};
      bool ok=rns_fp2_eq(rns_fp2_sqr(a),rns_fp2_mul(a,a));
      printf("Fp2 9: sqr(a) = a*a            : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 10: Fp2 mul_nr property: (1+u)*(a) using mul vs mul_nr
    { RnsFp two=rns_add(rns_one(),rns_one());
      RnsFp three=rns_add(two,rns_one());
      RnsFp2 a={two,three};
      RnsFp2 nr={rns_one(),rns_one()};  // 1+u (non-residue)
      RnsFp2 lhs=rns_fp2_mul(a,nr);     // a*(1+u) via full mul
      RnsFp2 rhs=rns_fp2_mul_nr(a);     // a*(1+u) via optimized mul_nr
      bool ok=rns_fp2_eq(lhs,rhs);
      printf("Fp2 10: a*(1+u) mul vs mul_nr  : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 11: Fp2 conj property: a * conj(a) is real (c1 = 0 in exact Fp...
    //     but in lazy-reduction RNS, c1 might be nonzero equivalent to 0 mod p)
    //     Instead: conj(a) + a = 2*c0 (real part doubled)
    { RnsFp two=rns_add(rns_one(),rns_one());
      RnsFp three=rns_add(two,rns_one());
      RnsFp2 a={two,three};
      RnsFp2 ac=rns_fp2_conj(a);
      RnsFp2 sum=rns_fp2_add(a,ac);  // should be {2*c0, 0}
      RnsFp2 expected={rns_add(two,two), rns_zero()};
      bool ok=rns_fp2_eq(sum,expected);
      printf("Fp2 11: a+conj(a) = 2*Re(a)    : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    // 12: Fp2 chained: (a*b)*(a*b) = a*a * b*b  (since commutative)
    { RnsFp two=rns_add(rns_one(),rns_one());
      RnsFp three=rns_add(two,rns_one());
      RnsFp five=rns_add(three,two);
      RnsFp2 a={two,three}; RnsFp2 b={five,rns_one()};
      RnsFp2 ab=rns_fp2_mul(a,b);
      RnsFp2 lhs=rns_fp2_mul(ab,ab);
      RnsFp2 rhs=rns_fp2_mul(rns_fp2_mul(a,a),rns_fp2_mul(b,b));
      bool ok=rns_fp2_eq(lhs,rhs);
      printf("Fp2 12: (a*b)^2 = a^2*b^2      : %s\n", ok?"PASS":"FAIL");
      if(ok)pass++;else fail++; }

    printf("\n=== RESULTS: %d PASS, %d FAIL ===\n", pass, fail);
    results[0] = pass;
    results[1] = fail;
}

__global__ void bench_kernel(uint64_t* out, int n) {
    if (threadIdx.x != 0) return;
    RnsFp m7; for(int i=0;i<RNS_K;i++){m7.r1[i]=RNS_ORACLE_MONT7_M1[i];m7.r2[i]=RNS_ORACLE_MONT7_M2[i];} m7.rr=RNS_ORACLE_MONT7_RED;

    // Fp mul bench
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    RnsFp acc = m7;
    for (int i = 0; i < n; i++) acc = rns_mul(acc, m7);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
    out[4] = acc.r1[0]; // anti-DCE

    // Fp2 mul bench
    RnsFp2 a2 = {m7, rns_add(m7, rns_one())};
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    RnsFp2 acc2 = a2;
    for (int i = 0; i < n; i++) acc2 = rns_fp2_mul(acc2, a2);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[1] = t1 - t0;
    out[5] = acc2.c0.r1[0]; // anti-DCE

    // Fp2 sqr bench
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    acc2 = a2;
    for (int i = 0; i < n; i++) acc2 = rns_fp2_sqr(acc2);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[2] = t1 - t0;
    out[6] = acc2.c0.r1[0];
}

int main() {
    int *d_results;
    cudaMalloc(&d_results, 2 * sizeof(int));
    test_kernel<<<1,1>>>(d_results);
    cudaDeviceSynchronize();
    int results[2];
    cudaMemcpy(results, d_results, 2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    // Benchmark
    printf("\n=== Benchmark: 10000 iterations ===\n");
    int N = 10000;
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

    printf("  Fp  mul: %.0f ns/op\n", (double)h[0]/(ghz*N));
    printf("  Fp2 mul: %.0f ns/op (3 Fp muls)\n", (double)h[1]/(ghz*N));
    printf("  Fp2 sqr: %.0f ns/op (2 Fp muls)\n", (double)h[2]/(ghz*N));
    printf("  GPU clock: %.2f GHz\n", ghz);
    printf("  Compare: sppark Fp mul was 328 ns/op\n");

    return results[1] > 0 ? 1 : 0;
}
