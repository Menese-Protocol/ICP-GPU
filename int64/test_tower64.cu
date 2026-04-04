// Oracle test: 64-bit tower Fp6+Fp12+pairing building blocks
#include <cstdio>
#include <cstdint>
#include "tower64.cuh"

__global__ void test_kernel(int* results) {
    if (threadIdx.x != 0) return;
    int pass=0, fail=0;
    Fp one=fp_one(), two=fp_add(one,one), three=fp_add(two,one), five=fp_add(three,two);
    Fp seven=fp_zero(); for(int i=0;i<7;i++) seven=fp_add(seven,one);
    Fp2 f2a={two,three}, f2b={five,one}, f2c={seven,two};
    Fp6 f6a={f2a,f2b,f2c}, f6b={f2c,f2a,f2b};

    // Fp6
    { bool ok=fp_eq(fp6_mul(fp6_one(),fp6_one()).c0.c0, fp6_one().c0.c0); printf("Fp6  1*1=1:     %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }
    { Fp6 a=f6a,b=f6b; bool ok=fp_eq(fp6_mul(a,b).c0.c0, fp6_mul(b,a).c0.c0); printf("Fp6  a*b=b*a:   %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }
    { Fp6 a=f6a,b=f6b,c={f2b,f2c,f2a};
      Fp6 l=fp6_mul(fp6_mul(a,b),c), r=fp6_mul(a,fp6_mul(b,c));
      bool ok=fp_eq(l.c0.c0,r.c0.c0)&&fp_eq(l.c1.c0,r.c1.c0); printf("Fp6  assoc:     %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }

    // Fp12
    { Fp12 o=fp12_one(); bool ok=fp_eq(fp12_mul(o,o).c0.c0.c0, o.c0.c0.c0); printf("Fp12 1*1=1:     %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }
    { Fp12 a={f6a,f6b},b={f6b,f6a}; bool ok=fp_eq(fp12_mul(a,b).c0.c0.c0, fp12_mul(b,a).c0.c0.c0); printf("Fp12 a*b=b*a:   %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }
    { Fp12 a={f6a,f6b}; Fp12 s=fp12_sqr(a), m=fp12_mul(a,a);
      bool ok=fp_eq(s.c0.c0.c0,m.c0.c0.c0)&&fp_eq(s.c1.c0.c0,m.c1.c0.c0); printf("Fp12 sqr=mul:   %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }
    { Fp12 o=fp12_one(); bool ok=fp_eq(cyclotomic_square(o).c0.c0.c0, o.c0.c0.c0); printf("Fp12 cyc(1)=1:  %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }

    // Frobenius
    { bool ok=fp_eq(fp12_frob(fp12_one()).c0.c0.c0, fp12_one().c0.c0.c0); printf("frob(1)=1:      %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }
    { Fp12 a={f6a,f6b}; Fp12 f=a; for(int i=0;i<6;i++) f=fp12_frob(f);
      bool ok=fp_eq(f.c0.c0.c0, fp12_conj(a).c0.c0.c0)&&fp_eq(f.c1.c0.c0, fp12_conj(a).c1.c0.c0);
      printf("frob^6=conj:    %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }

    // Fp12 inv
    { Fp12 a={f6a,f6b}; Fp12 ai=fp12_inv(a); Fp12 r=fp12_mul(a,ai);
      bool ok=fp_eq(r.c0.c0.c0, fp12_one().c0.c0.c0); printf("Fp12 inv:       %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }

    // final_exp(ONE)=ONE
    { Fp12 o=fp12_one(); Fp12 r=final_exp(o);
      bool ok=fp_eq(r.c0.c0.c0, o.c0.c0.c0); printf("final_exp(1)=1: %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }

    // cyc_exp(ONE)=ONE
    { Fp12 o=fp12_one(); Fp12 r=cyc_exp(o);
      bool ok=fp_eq(r.c0.c0.c0, o.c0.c0.c0); printf("cyc_exp(1)=1:   %s\n",ok?"PASS":"FAIL"); if(ok)pass++;else fail++; }

    printf("\n=== RESULTS: %d PASS, %d FAIL ===\n", pass, fail);
    results[0]=pass; results[1]=fail;
}

__global__ void bench(uint64_t* out, int n) {
    if (threadIdx.x != 0) return;
    Fp one=fp_one(), two=fp_add(one,one), three=fp_add(two,one), five=fp_add(three,two);
    Fp seven=fp_zero(); for(int i=0;i<7;i++) seven=fp_add(seven,one);
    Fp2 f2a={two,three}, f2b={five,one}, f2c={seven,two};
    Fp6 f6a={f2a,f2b,f2c}, f6b={f2c,f2a,f2b};
    Fp12 a12={f6a,f6b};
    unsigned long long t0,t1;

    asm volatile("mov.u64 %0, %%clock64;":"=l"(t0));
    Fp12 acc=a12; for(int i=0;i<n;i++) acc=fp12_mul(acc,a12);
    asm volatile("mov.u64 %0, %%clock64;":"=l"(t1));
    out[0]=t1-t0; out[4]=acc.c0.c0.c0.v[0];

    asm volatile("mov.u64 %0, %%clock64;":"=l"(t0));
    acc=a12; for(int i=0;i<n;i++) acc=fp12_sqr(acc);
    asm volatile("mov.u64 %0, %%clock64;":"=l"(t1));
    out[1]=t1-t0;

    asm volatile("mov.u64 %0, %%clock64;":"=l"(t0));
    acc=a12; for(int i=0;i<n;i++) acc=cyclotomic_square(acc);
    asm volatile("mov.u64 %0, %%clock64;":"=l"(t1));
    out[2]=t1-t0;

    asm volatile("mov.u64 %0, %%clock64;":"=l"(t0));
    Fp6 a6=f6a; for(int i=0;i<n;i++) a6=fp6_mul(a6,f6a);
    asm volatile("mov.u64 %0, %%clock64;":"=l"(t1));
    out[3]=t1-t0; out[5]=a6.c0.c0.v[0];
}

int main() {
    int *d; cudaMalloc(&d,2*sizeof(int));
    test_kernel<<<1,1>>>(d); cudaDeviceSynchronize();
    int r[2]; cudaMemcpy(r,d,2*sizeof(int),cudaMemcpyDeviceToHost);

    printf("\n=== Benchmark: 1000 iters ===\n");
    int N=1000;
    uint64_t *do_; cudaMalloc(&do_,8*sizeof(uint64_t));
    bench<<<1,1>>>(do_,N); cudaDeviceSynchronize();
    uint64_t h[8]; cudaMemcpy(h,do_,8*sizeof(uint64_t),cudaMemcpyDeviceToHost);
    int dev; cudaGetDevice(&dev); int clk; cudaDeviceGetAttribute(&clk,cudaDevAttrClockRate,dev);
    double ghz=clk/1e6;
    printf("  Fp6  mul:     %7.0f ns  (sppark: 6000ns)\n", h[3]/(ghz*N));
    printf("  Fp12 mul:     %7.0f ns  (sppark: 18000ns)\n", h[0]/(ghz*N));
    printf("  Fp12 sqr:     %7.0f ns  (sppark: ~12000ns)\n", h[1]/(ghz*N));
    printf("  Fp12 cyc_sqr: %7.0f ns  (sppark: ~8000ns)\n", h[2]/(ghz*N));

    return r[1]>0?1:0;
}
