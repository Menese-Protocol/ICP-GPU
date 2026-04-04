// Detailed BLS verify profiling — time each component
// Uses clock64() inside the kernel to measure miller loop vs final_exp

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

struct Fp{uint64_t v[6];};struct Fp2{Fp c0,c1;};struct Fp6{Fp2 c0,c1,c2;};struct Fp12{Fp6 c0,c1;};
struct G1Affine{Fp x,y;};

__device__ __constant__ uint64_t FP_P[6]={0xb9feffffffffaaabULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};
__device__ __constant__ uint64_t FP_ONE[6]={0x760900000002fffdULL,0xebf4000bc40c0002ULL,0x5f48985753c758baULL,0x77ce585370525745ULL,0x5c071a97a256ec6dULL,0x15f65ec3fa80e493ULL};
#define M0 0x89f3fffcfffcfffdULL
#define BLS_X 0xd201000000010000ULL
#define BLS_X_IS_NEG true
#define NUM_COEFFS 68
#define COEFF_U64S (NUM_COEFFS * 36)

#include "g2_coeffs.h"
#include "pk42_coeffs.h"

// Same proven field ops as bench_bls_verify.cu
__device__ Fp fp_zero(){Fp r={};return r;}
__device__ Fp fp_one(){Fp r;for(int i=0;i<6;i++)r.v[i]=FP_ONE[i];return r;}
__device__ bool fp_is_zero(const Fp&a){uint64_t ac=0;for(int i=0;i<6;i++)ac|=a.v[i];return ac==0;}
__device__ bool fp_eq(const Fp&a,const Fp&b){for(int i=0;i<6;i++)if(a.v[i]!=b.v[i])return false;return true;}
__device__ Fp fp_add(const Fp&a,const Fp&b){Fp r;unsigned __int128 c=0;for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)a.v[i]+b.v[i]+c;r.v[i]=(uint64_t)s;c=s>>64;}Fp t;unsigned __int128 bw=0;for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw;t.v[i]=(uint64_t)d;bw=(d>>127)&1;}return(bw==0)?t:r;}
__device__ Fp fp_sub(const Fp&a,const Fp&b){Fp r;unsigned __int128 bw=0;for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)a.v[i]-b.v[i]-bw;r.v[i]=(uint64_t)d;bw=(d>>127)&1;}if(bw){unsigned __int128 c=0;for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)r.v[i]+FP_P[i]+c;r.v[i]=(uint64_t)s;c=s>>64;}}return r;}
__device__ Fp fp_neg(const Fp&a){if(fp_is_zero(a))return a;return fp_sub(fp_zero(),a);}
__device__ __noinline__ Fp fp_mul(const Fp&a,const Fp&b){uint64_t t[7]={0};for(int i=0;i<6;i++){uint64_t carry=0;for(int j=0;j<6;j++){unsigned __int128 p=(unsigned __int128)a.v[j]*b.v[i]+t[j]+carry;t[j]=(uint64_t)p;carry=(uint64_t)(p>>64);}t[6]=carry;uint64_t m=t[0]*M0;unsigned __int128 rd=(unsigned __int128)m*FP_P[0]+t[0];carry=(uint64_t)(rd>>64);for(int j=1;j<6;j++){rd=(unsigned __int128)m*FP_P[j]+t[j]+carry;t[j-1]=(uint64_t)rd;carry=(uint64_t)(rd>>64);}t[5]=t[6]+carry;t[6]=(t[5]<carry)?1:0;}Fp r;for(int i=0;i<6;i++)r.v[i]=t[i];Fp s;unsigned __int128 bw=0;for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw;s.v[i]=(uint64_t)d;bw=(d>>127)&1;}return(bw==0)?s:r;}
__device__ Fp fp_sqr(const Fp&a){return fp_mul(a,a);}
__device__ Fp2 fp2_add(const Fp2&a,const Fp2&b){return{fp_add(a.c0,b.c0),fp_add(a.c1,b.c1)};}
__device__ Fp2 fp2_sub(const Fp2&a,const Fp2&b){return{fp_sub(a.c0,b.c0),fp_sub(a.c1,b.c1)};}
__device__ Fp2 fp2_neg(const Fp2&a){return{fp_neg(a.c0),fp_neg(a.c1)};}
__device__ __noinline__ Fp2 fp2_mul(const Fp2&a,const Fp2&b){Fp t0=fp_mul(a.c0,b.c0),t1=fp_mul(a.c1,b.c1);return{fp_sub(t0,t1),fp_sub(fp_sub(fp_mul(fp_add(a.c0,a.c1),fp_add(b.c0,b.c1)),t0),t1)};}
__device__ Fp2 fp2_mul_nr(const Fp2&a){return{fp_sub(a.c0,a.c1),fp_add(a.c0,a.c1)};}
__device__ Fp2 fp2_mul_fp(const Fp2&a,const Fp&s){return{fp_mul(a.c0,s),fp_mul(a.c1,s)};}
__device__ Fp2 fp2_conj(const Fp2&a){return{a.c0,fp_neg(a.c1)};}
__device__ Fp2 fp2_zero(){return{fp_zero(),fp_zero()};}
__device__ Fp2 fp2_one(){return{fp_one(),fp_zero()};}
__device__ Fp6 fp6_zero(){return{fp2_zero(),fp2_zero(),fp2_zero()};}
__device__ Fp6 fp6_one(){return{fp2_one(),fp2_zero(),fp2_zero()};}
__device__ Fp6 fp6_add(const Fp6&a,const Fp6&b){return{fp2_add(a.c0,b.c0),fp2_add(a.c1,b.c1),fp2_add(a.c2,b.c2)};}
__device__ Fp6 fp6_sub(const Fp6&a,const Fp6&b){return{fp2_sub(a.c0,b.c0),fp2_sub(a.c1,b.c1),fp2_sub(a.c2,b.c2)};}
__device__ Fp6 fp6_neg(const Fp6&a){return{fp2_neg(a.c0),fp2_neg(a.c1),fp2_neg(a.c2)};}
__device__ __noinline__ Fp6 fp6_mul(const Fp6&a,const Fp6&b){Fp2 a_a=fp2_mul(a.c0,b.c0),b_b=fp2_mul(a.c1,b.c1),c_c=fp2_mul(a.c2,b.c2);return{fp2_add(a_a,fp2_mul_nr(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c1,a.c2),fp2_add(b.c1,b.c2)),b_b),c_c))),fp2_add(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c1),fp2_add(b.c0,b.c1)),a_a),b_b),fp2_mul_nr(c_c)),fp2_add(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c2),fp2_add(b.c0,b.c2)),a_a),c_c),b_b)};}
__device__ Fp6 fp6_mul_by_v(const Fp6&a){return{fp2_mul_nr(a.c2),a.c0,a.c1};}
__device__ __noinline__ Fp12 fp12_mul(const Fp12&a,const Fp12&b){Fp6 aa=fp6_mul(a.c0,b.c0),bb=fp6_mul(a.c1,b.c1);return{fp6_add(aa,fp6_mul_by_v(bb)),fp6_sub(fp6_sub(fp6_mul(fp6_add(a.c0,a.c1),fp6_add(b.c0,b.c1)),aa),bb)};}
__device__ __noinline__ Fp12 fp12_sqr(const Fp12&a){Fp6 ab=fp6_mul(a.c0,a.c1),s1=fp6_add(a.c0,a.c1),s2=fp6_add(a.c0,fp6_mul_by_v(a.c1));return{fp6_sub(fp6_sub(fp6_mul(s2,s1),ab),fp6_mul_by_v(ab)),fp6_add(ab,ab)};}
__device__ Fp12 fp12_conj(const Fp12&a){return{a.c0,fp6_neg(a.c1)};}
__device__ Fp6 fp6_mul_by_01(const Fp6&s,const Fp2&c0,const Fp2&c1){Fp2 a=fp2_mul(s.c0,c0),b=fp2_mul(s.c1,c1);return{fp2_add(fp2_mul_nr(fp2_mul(s.c2,c1)),a),fp2_sub(fp2_sub(fp2_mul(fp2_add(c0,c1),fp2_add(s.c0,s.c1)),a),b),fp2_add(fp2_mul(s.c2,c0),b)};}
__device__ Fp6 fp6_mul_by_1(const Fp6&s,const Fp2&c1){return{fp2_mul_nr(fp2_mul(s.c2,c1)),fp2_mul(s.c0,c1),fp2_mul(s.c1,c1)};}
__device__ Fp12 fp12_mul_by_014(const Fp12&f,const Fp2&c0,const Fp2&c1,const Fp2&c4){Fp6 aa=fp6_mul_by_01(f.c0,c0,c1),bb=fp6_mul_by_1(f.c1,c4);return{fp6_add(fp6_mul_by_v(bb),aa),fp6_sub(fp6_sub(fp6_mul_by_01(fp6_add(f.c1,f.c0),c0,fp2_add(c1,c4)),aa),bb)};}
__device__ void lc(const uint64_t*co,int i,Fp2&c0,Fp2&c1,Fp2&c2){int b=i*36;for(int j=0;j<6;j++){c0.c0.v[j]=co[b+j];c0.c1.v[j]=co[b+6+j];}for(int j=0;j<6;j++){c1.c0.v[j]=co[b+12+j];c1.c1.v[j]=co[b+18+j];}for(int j=0;j<6;j++){c2.c0.v[j]=co[b+24+j];c2.c1.v[j]=co[b+30+j];}}

__device__ Fp2 fp2_inv(const Fp2&a){Fp n=fp_add(fp_sqr(a.c0),fp_sqr(a.c1));
    // fp_inv via exponentiation
    Fp r2=fp_one(),base=n;uint64_t exp[6]={0xb9feffffffffaaa9ULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};for(int w=0;w<6;w++)for(int bit=0;bit<64;bit++){if(w==5&&bit>=61)break;if((exp[w]>>bit)&1)r2=fp_mul(r2,base);base=fp_sqr(base);}
    Fp i=r2;return{fp_mul(a.c0,i),fp_neg(fp_mul(a.c1,i))};}
__device__ __noinline__ Fp6 fp6_inv(const Fp6&f){Fp2 c0s=fp2_mul(f.c0,f.c0),c1s=fp2_mul(f.c1,f.c1),c2s=fp2_mul(f.c2,f.c2);Fp2 c01=fp2_mul(f.c0,f.c1),c02=fp2_mul(f.c0,f.c2),c12=fp2_mul(f.c1,f.c2);Fp2 t0=fp2_sub(c0s,fp2_mul_nr(c12));Fp2 t1=fp2_sub(fp2_mul_nr(c2s),c01);Fp2 t2=fp2_sub(c1s,c02);Fp2 sc=fp2_add(fp2_mul(f.c0,t0),fp2_mul_nr(fp2_add(fp2_mul(f.c2,t1),fp2_mul(f.c1,t2))));Fp2 si=fp2_inv(sc);return{fp2_mul(t0,si),fp2_mul(t1,si),fp2_mul(t2,si)};}
__device__ __noinline__ Fp12 fp12_inv(const Fp12&f){Fp6 t=fp6_sub(fp6_mul(f.c0,f.c0),fp6_mul_by_v(fp6_mul(f.c1,f.c1)));Fp6 ti=fp6_inv(t);return{fp6_mul(f.c0,ti),fp6_neg(fp6_mul(f.c1,ti))};}
__device__ Fp6 fp6_frob(const Fp6&f){Fp2 c0=fp2_conj(f.c0),c1=fp2_conj(f.c1),c2=fp2_conj(f.c2);Fp2 fc1={fp_zero(),{0xcd03c9e48671f071ULL,0x5dab22461fcda5d2ULL,0x587042afd3851b95ULL,0x8eb60ebe01bacb9eULL,0x03f97d6e83d050d2ULL,0x18f0206554638741ULL}};c1=fp2_mul(c1,fc1);Fp2 fc2={{0x890dc9e4867545c3ULL,0x2af322533285a5d5ULL,0x50880866309b7e2cULL,0xa20d1b8c7e881024ULL,0x14e4f04fe2db9068ULL,0x14e56d3f1564853aULL},fp_zero()};c2=fp2_mul(c2,fc2);return{c0,c1,c2};}
__device__ Fp12 fp12_frob(const Fp12&f){Fp6 c0=fp6_frob(f.c0);Fp6 c1=fp6_frob(f.c1);Fp2 co={{0x07089552b319d465ULL,0xc6695f92b50a8313ULL,0x97e83cccd117228fULL,0xa35baecab2dc29eeULL,0x1ce393ea5daace4dULL,0x08f2220fb0fb66ebULL},{0xb2f66aad4ce5d646ULL,0x5842a06bfc497cecULL,0xcf4895d42599d394ULL,0xc11b9cba40a8e8d0ULL,0x2e3813cbe5a0de89ULL,0x110eefda88847fafULL}};c1={fp2_mul(c1.c0,co),fp2_mul(c1.c1,co),fp2_mul(c1.c2,co)};return{c0,c1};}

__device__ __noinline__ Fp12 cyc_exp(const Fp12&f){Fp12 t={fp6_one(),fp6_zero()};bool fd=false;for(int i=63;i>=0;i--){if(fd)t=fp12_sqr(t);bool bit=((BLS_X>>i)&1)==1;if(!fd){fd=bit;if(!bit)continue;}if(bit)t=fp12_mul(t,f);}return fp12_conj(t);}

// Timed BLS verify kernel with clock() instrumentation
__global__ void timed_bls_verify(long long* timings) {
    G1Affine sig, neg_hm;
    sig.x = {{0xd27b1adaea06e32cULL,0x9079033c644ae1d9ULL,0xf154b307a1249c34ULL,0xa365af8b574fe9d6ULL,0x375b89d156410186ULL,0x139b61eeb595cf47ULL}};
    sig.y = {{0xd973686bc9912933ULL,0x40d7e6761b92732fULL,0x6b43adf272a19617ULL,0x31388f4c360d31deULL,0x588138872a0f1626ULL,0x0a0e79be45d84809ULL}};
    neg_hm.x = {{0xbf6f80fad9849c75ULL,0x018298254a48192dULL,0xa8588f9235e2e40dULL,0x5508d390e218ff49ULL,0xf29c6756cc2dd13aULL,0x0d3056fc0db4365fULL}};
    neg_hm.y = {{0x25301550dae86c14ULL,0x3140795267108347ULL,0xc5d7e01597b162a4ULL,0x1c0d85a74c2e54c0ULL,0xf66a4c922bdcc305ULL,0x10d66042e1a3e5acULL}};

    long long t0 = clock64();

    // === MILLER LOOP ===
    Fp12 f={fp6_one(),fp6_zero()};int ci=0;bool found=false;
    for(int b=63;b>=0;b--){
        bool bit=(((BLS_X>>1)>>b)&1)==1;
        if(!found){found=bit;continue;}
        Fp2 c0,c1,c2;
        lc(G2_COEFFS,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,sig.x),fp2_mul_fp(c0,sig.y));
        lc(PK42_COEFFS,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,neg_hm.x),fp2_mul_fp(c0,neg_hm.y));ci++;
        if(bit){lc(G2_COEFFS,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,sig.x),fp2_mul_fp(c0,sig.y));
        lc(PK42_COEFFS,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,neg_hm.x),fp2_mul_fp(c0,neg_hm.y));ci++;}
        f=fp12_sqr(f);
    }
    Fp2 c0,c1,c2;
    lc(G2_COEFFS,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,sig.x),fp2_mul_fp(c0,sig.y));
    lc(PK42_COEFFS,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,neg_hm.x),fp2_mul_fp(c0,neg_hm.y));
    if(BLS_X_IS_NEG)f=fp12_conj(f);

    long long t1 = clock64();

    // === FINAL EXPONENTIATION ===
    // Easy part
    Fp12 t0_f=fp12_conj(f),t1_f=fp12_inv(f),t2_f=fp12_mul(t0_f,t1_f);t1_f=t2_f;
    t2_f=fp12_mul(fp12_frob(fp12_frob(t2_f)),t1_f);f=t2_f;

    long long t2 = clock64();

    // Hard part
    Fp12 tt1=fp12_conj(fp12_sqr(t2_f));
    Fp12 t3=cyc_exp(t2_f),t4=fp12_mul(tt1,t3);tt1=cyc_exp(t4);t4=fp12_conj(t4);f=fp12_mul(f,t4);
    t4=fp12_sqr(t3);Fp12 tt0=cyc_exp(tt1);t3=fp12_mul(t3,tt0);t3=fp12_frob(fp12_frob(t3));f=fp12_mul(f,t3);
    t4=fp12_mul(t4,cyc_exp(tt0));f=fp12_mul(f,cyc_exp(t4));t4=fp12_mul(t4,fp12_conj(t2_f));t2_f=fp12_mul(t2_f,tt1);
    t2_f=fp12_frob(fp12_frob(fp12_frob(t2_f)));f=fp12_mul(f,t2_f);t4=fp12_frob(t4);f=fp12_mul(f,t4);

    long long t3_t = clock64();

    timings[0] = t1 - t0;      // miller loop
    timings[1] = t2 - t1;      // easy part of final exp
    timings[2] = t3_t - t2;    // hard part of final exp
    timings[3] = t3_t - t0;    // total
}

int main() {
    printf("=== BLS Verify Internal Timing ===\n\n");

    // Get GPU clock rate
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int clockKHz; cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, device); double ghz = clockKHz / 1e6;
    printf("GPU: %s, clock: %.2f GHz\n\n", prop.name, ghz);

    long long* d_timings;
    cudaMalloc(&d_timings, 4 * sizeof(long long));

    // Warmup
    timed_bls_verify<<<1,1>>>(d_timings);
    cudaDeviceSynchronize();

    // Actual measurement
    timed_bls_verify<<<1,1>>>(d_timings);
    cudaDeviceSynchronize();

    long long h_timings[4];
    cudaMemcpy(h_timings, d_timings, 4*sizeof(long long), cudaMemcpyDeviceToHost);

    double to_ms = 1.0 / (ghz * 1e6); // cycles to ms
    printf("  Miller loop:     %12lld cycles = %.2fms (%.0f%%)\n",
           h_timings[0], h_timings[0]*to_ms, 100.0*h_timings[0]/h_timings[3]);
    printf("  Final exp easy:  %12lld cycles = %.2fms (%.0f%%)\n",
           h_timings[1], h_timings[1]*to_ms, 100.0*h_timings[1]/h_timings[3]);
    printf("  Final exp hard:  %12lld cycles = %.2fms (%.0f%%)\n",
           h_timings[2], h_timings[2]*to_ms, 100.0*h_timings[2]/h_timings[3]);
    printf("  TOTAL:           %12lld cycles = %.2fms\n",
           h_timings[3], h_timings[3]*to_ms);

    // Count fp_mul operations
    printf("\n  Estimated fp_mul count:\n");
    printf("    Miller: ~5600 fp_mul\n");
    printf("    Final exp easy: ~800 fp_mul (inv + frob + mul)\n");
    printf("    Final exp hard: ~8000 fp_mul (5× cyc_exp × 62 iter)\n");
    printf("    Total: ~14400 fp_mul\n");

    double total_ms = h_timings[3]*to_ms;
    printf("\n  Cycles per fp_mul (from total): %.0f\n", (double)h_timings[3]/14400);
    printf("  ns per fp_mul: %.0f\n", total_ms*1e6/14400);
    printf("  CPU comparison: 862µs / ~14400 fp_mul = ~60ns per fp_mul\n");

    cudaFree(d_timings);
    printf("\n=== Done ===\n");
    return 0;
}
