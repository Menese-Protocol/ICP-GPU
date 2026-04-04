// Profile WHERE the 38ms goes in GPU BLS verify
// Break down: kernel launch, miller loop, final exp, identity check

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>

struct Fp{uint64_t v[6];};struct Fp2{Fp c0,c1;};struct Fp6{Fp2 c0,c1,c2;};struct Fp12{Fp6 c0,c1;};
struct G1Affine{Fp x,y;};

// Minimal stubs just to measure launch overhead
__global__ void kernel_empty() {}
__global__ void kernel_trivial(int* out) { *out = 42; }
__global__ void kernel_fp_mul_only(uint64_t* out) {
    // Just do 1000 fp_mul to measure ALU throughput
    uint64_t a[6] = {0x760900000002fffdULL,0xebf4000bc40c0002ULL,0x5f48985753c758baULL,
                     0x77ce585370525745ULL,0x5c071a97a256ec6dULL,0x15f65ec3fa80e493ULL};
    uint64_t P[6] = {0xb9feffffffffaaabULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,
                     0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};
    uint64_t M0 = 0x89f3fffcfffcfffdULL;

    for(int iter=0; iter<1000; iter++) {
        uint64_t t[7]={0};
        for(int i=0;i<6;i++){
            uint64_t carry=0;
            for(int j=0;j<6;j++){
                unsigned __int128 p=(unsigned __int128)a[j]*a[i]+t[j]+carry;
                t[j]=(uint64_t)p; carry=(uint64_t)(p>>64);
            }
            t[6]=carry;
            uint64_t m=t[0]*M0;
            unsigned __int128 rd=(unsigned __int128)m*P[0]+t[0];
            carry=(uint64_t)(rd>>64);
            for(int j=1;j<6;j++){
                rd=(unsigned __int128)m*P[j]+t[j]+carry;
                t[j-1]=(uint64_t)rd; carry=(uint64_t)(rd>>64);
            }
            t[5]=t[6]+carry; t[6]=(t[5]<carry)?1:0;
        }
        for(int i=0;i<6;i++) a[i]=t[i];
    }
    for(int i=0;i<6;i++) out[i]=a[i];
}

int main() {
    printf("=== GPU BLS Overhead Profiling ===\n\n");

    // 1. Bare kernel launch overhead
    cudaEvent_t s,e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // Warmup
    kernel_empty<<<1,1>>>(); cudaDeviceSynchronize();

    cudaEventRecord(s);
    for(int i=0;i<100;i++) kernel_empty<<<1,1>>>();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e);
    printf("1. Kernel launch overhead: %.3fms per launch\n", ms/100);

    // 2. Kernel with trivial work
    int* d_out; cudaMalloc(&d_out, 4);
    kernel_trivial<<<1,1>>>(d_out); cudaDeviceSynchronize();
    cudaEventRecord(s);
    for(int i=0;i<100;i++) kernel_trivial<<<1,1>>>(d_out);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms,s,e);
    printf("2. Trivial kernel: %.3fms per call\n", ms/100);
    cudaFree(d_out);

    // 3. 1000 fp_mul (Montgomery) — measure ALU throughput
    uint64_t* d_fp; cudaMalloc(&d_fp, 48);
    kernel_fp_mul_only<<<1,1>>>(d_fp); cudaDeviceSynchronize();
    cudaEventRecord(s);
    for(int i=0;i<10;i++) kernel_fp_mul_only<<<1,1>>>(d_fp);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms,s,e);
    printf("3. 1000x fp_mul (1 thread): %.3fms per call (%.1fns per fp_mul)\n",
           ms/10, ms/10*1e6/1000);
    cudaFree(d_fp);

    // 4. Estimate: full BLS verify = multi_miller + final_exp
    // Miller loop: ~62 iterations × (2 ell ops per pair × 2 pairs) = ~248 ell ops
    //   Each ell = fp12_mul_by_014 ≈ 6 fp2_mul ≈ 18 fp_mul
    //   Plus fp12_sqr ≈ 6 fp2_mul ≈ 18 fp_mul per iteration
    //   Total miller: ~62 × (248/62 × 18 + 18) = ~62 × (72+18) = ~5580 fp_mul
    // Final exp: ~12 cyc_exp × 62 iterations × fp12_sqr+fp12_mul ≈ ~8000 fp_mul
    // Grand total: ~13000-15000 fp_mul per BLS verify

    float ns_per_fpmul = ms/10*1e6/1000;
    float est_bls_ms = 14000 * ns_per_fpmul / 1e6;
    printf("\n4. Estimated BLS verify from fp_mul count:\n");
    printf("   ~14000 fp_mul × %.1fns = %.1fms\n", ns_per_fpmul, est_bls_ms);
    printf("   Actual measured: ~38ms\n");
    printf("   Ratio: %.1fx (register pressure, memory stalls)\n", 38.0/est_bls_ms);

    // 5. Multiple threads — does it help?
    printf("\n5. fp_mul scaling with thread count:\n");
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    for(int ti=0; ti<8; ti++) {
        int T = thread_counts[ti];
        uint64_t* d_fp2; cudaMalloc(&d_fp2, T*48);
        kernel_fp_mul_only<<<1,T>>>(d_fp2); cudaDeviceSynchronize();
        cudaEventRecord(s);
        kernel_fp_mul_only<<<1,T>>>(d_fp2);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms,s,e);
        printf("   T=%3d: %.3fms total (%.3fms per thread)\n", T, ms, ms);
        cudaFree(d_fp2);
    }

    cudaEventDestroy(s); cudaEventDestroy(e);
    printf("\n=== Done ===\n");
    return 0;
}
