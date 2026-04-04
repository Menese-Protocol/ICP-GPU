// Full BLS verify test with sppark-based field tower
// Must match oracle exactly

#include "field_sppark.cuh"
#include "../cuda/g2_coeffs.h"
#include "../cuda/pk42_coeffs.h"
#include <cstdio>
#include <chrono>

#define BLS_X 0xd201000000010000ULL
#define BLS_X_IS_NEG true

// Helper to build fp_t from u64 array
__device__ __host__ fp_t fp_from_u64(const uint64_t v[6]) {
    fp_t r; uint32_t* w=(uint32_t*)&r;
    for(int i=0;i<6;i++){w[2*i]=(uint32_t)v[i];w[2*i+1]=(uint32_t)(v[i]>>32);}
    return r;
}

// Load G2 precomputed coefficient
__device__ void lc(const uint64_t*co,int idx,Fp2&c0,Fp2&c1,Fp2&c2){
    int b=idx*36;
    c0.c0=fp_from_u64(co+b); c0.c1=fp_from_u64(co+b+6);
    c1.c0=fp_from_u64(co+b+12); c1.c1=fp_from_u64(co+b+18);
    c2.c0=fp_from_u64(co+b+24); c2.c1=fp_from_u64(co+b+30);
}

// Multi-miller loop (2 pairs)
__device__ __noinline__ Fp12 multi_miller_2(const G1Affine&p1,const uint64_t*q1,const G1Affine&p2,const uint64_t*q2){
    Fp12 f=fp12_one();int ci=0;bool found=false;
    for(int b=63;b>=0;b--){bool bit=(((BLS_X>>1)>>b)&1)==1;if(!found){found=bit;continue;}
    Fp2 c0,c1,c2;lc(q1,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p1.x),fp2_mul_fp(c0,p1.y));
    lc(q2,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p2.x),fp2_mul_fp(c0,p2.y));ci++;
    if(bit){lc(q1,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p1.x),fp2_mul_fp(c0,p1.y));
    lc(q2,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p2.x),fp2_mul_fp(c0,p2.y));ci++;}
    f=fp12_sqr(f);}
    Fp2 c0,c1,c2;lc(q1,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p1.x),fp2_mul_fp(c0,p1.y));
    lc(q2,ci,c0,c1,c2);f=fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p2.x),fp2_mul_fp(c0,p2.y));
    if(BLS_X_IS_NEG)f=fp12_conj(f);return f;}

__device__ __noinline__ Fp12 cyc_exp(const Fp12&f){Fp12 t=fp12_one();bool fd=false;
    for(int i=63;i>=0;i--){if(fd)t=cyclotomic_square(t);bool bit=((BLS_X>>i)&1)==1;
    if(!fd){fd=bit;if(!bit)continue;}if(bit)t=fp12_mul(t,f);}return fp12_conj(t);}

__device__ __noinline__ Fp12 final_exp(const Fp12&fi){
    Fp12 f=fi,t0=fp12_conj(f),t1=fp12_inv(f),t2=fp12_mul(t0,t1);t1=t2;
    t2=fp12_mul(fp12_frob(fp12_frob(t2)),t1);f=t2;t1=fp12_conj(cyclotomic_square(t2));
    Fp12 t3=cyc_exp(t2),t4=fp12_mul(t1,t3);t1=cyc_exp(t4);t4=fp12_conj(t4);f=fp12_mul(f,t4);
    t4=cyclotomic_square(t3);t0=cyc_exp(t1);t3=fp12_mul(t3,t0);t3=fp12_frob(fp12_frob(t3));f=fp12_mul(f,t3);
    t4=fp12_mul(t4,cyc_exp(t0));f=fp12_mul(f,cyc_exp(t4));t4=fp12_mul(t4,fp12_conj(t2));t2=fp12_mul(t2,t1);
    t2=fp12_frob(fp12_frob(fp12_frob(t2)));f=fp12_mul(f,t2);t4=fp12_frob(t4);f=fp12_mul(f,t4);return f;}

__device__ bool bls_verify(const G1Affine&sig,const G1Affine&nhm,const uint64_t*g2c,const uint64_t*pkc){
    Fp12 ml=multi_miller_2(sig,g2c,nhm,pkc);Fp12 r=final_exp(ml);
    uint64_t v[6]; fp_to_u64(r.c0.c0.c0, v);
    uint64_t one[6]={0x760900000002fffdULL,0xebf4000bc40c0002ULL,0x5f48985753c758baULL,
                     0x77ce585370525745ULL,0x5c071a97a256ec6dULL,0x15f65ec3fa80e493ULL};
    for(int i=0;i<6;i++) if(v[i]!=one[i]) return false;
    // Check rest is zero
    fp_t* fps=(fp_t*)&r;
    for(int i=1;i<12;i++) if(!fps[i].is_zero()) return false;
    return true;
}

__global__ void test() {
    uint64_t sx[6]={0xd27b1adaea06e32cULL,0x9079033c644ae1d9ULL,0xf154b307a1249c34ULL,0xa365af8b574fe9d6ULL,0x375b89d156410186ULL,0x139b61eeb595cf47ULL};
    uint64_t sy[6]={0xd973686bc9912933ULL,0x40d7e6761b92732fULL,0x6b43adf272a19617ULL,0x31388f4c360d31deULL,0x588138872a0f1626ULL,0x0a0e79be45d84809ULL};
    uint64_t nx[6]={0xbf6f80fad9849c75ULL,0x018298254a48192dULL,0xa8588f9235e2e40dULL,0x5508d390e218ff49ULL,0xf29c6756cc2dd13aULL,0x0d3056fc0db4365fULL};
    uint64_t ny[6]={0x25301550dae86c14ULL,0x3140795267108347ULL,0xc5d7e01597b162a4ULL,0x1c0d85a74c2e54c0ULL,0xf66a4c922bdcc305ULL,0x10d66042e1a3e5acULL};
    G1Affine sig={fp_from_u64(sx),fp_from_u64(sy)};
    G1Affine nhm={fp_from_u64(nx),fp_from_u64(ny)};
    bool valid = bls_verify(sig, nhm, G2_COEFFS, PK42_COEFFS);
    printf("BLS verify (sppark): %s\n", valid ? "VALID ✓" : "INVALID ✗");
}

__global__ void bench_kernel(const G1Affine*sigs,const G1Affine*nhms,const uint64_t*g2c,const uint64_t*pkc,bool*res,int M){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=M)return;
    res[idx]=bls_verify(sigs[idx],nhms[idx],g2c,pkc+(uint64_t)idx*2448);}

int main(){
    printf("=== sppark Full BLS Verify ===\n");
    test<<<1,1>>>();cudaDeviceSynchronize();
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){printf("CUDA ERROR: %s\n",cudaGetErrorString(err));return 1;}

    // Benchmark
    printf("\n=== Batch Benchmark ===\n");
    uint64_t sx[6]={0xd27b1adaea06e32cULL,0x9079033c644ae1d9ULL,0xf154b307a1249c34ULL,0xa365af8b574fe9d6ULL,0x375b89d156410186ULL,0x139b61eeb595cf47ULL};
    uint64_t sy[6]={0xd973686bc9912933ULL,0x40d7e6761b92732fULL,0x6b43adf272a19617ULL,0x31388f4c360d31deULL,0x588138872a0f1626ULL,0x0a0e79be45d84809ULL};
    uint64_t nx[6]={0xbf6f80fad9849c75ULL,0x018298254a48192dULL,0xa8588f9235e2e40dULL,0x5508d390e218ff49ULL,0xf29c6756cc2dd13aULL,0x0d3056fc0db4365fULL};
    uint64_t ny[6]={0x25301550dae86c14ULL,0x3140795267108347ULL,0xc5d7e01597b162a4ULL,0x1c0d85a74c2e54c0ULL,0xf66a4c922bdcc305ULL,0x10d66042e1a3e5acULL};

    int sizes[]={1,40,100,140,200,500};
    for(int bi=0;bi<6;bi++){
        int M=sizes[bi];
        G1Affine h_sig,h_nhm;
        h_sig.x=fp_from_u64(sx);h_sig.y=fp_from_u64(sy);
        h_nhm.x=fp_from_u64(nx);h_nhm.y=fp_from_u64(ny);

        G1Affine*dS,*dN;bool*dR;uint64_t*dG,*dP;
        cudaMalloc(&dS,M*sizeof(G1Affine));cudaMalloc(&dN,M*sizeof(G1Affine));
        cudaMalloc(&dR,M);cudaMalloc(&dG,2448*8);cudaMalloc(&dP,(uint64_t)M*2448*8);

        G1Affine*hS=new G1Affine[M],*hN=new G1Affine[M];
        for(int i=0;i<M;i++){hS[i]=h_sig;hN[i]=h_nhm;}
        uint64_t h_pk[2448];uint64_t*pk_src;cudaGetSymbolAddress((void**)&pk_src,PK42_COEFFS);
        cudaMemcpy(h_pk,pk_src,2448*8,cudaMemcpyDeviceToHost);
        uint64_t*hP=new uint64_t[(uint64_t)M*2448];
        for(int i=0;i<M;i++)memcpy(hP+(uint64_t)i*2448,h_pk,2448*8);

        cudaMemcpy(dS,hS,M*sizeof(G1Affine),cudaMemcpyHostToDevice);
        cudaMemcpy(dN,hN,M*sizeof(G1Affine),cudaMemcpyHostToDevice);
        uint64_t*g2s;cudaGetSymbolAddress((void**)&g2s,G2_COEFFS);
        cudaMemcpy(dG,g2s,2448*8,cudaMemcpyDeviceToDevice);
        cudaMemcpy(dP,hP,(uint64_t)M*2448*8,cudaMemcpyHostToDevice);

        int thr=32,blk=(M+thr-1)/thr;
        bench_kernel<<<blk,thr>>>(dS,dN,dG,dP,dR,M);cudaDeviceSynchronize();
        int rounds=(M<=40)?5:2;
        auto start=std::chrono::high_resolution_clock::now();
        for(int r=0;r<rounds;r++)bench_kernel<<<blk,thr>>>(dS,dN,dG,dP,dR,M);
        cudaDeviceSynchronize();
        double ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-start).count()/rounds;
        double cpu=M*0.348;
        printf("M=%-4d  %7.1fms  per=%.2fms  CPU=%.1fms  %.1fx\n",M,ms,ms/M,cpu,cpu/ms);
        delete[]hS;delete[]hN;delete[]hP;
        cudaFree(dS);cudaFree(dN);cudaFree(dR);cudaFree(dG);cudaFree(dP);
    }
    printf("\n=== Done ===\n");return 0;
}
