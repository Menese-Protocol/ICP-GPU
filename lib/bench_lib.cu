// Quick benchmark of optimized library
#include "pairing.cuh"
#include "../cuda/g2_coeffs.h"
#include "../cuda/pk42_coeffs.h"
#include <cstdio>
#include <chrono>

__global__ void bench_verify(const G1Affine* sigs, const G1Affine* nhms,
                              const uint64_t* g2c, const uint64_t* pkc,
                              bool* res, int M) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=M) return;
    res[idx] = bls_verify(sigs[idx], nhms[idx], g2c, pkc+(uint64_t)idx*G2_COEFF_U64S);
}

int main() {
    printf("=== Optimized Library Benchmark ===\n\n");

    G1Affine sig={{0xd27b1adaea06e32cULL,0x9079033c644ae1d9ULL,0xf154b307a1249c34ULL,0xa365af8b574fe9d6ULL,0x375b89d156410186ULL,0x139b61eeb595cf47ULL}},
             nhm={{0xbf6f80fad9849c75ULL,0x018298254a48192dULL,0xa8588f9235e2e40dULL,0x5508d390e218ff49ULL,0xf29c6756cc2dd13aULL,0x0d3056fc0db4365fULL}};
    sig.y={{0xd973686bc9912933ULL,0x40d7e6761b92732fULL,0x6b43adf272a19617ULL,0x31388f4c360d31deULL,0x588138872a0f1626ULL,0x0a0e79be45d84809ULL}};
    nhm.y={{0x25301550dae86c14ULL,0x3140795267108347ULL,0xc5d7e01597b162a4ULL,0x1c0d85a74c2e54c0ULL,0xf66a4c922bdcc305ULL,0x10d66042e1a3e5acULL}};

    int sizes[]={1,40,100,140,200,500};
    printf("%-6s  %-10s  %-10s  %-10s  %-8s\n","M","GPU","per verify","CPU seq","Speedup");

    for(int bi=0;bi<6;bi++){
        int M=sizes[bi];
        G1Affine*dS,*dN;bool*dR;uint64_t*dG,*dP;
        cudaMalloc(&dS,M*sizeof(G1Affine));cudaMalloc(&dN,M*sizeof(G1Affine));
        cudaMalloc(&dR,M*sizeof(bool));cudaMalloc(&dG,G2_COEFF_U64S*8);
        cudaMalloc(&dP,(uint64_t)M*G2_COEFF_U64S*8);

        G1Affine*hS=new G1Affine[M],*hN=new G1Affine[M];
        uint64_t*hP=new uint64_t[(uint64_t)M*G2_COEFF_U64S];
        for(int i=0;i<M;i++){hS[i]=sig;hN[i]=nhm;}
        uint64_t h_pk[G2_COEFF_U64S];
        uint64_t*pk_src;cudaGetSymbolAddress((void**)&pk_src,PK42_COEFFS);
        cudaMemcpy(h_pk,pk_src,G2_COEFF_U64S*8,cudaMemcpyDeviceToHost);
        for(int i=0;i<M;i++)memcpy(hP+(uint64_t)i*G2_COEFF_U64S,h_pk,G2_COEFF_U64S*8);

        cudaMemcpy(dS,hS,M*sizeof(G1Affine),cudaMemcpyHostToDevice);
        cudaMemcpy(dN,hN,M*sizeof(G1Affine),cudaMemcpyHostToDevice);
        uint64_t*g2_src;cudaGetSymbolAddress((void**)&g2_src,G2_COEFFS);
        cudaMemcpy(dG,g2_src,G2_COEFF_U64S*8,cudaMemcpyDeviceToDevice);
        cudaMemcpy(dP,hP,(uint64_t)M*G2_COEFF_U64S*8,cudaMemcpyHostToDevice);

        int thr=32,blk=(M+thr-1)/thr;
        bench_verify<<<blk,thr>>>(dS,dN,dG,dP,dR,M);cudaDeviceSynchronize();

        int rounds=(M<=40)?5:2;
        auto start=std::chrono::high_resolution_clock::now();
        for(int r=0;r<rounds;r++)bench_verify<<<blk,thr>>>(dS,dN,dG,dP,dR,M);
        cudaDeviceSynchronize();
        double ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-start).count()/rounds;
        double cpu=M*0.348;

        printf("M=%-4d  %7.1fms  %8.2fms  %7.1fms    %.1fx\n",M,ms,ms/M,cpu,cpu/ms);
        delete[]hS;delete[]hN;delete[]hP;
        cudaFree(dS);cudaFree(dN);cudaFree(dR);cudaFree(dG);cudaFree(dP);
    }
    printf("\n=== Done ===\n");
    return 0;
}
