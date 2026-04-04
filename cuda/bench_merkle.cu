// Merkle SHA-256 Benchmark — compute-only timing (data pre-loaded on GPU)
// Oracle-verified bit-exact against Rust sha2 crate

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include "/workspace/ic/rs/crypto/internal/crypto_lib/gpu_crypto/cuda/sha256.cuh"

#define SUB_CHUNK_SIZE 4096
#define CHUNK_SIZE     1048576
#define NUM_LEAVES     256
#define TREE_ROUNDS    8

__global__ void kernel_merkle_leaves(const uint8_t* data, uint8_t* leaf_hashes, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = gid / NUM_LEAVES;
    int leaf  = gid % NUM_LEAVES;
    if (chunk >= N) return;
    sha256(data + (uint64_t)chunk * CHUNK_SIZE + (uint64_t)leaf * SUB_CHUNK_SIZE,
           SUB_CHUNK_SIZE,
           leaf_hashes + ((uint64_t)chunk * NUM_LEAVES + leaf) * 32);
}

__global__ void kernel_merkle_combine(const uint8_t* in_h, uint8_t* out_h, int N, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int half = count / 2;
    int chunk = gid / half;
    int pair  = gid % half;
    if (chunk >= N) return;
    uint8_t combined[64];
    const uint8_t* L = in_h + ((uint64_t)chunk * count + 2*pair) * 32;
    const uint8_t* R = in_h + ((uint64_t)chunk * count + 2*pair + 1) * 32;
    for(int i=0;i<32;i++){combined[i]=L[i];combined[32+i]=R[i];}
    sha256(combined, 64, out_h + ((uint64_t)chunk * half + pair) * 32);
}

int main() {
    static const uint8_t MERKLE_ROOT[32] = {0x4b,0x89,0x4c,0x45,0x82,0xe7,0xb5,0x78,0x8c,0xfe,0x60,0xc8,0x83,0x17,0xd2,0xb4,0xcb,0x2f,0x6c,0xb4,0x9c,0x63,0x7d,0x43,0x5c,0xe6,0x0e,0xa5,0xc8,0x97,0xa9,0x8d};

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("=== Merkle SHA-256 Compute-Only Benchmark ===\n");
    printf("GPU VRAM: %.1f GiB free\n\n", free_mem/1e9);

    uint8_t* h_chunk = (uint8_t*)malloc(CHUNK_SIZE);
    for(size_t i=0;i<CHUNK_SIZE;i+=8){uint64_t v=(uint64_t)i*0x9E3779B97F4A7C15ULL;memcpy(h_chunk+i,&v,8);}

    printf("%-25s %10s %10s %10s %8s\n","Workload","Leaves","Tree","Total","vs 16T");
    printf("%-25s %10s %10s %10s %8s\n","---","(ms)","(ms)","(ms)","---");

    int bench_sizes[] = {10, 50, 100, 500, 1000, 2000};
    for(int bi=0; bi<6; bi++) {
        int M = bench_sizes[bi];
        size_t data_sz = (size_t)M * CHUNK_SIZE;
        size_t buf_sz  = (size_t)M * NUM_LEAVES * 32;
        size_t needed  = data_sz + buf_sz * 2;
        if(needed > free_mem * 0.85) { printf("%-25s SKIP\n",""); continue; }

        // Allocate
        uint8_t *d_data, *d_a, *d_b;
        cudaMalloc(&d_data, data_sz);
        cudaMalloc(&d_a, buf_sz);
        cudaMalloc(&d_b, buf_sz);

        // Upload once
        for(int i=0; i<M; i++)
            cudaMemcpy(d_data + (size_t)i*CHUNK_SIZE, h_chunk, CHUNK_SIZE, cudaMemcpyHostToDevice);

        int total_leaves = M * NUM_LEAVES;

        // Warmup
        kernel_merkle_leaves<<<(total_leaves+255)/256, 256>>>(d_data, d_a, M);
        int cnt = NUM_LEAVES; uint8_t *src=d_a, *dst=d_b;
        while(cnt>1){int h=cnt/2;kernel_merkle_combine<<<(M*h+255)/256,256>>>(src,dst,M,cnt);cnt=h;uint8_t*t=src;src=dst;dst=t;}
        cudaDeviceSynchronize();

        // Verify
        uint8_t h_root[32];
        cudaMemcpy(h_root, src, 32, cudaMemcpyDeviceToHost);
        if(memcmp(h_root, MERKLE_ROOT, 32)!=0){printf("ORACLE MISMATCH at M=%d!\n",M);return 1;}

        // Time Phase 1 (leaves)
        cudaEvent_t s1,e1,s2,e2;
        cudaEventCreate(&s1);cudaEventCreate(&e1);
        cudaEventCreate(&s2);cudaEventCreate(&e2);
        int rounds = (M<=100)?10:3;

        cudaEventRecord(s1);
        for(int r=0;r<rounds;r++)
            kernel_merkle_leaves<<<(total_leaves+255)/256, 256>>>(d_data, d_a, M);
        cudaEventRecord(e1);cudaEventSynchronize(e1);
        float leaves_ms; cudaEventElapsedTime(&leaves_ms,s1,e1); leaves_ms/=rounds;

        // Time Phase 2 (tree)
        cudaEventRecord(s2);
        for(int r=0;r<rounds;r++){
            cnt=NUM_LEAVES; src=d_a; dst=d_b;
            while(cnt>1){int h=cnt/2;kernel_merkle_combine<<<(M*h+255)/256,256>>>(src,dst,M,cnt);cnt=h;uint8_t*t=src;src=dst;dst=t;}
        }
        cudaEventRecord(e2);cudaEventSynchronize(e2);
        float tree_ms; cudaEventElapsedTime(&tree_ms,s2,e2); tree_ms/=rounds;

        float total_ms = leaves_ms + tree_ms;
        double cpu_ms = ((double)M * CHUNK_SIZE / 1e9) / 33.9 * 1000.0;

        char desc[64];
        snprintf(desc,sizeof(desc),"%d x 1MiB (%.1f GiB)", M, (double)M*CHUNK_SIZE/(1024.0*1024.0*1024.0));
        printf("%-25s %10.2f %10.2f %10.2f %7.1fx\n", desc, leaves_ms, tree_ms, total_ms, cpu_ms/total_ms);

        cudaFree(d_data);cudaFree(d_a);cudaFree(d_b);
        cudaEventDestroy(s1);cudaEventDestroy(e1);
        cudaEventDestroy(s2);cudaEventDestroy(e2);
    }

    free(h_chunk);
    printf("\n=== Done ===\n");
    return 0;
}
