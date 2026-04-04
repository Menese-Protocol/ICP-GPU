// GPU Merkle SHA-256 — Oracle-verified implementation
//
// Spec:
//   1 MiB chunk → 256 × 4 KiB leaves → SHA-256 each → Merkle tree → 32-byte root
//   Phase 1: 256 threads hash 4KiB sub-chunks in parallel (per chunk)
//   Phase 2: 8 rounds of tree combining: 128→64→32→16→8→4→2→1
//
// Oracle test vectors (from Rust sha2 crate):
//   LEAF_0:      9ae9f134694b752cde42498d6a717c39c4301aaa0254a266d1152c76bf882608
//   LEAF_255:    b26c580fcc526bf870704310129a51b49b8fb610f14cb670c8b4d9df93da5677
//   MERKLE_ROOT: 4b894c4582e7b5788cfe60c88317d2b4cb2f6cb49c637d435ce60ea5c897a98d

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include "/workspace/ic/rs/crypto/internal/crypto_lib/gpu_crypto/cuda/sha256.cuh"

#define SUB_CHUNK_SIZE 4096
#define CHUNK_SIZE     1048576
#define NUM_LEAVES     256  // CHUNK_SIZE / SUB_CHUNK_SIZE
#define TREE_ROUNDS    8    // log2(256)

// Phase 1: Hash leaves — 256 threads per chunk
// Each thread: SHA-256 of one 4KiB sub-chunk
__global__ void kernel_merkle_leaves(
    const uint8_t* data,    // N × 1MiB chunks
    uint8_t* leaf_hashes,   // N × 256 × 32 bytes
    int N
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_idx = global_id / NUM_LEAVES;
    int leaf_idx  = global_id % NUM_LEAVES;
    if (chunk_idx >= N) return;

    const uint8_t* src = data + (uint64_t)chunk_idx * CHUNK_SIZE + (uint64_t)leaf_idx * SUB_CHUNK_SIZE;
    uint8_t* dst = leaf_hashes + ((uint64_t)chunk_idx * NUM_LEAVES + leaf_idx) * 32;
    sha256(src, SUB_CHUNK_SIZE, dst);
}

// Phase 2: Merkle combine — each thread combines one pair
// Input: count hashes per chunk → output: count/2 hashes per chunk
// SHA-256(left_32_bytes || right_32_bytes) → 32 bytes
__global__ void kernel_merkle_combine(
    const uint8_t* in_hashes,   // N × count × 32
    uint8_t* out_hashes,        // N × (count/2) × 32
    int N,
    int count                   // number of input hashes per chunk (must be even)
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int half = count / 2;
    int chunk_idx = global_id / half;
    int pair_idx  = global_id % half;
    if (chunk_idx >= N) return;

    const uint8_t* left  = in_hashes + ((uint64_t)chunk_idx * count + 2 * pair_idx) * 32;
    const uint8_t* right = in_hashes + ((uint64_t)chunk_idx * count + 2 * pair_idx + 1) * 32;
    uint8_t combined[64];
    for (int i = 0; i < 32; i++) {
        combined[i]      = left[i];
        combined[32 + i] = right[i];
    }
    uint8_t* dst = out_hashes + ((uint64_t)chunk_idx * half + pair_idx) * 32;
    sha256(combined, 64, dst);
}

// Host function: compute Merkle root for N chunks
// Returns root hashes in h_roots (N × 32 bytes, host memory)
void gpu_merkle_hash(
    const uint8_t* h_data,  // host: N × 1MiB
    uint8_t* h_roots,       // host: N × 32 bytes output
    int N
) {
    size_t data_size = (size_t)N * CHUNK_SIZE;
    size_t leaves_size = (size_t)N * NUM_LEAVES * 32;

    // Allocate GPU memory
    uint8_t *d_data, *d_buf_a, *d_buf_b;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_buf_a, leaves_size);  // holds current level
    cudaMalloc(&d_buf_b, leaves_size);  // holds next level

    // Upload data
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

    // Phase 1: leaf hashing (256 threads per chunk)
    int total_leaves = N * NUM_LEAVES;
    int threads = 256;
    int blocks = (total_leaves + threads - 1) / threads;
    kernel_merkle_leaves<<<blocks, threads>>>(d_data, d_buf_a, N);

    // Phase 2: tree combining (8 rounds)
    int count = NUM_LEAVES; // 256
    uint8_t *src = d_buf_a, *dst = d_buf_b;
    while (count > 1) {
        int half = count / 2;
        int total_pairs = N * half;
        int blk = (total_pairs + 255) / 256;
        kernel_merkle_combine<<<blk, 256>>>(src, dst, N, count);
        count = half;
        // Swap
        uint8_t* tmp = src; src = dst; dst = tmp;
    }

    // Download roots (1 hash per chunk, currently in src buffer)
    cudaMemcpy(h_roots, src, N * 32, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_buf_a);
    cudaFree(d_buf_b);
}

// ==================== Test + Benchmark ====================

static void fill_deterministic(uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; i += 8) {
        uint64_t val = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
        size_t remaining = len - i;
        if (remaining >= 8) memcpy(data + i, &val, 8);
        else memcpy(data + i, &val, remaining);
    }
}

int main() {
    printf("=== GPU Merkle SHA-256 — Oracle Test ===\n\n");

    // Oracle test vectors
    static const uint8_t LEAF_0[32] = {0x9a,0xe9,0xf1,0x34,0x69,0x4b,0x75,0x2c,0xde,0x42,0x49,0x8d,0x6a,0x71,0x7c,0x39,0xc4,0x30,0x1a,0xaa,0x02,0x54,0xa2,0x66,0xd1,0x15,0x2c,0x76,0xbf,0x88,0x26,0x08};
    static const uint8_t LEAF_255[32] = {0xb2,0x6c,0x58,0x0f,0xcc,0x52,0x6b,0xf8,0x70,0x70,0x43,0x10,0x12,0x9a,0x51,0xb4,0x9b,0x8f,0xb6,0x10,0xf1,0x4c,0xb6,0x70,0xc8,0xb4,0xd9,0xdf,0x93,0xda,0x56,0x77};
    static const uint8_t MERKLE_ROOT[32] = {0x4b,0x89,0x4c,0x45,0x82,0xe7,0xb5,0x78,0x8c,0xfe,0x60,0xc8,0x83,0x17,0xd2,0xb4,0xcb,0x2f,0x6c,0xb4,0x9c,0x63,0x7d,0x43,0x5c,0xe6,0x0e,0xa5,0xc8,0x97,0xa9,0x8d};

    // Generate test data
    uint8_t* h_data = (uint8_t*)malloc(CHUNK_SIZE);
    fill_deterministic(h_data, CHUNK_SIZE);

    // ===== Test 1: Verify leaf hashes =====
    printf("Test 1: Leaf hashes (256 × SHA-256 of 4KiB)\n");
    {
        uint8_t *d_data, *d_leaves;
        cudaMalloc(&d_data, CHUNK_SIZE);
        cudaMalloc(&d_leaves, NUM_LEAVES * 32);
        cudaMemcpy(d_data, h_data, CHUNK_SIZE, cudaMemcpyHostToDevice);

        kernel_merkle_leaves<<<1, 256>>>(d_data, d_leaves, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { printf("  CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }

        uint8_t h_leaves[NUM_LEAVES * 32];
        cudaMemcpy(h_leaves, d_leaves, NUM_LEAVES * 32, cudaMemcpyDeviceToHost);

        bool leaf0_ok  = (memcmp(h_leaves, LEAF_0, 32) == 0);
        bool leaf255_ok = (memcmp(h_leaves + 255 * 32, LEAF_255, 32) == 0);

        printf("  leaf[0]:   "); for(int j=0;j<32;j++) printf("%02x",h_leaves[j]); printf(" %s\n", leaf0_ok?"PASS":"FAIL");
        printf("  leaf[255]: "); for(int j=0;j<32;j++) printf("%02x",h_leaves[255*32+j]); printf(" %s\n", leaf255_ok?"PASS":"FAIL");

        if (!leaf0_ok || !leaf255_ok) { printf("  LEAF TEST FAILED — stopping.\n"); return 1; }

        cudaFree(d_data); cudaFree(d_leaves);
    }

    // ===== Test 2: Full Merkle root =====
    printf("\nTest 2: Full Merkle root (256 leaves → 8 rounds → root)\n");
    {
        uint8_t h_root[32];
        gpu_merkle_hash(h_data, h_root, 1);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { printf("  CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }

        bool root_ok = (memcmp(h_root, MERKLE_ROOT, 32) == 0);
        printf("  GPU root:    "); for(int j=0;j<32;j++) printf("%02x",h_root[j]); printf("\n");
        printf("  Oracle root: "); for(int j=0;j<32;j++) printf("%02x",MERKLE_ROOT[j]); printf("\n");
        printf("  Match: %s\n", root_ok ? "PASS" : "FAIL");

        if (!root_ok) { printf("  ROOT TEST FAILED — stopping.\n"); return 1; }
    }

    // ===== Test 3: Batch (3 identical chunks) =====
    printf("\nTest 3: Batch (3 identical chunks)\n");
    {
        int N = 3;
        uint8_t* h_batch = (uint8_t*)malloc(N * CHUNK_SIZE);
        for (int i = 0; i < N; i++) memcpy(h_batch + i * CHUNK_SIZE, h_data, CHUNK_SIZE);

        uint8_t h_roots[3 * 32];
        gpu_merkle_hash(h_batch, h_roots, N);

        bool all_ok = true;
        for (int i = 0; i < N; i++) {
            bool ok = (memcmp(h_roots + i * 32, MERKLE_ROOT, 32) == 0);
            if (!ok) all_ok = false;
            printf("  chunk[%d]: %s\n", i, ok ? "PASS" : "FAIL");
        }
        if (!all_ok) { printf("  BATCH TEST FAILED — stopping.\n"); free(h_batch); return 1; }
        free(h_batch);
    }

    printf("\n=== ALL ORACLE TESTS PASS ===\n");

    // ===== Benchmark =====
    printf("\n=== Benchmark ===\n");
    printf("%-30s  %10s  %10s  %8s\n", "Workload", "GPU(ms)", "GB/s(raw)", "vs 16T");

    int bench_sizes[] = {10, 50, 100, 500, 1000, 2000};
    for (int bi = 0; bi < 6; bi++) {
        int M = bench_sizes[bi];
        size_t total_bytes = (size_t)M * CHUNK_SIZE;

        // Check available memory
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t needed = total_bytes + (size_t)M * NUM_LEAVES * 32 * 2; // data + 2 buffers
        if (needed > free_mem * 0.9) {
            printf("%-30s  SKIP (need %.1f GiB, have %.1f GiB)\n",
                   "", needed/1e9, free_mem/1e9);
            continue;
        }

        uint8_t* h_batch = (uint8_t*)malloc(total_bytes);
        if (!h_batch) continue;
        for (int i = 0; i < M; i++) memcpy(h_batch + (size_t)i * CHUNK_SIZE, h_data, CHUNK_SIZE);
        uint8_t* h_roots = (uint8_t*)malloc(M * 32);

        // Warmup
        gpu_merkle_hash(h_batch, h_roots, M);

        // Timed
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        int rounds = (M <= 100) ? 10 : 3;

        cudaEventRecord(s);
        for (int r = 0; r < rounds; r++)
            gpu_merkle_hash(h_batch, h_roots, M);
        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float tot_ms; cudaEventElapsedTime(&tot_ms, s, e);
        float gpu_ms = tot_ms / rounds;
        double gbps = (total_bytes / 1e9) / (gpu_ms / 1e3);
        // CPU 16-thread flat SHA-256: 33.9 GB/s
        double cpu_ms = (total_bytes / 1e9) / 33.9 * 1000.0;

        char desc[64];
        snprintf(desc, sizeof(desc), "%d x 1MiB (%.1f GiB)", M, total_bytes / (1024.0*1024.0*1024.0));
        printf("%-30s  %10.2f  %10.1f  %7.1fx\n", desc, gpu_ms, gbps, cpu_ms / gpu_ms);

        // Verify correctness of last run
        bool ok = (memcmp(h_roots, MERKLE_ROOT, 32) == 0);
        if (!ok) printf("  WARNING: root[0] mismatch after benchmark!\n");

        free(h_batch); free(h_roots);
        cudaEventDestroy(s); cudaEventDestroy(e);
    }

    free(h_data);
    printf("\n=== Done ===\n");
    return 0;
}
