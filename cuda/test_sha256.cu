// GPU SHA-256 Batch Hashing
// Each thread hashes one independent chunk
// Test against NIST vectors, then benchmark

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>

// SHA-256 constants
__device__ __constant__ uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __constant__ uint32_t H_INIT[8] = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ uint32_t sig0(uint32_t x) { return rotr(x,2) ^ rotr(x,13) ^ rotr(x,22); }
__device__ __forceinline__ uint32_t sig1(uint32_t x) { return rotr(x,6) ^ rotr(x,11) ^ rotr(x,25); }
__device__ __forceinline__ uint32_t ssig0(uint32_t x) { return rotr(x,7) ^ rotr(x,18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t ssig1(uint32_t x) { return rotr(x,17) ^ rotr(x,19) ^ (x >> 10); }

// SHA-256 hash of a single message (up to 55 bytes for single-block)
// For IC state chunks (typically 1MB), we'd process multiple blocks per thread
__device__ void sha256_single_block(const uint8_t* msg, uint32_t len, uint8_t* hash) {
    // Pad message into single 64-byte block (len <= 55)
    uint8_t block[64];
    memset(block, 0, 64);
    memcpy(block, msg, len);
    block[len] = 0x80;
    // Length in bits, big-endian, at end
    uint64_t bitlen = (uint64_t)len * 8;
    block[56] = (bitlen >> 56) & 0xff;
    block[57] = (bitlen >> 48) & 0xff;
    block[58] = (bitlen >> 40) & 0xff;
    block[59] = (bitlen >> 32) & 0xff;
    block[60] = (bitlen >> 24) & 0xff;
    block[61] = (bitlen >> 16) & 0xff;
    block[62] = (bitlen >> 8) & 0xff;
    block[63] = bitlen & 0xff;

    // Parse block into 16 words
    uint32_t W[64];
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16) |
                ((uint32_t)block[i*4+2] << 8) | block[i*4+3];
    }
    for (int i = 16; i < 64; i++) {
        W[i] = ssig1(W[i-2]) + W[i-7] + ssig0(W[i-15]) + W[i-16];
    }

    uint32_t a=H_INIT[0],b=H_INIT[1],c=H_INIT[2],d=H_INIT[3];
    uint32_t e=H_INIT[4],f=H_INIT[5],g=H_INIT[6],h=H_INIT[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sig1(e) + ch(e,f,g) + K[i] + W[i];
        uint32_t t2 = sig0(a) + maj(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }

    a+=H_INIT[0]; b+=H_INIT[1]; c+=H_INIT[2]; d+=H_INIT[3];
    e+=H_INIT[4]; f+=H_INIT[5]; g+=H_INIT[6]; h+=H_INIT[7];

    // Output big-endian
    uint32_t out[8] = {a,b,c,d,e,f,g,h};
    for (int i = 0; i < 8; i++) {
        hash[i*4]   = (out[i] >> 24) & 0xff;
        hash[i*4+1] = (out[i] >> 16) & 0xff;
        hash[i*4+2] = (out[i] >> 8) & 0xff;
        hash[i*4+3] = out[i] & 0xff;
    }
}

// Multi-block SHA-256 for arbitrary length messages
__device__ void sha256(const uint8_t* msg, uint64_t len, uint8_t* hash) {
    uint32_t state[8];
    for (int i = 0; i < 8; i++) state[i] = H_INIT[i];

    uint64_t processed = 0;
    // Process full 64-byte blocks
    while (processed + 64 <= len) {
        uint32_t W[64];
        for (int i = 0; i < 16; i++) {
            int off = processed + i * 4;
            W[i] = ((uint32_t)msg[off] << 24) | ((uint32_t)msg[off+1] << 16) |
                    ((uint32_t)msg[off+2] << 8) | msg[off+3];
        }
        for (int i = 16; i < 64; i++)
            W[i] = ssig1(W[i-2]) + W[i-7] + ssig0(W[i-15]) + W[i-16];

        uint32_t a=state[0],b=state[1],c=state[2],d=state[3];
        uint32_t e=state[4],f=state[5],g=state[6],h=state[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h + sig1(e) + ch(e,f,g) + K[i] + W[i];
            uint32_t t2 = sig0(a) + maj(a,b,c);
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
        state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
        processed += 64;
    }

    // Final block(s) with padding
    uint8_t block[128]; // At most 2 blocks for padding
    memset(block, 0, 128);
    uint64_t remaining = len - processed;
    memcpy(block, msg + processed, remaining);
    block[remaining] = 0x80;

    int blocks = (remaining < 56) ? 1 : 2;
    uint64_t bitlen = len * 8;
    int last = blocks * 64 - 8;
    block[last]   = (bitlen >> 56) & 0xff;
    block[last+1] = (bitlen >> 48) & 0xff;
    block[last+2] = (bitlen >> 40) & 0xff;
    block[last+3] = (bitlen >> 32) & 0xff;
    block[last+4] = (bitlen >> 24) & 0xff;
    block[last+5] = (bitlen >> 16) & 0xff;
    block[last+6] = (bitlen >> 8) & 0xff;
    block[last+7] = bitlen & 0xff;

    for (int blk = 0; blk < blocks; blk++) {
        uint32_t W[64];
        for (int i = 0; i < 16; i++) {
            int off = blk * 64 + i * 4;
            W[i] = ((uint32_t)block[off] << 24) | ((uint32_t)block[off+1] << 16) |
                    ((uint32_t)block[off+2] << 8) | block[off+3];
        }
        for (int i = 16; i < 64; i++)
            W[i] = ssig1(W[i-2]) + W[i-7] + ssig0(W[i-15]) + W[i-16];

        uint32_t a=state[0],b=state[1],c=state[2],d=state[3];
        uint32_t e=state[4],f=state[5],g=state[6],h=state[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h + sig1(e) + ch(e,f,g) + K[i] + W[i];
            uint32_t t2 = sig0(a) + maj(a,b,c);
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
        state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
    }

    for (int i = 0; i < 8; i++) {
        hash[i*4]   = (state[i] >> 24) & 0xff;
        hash[i*4+1] = (state[i] >> 16) & 0xff;
        hash[i*4+2] = (state[i] >> 8) & 0xff;
        hash[i*4+3] = state[i] & 0xff;
    }
}

// ==================== Test vectors ====================
__global__ void test_sha256() {
    uint8_t hash[32];

    // NIST test vector 1: SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    uint8_t msg1[] = "abc";
    sha256(msg1, 3, hash);
    printf("TEST 1 SHA256(\"abc\"):\n  ");
    for (int i = 0; i < 32; i++) printf("%02x", hash[i]);
    bool ok1 = (hash[0]==0xba && hash[1]==0x78 && hash[31]==0xad);
    printf("\n  %s (expect: ba7816bf...)\n", ok1 ? "PASS" : "FAIL");

    // NIST test vector 2: SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    sha256((uint8_t*)"", 0, hash);
    printf("TEST 2 SHA256(\"\"):\n  ");
    for (int i = 0; i < 32; i++) printf("%02x", hash[i]);
    bool ok2 = (hash[0]==0xe3 && hash[1]==0xb0 && hash[31]==0x55);
    printf("\n  %s (expect: e3b0c442...)\n", ok2 ? "PASS" : "FAIL");

    // Test vector 3: 56 bytes (two-block boundary)
    uint8_t msg3[56];
    memset(msg3, 'a', 56);
    sha256(msg3, 56, hash);
    printf("TEST 3 SHA256(56x'a'):\n  ");
    for (int i = 0; i < 32; i++) printf("%02x", hash[i]);
    // Expected: b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a
    bool ok3 = (hash[0]==0xb3 && hash[1]==0x54);
    printf("\n  %s (expect: b35439a4...)\n", ok3 ? "PASS" : "FAIL");

    // Test vector 4: 64 bytes (exactly one full block + padding block)
    uint8_t msg4[64];
    memset(msg4, 'b', 64);
    sha256(msg4, 64, hash);
    printf("TEST 4 SHA256(64x'b'):\n  ");
    for (int i = 0; i < 32; i++) printf("%02x", hash[i]);
    // Expected: 23a4faa5310a85a2ad07a1d50d0e3e3f76f7f4f835a5e8fb4aaee2dd2f076950
    bool ok4 = (hash[0]==0x23 && hash[1]==0xa4);
    printf("\n  %s (expect: 23a4faa5...)\n", ok4 ? "PASS" : "FAIL");

    // Test vector 5: 1000 bytes
    uint8_t msg5[1000];
    memset(msg5, 'c', 1000);
    sha256(msg5, 1000, hash);
    printf("TEST 5 SHA256(1000x'c'):\n  ");
    for (int i = 0; i < 32; i++) printf("%02x", hash[i]);
    printf("\n  (no reference — just checking it runs)\n");
}

// ==================== Batch kernel ====================
// Each thread hashes one chunk of `chunk_size` bytes
__global__ void batch_sha256(const uint8_t* chunks, uint8_t* hashes,
                              int N, int chunk_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    sha256(chunks + (uint64_t)idx * chunk_size, chunk_size, hashes + idx * 32);
}

// ==================== Benchmark ====================
int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  GPU SHA-256 Batch Hashing Benchmark                   ║\n");
    printf("║  For IC state manifest chunk hashing                   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    test_sha256<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }

    printf("\n=== Batch SHA-256 Benchmark ===\n");
    printf("IC state chunks are typically 1MB each\n");
    printf("Checkpoint can have 1000-100000 chunks\n\n");

    // Test with different chunk sizes and batch counts
    int chunk_sizes[] = {1024, 4096, 65536, 1048576}; // 1KB, 4KB, 64KB, 1MB
    const char* size_names[] = {"1KB", "4KB", "64KB", "1MB"};
    int batch_counts[] = {100, 1000, 10000, 50000};

    for (int si = 0; si < 4; si++) {
        int cs = chunk_sizes[si];
        printf("--- Chunk size: %s ---\n", size_names[si]);

        for (int bi = 0; bi < 4; bi++) {
            int N = batch_counts[bi];
            uint64_t total_bytes = (uint64_t)N * cs;

            // Skip if too much memory (>8GB)
            if (total_bytes > 8ULL * 1024 * 1024 * 1024) {
                printf("  n=%-6d  SKIPPED (%.1fGB > 8GB limit)\n", N, total_bytes / 1e9);
                continue;
            }

            uint8_t *d_chunks, *d_hashes;
            cudaMalloc(&d_chunks, total_bytes);
            cudaMalloc(&d_hashes, (uint64_t)N * 32);
            cudaMemset(d_chunks, 0x42, total_bytes); // Fill with dummy data

            int threads = 256;
            int blocks = (N + threads - 1) / threads;

            // Warm up
            batch_sha256<<<blocks, threads>>>(d_chunks, d_hashes, N, cs);
            cudaDeviceSynchronize();

            // Benchmark
            int rounds = (N <= 1000) ? 10 : 3;
            auto start = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < rounds; r++) {
                batch_sha256<<<blocks, threads>>>(d_chunks, d_hashes, N, cs);
            }
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count() / rounds;
            double gbps = (total_bytes / 1e9) / (ms / 1e3);
            double hashes_per_sec = N / (ms / 1e3);

            printf("  n=%-6d  %8.1fms  %.1f Mhash/s  %.2f GB/s\n",
                   N, ms, hashes_per_sec / 1e6, gbps);

            cudaFree(d_chunks);
            cudaFree(d_hashes);
        }
        printf("\n");
    }

    printf("=== Done ===\n");
    return 0;
}
