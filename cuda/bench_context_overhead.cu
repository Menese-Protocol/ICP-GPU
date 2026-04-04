// CUDA CONTEXT OVERHEAD BENCHMARK
// Oracle: Measure cold init vs warm context for GPU crypto operations
//
// Three scenarios:
// 1. Cold start: cuInit + cuDeviceGet + cuCtxCreate (first ever GPU call)
// 2. Context switch: cuCtxPushCurrent / cuCtxPopCurrent
// 3. Warm (persistent context): kernel launch on existing context
//
// Also measures: cudaMalloc/Free overhead, cudaMemcpy for typical sizes

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <dlfcn.h>

// Simple kernel for timing
__global__ void kernel_nop() {}

struct Fp { uint64_t v[6]; };
__device__ __constant__ uint64_t FP_P_CTX[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};

#define N_ITERS 100

double now_ms() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

int main() {
    printf("=== CUDA CONTEXT & ALLOCATION OVERHEAD ===\n\n");

    // ==================== 1. First cudaInit (driver lazy init) ====================
    // The very first CUDA call triggers driver initialization
    // We can't easily measure this in the same process after it's done,
    // but we can measure cuCtxCreate on a new context

    // Note: By the time main() runs, cudart may have already initialized.
    // Measure the full pipeline overhead instead.

    printf("--- 1. Context operations ---\n");

    // Measure cudaSetDevice (should be near-free if already set)
    {
        double t0 = now_ms();
        for (int i = 0; i < N_ITERS; i++) {
            cudaSetDevice(0);
        }
        double t1 = now_ms();
        printf("  cudaSetDevice(0):              %.1f μs/call  (n=%d)\n", (t1-t0)/N_ITERS * 1000, N_ITERS);
    }

    // Measure cudaStreamCreate/Destroy
    {
        double t0 = now_ms();
        for (int i = 0; i < N_ITERS; i++) {
            cudaStream_t s;
            cudaStreamCreate(&s);
            cudaStreamDestroy(s);
        }
        double t1 = now_ms();
        printf("  cudaStreamCreate+Destroy:      %.1f μs/call  (n=%d)\n", (t1-t0)/N_ITERS * 1000, N_ITERS);
    }

    // Measure just cudaStreamCreate (keep stream)
    {
        cudaStream_t streams[N_ITERS];
        double t0 = now_ms();
        for (int i = 0; i < N_ITERS; i++) {
            cudaStreamCreate(&streams[i]);
        }
        double t1 = now_ms();
        printf("  cudaStreamCreate only:         %.1f μs/call  (n=%d)\n", (t1-t0)/N_ITERS * 1000, N_ITERS);
        for (int i = 0; i < N_ITERS; i++) cudaStreamDestroy(streams[i]);
    }

    printf("\n--- 2. Memory allocation/free ---\n");

    // Typical IC DKG sizes
    struct AllocTest {
        size_t bytes;
        const char* label;
    };
    AllocTest alloc_tests[] = {
        {48 * 448, "DKG compressed (48B × 448)"},
        {96 * 448, "DKG affine (96B × 448)"},
        {32 * 448, "DKG scalars (32B × 448)"},
        {1024 * 1024, "1 MiB (checkpoint chunk)"},
        {32 * 500, "BLS hashes (500 sigs)"},
    };
    int n_alloc_tests = sizeof(alloc_tests) / sizeof(alloc_tests[0]);

    for (int i = 0; i < n_alloc_tests; i++) {
        void* d_ptr;
        double t0 = now_ms();
        for (int j = 0; j < N_ITERS; j++) {
            cudaMalloc(&d_ptr, alloc_tests[i].bytes);
            cudaFree(d_ptr);
        }
        double t1 = now_ms();
        printf("  cudaMalloc+Free %-25s: %7.1f μs/call  (n=%d)\n",
               alloc_tests[i].label, (t1-t0)/N_ITERS * 1000, N_ITERS);
    }

    // Pre-allocated pool: measure cudaMalloc once then reuse
    printf("\n--- 3. Pre-allocated pool vs on-demand ---\n");
    {
        // On-demand: alloc + memcpy + kernel + memcpy + free
        size_t sz = 96 * 448;
        uint8_t* h_data = (uint8_t*)malloc(sz);
        uint8_t* h_out = (uint8_t*)malloc(sz);
        memset(h_data, 0x42, sz);

        // On-demand
        double t0 = now_ms();
        for (int i = 0; i < N_ITERS; i++) {
            void *d_in, *d_out;
            cudaMalloc(&d_in, sz);
            cudaMalloc(&d_out, sz);
            cudaMemcpy(d_in, h_data, sz, cudaMemcpyHostToDevice);
            kernel_nop<<<1,1>>>();
            cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            cudaFree(d_in);
            cudaFree(d_out);
        }
        double t1 = now_ms();
        double on_demand_us = (t1 - t0) / N_ITERS * 1000;

        // Pre-allocated
        void *d_in_p, *d_out_p;
        cudaMalloc(&d_in_p, sz);
        cudaMalloc(&d_out_p, sz);

        double t2 = now_ms();
        for (int i = 0; i < N_ITERS; i++) {
            cudaMemcpy(d_in_p, h_data, sz, cudaMemcpyHostToDevice);
            kernel_nop<<<1,1>>>();
            cudaMemcpy(h_out, d_out_p, sz, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
        double t3 = now_ms();
        double pre_alloc_us = (t3 - t2) / N_ITERS * 1000;

        printf("  On-demand (alloc+copy+kernel+copy+free): %7.1f μs/call\n", on_demand_us);
        printf("  Pre-allocated (copy+kernel+copy):        %7.1f μs/call\n", pre_alloc_us);
        printf("  Savings:                                 %7.1f μs (%.1fx)\n",
               on_demand_us - pre_alloc_us, on_demand_us / pre_alloc_us);

        cudaFree(d_in_p); cudaFree(d_out_p);
        free(h_data); free(h_out);
    }

    printf("\n--- 4. Pinned vs pageable host memory ---\n");
    {
        size_t sz = 96 * 448;  // DKG affine points
        void* d_buf;
        cudaMalloc(&d_buf, sz);

        // Pageable
        uint8_t* h_page = (uint8_t*)malloc(sz);
        memset(h_page, 0x42, sz);

        double t0 = now_ms();
        for (int i = 0; i < N_ITERS; i++) {
            cudaMemcpy(d_buf, h_page, sz, cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
        double t1 = now_ms();
        double pageable_us = (t1-t0)/N_ITERS * 1000;

        // Pinned
        uint8_t* h_pin;
        cudaMallocHost(&h_pin, sz);
        memset(h_pin, 0x42, sz);

        double t2 = now_ms();
        for (int i = 0; i < N_ITERS; i++) {
            cudaMemcpy(d_buf, h_pin, sz, cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
        double t3 = now_ms();
        double pinned_us = (t3-t2)/N_ITERS * 1000;

        printf("  H2D pageable (42 KB):  %7.1f μs/call\n", pageable_us);
        printf("  H2D pinned   (42 KB):  %7.1f μs/call\n", pinned_us);
        printf("  Speedup:               %.1fx\n", pageable_us / pinned_us);

        free(h_page); cudaFreeHost(h_pin); cudaFree(d_buf);
    }

    printf("\n--- 5. dlopen overhead (simulated) ---\n");
    {
        // The IC uses dlopen to load libgpu_msm.so at runtime
        // Measure dlopen + dlsym cost
        double t0 = now_ms();
        for (int i = 0; i < 10; i++) {
            void* handle = dlopen("libcudart.so", RTLD_LAZY);
            if (handle) {
                dlsym(handle, "cudaDeviceSynchronize");
                dlclose(handle);
            }
        }
        double t1 = now_ms();
        printf("  dlopen+dlsym+dlclose:  %7.1f μs/call  (n=10)\n", (t1-t0)/10 * 1000);
    }

    printf("\n=== SUMMARY ===\n");
    printf("If persistent context saves >100μs per crypto call, it's worth integrating.\n");
    printf("DKG does ~122 dealings/round × cost/dealing = total overhead.\n");

    return 0;
}
