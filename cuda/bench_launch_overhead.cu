// ORACLE STEP 1: Measure exact GPU kernel launch overhead
// Tests: empty kernel, Fp multiply, BLS verify, decompress, MSM
// Each measured with cudaEvent timing at microsecond resolution
//
// This is our "before" baseline for CUDA Graphs optimization

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

// ==================== Minimal kernels for overhead measurement ====================

// Test 1: Empty kernel — pure launch overhead
__global__ void kernel_empty() {}

// Test 2: Trivial work — single uint64 add
__global__ void kernel_trivial(uint64_t* out) {
    out[0] = out[0] + 1;
}

// Test 3: Fp multiply (representative crypto work unit)
struct Fp { uint64_t v[6]; };
__device__ __constant__ uint64_t FP_P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};
#define M0 0x89f3fffcfffcfffdULL

__device__ Fp fp_mul_dev(const Fp& a, const Fp& b) {
    uint64_t t[7] = {0};
    for (int i = 0; i < 6; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 6; j++) {
            unsigned __int128 p = (unsigned __int128)a.v[j] * b.v[i] + t[j] + carry;
            t[j] = (uint64_t)p; carry = (uint64_t)(p >> 64);
        }
        t[6] = carry;
        uint64_t m = t[0] * M0;
        unsigned __int128 rd = (unsigned __int128)m * FP_P[0] + t[0];
        carry = (uint64_t)(rd >> 64);
        for (int j = 1; j < 6; j++) {
            rd = (unsigned __int128)m * FP_P[j] + t[j] + carry;
            t[j-1] = (uint64_t)rd; carry = (uint64_t)(rd >> 64);
        }
        t[5] = t[6] + carry; t[6] = (t[5] < carry) ? 1 : 0;
    }
    Fp r; for (int i = 0; i < 6; i++) r.v[i] = t[i];
    Fp s; unsigned __int128 bw = 0;
    for (int i = 0; i < 6; i++) {
        unsigned __int128 d = (unsigned __int128)r.v[i] - FP_P[i] - bw;
        s.v[i] = (uint64_t)d; bw = (d >> 127) & 1;
    }
    return (bw == 0) ? s : r;
}

__global__ void kernel_fp_mul(Fp* out, const Fp* a, const Fp* b) {
    out[0] = fp_mul_dev(a[0], b[0]);
}

// Test 4: Batch Fp multiply (N threads, simulates batch crypto)
__global__ void kernel_fp_mul_batch(Fp* out, const Fp* a, const Fp* b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = fp_mul_dev(a[tid], b[tid]);
    }
}

// ==================== Measurement helpers ====================

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

struct TimingResult {
    float min_us, max_us, avg_us, median_us;
};

// Sort helper for median
void sort_floats(float* arr, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (arr[j] < arr[i]) { float t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
}

// ==================== Main benchmark ====================

void bench_kernel(const char* name, int warmup, int iters,
                  void (*launch_fn)(cudaStream_t), cudaStream_t stream) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        launch_fn(stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Measure each launch individually
    float* times = new float[iters];
    for (int i = 0; i < iters; i++) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        launch_fn(stream);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&times[i], start, stop));
        times[i] *= 1000.0f; // ms → μs
    }

    // Statistics
    sort_floats(times, iters);
    float sum = 0;
    for (int i = 0; i < iters; i++) sum += times[i];

    printf("  %-30s  min=%8.1f μs  median=%8.1f μs  avg=%8.1f μs  max=%8.1f μs  (n=%d)\n",
           name, times[0], times[iters/2], sum/iters, times[iters-1], iters);

    delete[] times;
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ==================== CUDA Graph measurement ====================

void bench_graph(const char* name, int warmup, int iters,
                 void (*launch_fn)(cudaStream_t), cudaStream_t stream) {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Capture the kernel launch into a graph
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_fn(stream);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Measure
    float* times = new float[iters];
    for (int i = 0; i < iters; i++) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&times[i], start, stop));
        times[i] *= 1000.0f;
    }

    sort_floats(times, iters);
    float sum = 0;
    for (int i = 0; i < iters; i++) sum += times[i];

    printf("  %-30s  min=%8.1f μs  median=%8.1f μs  avg=%8.1f μs  max=%8.1f μs  (n=%d) [GRAPH]\n",
           name, times[0], times[iters/2], sum/iters, times[iters-1], iters);

    delete[] times;
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
}

// ==================== Kernel launchers (for function pointers) ====================

static uint64_t* d_trivial;
static Fp *d_fp_a, *d_fp_b, *d_fp_out;
static Fp *d_fp_batch_a, *d_fp_batch_b, *d_fp_batch_out;
static int batch_n;

void launch_empty(cudaStream_t s) { kernel_empty<<<1,1,0,s>>>(); }
void launch_trivial(cudaStream_t s) { kernel_trivial<<<1,1,0,s>>>(d_trivial); }
void launch_fp_mul(cudaStream_t s) { kernel_fp_mul<<<1,1,0,s>>>(d_fp_out, d_fp_a, d_fp_b); }
void launch_fp_mul_batch(cudaStream_t s) {
    int threads = 256;
    int blocks = (batch_n + threads - 1) / threads;
    kernel_fp_mul_batch<<<blocks, threads, 0, s>>>(d_fp_batch_out, d_fp_batch_a, d_fp_batch_b, batch_n);
}

int main() {
    printf("=== GPU KERNEL LAUNCH OVERHEAD BENCHMARK ===\n");
    printf("GPU: RTX PRO 6000 Blackwell (sm_120)\n\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // ---- Allocate ----
    CHECK_CUDA(cudaMalloc(&d_trivial, sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_fp_a, sizeof(Fp)));
    CHECK_CUDA(cudaMalloc(&d_fp_b, sizeof(Fp)));
    CHECK_CUDA(cudaMalloc(&d_fp_out, sizeof(Fp)));

    // Init Fp values (just need non-zero)
    Fp one = {{0x760900000002fffdULL, 0xebf4000bc40c0002ULL,
               0x5f48985753c758baULL, 0x77ce585370525745ULL,
               0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL}};
    CHECK_CUDA(cudaMemcpy(d_fp_a, &one, sizeof(Fp), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_fp_b, &one, sizeof(Fp), cudaMemcpyHostToDevice));

    // Batch Fp
    int batch_sizes[] = {1, 16, 64, 256, 448, 1024};
    int num_batch_sizes = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    int warmup = 50;
    int iters = 200;

    // ==================== REGULAR LAUNCHES ====================
    printf("--- Regular kernel launches (no graph) ---\n");
    bench_kernel("empty kernel", warmup, iters, launch_empty, stream);
    bench_kernel("trivial (1 add)", warmup, iters, launch_trivial, stream);
    bench_kernel("fp_mul (1 thread)", warmup, iters, launch_fp_mul, stream);

    for (int bi = 0; bi < num_batch_sizes; bi++) {
        batch_n = batch_sizes[bi];
        CHECK_CUDA(cudaMalloc(&d_fp_batch_a, batch_n * sizeof(Fp)));
        CHECK_CUDA(cudaMalloc(&d_fp_batch_b, batch_n * sizeof(Fp)));
        CHECK_CUDA(cudaMalloc(&d_fp_batch_out, batch_n * sizeof(Fp)));
        // Fill with ones
        for (int i = 0; i < batch_n; i++) {
            CHECK_CUDA(cudaMemcpy(d_fp_batch_a + i, &one, sizeof(Fp), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_fp_batch_b + i, &one, sizeof(Fp), cudaMemcpyHostToDevice));
        }

        char label[64];
        snprintf(label, sizeof(label), "fp_mul_batch (N=%d)", batch_n);
        bench_kernel(label, warmup, iters, launch_fp_mul_batch, stream);

        CHECK_CUDA(cudaFree(d_fp_batch_a));
        CHECK_CUDA(cudaFree(d_fp_batch_b));
        CHECK_CUDA(cudaFree(d_fp_batch_out));
    }

    // ==================== CUDA GRAPH LAUNCHES ====================
    printf("\n--- CUDA Graph launches ---\n");
    bench_graph("empty kernel", warmup, iters, launch_empty, stream);
    bench_graph("trivial (1 add)", warmup, iters, launch_trivial, stream);
    bench_graph("fp_mul (1 thread)", warmup, iters, launch_fp_mul, stream);

    for (int bi = 0; bi < num_batch_sizes; bi++) {
        batch_n = batch_sizes[bi];
        CHECK_CUDA(cudaMalloc(&d_fp_batch_a, batch_n * sizeof(Fp)));
        CHECK_CUDA(cudaMalloc(&d_fp_batch_b, batch_n * sizeof(Fp)));
        CHECK_CUDA(cudaMalloc(&d_fp_batch_out, batch_n * sizeof(Fp)));
        for (int i = 0; i < batch_n; i++) {
            CHECK_CUDA(cudaMemcpy(d_fp_batch_a + i, &one, sizeof(Fp), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_fp_batch_b + i, &one, sizeof(Fp), cudaMemcpyHostToDevice));
        }

        char label[64];
        snprintf(label, sizeof(label), "fp_mul_batch (N=%d)", batch_n);
        bench_graph(label, warmup, iters, launch_fp_mul_batch, stream);

        CHECK_CUDA(cudaFree(d_fp_batch_a));
        CHECK_CUDA(cudaFree(d_fp_batch_b));
        CHECK_CUDA(cudaFree(d_fp_batch_out));
    }

    // ==================== SPEEDUP SUMMARY ====================
    printf("\n--- Comparison summary (run above numbers through analysis) ---\n");
    printf("Compare median regular vs median graph for each kernel.\n");
    printf("Graph overhead reduction = (regular - graph) / regular * 100%%\n");

    // Cleanup
    CHECK_CUDA(cudaFree(d_trivial));
    CHECK_CUDA(cudaFree(d_fp_a));
    CHECK_CUDA(cudaFree(d_fp_b));
    CHECK_CUDA(cudaFree(d_fp_out));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
