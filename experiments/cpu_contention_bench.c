/*
 * IC NODE CPU CONTENTION BENCHMARK
 *
 * Simulates the IC replica thread architecture on a constrained-core machine.
 * Measures canister execution throughput under different DKG crypto load scenarios.
 *
 * IC MAINNET THREAD ARCHITECTURE (extracted from dfinity/ic source):
 *   consensus.rs:85           MAX_CONSENSUS_THREADS = 16
 *   idkg/src/lib.rs:238       MAX_IDKG_THREADS = 16
 *   execution_environment.rs:80  NUMBER_OF_EXECUTION_THREADS = 4
 *   execution_environment.rs:127 QUERY_EXECUTION_THREADS_TOTAL = 4
 *   replica/main.rs:95        P2P workers = max(cpu_count/4, 2)
 *
 * SIMULATION:
 *   Mode "cpu-only":    16 DKG threads (busy) + 16 consensus + 4 execution + 4 query
 *   Mode "gpu-offload": 16 DKG threads (idle) + 16 consensus + 4 execution + 4 query
 *
 * DKG crypto workload: modular exponentiation chain (simulates BLS12-381 fp_mul)
 * Consensus workload:  moderate hash-like computation (lighter than DKG)
 * Execution workload:  instruction counter (measures achieved throughput)
 *
 * OUTPUT: total operations completed by 4 execution threads (stdout, integer)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>

/* ======================== Configuration ======================== */

/* IC mainnet thread counts (from source) */
#define IDKG_THREADS      16   /* MAX_IDKG_THREADS */
#define CONSENSUS_THREADS 16   /* MAX_CONSENSUS_THREADS */
#define EXEC_THREADS       4   /* NUMBER_OF_EXECUTION_THREADS */
#define QUERY_THREADS      4   /* QUERY_EXECUTION_THREADS_TOTAL */

/* Total threads = 40 (matches real IC node) */
#define TOTAL_THREADS (IDKG_THREADS + CONSENSUS_THREADS + EXEC_THREADS + QUERY_THREADS)

/* Global control */
static atomic_int g_running = 1;
static atomic_uint_fast64_t g_exec_ops = 0;  /* canister execution throughput */
static atomic_uint_fast64_t g_query_ops = 0; /* query execution throughput */

/* ======================== Workloads ======================== */

/*
 * DKG crypto workload: simulates BLS12-381 field operations.
 * Real DKG does point decompression (fp_sqrt = 381-bit exponentiation)
 * and MSM (scalar-point multiply chains).
 * We simulate with 64-bit modular multiply chains.
 */
static void dkg_crypto_work(void) {
    /* Simulated modular multiplication (simplified BLS12-381 Fp) */
    volatile uint64_t a = 0x760900000002fffdULL;
    volatile uint64_t b = 0xebf4000bc40c0002ULL;
    volatile uint64_t p = 0xb9feffffffffaaabULL;

    for (int i = 0; i < 10000; i++) {
        /* Chain of modular multiplies (simulates fp_mul) */
        unsigned __int128 prod = (unsigned __int128)a * b;
        a = (uint64_t)(prod % p);
        b = a ^ 0x5f48985753c758baULL;
    }
}

/*
 * Consensus workload: lighter than DKG.
 * Real consensus does block validation, notarization checks, artifact management.
 * More I/O and hash operations, less heavy crypto.
 */
static void consensus_work(void) {
    volatile uint64_t state = 0x6a09e667bb67ae85ULL;
    for (int i = 0; i < 2000; i++) {
        /* Simulated hash-like mixing (SHA-256 style) */
        state = (state >> 17) | (state << 47);
        state ^= state >> 23;
        state *= 0x9E3779B97F4A7C15ULL;
    }
}

/*
 * Canister execution workload: compute-bound instruction execution.
 * Real execution runs WASM instructions via Wasmtime.
 * We measure how many "instruction batches" complete per second.
 * Each batch represents MAX_INSTRUCTIONS_PER_SLICE = 2 billion / slices.
 */
static void execution_work(atomic_uint_fast64_t *counter) {
    /* Simulate WASM execution: arithmetic + memory access pattern */
    volatile uint64_t acc = 0;
    volatile uint64_t mem[64];
    memset((void*)mem, 0x42, sizeof(mem));

    for (int i = 0; i < 5000; i++) {
        /* Mix of arithmetic and memory ops (WASM-like) */
        acc += mem[i & 63];
        mem[(i + 17) & 63] = acc ^ (acc >> 3);
        acc = (acc * 6364136223846793005ULL) + 1442695040888963407ULL;
    }
    atomic_fetch_add(counter, 1);
}

/* ======================== Thread functions ======================== */

static void* thread_idkg(void *arg) {
    int idle = *(int*)arg;  /* 1 = GPU offloaded (sleep), 0 = CPU active */
    while (atomic_load(&g_running)) {
        if (idle) {
            /* GPU handles crypto — thread is essentially idle */
            usleep(10000);  /* 10ms sleep, simulating GPU-offloaded */
        } else {
            /* CPU doing DKG crypto verification */
            dkg_crypto_work();
        }
    }
    return NULL;
}

static void* thread_consensus(void *arg) {
    (void)arg;
    while (atomic_load(&g_running)) {
        consensus_work();
    }
    return NULL;
}

static void* thread_execution(void *arg) {
    (void)arg;
    while (atomic_load(&g_running)) {
        execution_work(&g_exec_ops);
    }
    return NULL;
}

static void* thread_query(void *arg) {
    (void)arg;
    while (atomic_load(&g_running)) {
        execution_work(&g_query_ops);
    }
    return NULL;
}

/* ======================== Main ======================== */

int main(int argc, char *argv[]) {
    int duration = 10;
    int gpu_offload = 0;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (strcmp(argv[i+1], "gpu-offload") == 0) gpu_offload = 1;
            i++;
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            duration = atoi(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    if (verbose) {
        fprintf(stderr, "=== IC CPU Contention Bench ===\n");
        fprintf(stderr, "Mode: %s\n", gpu_offload ? "GPU-offloaded" : "CPU-only");
        fprintf(stderr, "Duration: %ds\n", duration);
        fprintf(stderr, "Threads: %d IDKG(%s) + %d consensus + %d exec + %d query = %d total\n",
                IDKG_THREADS, gpu_offload ? "IDLE" : "BUSY",
                CONSENSUS_THREADS, EXEC_THREADS, QUERY_THREADS, TOTAL_THREADS);
    }

    pthread_t threads[TOTAL_THREADS];
    int tidx = 0;

    /* Spawn IDKG threads */
    int idkg_idle = gpu_offload;
    for (int i = 0; i < IDKG_THREADS; i++) {
        pthread_create(&threads[tidx++], NULL, thread_idkg, &idkg_idle);
    }

    /* Spawn consensus threads */
    for (int i = 0; i < CONSENSUS_THREADS; i++) {
        pthread_create(&threads[tidx++], NULL, thread_consensus, NULL);
    }

    /* Spawn execution threads */
    for (int i = 0; i < EXEC_THREADS; i++) {
        pthread_create(&threads[tidx++], NULL, thread_execution, NULL);
    }

    /* Spawn query threads */
    for (int i = 0; i < QUERY_THREADS; i++) {
        pthread_create(&threads[tidx++], NULL, thread_query, NULL);
    }

    /* Run for specified duration */
    sleep(duration);
    atomic_store(&g_running, 0);

    /* Join all threads */
    for (int i = 0; i < TOTAL_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    uint64_t exec_total = atomic_load(&g_exec_ops);
    uint64_t query_total = atomic_load(&g_query_ops);

    if (verbose) {
        fprintf(stderr, "Execution ops: %lu (%.0f ops/sec)\n",
                exec_total, (double)exec_total / duration);
        fprintf(stderr, "Query ops:     %lu (%.0f ops/sec)\n",
                query_total, (double)query_total / duration);
        fprintf(stderr, "Combined:      %lu (%.0f ops/sec)\n",
                exec_total + query_total, (double)(exec_total + query_total) / duration);
    }

    /* Output just the exec ops/sec for script consumption */
    printf("%lu\n", exec_total / duration);

    return 0;
}
