#!/bin/bash
# ============================================================================
# IC NODE CPU CONTENTION EXPERIMENT
# ============================================================================
#
# HYPOTHESIS: On a 12-core machine, GPU offloading DKG crypto reduces CPU
# contention, allowing canister execution threads to get more CPU time.
#
# IC MAINNET THREAD ARCHITECTURE (from source code):
#   - IDKG/DKG crypto:    16 rayon threads (MAX_IDKG_THREADS)
#   - Consensus:          16 rayon threads (MAX_CONSENSUS_THREADS)
#   - Canister execution:  4 threads (NUMBER_OF_EXECUTION_THREADS)
#   - Query execution:     4 threads (QUERY_EXECUTION_THREADS_TOTAL)
#   - P2P:                cpu_count/4 tokio workers
#   - HTTP:               separate tokio runtime
#   Total: ~40+ threads competing for physical cores
#
# IC MAINNET PARAMETERS (from rs/limits/src/lib.rs):
#   - DKG_INTERVAL_HEIGHT:         499 blocks
#   - DKG_DEALINGS_PER_BLOCK:      1
#   - MAX_INSTRUCTIONS_PER_ROUND:  4 billion
#   - NUMBER_OF_EXECUTION_THREADS: 4
#   - UNIT_DELAY_APP_SUBNET:       1000ms (1 block/sec)
#   - scheduler_cores:             4
#
# EXPERIMENT DESIGN:
#   Simulate the CPU contention pattern on a 12-core machine (our VPS-3).
#
#   Phase A (CPU-only): Spawn threads matching IC's production config.
#     - 16 threads doing DKG crypto work (fp_mul chains simulating BLS)
#     - 16 threads doing consensus work (moderate CPU)
#     - 4 threads doing "canister execution" (compute-bound, measures throughput)
#     - Measure: instructions/second achieved by the 4 canister threads
#
#   Phase B (GPU-offloaded): Same, but DKG crypto threads are idle (GPU handles it).
#     - 16 DKG threads: SLEEPING (GPU does the work)
#     - 16 consensus threads: same as Phase A
#     - 4 canister threads: same workload
#     - Measure: instructions/second achieved by the 4 canister threads
#
#   Phase C (smaller machine simulation): Restrict to N cores via taskset.
#     - Test at 8, 12, 16, 24, 32 cores
#     - For each: measure canister throughput with and without DKG contention
#
# OUTPUT: Canister throughput improvement from GPU offloading at each core count
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/cpu_contention_bench"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Build it first: cd ${SCRIPT_DIR} && make"
    exit 1
fi

echo "=== IC NODE CPU CONTENTION EXPERIMENT ==="
echo "Host: $(hostname), $(nproc) cores, $(uname -m)"
echo "Date: $(date -u)"
echo ""

# Test at different core counts to simulate different hardware
CORE_COUNTS="4 8 12 16 24 32"
DURATION=10  # seconds per phase

echo "Testing core counts: $CORE_COUNTS"
echo "Duration per phase: ${DURATION}s"
echo ""

printf "%-8s  %14s  %14s  %10s  %s\n" \
    "Cores" "CPU-only" "GPU-offload" "Improve" "Notes"
printf "%-8s  %14s  %14s  %10s  %s\n" \
    "---" "(ops/sec)" "(ops/sec)" "" "---"

MAX_CORES=$(nproc)

for CORES in $CORE_COUNTS; do
    if [ "$CORES" -gt "$MAX_CORES" ]; then
        printf "%-8s  %14s  %14s  %10s  %s\n" \
            "$CORES" "SKIP" "SKIP" "-" "only $MAX_CORES cores available"
        continue
    fi

    # Build CPU mask for taskset (first N cores)
    LAST_CORE=$((CORES - 1))
    CPUMASK="0-${LAST_CORE}"

    # Phase A: CPU-only (DKG threads active)
    OPS_CPU=$(taskset -c "$CPUMASK" "$BINARY" --mode cpu-only --duration "$DURATION" 2>/dev/null)

    # Phase B: GPU-offloaded (DKG threads sleeping)
    OPS_GPU=$(taskset -c "$CPUMASK" "$BINARY" --mode gpu-offload --duration "$DURATION" 2>/dev/null)

    # Calculate improvement
    if [ "$OPS_CPU" -gt 0 ]; then
        IMPROVE=$(echo "scale=1; ($OPS_GPU - $OPS_CPU) * 100 / $OPS_CPU" | bc)
        RATIO=$(echo "scale=2; $OPS_GPU / $OPS_CPU" | bc)
    else
        IMPROVE="N/A"
        RATIO="N/A"
    fi

    NOTES=""
    if [ "$CORES" -eq 12 ]; then
        NOTES="← Our VPS-3 (RTX PRO 6000)"
    elif [ "$CORES" -eq 32 ]; then
        NOTES="← DFINITY Gen2 spec"
    fi

    printf "%-8s  %14s  %14s  %9s%%  %s\n" \
        "$CORES" "$OPS_CPU" "$OPS_GPU" "$IMPROVE" "$NOTES"
done

echo ""
echo "=== INTERPRETATION ==="
echo "If GPU-offload throughput at 12 cores >= CPU-only throughput at 32 cores,"
echo "then a 12-core + GPU node can match a 32-core DFINITY Gen2 node."
echo ""
echo "IC Mainnet Parameters Used:"
echo "  IDKG threads:     16 (MAX_IDKG_THREADS from idkg/src/lib.rs)"
echo "  Consensus threads: 16 (MAX_CONSENSUS_THREADS from consensus.rs)"
echo "  Execution threads: 4 (NUMBER_OF_EXECUTION_THREADS from execution_environment.rs)"
echo "  DKG interval:      499 blocks (DKG_INTERVAL_HEIGHT from limits/src/lib.rs)"
echo "  Block rate:        1/sec (UNIT_DELAY_APP_SUBNET)"
