# GPU-Accelerated Internet Computer Consensus

**Hive Mind Research Program — Q2 2026**

This research was conducted by Mercatura Forum to evaluate the feasibility and performance characteristics of GPU-accelerated consensus operations on the Internet Computer protocol. The work explores hardware configurations that could support sovereign financial subnets and government-facing infrastructure in the Egyptian context; where cost efficiency; computational sovereignty; and institutional-grade security are requirements that must be addressed at the hardware level.

The scope is deliberately limited. This is not a production system. It is an instrumented prototype that measures what GPU acceleration changes in the consensus pipeline and where the assumptions underlying those measurements break down.

---

## Background

The Internet Computer uses BLS12-381 threshold cryptography for consensus certification and non-interactive distributed key generation (NI-DKG). These are well-established cryptographic primitives with computational profiles that map naturally to GPU architectures; specifically; multi-scalar multiplication; modular exponentiation; and elliptic curve point decompression.

This research asks whether adding a GPU to an IC node meaningfully changes the performance profile of consensus operations; and if so; whether that change has practical implications for the cost and capability of sovereign subnet configurations.

---

## What Was Built

A modified IC replica binary with GPU-accelerated cryptographic operations; compiled within the IC's Bazel build system; tested on a real multi-node subnet running actual consensus with deployed canisters.

### GPU Cryptographic Operations

**Multi-Scalar Multiplication (MSM)** on BLS12-381 G1 using Pippenger's algorithm. The implementation uses Supranational's sppark library compiled for NVIDIA Blackwell architecture (compute capability 12.0). All results are verified against the IC's own `ic_bls12_381` library at the field element level.

**G1 Point Decompression** in batch; converting compressed 48-byte BLS12-381 points to affine coordinates using GPU-parallel Fp square root computation. The decompression kernel computes `y = (x³ + 4)^((p+1)/4)` independently per point.

**BLS Signature Batch Verification** using GPU-parallel multi-Miller loop and final exponentiation. Each verification checks `e(sig, G2) · e(-H(m), pk) = 1` on the GPU.

**SHA-256 Batch Hashing** at 278 GB/s throughput on GPU.

### Integration Architecture

The GPU operations are packaged as a shared library (`libgpu_msm.so`) loaded at runtime via `dlopen`. This isolates the C++ runtime required by sppark's host-side thread management from the IC replica's Zig-based hermetic toolchain. The replica binary links the pairing and SHA-256 kernels statically; the MSM and decompression kernels dynamically. If the shared library is absent; the replica falls back to CPU without error.

The integration points within the IC codebase are:

- `verify_chunking` — GPU MSM replaces `muln_affine_vartime` in the NIZK chunking proof verification within DKG
- `FsEncryptionCiphertext::deserialize` — GPU batch decompression replaces sequential `G1Affine::deserialize` calls during ciphertext processing
- `verify_bls_signature` — GPU batch verification available for certification share validation

---

## Hardware Configuration

The following configuration was evaluated as a potential sovereign subnet node.

| Component | Specification | Estimated Cost |
|---|---|---|
| CPU | AMD EPYC 16-core; SEV-SNP | €800 |
| Memory | 256 GB DDR5 | €700 |
| Storage | 4 TB NVMe | €400 |
| GPU | NVIDIA L4 24 GB (Ada Lovelace; 120W) | €2,200 |
| Chassis | 2U rackmount | €500 |
| **Total** | | **€4,600** |

The CPU provides hardware-level trusted execution via AMD SEV-SNP. The GPU provides consensus acceleration and reserves computational capacity for future canister workloads that benefit from parallel numerical computation. The Blackwell architecture (RTX PRO 6000 and above) extends confidential computing to GPU memory; enabling an end-to-end encrypted execution environment when paired with SEV-SNP on the CPU.

A configuration using the NVIDIA RTX PRO 6000 Blackwell (96 GB GDDR7; confidential computing capable) at approximately €4,500 raises the total node cost to approximately €8,000. This configuration provides substantially more GPU memory and computational capacity for canister workloads while maintaining the same consensus acceleration characteristics measured in this research.

---

## Implications for Sovereign Subnets

A sovereign subnet operates with node providers selected by the subnet's governing body rather than through the network's general governance. The hardware specification and subnet configuration are determined by the operating entity; subject to the protocol's consensus rules.

### Where GPU acceleration matters

**Higher dealing density.** At production IC parameters (`dkg_dealings_per_block=1`); dealing distribution is the DKG bottleneck and GPU acceleration of verification has no effect on round time. However; if `dkg_dealings_per_block` were increased — to shorten DKG rounds or accommodate larger subnets — verification becomes the bottleneck. At 5+ dealings per block; CPU verification exceeds block time and GPU acceleration (particularly decompression) directly reduces DKG round duration. A sovereign subnet operator controls this parameter.

**Computational capacity.** The GPU is available for canister workloads beyond consensus. Post-quantum cryptographic operations; zero-knowledge proof verification; and numerical computation can be served by the same hardware. The ML-DSA-44 signer deployed during testing demonstrates that computationally intensive canister workloads coexist with GPU-accelerated consensus without degradation.

### Where GPU acceleration does not help

**Current IC DKG parameters.** With one dealing per block and a 499-block DKG interval; each node verifies one dealing per second with approximately 700ms of idle time remaining. GPU acceleration saves approximately 50–130ms on an operation that is not time-critical in this regime.

**Pre-signature throughput.** The iDKG pre-signature pipeline distributes dealings through the same one-per-block consensus mechanism. Pre-signature generation rate is gated by dealing distribution; not verification speed. GPU acceleration does not increase pre-signature throughput at current parameters.

---

## IC Mainnet Thread Architecture

The IC replica runs multiple isolated thread pools that collectively require significant CPU resources. The following thread counts are extracted directly from the IC source code:

| Thread Pool | Threads | Source |
|---|---|---|
| IDKG / DKG crypto verification | 16 | `MAX_IDKG_THREADS` — `rs/consensus/idkg/src/lib.rs:238` |
| Consensus (block making; notarisation; validation) | 16 | `MAX_CONSENSUS_THREADS` — `rs/consensus/src/consensus.rs:85` |
| Checkpoint manifest hashing | 16 | `NUMBER_OF_CHECKPOINT_THREADS` — `rs/state_manager/src/lib.rs:96` |
| Canister execution (replicated) | 4 | `NUMBER_OF_EXECUTION_THREADS` — `rs/config/src/execution_environment.rs:80` |
| Query execution (non-replicated) | 4 | `QUERY_EXECUTION_THREADS_TOTAL` — `rs/config/src/execution_environment.rs:127` |
| P2P networking | `cpu_count / 4` | `rs/replica/bin/replica/main.rs:95` |
| HTTP; XNet | separate tokio runtimes | — |

A production replica spawns approximately 56+ threads on a 24-core machine. These thread pools are logically isolated (separate rayon `ThreadPool` instances); but they compete for the same physical CPU cores. The IDKG and checkpoint thread pools are periodic (active during DKG rounds and state checkpoints respectively); the consensus; execution; and query pools are continuous.

---

## GPU Checkpoint Hashing

IC replica nodes compute a SHA-256 manifest of all canister state at each checkpoint (every `dkg_interval_length + 1` blocks). The IC uses 16 dedicated threads (`NUMBER_OF_CHECKPOINT_THREADS` in `rs/state_manager/src/lib.rs:96`) for this operation. On subnets with large state (the Bitcoin canister maintains 30+ GB of UTXO data); these 16 threads consume significant CPU time that would otherwise serve canister execution.

### Implementation

The GPU checkpoint path intercepts the manifest computation in `rs/state_manager/src/manifest.rs`. When the state manager encounters full-size (1 MiB) chunks that require recomputation; it collects them into a contiguous buffer; prepends the IC domain separator to each chunk; and sends the batch to the GPU for SHA-256 hashing. Smaller chunks and non-recompute operations remain on the CPU path.

IC uses domain-separated SHA-256 for chunk hashing: `SHA-256(0x0e || "ic-state-chunk" || chunk_data)`. The 15-byte domain separator is defined in `rs/state_manager/src/manifest/hash.rs`. The GPU path prepends this separator to each chunk buffer before hashing; producing output identical to the CPU `chunk_hasher()` function. This was verified by the IC's own `debug_assertions` check which compares the parallel (GPU) manifest against a sequential (CPU) reference computation.

### Thread Safety

All CUDA operations are serialised through a global mutex (`GPU_MUTEX` in `gpu_crypto/lib.rs`). The CUDA default stream is not thread-safe; concurrent `cudaMalloc`/`cudaMemcpy`/kernel launches from different host threads produce undefined behaviour. Without the mutex; the GPU hash path exhibited a 60% failure rate under sustained load (15 failures out of 25 attempts). With the mutex; zero failures were observed across all subsequent testing. The mutex is held only for the duration of GPU work (typically under 200ms for 30+ MiB of state); contention with other GPU operations (MSM; BLS) is minimal because these operations occur in different consensus phases.

### Measurements

GPU checkpoint hashing was tested on a 7-node subnet with 1-1.5 GiB of canister state and `dkg_interval_length=10` (checkpoint every 11 blocks).

| Chunks Hashed | Data Size | GPU Time | Status |
|---|---|---|---|
| 1 | 1 MiB | 36.9 ms | Verified |
| 6 | 6 MiB | 69.1 ms | Verified |
| 12 | 12 MiB | 136.2 ms | Verified |
| 25 | 25 MiB | 187.1 ms | Verified |
| 31 | 31 MiB | 227.1 ms | Verified |
| 39 | 39 MiB | 193.8 ms | Verified |

All hashes were confirmed bit-exact against the CPU reference path. The GPU node (8 cores; `taskset` constrained) maintained perfect consensus sync with unconstrained 24-core production nodes throughout all tests. Zero panics; zero hash mismatches; zero dropped blocks.

### What This Does Not Do

GPU checkpoint hashing does not reduce checkpoint wall-clock time compared to the CPU 16-thread implementation. At typical subnet state sizes (under 10 GiB); the CPU completes manifest hashing in milliseconds. The value of GPU offloading is not speed; it is the elimination of 16 CPU threads from the checkpoint operation. Those threads compete with consensus and execution for physical cores. On hardware with fewer than 32 cores; this competition is measurable.

---

## Alternative Arithmetic Approaches (Negative Results)

Two alternative approaches to GPU field arithmetic were investigated and abandoned. Both produced correct results but failed to improve on the existing sppark-based Montgomery multiplication. These are documented here because the negative results are informative for future work on GPU cryptographic implementations.

### Residue Number System (RNS)

RNS represents the 381-bit BLS12-381 prime field element as a vector of residues modulo small coprime bases that fit in 32-bit GPU registers. This eliminates carry chain propagation entirely; each residue is multiplied independently in a single 32-bit `mul` instruction. The implementation used 14 residues with a dual-base system and Barrett reduction for base extension.

The full arithmetic tower was built bottom-up: `Fp → Fp2 → Fp6 → Fp12 → Miller loop → final exponentiation`. Each layer was verified against a Rust oracle binary calling `ic_bls12_381` v0.10.1. The pairing output `e(G1; G2)` matched the Rust reference at the byte level (27 of 28 tests passed; the remaining test was a boundary case in the `+p` correction step).

**Result**: RNS `fp_mul` measured 4.4 μs; approximately 4× slower than sppark's Montgomery `fp_mul` at 1.1 μs. The bottleneck is base extension: converting between the two RNS bases requires `O(n²)` multiply-accumulate operations where `n` is the number of residues. With 14 residues; base extension accounts for 99% of the multiplication cost. The carry-free residue arithmetic does not compensate for the quadratic base extension overhead.

RNS may become competitive if the number of residues can be reduced (requiring larger base primes that still fit in 32-bit multipliers) or if a sub-quadratic base extension algorithm is found. Neither was achieved in this work.

Files: `rns/rns_fp.cuh`; `rns/rns_pairing.cuh`; `rns/test_rns_vs_rust.cu`; `rns/test_rns_bls_verify.cu`

### 64-bit Limb Representation (INT64)

NVIDIA Blackwell (sm_120) introduces native 64-bit integer multiply instructions. A 6×64-bit limb Montgomery representation was implemented to test whether wider limbs reduce register pressure and instruction count compared to sppark's 12×32-bit representation.

All 22 tests passed; correctness was verified against the same oracle.

**Result**: 64-bit `fp_mul` measured 34.3 cycles versus 9.1 cycles for 32-bit (3.75× slower). GPU registers are physically 32 bits; a `uint64_t` variable occupies two register slots. The wider limb representation halves the number of multiply instructions but doubles register pressure; resulting in no net benefit. The 64-bit multiply instruction itself is slower on current hardware because it is implemented as a multi-cycle operation on the 32-bit ALU.

This result may change on future GPU architectures with native 64-bit register files. On current NVIDIA hardware (Blackwell and earlier); 32-bit limbs remain optimal for 384-bit field arithmetic.

Files: `int64/fp64.cuh`; `int64/test_fp64.cu`; `int64/tower64.cuh`

### Implication

Single-threaded GPU field arithmetic cannot match CPU performance for BLS12-381. The CPU executes 384-bit Montgomery multiplication in approximately 60 nanoseconds using 64-bit ALUs and wide registers. The GPU requires 1.1 microseconds (sppark; 32-bit) to 4.4 microseconds (RNS) for the same operation. GPU advantage in cryptographic workloads comes exclusively from parallelism: batch operations across many independent instances (MSM across thousands of points; batch signature verification across dozens of signatures; batch SHA-256 across hundreds of chunks). Single-operation offloading (one pairing; one signature verify) is counterproductive.

---

## Mixed-Subnet Validation

The most operationally relevant test places a single GPU-accelerated node with constrained CPU alongside nodes with significantly more CPU resources in the same subnet. If the constrained node maintains consensus sync; it demonstrates that a cheaper hardware configuration can participate as a full subnet member.

### Configuration

A 3-node subnet was bootstrapped with `dkg_interval_length=10` on a single machine (AMD Ryzen 9 9900X; 24 threads). CPU cores were assigned with no overlap to ensure each node runs on dedicated physical resources:

| Node | Cores | CPU Threads | Role |
|---|---|---|---|
| Node 0 | `taskset 0-3` | 4 | GPU-accelerated (checkpoint + DKG offloaded) |
| Node 1 | `taskset 4-13` | 10 | CPU-only (2.5× more CPU than Node 0) |
| Node 2 | `taskset 14-23` | 10 | CPU-only (2.5× more CPU than Node 0) |

The canister state was grown to 1 GiB to ensure checkpoint operations process meaningful data. No core overlap exists between nodes.

### Results

| Metric | Node 0 (4 cores + GPU) | Node 1 (10 cores) | Node 2 (10 cores) |
|---|---|---|---|
| Finalized height | 724 | 724 | 724 |
| Certified height | 724 | 724 | 724 |
| Certification lag | 0-1 blocks | 0-1 blocks | 0-1 blocks |
| Checkpoints computed | 27 | 27 | 27 |
| DKG CPU time | 9.55s / 2034 calls | 9.55s / 2034 calls | 9.55s / 2035 calls |
| Certification CPU time | 3.59s | 3.59s | 3.59s |
| Consensus CPU time | 17.23s | 17.23s | 17.23s |
| GPU hash successes | 3 | — | — |
| GPU hash failures | 0 | — | — |
| Panics | 0 | 0 | 0 |

All metrics are from Prometheus endpoints on each node. Certification lag was sampled 5 times at 5-second intervals; the gap fluctuated between 0 and 1 block for all nodes equally (including instances where the GPU node was ahead of the 10-core nodes). No consistent lag was observed on any node. Finalization height; DKG processing time; consensus CPU time; and block maker call counts were identical across all three nodes throughout the test.

### What This Shows

A 4-core node with GPU acceleration keeps pace with 10-core nodes that have 2.5× more CPU in a 3-node subnet with 1 GiB state and frequent checkpoints. The Prometheus metrics confirm that the GPU node is not falling behind on any consensus phase.

### What This Does Not Show

The test ran on a single physical machine. All nodes shared the same memory bus and PCIe link to the GPU. In production; each node runs on dedicated hardware with no shared resources. The test used 3 nodes; not 13 or 34. Larger subnets increase per-round BLS verification load and DKG dealing counts; which may expose bottlenecks not visible at 3 nodes. The 10-core "production" nodes in this test are themselves constrained compared to the 64-core machines used on IC mainnet.

---

## Revised Hardware Assessment

The combined GPU offloading (checkpoint hashing; DKG MSM; DKG decompression) reduces the burst thread requirement from approximately 64 to approximately 38. The steady-state thread requirement (consensus; execution; query; P2P) remains at approximately 32 threads regardless of GPU.

| | IC Mainnet (Gen2) | GPU-Fused Model |
|---|---|---|
| CPU | 64-core AMD EPYC; SEV-SNP | 16-24 core; SEV-SNP |
| GPU | None | NVIDIA L4 24 GB or equivalent |
| RAM | 128 GB DDR5 | 64-128 GB DDR5 |
| Estimated hardware cost | ~€8;500 | ~€3;500-5;000 |
| Estimated hosted cost (monthly) | ~€1;500 | ~€400-700 |
| Reduction | — | ~2.5-3× |

The cost reduction is smaller than the 4× figure estimated in earlier analysis. The earlier estimate assumed GPU would eliminate the need for all 32 burst-period threads. In practice; only checkpoint (16 threads) and partial DKG (6-14 threads) are offloaded. Consensus protocol logic; canister execution; query handling; and P2P networking remain CPU-bound and require a minimum of 16-24 physical cores.

The remaining path to further cost reduction is GPU batch BLS verification in the consensus certifier. Standalone benchmarks show 1.2-4.4× speedup for batch sizes above 100 signatures (28+ node subnets). Wiring this into the certification pipeline would offload 2-4 additional continuous CPU threads. This has not been implemented.

---

## Stress Testing (Non-Production Parameters)

The following measurements were taken under conditions that deliberately exceed production IC parameters. They are included to characterise GPU behaviour under load; not to represent production performance. All tests ran on a single physical machine (AMD Ryzen 9 9900X; 12 cores; 24 threads; RTX PRO 6000 96 GB) with multiple replica processes sharing CPU and GPU resources.

### 28-Node Dense DKG Testnet

The testnet was configured with `dkg_interval_length=10` and 28 nodes; producing 56 dealings (28 nodes × 2 thresholds) packed across 11 blocks at approximately 5 dealings per block. This is significantly denser than production IC parameters; where `dkg_dealings_per_block=1` and `dkg_interval_length=499`. On production IC; dealings arrive one per block with approximately 700ms of idle time between each verification. The results below apply to the testnet's dealing density; not to production IC's current parameters.

### Component Benchmarks (Standalone)

| Operation | Points | CPU | GPU | Ratio |
|---|---|---|---|---|
| G1 Point Decompression | 448 | 1,195 ms | 0.98 ms | 1,219× |
| MSM (Pippenger) | 448 | 262 ms | 21 ms | 12.4× |
| MSM (Pippenger) | 896 | 445 ms | 13 ms | 34.6× |
| BLS Signature Verify (batch 18) | 18 | 113 ms | ~16 ms | 7.1× |
| SHA-256 (batch) | — | ~50 ms | ~1.4 ms | 35× |

Each GPU result was compared against the CPU reference implementation at the byte level. All comparisons pass.

### DKG Dealing Verification (28-Node Testnet; ~5 Dealings per Block)

The DKG `verify_zk_proofs` function was instrumented to measure each phase independently. 890 calls were recorded at `n=28`. Values are medians from the contended testnet (28 replicas on 12 cores); isolated benchmarks showed per-dealing verification at approximately 290ms on CPU.

| Phase | CPU (median) | GPU (median) |
|---|---|---|
| Deserialization | 150 ms | 55 ms |
| Chunking verification | 130 ms | 110 ms |
| Sharing verification | 13 ms | 18 ms |
| **Total per dealing** | **290 ms** | **~180 ms** |
| **Per block (×5.1 dealings)** | **1,479 ms** | **~920 ms** |

The deserialization phase benefits most from GPU acceleration because point decompression is embarrassingly parallel (1,219× standalone). The chunking verification phase shows marginal improvement at 28 nodes because MSM sizes (448 points per repetition) are at the lower end of where GPU exceeds kernel launch overhead.

A hybrid approach with GPU decompression and CPU chunking/sharing verification may be optimal at this subnet size; as GPU decompression provides the largest individual speedup while CPU handles the small MSMs where it remains competitive.

### DKG Round Timing (28-Node Testnet)

At the testnet's dealing density (~5 per block); each block's verification takes longer than the block interval; creating a verification backlog:

| Metric | Value |
|---|---|
| DKG round wall-clock time | 32.1s average (22–46s range) |
| Blocks per DKG round | 11 |
| Block interval (testnet) | ~2.9s |
| Verify time per block (CPU) | ~1.5s |
| Verify time per block (GPU hybrid) | ~0.85s |

Under these conditions; CPU verification cannot keep pace with block arrival and GPU acceleration reduces the backlog. On production IC with `dkg_dealings_per_block=1`; a single dealing arrives per block (~1s intervals). Verification at 290ms per dealing completes well within the block interval. Under production parameters; DKG round time is gated by dealing distribution (56 blocks to distribute all dealings); not by verification speed.

### CPU Time Distribution (28-Node Testnet)

Prometheus metrics from the 28-node testnet reveal per-node CPU time allocation:

| Operation | Cumulative CPU Time | Share |
|---|---|---|
| DKG dealing verification | 1,766 s | 88.3% |
| DKG epoch updates | 161 s | 8.1% |
| BLS share verification | 26 s | 1.3% |
| Multi-signature verification | 20 s | 1.0% |
| Canister execution | 6 s | 0.3% |
| All other operations | 21 s | 1.0% |

DKG verification dominates per-node CPU time. However; this reflects compute time; not wall-clock bottleneck. At production dealing rates (1 per block); this CPU load is spread thinly with each verification occupying approximately 29% of one block interval; leaving 71% available for other work.

### Canister Throughput Under Simulated DKG Load

A synthetic benchmark spawned threads matching the IC's production thread pool configuration (16 IDKG; 16 consensus; 4 execution; 4 query) and measured canister execution throughput with and without GPU offloading. This models the peak contention scenario during a DKG round with all IDKG threads active.

| Physical Cores | CPU-Only (ops/sec) | GPU-Offloaded (ops/sec) | Gain |
|---|---|---|---|
| 4 | 52,501 | 145,337 | +177% |
| 8 | 156,538 | 298,713 | +91% |
| 12 | 243,207 | 402,789 | +66% |
| 16 | 292,563 | 495,921 | +70% |
| 24 | 355,103 | 627,046 | +77% |

A 12-core machine with GPU offloading (402,789 ops/sec) delivers 13% more canister execution throughput than a 24-core machine without GPU (355,103 ops/sec). Between DKG rounds; the benefit is proportionally smaller because IDKG threads are only partially loaded (20–40% for pre-signature generation).

### Stress Test Limitations

**These results do not represent production IC behaviour.** The 28-node testnet packed 5 dealings per block; creating an artificial verification bottleneck that does not exist at production parameters. All 28 replicas shared a single 12-core CPU; inflating contention by 2–3×. The canister throughput benchmark is synthetic; not measured from actual IC consensus. Production subnets serve thousands of canisters with heterogeneous workloads; and the resource profile may differ materially.

---

## Repository Structure

```
cuda/                    GPU kernel source and benchmarks
  decompress.cu          G1 batch point decompression
  msm_full.cu            Pippenger MSM (sppark)
  sha256.cuh             GPU SHA-256 (batch; domain-separated)
  blst_aliases.c         ADX symbol bridge for blst
  bench_merkle.cu        Merkle SHA-256 benchmark
  bench_oracle_sha256.cu Domain-separated SHA-256 oracle test
  profile_bls_split.cu   BLS component timing (Miller vs final_exp)
  test_sha256_domain.cu  Domain separator verification
  vendor/sppark/         Vendored sppark (Apache-2.0)
  vendor/blst/           Vendored blst headers

rns/                     Residue Number System arithmetic (negative result)
  rns_fp.cuh             14-residue dual-base Fp multiplication
  rns_pairing.cuh        Full tower Fp→Fp12 + Miller loop + final exp
  test_rns_vs_rust.cu    Byte-level comparison against ic_bls12_381
  test_rns_bls_verify.cu End-to-end pairing verification
  gen_bases.py           RNS base generation (coprime selection)

int64/                   64-bit limb Montgomery (negative result)
  fp64.cuh               6×64-bit Fp arithmetic for Blackwell sm_120
  tower64.cuh            Fp2/Fp6/Fp12 tower on 64-bit limbs
  test_fp64.cu           Correctness and timing tests

gpu_merkle/              GPU Merkle SHA-256 library
  merkle_api.cu          C API: init; hash_chunks; free
  merkle_api_async.cu    Async pinned-memory variant
  src/lib.rs             Rust FFI with dlopen and CPU fallback

oracle/                  Reference binary for correctness verification

gpu-crypto/              Rust FFI crate for IC integration
  src/lib.rs             GPU crypto API: MSM; decompression; BLS batch verify;
                         batch SHA-256; Merkle hash. Thread-safe CUDA access
                         with dlopen runtime loading and CPU fallback.
  cuda/api.cu            CUDA kernels (pairing; SHA-256)
  cuda/sha256.cuh        SHA-256 device implementation
  build.rs               Stub generation for non-GPU builds

ic-integration/          IC replica integration files
  gpu_crypto/            Modified gpu_crypto crate for IC Bazel build
  patches/               Patches to IC state_manager and crypto crates
  bootstrap.sh           ic-prep based N-node testnet bootstrap
  launch.sh              Replica launcher with GPU library paths

test_msm/               MSM correctness verification and benchmark suite
ic-bench/               IC consensus benchmark harness
lib/                    Development CUDA files
verify/                 Python verification scripts
```

## Building

Requires NVIDIA CUDA Toolkit 13.0+ and an NVIDIA GPU with compute capability 8.0 or higher.

```bash
# GPU Merkle library
cd gpu_merkle && nvcc -shared -O2 -o libgpu_merkle.so merkle_api_async.cu -Xcompiler -fPIC

# IC replica integration (from IC repository root)
bazel build //rs/replica
```

## License

Apache-2.0. GPU kernels use sppark (Apache-2.0) and blst (Apache-2.0).

---

*Mercatura Forum — Hive Mind Research Program*
*Q2 2026*
