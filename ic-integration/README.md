# IC Replica Integration

This directory contains the modifications required to integrate GPU-accelerated cryptographic operations into the Internet Computer replica binary.

## Contents

- `gpu_crypto/` — New crate added to `rs/crypto/internal/crypto_lib/gpu_crypto/` in the IC repository. Contains the Rust FFI layer; CUDA kernels for BLS pairing; MSM; and point decompression; and the Bazel build configuration.

- `patches/ic-gpu-consensus.patch` — Patch against the IC repository (`dfinity/ic`) containing modifications to existing files:
  - `certifier.rs` — Batch share collection with timing instrumentation in the certification pipeline
  - `lib.rs` (bls12_381 type) — GPU MSM and GPU decompression modules with CPU fallback
  - `nizk_chunking.rs` — GPU MSM calls replacing `muln_affine_vartime` in DKG chunking proof verification
  - `forward_secure.rs` — GPU batch decompression in ciphertext deserialization
  - `encryption.rs` — DKG phase timing instrumentation
  - `nizk_sharing.rs` — DKG sharing verification timing instrumentation

- `bootstrap.sh` — Initialises a local multi-node IC testnet using `ic-prep`
- `launch.sh` — Launches N replica processes on unique loopback IPs
- `collect_metrics.sh` — Collects Prometheus crypto timing metrics from running nodes

## Applying the Patch

```bash
cd <ic-repository-root>

# Copy the gpu_crypto crate
cp -r <this-dir>/gpu_crypto rs/crypto/internal/crypto_lib/gpu_crypto

# Apply modifications to existing files
git apply <this-dir>/patches/ic-gpu-consensus.patch

# Build
bazel build //rs/replica
```

## Running the Testnet

```bash
# Bootstrap a 28-node subnet
bash bootstrap.sh 28

# Launch all nodes (requires sandbox binaries next to replica)
bash launch.sh 28

# Collect metrics after consensus stabilises
bash collect_metrics.sh
```

The replica loads `libgpu_msm.so` via `dlopen` at runtime. Place the shared library in the same directory as the replica binary or in `LD_LIBRARY_PATH`. If absent; the replica falls back to CPU for all GPU-accelerated operations.
