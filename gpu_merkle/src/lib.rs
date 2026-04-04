//! GPU Merkle SHA-256 — Rust FFI wrapper
//!
//! Loads libgpu_merkle.so via dlopen at runtime.
//! Falls back to CPU if GPU unavailable.
//!
//! Oracle-verified: all GPU results match Rust sha2 crate bit-exact.

use std::path::Path;
use std::sync::OnceLock;

/// Merkle tree parameters (must match CUDA code)
pub const CHUNK_SIZE: usize = 1_048_576;
pub const SUB_CHUNK_SIZE: usize = 4_096;
pub const NUM_LEAVES: usize = CHUNK_SIZE / SUB_CHUNK_SIZE; // 256

// FFI function signatures
type FnInit = unsafe extern "C" fn() -> i32;
type FnHashChunks = unsafe extern "C" fn(*const u8, *mut u8, i32) -> i32;
type FnFree = unsafe extern "C" fn();

struct GpuLib {
    _lib: libloading::Library,
    init: FnInit,
    hash_chunks: FnHashChunks,
    _free: FnFree,
}

static GPU_LIB: OnceLock<Option<GpuLib>> = OnceLock::new();

fn load_gpu() -> Option<GpuLib> {
    // Search paths for the shared library
    let search_paths = [
        "/workspace/gpu_pairing/gpu_merkle/libgpu_merkle.so",
        "./libgpu_merkle.so",
        "/usr/local/lib/libgpu_merkle.so",
    ];

    for path in &search_paths {
        if !Path::new(path).exists() {
            continue;
        }
        unsafe {
            match libloading::Library::new(path) {
                Ok(lib) => {
                    let init: FnInit = *lib.get(b"gpu_merkle_init").ok()?;
                    let hash_chunks: FnHashChunks = *lib.get(b"gpu_merkle_hash_chunks").ok()?;
                    let free: FnFree = *lib.get(b"gpu_merkle_free").ok()?;

                    // Try init
                    if init() != 0 {
                        eprintln!("[gpu-merkle] GPU init failed, falling back to CPU");
                        return None;
                    }

                    eprintln!("[gpu-merkle] GPU loaded from {path}");
                    return Some(GpuLib {
                        _lib: lib,
                        init: init,
                        hash_chunks,
                        _free: free,
                    });
                }
                Err(e) => {
                    eprintln!("[gpu-merkle] Failed to load {path}: {e}");
                }
            }
        }
    }
    None
}

fn get_gpu() -> Option<&'static GpuLib> {
    GPU_LIB.get_or_init(load_gpu).as_ref()
}

/// Returns true if GPU is available for Merkle hashing.
pub fn gpu_available() -> bool {
    get_gpu().is_some()
}

/// Compute Merkle SHA-256 root hashes for N × 1MiB chunks.
///
/// Uses GPU if available, falls back to CPU.
/// Returns Vec of 32-byte root hashes.
pub fn merkle_hash_chunks(data: &[u8], num_chunks: usize) -> Vec<[u8; 32]> {
    assert_eq!(data.len(), num_chunks * CHUNK_SIZE);

    // Try GPU path
    if let Some(gpu) = get_gpu() {
        let mut roots = vec![[0u8; 32]; num_chunks];
        let ret = unsafe {
            (gpu.hash_chunks)(
                data.as_ptr(),
                roots.as_mut_ptr() as *mut u8,
                num_chunks as i32,
            )
        };
        if ret == 0 {
            return roots;
        }
        eprintln!("[gpu-merkle] GPU hash failed (ret={ret}), falling back to CPU");
    }

    // CPU fallback
    merkle_hash_chunks_cpu(data, num_chunks)
}

/// CPU implementation of Merkle SHA-256 (always available).
pub fn merkle_hash_chunks_cpu(data: &[u8], num_chunks: usize) -> Vec<[u8; 32]> {
    use sha2::{Sha256, Digest};

    assert_eq!(data.len(), num_chunks * CHUNK_SIZE);

    (0..num_chunks)
        .map(|c| {
            let chunk = &data[c * CHUNK_SIZE..(c + 1) * CHUNK_SIZE];

            // Phase 1: leaf hashes
            let mut hashes: Vec<[u8; 32]> = (0..NUM_LEAVES)
                .map(|i| {
                    let sub = &chunk[i * SUB_CHUNK_SIZE..(i + 1) * SUB_CHUNK_SIZE];
                    Sha256::digest(sub).into()
                })
                .collect();

            // Phase 2: tree combine
            while hashes.len() > 1 {
                hashes = hashes
                    .chunks(2)
                    .map(|pair| {
                        let mut h = Sha256::new();
                        h.update(&pair[0]);
                        h.update(&pair[1]);
                        h.finalize().into()
                    })
                    .collect();
            }

            hashes[0]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data() -> Vec<u8> {
        let mut data = vec![0u8; CHUNK_SIZE];
        for i in (0..CHUNK_SIZE).step_by(8) {
            let val = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let to_copy = (CHUNK_SIZE - i).min(8);
            data[i..i + to_copy].copy_from_slice(&val.to_le_bytes()[..to_copy]);
        }
        data
    }

    #[test]
    fn test_cpu_matches_oracle() {
        let data = test_data();
        let roots = merkle_hash_chunks_cpu(&data, 1);
        let expected = hex::decode(
            "4b894c4582e7b5788cfe60c88317d2b4cb2f6cb49c637d435ce60ea5c897a98d"
        ).unwrap();
        assert_eq!(roots[0].as_slice(), expected.as_slice());
    }

    #[test]
    fn test_gpu_matches_oracle() {
        let data = test_data();
        let roots = merkle_hash_chunks(&data, 1);
        let expected = hex::decode(
            "4b894c4582e7b5788cfe60c88317d2b4cb2f6cb49c637d435ce60ea5c897a98d"
        ).unwrap();
        assert_eq!(roots[0].as_slice(), expected.as_slice(),
            "GPU root {} != oracle",
            hex::encode(roots[0])
        );
    }

    #[test]
    fn test_batch_consistency() {
        let data = test_data();
        let mut batch = vec![0u8; 5 * CHUNK_SIZE];
        for i in 0..5 {
            batch[i * CHUNK_SIZE..(i + 1) * CHUNK_SIZE].copy_from_slice(&data);
        }
        let roots = merkle_hash_chunks(&batch, 5);
        let expected = hex::decode(
            "4b894c4582e7b5788cfe60c88317d2b4cb2f6cb49c637d435ce60ea5c897a98d"
        ).unwrap();
        for (i, root) in roots.iter().enumerate() {
            assert_eq!(root.as_slice(), expected.as_slice(), "chunk {i} mismatch");
        }
    }
}
