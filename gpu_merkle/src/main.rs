//! IC Checkpoint Production Simulation
//!
//! V4 (current): CPU 16-thread domain-separated flat SHA-256
//! V5 (proposed): GPU Merkle SHA-256
//!
//! Production parameters: CHUNK_SIZE=1MiB, CHECKPOINT_THREADS=16

use gpu_merkle::*;
use sha2::{Sha256, Digest as Sha2Digest};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const IC_CHECKPOINT_THREADS: usize = 16;

fn generate_state(num_chunks: usize, seed: u64) -> Vec<u8> {
    let mut data = vec![0u8; num_chunks * CHUNK_SIZE];
    for i in 0..num_chunks {
        let base = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(i as u64);
        let offset = i * CHUNK_SIZE;
        for j in (0..CHUNK_SIZE).step_by(8) {
            let val = base.wrapping_add(j as u64).wrapping_mul(0x517CC1B727220A95);
            let to_copy = (CHUNK_SIZE - j).min(8);
            data[offset + j..offset + j + to_copy].copy_from_slice(&val.to_le_bytes()[..to_copy]);
        }
    }
    data
}

/// V4: CPU 16-thread domain-separated flat SHA-256 (exact IC code path)
fn v4_cpu(data: &[u8], num_chunks: usize, num_threads: usize) -> (Vec<[u8; 32]>, f64) {
    let counter = AtomicUsize::new(0);
    let results: Vec<std::sync::Mutex<[u8; 32]>> =
        (0..num_chunks).map(|_| std::sync::Mutex::new([0u8; 32])).collect();

    let start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..num_threads {
            scope.spawn(|| {
                loop {
                    let idx = counter.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_chunks { break; }
                    let chunk = &data[idx * CHUNK_SIZE..(idx + 1) * CHUNK_SIZE];
                    let mut h = Sha256::new();
                    h.update([14u8]);
                    h.update(b"ic-state-chunk");
                    h.update(chunk);
                    *results[idx].lock().unwrap() = h.finalize().into();
                }
            });
        }
    });
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let hashes: Vec<[u8; 32]> = results.into_iter().map(|m| m.into_inner().unwrap()).collect();
    (hashes, ms)
}

/// V5: GPU Merkle (end-to-end including H2D transfer)
fn v5_gpu(data: &[u8], num_chunks: usize) -> (Vec<[u8; 32]>, f64) {
    let start = Instant::now();
    let roots = merkle_hash_chunks(data, num_chunks);
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (roots, ms)
}

/// V5: CPU Merkle fallback
fn v5_cpu(data: &[u8], num_chunks: usize) -> (Vec<[u8; 32]>, f64) {
    let start = Instant::now();
    let roots = merkle_hash_chunks_cpu(data, num_chunks);
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (roots, ms)
}

/// Contention test: measure canister ops while checkpoint runs
fn contention_test(
    data: &[u8],
    num_chunks: usize,
    total_cores: usize,
    use_gpu: bool,
) -> (f64, u64) {
    let canister_ops = AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicBool::new(false);
    let canister_threads = 4.min(total_cores);

    let counter = AtomicUsize::new(0);

    let start = Instant::now();
    std::thread::scope(|scope| {
        // Canister threads
        for _ in 0..canister_threads {
            scope.spawn(|| {
                let mut x: u64 = 0x12345;
                while !done.load(Ordering::Relaxed) {
                    for _ in 0..1000 {
                        x = x.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
                    }
                    canister_ops.fetch_add(1000, Ordering::Relaxed);
                }
                std::hint::black_box(x);
            });
        }

        if use_gpu {
            let _ = merkle_hash_chunks(data, num_chunks);
        } else {
            let ckpt_threads = IC_CHECKPOINT_THREADS.min(total_cores.saturating_sub(canister_threads));
            for _ in 0..ckpt_threads {
                scope.spawn(|| {
                    loop {
                        let idx = counter.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_chunks { break; }
                        let chunk = &data[idx * CHUNK_SIZE..(idx + 1) * CHUNK_SIZE];
                        let mut h = Sha256::new();
                        h.update([14u8]);
                        h.update(b"ic-state-chunk");
                        h.update(chunk);
                        let _: [u8; 32] = h.finalize().into();
                    }
                });
            }
        }

        done.store(true, Ordering::Relaxed);
    });

    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let ops = canister_ops.load(Ordering::Relaxed) as u64;
    (ms, ops)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  IC Checkpoint Production Simulation                        ║");
    println!("║  V4 (CPU 16T flat SHA-256) vs V5 (GPU Merkle SHA-256)      ║");
    println!("║  Production parameters: 1MiB chunks, 16 checkpoint threads  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("GPU: {}\n", if gpu_available() { "AVAILABLE" } else { "NOT FOUND" });

    // ===== Correctness =====
    println!("=== CORRECTNESS ===");
    let test_data = generate_state(5, 42);
    let (v4h, _) = v4_cpu(&test_data, 5, IC_CHECKPOINT_THREADS);
    let (v5h, _) = v5_gpu(&test_data, 5);
    let oracle = merkle_hash_chunks_cpu(&test_data, 5);
    let ok = v5h.iter().zip(oracle.iter()).all(|(a, b)| a == b);
    println!("  V5 GPU matches Merkle oracle: {}", if ok { "PASS" } else { "FAIL" });
    println!("  V4 != V5 (different algo): {}\n", v4h[0] != v5h[0]);
    assert!(ok);

    // ===== Production timing for different subnet sizes =====
    println!("=== PRODUCTION CHECKPOINT TIMING ===");
    println!("  (End-to-end: includes data read + hash + result collection)\n");

    let scenarios: Vec<(&str, usize, usize)> = vec![
        ("13-node small (100 MiB)",    100,  13),
        ("13-node medium (500 MiB)",   500,  13),
        ("28-node NNS (1 GiB)",       1024,  28),
        ("13-node large (2 GiB)",     2048,  13),
    ];

    println!("{:<30} {:>10} {:>10} {:>10} {:>8}",
             "Scenario", "V4 CPU", "V5 GPU", "V5 CPU-M", "GPU/CPU");
    println!("{}", "-".repeat(72));

    for (name, chunks, _nodes) in &scenarios {
        let n = *chunks;
        if n * CHUNK_SIZE > 3_000_000_000 { // 3GB limit (VRAM)
            println!("{:<30} SKIP (VRAM limit)", name);
            continue;
        }
        let data = generate_state(n, 1337);

        // Warmup
        let _ = v5_gpu(&data, n);

        // Measure (average of 3)
        let mut v4_ms = 0.0;
        let mut v5g_ms = 0.0;
        let mut v5c_ms = 0.0;
        let runs = 3;
        for _ in 0..runs {
            let (_, t) = v4_cpu(&data, n, IC_CHECKPOINT_THREADS);
            v4_ms += t;
            let (_, t) = v5_gpu(&data, n);
            v5g_ms += t;
            let (_, t) = v5_cpu(&data, n);
            v5c_ms += t;
        }
        v4_ms /= runs as f64;
        v5g_ms /= runs as f64;
        v5c_ms /= runs as f64;

        let ratio = v4_ms / v5g_ms;
        println!("{:<30} {:>9.1}ms {:>9.1}ms {:>9.1}ms {:>7.1}x",
                 name, v4_ms, v5g_ms, v5c_ms, ratio);
    }

    // ===== Contention: Checkpoint + Canister on limited cores =====
    println!("\n=== CPU CONTENTION: Checkpoint vs Canister Execution ===");
    println!("  500 MiB state, 4 canister threads competing with 16 checkpoint threads\n");

    let state = generate_state(500, 9999);

    println!("{:<20} {:>10} {:>14} {:>10} {:>14} {:>10}",
             "Config", "V4 time", "V4 can_ops", "V5 time", "V5 can_ops", "ops gain");
    println!("{}", "-".repeat(82));

    for (cores, name) in [(4, "4-core"), (8, "8-core"), (12, "12-core"), (16, "16-core"), (24, "24-core")] {
        // V4: CPU checkpoint + canister threads competing
        let (v4_ms, v4_ops) = contention_test(&state, 500, cores, false);
        // V5: GPU checkpoint, canister threads uncontested
        let (v5_ms, v5_ops) = contention_test(&state, 500, cores, true);

        let gain = if v4_ops > 0 { v5_ops as f64 / v4_ops as f64 } else { f64::INFINITY };
        println!("{:<20} {:>9.1}ms {:>14} {:>9.1}ms {:>14} {:>9.1}x",
                 name, v4_ms, v4_ops, v5_ms, v5_ops, gain);
    }

    // ===== 13-node and 28-node specific production configs =====
    println!("\n=== IC SUBNET CONFIGS (Production Parameters) ===");
    println!("  checkpoint_interval=499 blocks, scheduler_cores=4\n");

    for (subnet, nodes, state_mib) in [
        ("13-node App subnet", 13, 500),
        ("28-node NNS subnet", 28, 1024),
        ("13-node Bitcoin", 13, 2048),
    ] {
        let data = generate_state(state_mib, nodes as u64);

        // Measure V4 with production thread counts
        let (_, v4_ms) = v4_cpu(&data, state_mib, IC_CHECKPOINT_THREADS);
        let _ = v5_gpu(&data, state_mib); // warmup
        let (_, v5_ms) = v5_gpu(&data, state_mib);

        // Per-round BLS time
        let bls_per_round = (3 * nodes + 5) as f64 * 0.862; // ms

        println!("  {} ({} MiB state, {} nodes):", subnet, state_mib, nodes);
        println!("    Checkpoint: V4={:.1}ms  V5={:.1}ms  ({:.1}x)", v4_ms, v5_ms, v4_ms/v5_ms);
        println!("    BLS/round:  {:.1}ms CPU seq", bls_per_round);
        println!("    Checkpoint every: ~{:.0}s (499 blocks × ~1s/block)", 499.0);
        println!("    Checkpoint as % of interval: V4={:.3}%  V5={:.3}%",
                 v4_ms / (499.0 * 1000.0) * 100.0,
                 v5_ms / (499.0 * 1000.0) * 100.0);
        println!();
    }

    println!("=== SIMULATION COMPLETE ===");
}
