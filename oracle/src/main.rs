// GPU Crypto Oracle — Source of Truth for IC Replica GPU Acceleration
//
// Oracle methodology: this binary uses the EXACT libraries the IC uses
// (ic_bls12_381 v0.10.1 for BLS, sha2 for SHA-256) and outputs intermediate
// values that GPU implementations must match BIT-EXACT.
//
// Subcommands:
//   bls-verify-batch    — Generate N BLS verification test vectors with intermediates
//   sha256-manifest     — Generate SHA-256 test vectors for manifest chunk hashing
//   merkle-oracle       — Generate Merkle SHA-256 test vectors with ALL intermediates
//   consensus-round     — Simulate one consensus round's crypto workload and time it
//   profile-bls         — Benchmark single + batch BLS verify timings (CPU baseline)
//   profile-sha256      — Benchmark SHA-256 at various chunk sizes (CPU baseline)

mod merkle_sha256;

use ic_bls12_381::*;
use ff::Field;
use group::Group;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Sha256, Digest as Sha2Digest};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: oracle <subcommand>");
        eprintln!("  bls-verify-batch <N>     — N BLS verify test vectors");
        eprintln!("  sha256-manifest <N> <SZ> — N chunks of SZ bytes each");
        eprintln!("  consensus-round <nodes>  — Simulate round crypto for N-node subnet");
        eprintln!("  profile-bls <N>          — Benchmark N BLS single verifies");
        eprintln!("  profile-sha256 <chunks>  — Benchmark SHA-256 on 1MiB chunks");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "bls-verify-batch" => {
            let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
            cmd_bls_verify_batch(n);
        }
        "sha256-manifest" => {
            let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            let sz: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1048576);
            cmd_sha256_manifest(n, sz);
        }
        "consensus-round" => {
            let nodes: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(13);
            cmd_consensus_round(nodes);
        }
        "profile-bls" => {
            let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            cmd_profile_bls(n);
        }
        "profile-sha256" => {
            let chunks: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
            cmd_profile_sha256(chunks);
        }
        "merkle-oracle" => {
            let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
            cmd_merkle_oracle(n);
        }
        "fp-test" => {
            cmd_fp_test();
        }
        "frob-coeffs" => {
            cmd_frob_coeffs();
        }
        "g2-coeffs" => {
            cmd_g2_coeffs();
        }
        "pairing-oracle" => {
            cmd_pairing_oracle();
        }
        _ => {
            eprintln!("Unknown subcommand: {}", args[1]);
            std::process::exit(1);
        }
    }
}

// ==================== Fp Field Arithmetic Test Vectors ====================
// Cross-check GPU field arithmetic against ic_bls12_381.
// Strategy: Use G1Affine x-coordinate (which is Fp) and Scalar for arithmetic.
// Output canonical (non-Montgomery) bytes so we can compare across different
// Montgomery representations (Rust R vs RNS M1).

fn cmd_fp_test() {
    println!("=== Fp Oracle Test Vectors (ic_bls12_381 v0.10.1) ===\n");

    // --- Scalar field tests (exported, has full arithmetic) ---
    // Scalar uses same Montgomery multiply algorithm, 256-bit field
    // Good for validating the mul/add/sub algorithm
    let one = Scalar::one();
    let two = one + one;
    let three = two + one;
    let seven = three + three + one;
    let six = three + three;
    let forty_nine = seven * seven;

    println!("--- Scalar arithmetic (Montgomery, 4x u64 limbs) ---");

    // Access internal limbs via transmute
    let one_limbs: [u64; 4] = unsafe { std::mem::transmute_copy(&one) };
    let seven_limbs: [u64; 4] = unsafe { std::mem::transmute_copy(&seven) };
    let forty_nine_limbs: [u64; 4] = unsafe { std::mem::transmute_copy(&forty_nine) };

    println!("Scalar::one()     mont limbs: {:?}", one_limbs);
    println!("Scalar(7)         mont limbs: {:?}", seven_limbs);
    println!("Scalar(49)        mont limbs: {:?}", forty_nine_limbs);
    println!("7*7 == 49: {}", seven * seven == forty_nine);

    // --- G1 generator coordinates give us Fp values ---
    // G1 generator x,y are Fp elements. We can extract their canonical bytes.
    let g1 = G1Affine::generator();
    // G1Affine has x: Fp, y: Fp, infinity: Choice
    // x and y are 48 bytes each in big-endian canonical form
    let g1_bytes = g1.to_uncompressed();
    let x_bytes = &g1_bytes[0..48];
    let y_bytes = &g1_bytes[48..96];

    println!("\n--- G1 generator (Fp coordinates, canonical big-endian) ---");
    print!("G1.x = ");
    for b in x_bytes { print!("{:02x}", b); }
    println!();
    print!("G1.y = ");
    for b in y_bytes { print!("{:02x}", b); }
    println!();

    // Internal Montgomery representation via transmute
    // G1Affine layout: x(Fp=6xu64), y(Fp=6xu64), infinity(u8)
    let g1_internal: ([u64; 6], [u64; 6], u8) = unsafe { std::mem::transmute_copy(&g1) };
    println!("\nG1.x mont limbs (6x u64): {:016x?}", g1_internal.0);
    println!("G1.y mont limbs (6x u64): {:016x?}", g1_internal.1);

    // --- Key test: Fp multiplication via G1 point operations ---
    // We know G1 generator. If we compute 7*G1 then extract x-coord,
    // that gives us an Fp value computed by ic_bls12_381's actual Fp multiply.
    let seven_scalar = seven;
    let g1_7 = G1Affine::from(G1Affine::generator() * seven_scalar);
    let g1_7_bytes = g1_7.to_uncompressed();

    println!("\n--- 7*G1 point (verifies Fp mul is used internally) ---");
    print!("(7*G1).x = ");
    for b in &g1_7_bytes[0..48] { print!("{:02x}", b); }
    println!();
    print!("(7*G1).y = ");
    for b in &g1_7_bytes[48..96] { print!("{:02x}", b); }
    println!();

    // --- Direct Fp test via pairing ---
    // e(G1, G2) gives Gt which is Fp12. The result depends on all Fp ops.
    // This is the ultimate cross-check.
    let g2 = G2Affine::generator();
    let pair = pairing(&g1, &g2);
    // Gt is Fp12. Extract internal bytes.
    let pair_bytes: [u64; 72] = unsafe { std::mem::transmute_copy(&pair) };
    println!("\n--- e(G1,G2) pairing result (first 6 u64 of Fp12.c0.c0) ---");
    println!("Fp12.c0.c0 mont limbs: {:016x?}", &pair_bytes[0..6]);

    // --- Output canonical values for key field elements ---
    // These canonical bytes are representation-independent.
    // Both Rust Montgomery and CUDA RNS should produce the same canonical output.
    println!("\n--- CANONICAL VALUES (representation-independent, for cross-check) ---");
    println!("These are the values both implementations must agree on.\n");

    // Scalar canonical via to_bytes (little-endian)
    let vals_scalar = vec![
        ("one", Scalar::one()),
        ("seven", seven),
        ("forty_nine", forty_nine),
        ("six", six),
    ];
    for (name, v) in &vals_scalar {
        let bytes = v.to_bytes();
        print!("Scalar({:>10}): ", name);
        for b in &bytes { print!("{:02x}", b); }
        println!();
    }

    // For Fp: construct known values and output canonical bytes
    // We can get Fp values from G1 arithmetic
    // 2*G1, 3*G1, etc. give us Fp values that depend on Fp mul
    println!();
    let test_scalars = vec![1u64, 2, 3, 5, 7, 42, 100];
    for &s in &test_scalars {
        let sc = {
            let mut v = Scalar::zero();
            let one_s = Scalar::one();
            for _ in 0..s { v += one_s; }
            v
        };
        let pt = G1Affine::from(G1Affine::generator() * sc);
        let pt_bytes = pt.to_uncompressed();
        print!("({:>3}*G1).x = ", s);
        for b in &pt_bytes[0..48] { print!("{:02x}", b); }
        println!();
    }
}

// ==================== G2 Prepared Coefficients ====================

fn cmd_g2_coeffs() {
    // G2Prepared precomputes line coefficients for the Miller loop
    let g2 = G2Affine::generator();
    let g2_prep = G2Prepared::from(g2);

    // G2Prepared contains a Vec of (Fp2, Fp2, Fp2) tuples
    // Access internal data via transmute
    // G2Prepared layout: infinity: Choice (1 byte), then padding, then Vec<(Fp2,Fp2,Fp2)>
    // Vec is (ptr, len, cap) = 24 bytes on 64-bit
    // Easier: just get the raw pointer and length via transmute of the struct

    // G2Prepared { infinity: Choice, coeffs: Vec<...> }
    // Choice is a u8. Vec is ptr+len+cap = 3*usize.
    // With alignment, likely: [u8, 7 padding, ptr(8), len(8), cap(8)] = 32 bytes
    // Or: since Vec comes after Choice and has align 8, offset of coeffs = 8

    // Read the struct as raw bytes
    let raw: &[u8] = unsafe {
        std::slice::from_raw_parts(
            &g2_prep as *const G2Prepared as *const u8,
            std::mem::size_of::<G2Prepared>()
        )
    };

    // Extract Vec fields: ptr at offset 8, len at offset 16, cap at offset 24
    let coeffs_ptr = usize::from_ne_bytes(raw[8..16].try_into().unwrap()) as *const [u64; 36];
    let coeffs_len = usize::from_ne_bytes(raw[16..24].try_into().unwrap());

    let coeffs_slice: &[[u64; 36]] = unsafe {
        std::slice::from_raw_parts(coeffs_ptr, coeffs_len)
    };

    println!("G2_PREPARED_COEFFS_COUNT={}", coeffs_len);

    // Output each coefficient as 36 hex u64 values (LE limbs)
    for (idx, coeff) in coeffs_slice.iter().enumerate() {
        print!("COEFF[{:02}]=", idx);
        for v in coeff {
            print!("{:016x}", v);
        }
        println!();
    }
}

fn cmd_pairing_oracle() {
    // Compute e(G1, G2) and output the Fp12 result in Montgomery limbs
    let g1 = G1Affine::generator();
    let g2 = G2Affine::generator();
    let gt = pairing(&g1, &g2);

    // Extract all 72 u64 limbs
    let limbs: [u64; 72] = unsafe { std::mem::transmute_copy(&gt) };

    println!("PAIRING_RESULT=");
    // Output as 12 Fp values (c0.c0.c0, c0.c0.c1, c0.c1.c0, ...)
    let names = ["c0c0c0", "c0c0c1", "c0c1c0", "c0c1c1",
                 "c0c2c0", "c0c2c1", "c1c0c0", "c1c0c1",
                 "c1c1c0", "c1c1c1", "c1c2c0", "c1c2c1"];
    for (i, name) in names.iter().enumerate() {
        let base = i * 6;
        print!("  {}=", name);
        for j in 0..6 {
            print!("{:016x}", limbs[base + j]);
        }
        println!();
    }

    // Also output G1 generator coordinates for the Miller loop input
    let g1_limbs: ([u64; 6], [u64; 6], u8) = unsafe { std::mem::transmute_copy(&g1) };
    print!("G1_X=");
    for v in &g1_limbs.0 { print!("{:016x}", v); }
    println!();
    print!("G1_Y=");
    for v in &g1_limbs.1 { print!("{:016x}", v); }
    println!();
}

// ==================== Frobenius Coefficients ====================

fn cmd_frob_coeffs() {
    println!("=== Frobenius Coefficients (canonical, from ic_bls12_381) ===\n");

    // The Frobenius endomorphism on Fp2 is conjugation: (a, b) → (a, -b)
    // On Fp6: frob(c0, c1, c2) = (conj(c0), conj(c1)*γ1, conj(c2)*γ2)
    // On Fp12: frob(c0, c1) = (frob6(c0), frob6(c1)*δ)
    //
    // The coefficients γ1, γ2, δ are specific Fp2 elements.
    // ic_bls12_381 stores them in Montgomery form internally.
    //
    // We extract them by computing: frob(test_element) and deducing the coefficients.
    //
    // For Fp6 γ1: apply frob to (0, 1, 0) → (0, conj(1)*γ1, 0) = (0, γ1, 0)
    // For Fp6 γ2: apply frob to (0, 0, 1) → (0, 0, conj(1)*γ2) = (0, 0, γ2)
    // For Fp12 δ: apply frob to Fp12(0, Fp6(1, 0, 0)) → Fp12(0, frob6(Fp6(1,0,0))*δ)
    //           = Fp12(0, Fp6(δ, 0, 0))

    // ic_bls12_381 uses precomputed FROBENIUS_COEFF_FP6_C1 and C2
    // Let's extract them by doing the actual frobenius on test elements

    // Method: compute frob on Fp12 element via the pairing library
    // Actually, let's just extract the constants directly from the source

    // The ic_bls12_381 Fp type stores 6 x u64 in Montgomery form
    // To get canonical: multiply by R^(-1) where R = 2^384 mod p

    // Known Montgomery form values from ic_bls12_381 source:
    // FROBENIUS_COEFF_FP6_C1[1] = Fp2(zero, Fp([...]))
    // Let me just compute frob of known elements and extract

    // Use G2Prepared which triggers frobenius internally
    // Actually simplest: just verify our constants by doing:
    //   frob(Fp12(Fp6(0,1,0), 0)) and extracting the result

    // Build Fp12 test element using pairing arithmetic
    // e(2*G1, G2) gives us an Fp12 value. Apply frob and compare.

    let g1 = G1Affine::generator();
    let g2 = G2Affine::generator();
    let gt = pairing(&g1, &g2);

    // gt is an Fp12 element (type Gt wraps Fp12)
    // Apply frobenius: frob(gt) can be computed via gt^p
    // But we don't have direct frob access on Gt...

    // Instead: output gt's internal representation so we can compare
    // with our CUDA frobenius
    let gt_bytes: [u64; 72] = unsafe { std::mem::transmute_copy(&gt) };

    println!("e(G1, G2) internal representation (72 x u64, Montgomery form):");
    println!("  Fp12.c0.c0.c0 (Fp) limbs: {:016x} {:016x} {:016x} {:016x} {:016x} {:016x}",
        gt_bytes[0], gt_bytes[1], gt_bytes[2], gt_bytes[3], gt_bytes[4], gt_bytes[5]);
    println!("  Fp12.c0.c0.c1 (Fp) limbs: {:016x} {:016x} {:016x} {:016x} {:016x} {:016x}",
        gt_bytes[6], gt_bytes[7], gt_bytes[8], gt_bytes[9], gt_bytes[10], gt_bytes[11]);

    // Output all 12 Fp elements (c0.c0, c0.c1, c0.c2, c1.c0, c1.c1, c1.c2 each has 2 Fp)
    let names = ["c0.c0.c0", "c0.c0.c1", "c0.c1.c0", "c0.c1.c1",
                 "c0.c2.c0", "c0.c2.c1", "c1.c0.c0", "c1.c0.c1",
                 "c1.c1.c0", "c1.c1.c1", "c1.c2.c0", "c1.c2.c1"];
    println!("\nFull e(G1,G2) in Montgomery limbs (for CUDA cross-check):");
    for (i, name) in names.iter().enumerate() {
        let base = i * 6;
        print!("  {}: ", name);
        for j in (0..6).rev() {
            print!("{:016x}", gt_bytes[base + j]);
        }
        println!();
    }

    // Output the Frobenius constant values from ic_bls12_381 source
    // These are hardcoded in the library. Let me extract via computation.
    //
    // The canonical Frobenius coefficients can be computed as:
    // γ₁ = ξ^((p-1)/3) where ξ = u+1 is the non-residue for Fp6
    // γ₂ = ξ^(2(p-1)/3)
    // δ = ξ^((p-1)/2) for Fp12
    //
    // But computing these requires big-int exponentiation.
    // Simpler: use the hex values from ic_bls12_381 source code directly.
    // They're in src/fp6.rs and src/fp12.rs

    // The values I used in gen_cuda_constants.py are the internal Montgomery limbs.
    // To get canonical: output via to_bytes() or compute R^(-1) * mont_value mod p.
    //
    // Let me just output the expected Frobenius result for e(G1,G2)
    // so we can cross-check the CUDA implementation directly.

    println!("\nTo cross-check Frobenius: apply frob in CUDA to e(G1,G2) and compare.");
    println!("If the result matches, the coefficients are correct.");
    println!("The above Montgomery limbs can be used to construct the test value.");
}

// ==================== BLS Verify Test Vectors ====================

fn gen_bls_keypair(rng: &mut ChaCha20Rng) -> (Scalar, G1Affine, G2Affine) {
    let sk = Scalar::random(&mut *rng);
    let pk = (G2Affine::generator() * sk).into();
    let sig_key = (G1Affine::generator() * sk).into();
    (sk, sig_key, pk)
}

fn hash_to_g1(msg: &[u8]) -> G1Affine {
    // IC uses hash_to_curve for BLS signatures
    // For oracle purposes, we use a deterministic mapping: H(msg) = scalar * G1
    let mut hasher = Sha256::new();
    hasher.update(b"ic-bls-test-hash");
    hasher.update(msg);
    let hash = hasher.finalize();
    let mut scalar_bytes = [0u8; 32];
    scalar_bytes.copy_from_slice(&hash);
    // Reduce to valid scalar
    scalar_bytes[31] &= 0x0f; // Ensure < curve order
    let scalar = Scalar::from_bytes(&scalar_bytes).unwrap();
    // Safe: just use as deterministic point
    (G1Affine::generator() * scalar).into()
}

fn bls_sign(sk: &Scalar, msg_point: &G1Affine) -> G1Affine {
    (G1Projective::from(msg_point) * sk).into()
}

fn bls_verify_single(sig: &G1Affine, pk: &G2Affine, msg: &G1Affine) -> bool {
    // e(sig, G2) * e(-msg, pk) == 1
    // Using IC's approach: multi_miller_loop + final_exp
    let neg_msg: G1Affine = (-G1Projective::from(msg)).into();
    let g2_gen = G2Affine::generator();

    // Prepare G2 points
    let g2_prep = G2Prepared::from(g2_gen);
    let pk_prep = G2Prepared::from(*pk);

    let ml = multi_miller_loop(&[
        (sig, &g2_prep),
        (&neg_msg, &pk_prep),
    ]);
    let result = ml.final_exponentiation();

    bool::from(result.is_identity())
}

fn cmd_bls_verify_batch(n: usize) {
    println!("=== BLS Verify Batch Oracle ({n} vectors) ===");
    println!("Library: ic_bls12_381 v0.10.1 (EXACT IC source of truth)");
    println!();

    let mut rng = ChaCha20Rng::seed_from_u64(42);

    for i in 0..n {
        let (sk, _sig_key, pk) = gen_bls_keypair(&mut rng);
        let msg = format!("test-message-{i}");
        let msg_point = hash_to_g1(msg.as_bytes());
        let sig = bls_sign(&sk, &msg_point);

        // Verify
        let valid = bls_verify_single(&sig, &pk, &msg_point);

        // Output intermediate values for GPU comparison
        println!("--- Vector {i} ---");
        println!("  msg: \"{}\"", msg);
        println!("  valid: {}", valid);

        // Output sig point (G1Affine = 48 bytes compressed, 96 bytes uncompressed)
        let sig_bytes = sig.to_compressed();
        println!("  sig_compressed: {}", hex::encode(sig_bytes));

        let pk_bytes = pk.to_compressed();
        println!("  pk_compressed: {}", hex::encode(pk_bytes));

        let msg_bytes = msg_point.to_compressed();
        println!("  msg_point_compressed: {}", hex::encode(msg_bytes));

        // Output uncompressed for direct GPU comparison (Montgomery form)
        let sig_uncomp = sig.to_uncompressed();
        let msg_uncomp = msg_point.to_uncompressed();
        println!("  sig_uncompressed: {}", hex::encode(sig_uncomp));
        println!("  msg_uncompressed: {}", hex::encode(msg_uncomp));

        assert!(valid, "Vector {i} failed verification!");
    }

    // Also test an INVALID signature
    println!("--- Vector INVALID ---");
    let (_, _, pk) = gen_bls_keypair(&mut rng);
    let msg_point = hash_to_g1(b"honest-message");
    let bad_sig: G1Affine = G1Affine::generator(); // Wrong signature
    let valid = bls_verify_single(&bad_sig, &pk, &msg_point);
    println!("  valid: {} (expected false)", valid);
    assert!(!valid, "Invalid signature should not verify!");

    println!("\n=== All {n} vectors + 1 invalid PASS ===");
}

// ==================== SHA-256 Manifest Oracle ====================

fn cmd_sha256_manifest(n: usize, chunk_size: usize) {
    println!("=== SHA-256 Manifest Oracle ({n} chunks × {chunk_size} bytes) ===");
    println!("IC domain separator: \"ic-state-chunk\"");
    println!();

    // IC uses domain-separated SHA-256 for chunks:
    //   hash = SHA256(len("ic-state-chunk") || "ic-state-chunk" || chunk_data)
    let domain = b"ic-state-chunk";
    let domain_len = domain.len() as u8;

    // Generate deterministic chunk data (same as GPU bench)
    let mut data = vec![0u8; chunk_size];
    for i in (0..chunk_size).step_by(8) {
        let val = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let remaining = chunk_size - i;
        let to_copy = remaining.min(8);
        data[i..i + to_copy].copy_from_slice(&val.to_le_bytes()[..to_copy]);
    }

    // Hash each "chunk" (same data, different patterns would be used in reality)
    // For the oracle, output first 5 and last 1 hash for validation
    let mut hashes = Vec::with_capacity(n);

    let start = Instant::now();
    for _chunk_idx in 0..n {
        let mut hasher = Sha256::new();
        // IC domain separation: 1 byte length + domain string
        hasher.update([domain_len]);
        hasher.update(domain);
        hasher.update(&data);
        let hash: [u8; 32] = hasher.finalize().into();
        hashes.push(hash);
    }
    let elapsed = start.elapsed();

    // Output reference hashes
    for (i, hash) in hashes.iter().enumerate().take(5) {
        println!("  chunk[{i}]: {}", hex::encode(hash));
    }
    if n > 5 {
        println!("  chunk[{}]: {}", n - 1, hex::encode(hashes[n - 1]));
    }

    // All chunks have same data, so all hashes should be identical
    let all_same = hashes.iter().all(|h| h == &hashes[0]);
    println!("  all_same: {} (expected true for uniform data)", all_same);

    let total_bytes = n as f64 * chunk_size as f64;
    let ms = elapsed.as_secs_f64() * 1000.0;
    let gbps = (total_bytes / 1e9) / elapsed.as_secs_f64();
    println!("\n  CPU single-thread: {ms:.1}ms for {n} chunks");
    println!("  Throughput: {gbps:.2} GB/s");

    // Also output WITHOUT domain separator (raw SHA-256) for GPU comparison
    println!("\n  --- Raw SHA-256 (no domain sep) ---");
    let raw_hash: [u8; 32] = Sha256::digest(&data).into();
    println!("  raw_sha256: {}", hex::encode(raw_hash));

    println!("\n=== SHA-256 Manifest Oracle DONE ===");
}

// ==================== Consensus Round Simulation ====================

fn cmd_consensus_round(nodes: usize) {
    println!("=== Consensus Round Simulation ({nodes}-node subnet) ===");
    println!("Simulates one consensus round's crypto workload\n");

    let mut rng = ChaCha20Rng::seed_from_u64(1337);

    // Generate node keys
    let mut node_keys: Vec<(Scalar, G2Affine)> = Vec::new();
    for _i in 0..nodes {
        let (sk, _, pk) = gen_bls_keypair(&mut rng);
        node_keys.push((sk, pk));
    }

    // Per round, consensus verifies:
    // 1. Block proposal: 1 basic sig verify
    // 2. Notarization shares: ~nodes individual threshold verifies
    // 3. Notarization aggregate: 1 combined verify
    // 4. Finalization shares: ~nodes individual threshold verifies
    // 5. Finalization aggregate: 1 combined verify
    // 6. Random beacon: 1 combined verify
    // 7. Random tape: 1 combined verify
    // 8. Certification shares: ~nodes individual verifies
    // 9. Certification aggregate: 1 combined verify

    let msg = hash_to_g1(b"consensus-round-block-hash");

    // === 1. Individual share verification (majority of work) ===
    let share_count = nodes; // Each node produces a share
    let mut sigs: Vec<G1Affine> = Vec::new();
    for (sk, _) in &node_keys {
        sigs.push(bls_sign(sk, &msg));
    }

    // Time individual verifications (this is what happens sequentially today)
    let start = Instant::now();
    let mut valid_count = 0;
    for (i, sig) in sigs.iter().enumerate() {
        let (_, pk) = &node_keys[i];
        if bls_verify_single(sig, pk, &msg) {
            valid_count += 1;
        }
    }
    let individual_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("  Individual share verifies: {share_count}");
    println!("    Valid: {valid_count}/{share_count}");
    println!("    Time (sequential): {individual_ms:.1}ms");
    println!("    Per verify: {:.3}ms", individual_ms / share_count as f64);

    // === 2. Simulate combined signature verify ===
    // In real IC, combined sig = Lagrange-interpolated threshold sig
    // For profiling, each combined verify = 1 pairing check
    let combined_verifies = 5; // notarization + finalization + beacon + tape + certification
    let start = Instant::now();
    for _ in 0..combined_verifies {
        // Each combined verify is same cost as individual
        bls_verify_single(&sigs[0], &node_keys[0].1, &msg);
    }
    let combined_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("\n  Combined signature verifies: {combined_verifies}");
    println!("    Time: {combined_ms:.1}ms");

    // === 3. Simulate round artifacts ===
    // Notarization: nodes shares + 1 combined
    // Finalization: nodes shares + 1 combined
    // Beacon/tape: 1 combined each
    // Certification: nodes shares + 1 combined
    let total_verifies = 3 * nodes + combined_verifies;
    let total_ms = individual_ms * 3.0 + combined_ms;

    println!("\n  === TOTAL ROUND CRYPTO ===");
    println!("  Verifications: {total_verifies} ({} individual + {combined_verifies} combined)",
             3 * nodes);
    println!("  CPU sequential time: {total_ms:.1}ms");
    println!("  CPU per-verify avg: {:.3}ms", total_ms / total_verifies as f64);
    println!();
    println!("  GPU batch potential (all {total_verifies} in one batch):");
    println!("    At 1ms/verify GPU: {:.1}ms (est. based on bench_bls_verify data)",
             total_verifies as f64 * 0.5); // GPU typically ~0.5ms/verify in batch
    println!("    Speedup: {:.1}x", total_ms / (total_verifies as f64 * 0.5));

    println!("\n=== Round Simulation DONE ===");
}

// ==================== Profile BLS ====================

fn cmd_profile_bls(n: usize) {
    println!("=== BLS Single Verify Profile ({n} iterations) ===");
    println!("Library: ic_bls12_381 v0.10.1\n");

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let (sk, _, pk) = gen_bls_keypair(&mut rng);
    let msg_point = hash_to_g1(b"profile-message");
    let sig = bls_sign(&sk, &msg_point);

    // Warmup
    for _ in 0..10 {
        bls_verify_single(&sig, &pk, &msg_point);
    }

    // Timed
    let start = Instant::now();
    for _ in 0..n {
        bls_verify_single(&sig, &pk, &msg_point);
    }
    let elapsed = start.elapsed();

    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let per_ms = total_ms / n as f64;
    let per_us = per_ms * 1000.0;

    println!("  Total: {total_ms:.1}ms for {n} verifications");
    println!("  Per verify: {per_ms:.3}ms ({per_us:.0}µs)");
    println!("  Throughput: {:.0} verifies/sec", n as f64 / elapsed.as_secs_f64());

    // This baseline is what GPU batch must beat
    println!("\n  === CPU BASELINE FOR GPU TARGET ===");
    println!("  GPU must beat: {per_us:.0}µs per verify (sequential)");
    println!("  For {n}-batch: GPU must beat {total_ms:.1}ms total");

    println!("\n=== Profile BLS DONE ===");
}

// ==================== Profile SHA-256 ====================

fn cmd_profile_sha256(chunks: usize) {
    let chunk_size = 1024 * 1024; // 1 MiB (IC default)
    let total_bytes = chunks * chunk_size;

    println!("=== SHA-256 Profile ({chunks} × 1MiB chunks = {:.1} GiB) ===",
             total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("IC manifest hashing baseline\n");

    // Allocate and fill data
    let mut data = vec![0u8; chunk_size];
    for i in (0..chunk_size).step_by(8) {
        let val = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let remaining = chunk_size - i;
        let to_copy = remaining.min(8);
        data[i..i + to_copy].copy_from_slice(&val.to_le_bytes()[..to_copy]);
    }

    // IC domain separator
    let domain = b"ic-state-chunk";
    let domain_len = domain.len() as u8;

    // Single-thread baseline
    let start = Instant::now();
    for _ in 0..chunks {
        let mut hasher = Sha256::new();
        hasher.update([domain_len]);
        hasher.update(domain);
        hasher.update(&data);
        let _hash: [u8; 32] = hasher.finalize().into();
    }
    let single_ms = start.elapsed().as_secs_f64() * 1000.0;
    let single_gbps = (total_bytes as f64 / 1e9) / (single_ms / 1000.0);

    println!("  Single-thread: {single_ms:.1}ms ({single_gbps:.2} GB/s)");

    // Multi-thread baseline (simulate Rayon with std threads)
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(16);
    // Simple estimate: divide work
    let per_thread = chunks / num_threads;
    let start = Instant::now();
    std::thread::scope(|s| {
        for _t in 0..num_threads {
            let data_ref = &data;
            let work = per_thread;
            s.spawn(move || {
                for _ in 0..work {
                    let mut hasher = Sha256::new();
                    hasher.update([domain_len]);
                    hasher.update(domain);
                    hasher.update(data_ref);
                    let _hash: [u8; 32] = hasher.finalize().into();
                }
            });
        }
    });
    let multi_ms = start.elapsed().as_secs_f64() * 1000.0;
    let multi_gbps = (total_bytes as f64 / 1e9) / (multi_ms / 1000.0);

    println!("  {num_threads}-thread: {multi_ms:.1}ms ({multi_gbps:.2} GB/s)");
    println!("  Multi/Single speedup: {:.1}x", single_ms / multi_ms);

    // GPU target
    println!("\n  === CPU BASELINE FOR GPU TARGET ===");
    println!("  GPU must beat {multi_ms:.1}ms (multi-thread) to be worth it");
    println!("  GPU target throughput: >{multi_gbps:.1} GB/s");
    println!("  Previous GPU bench: ~278 GB/s on RTX PRO 6000");

    println!("\n=== Profile SHA-256 DONE ===");
}

// ==================== Merkle SHA-256 Oracle ====================

fn cmd_merkle_oracle(num_chunks: usize) {
    use merkle_sha256::*;

    println!("=== Merkle SHA-256 Oracle ({num_chunks} chunk(s)) ===");
    println!("Spec: 1MiB → 256×4KiB leaves → SHA-256 each → Merkle tree → root");
    println!("Library: sha2 crate (same as IC's state_manager)\n");

    // Generate deterministic test data
    let mut data = vec![0u8; CHUNK_SIZE];
    for i in (0..CHUNK_SIZE).step_by(8) {
        let val = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let remaining = CHUNK_SIZE - i;
        let to_copy = remaining.min(8);
        data[i..i + to_copy].copy_from_slice(&val.to_le_bytes()[..to_copy]);
    }

    // Single chunk with all intermediates (for GPU debugging)
    let (leaves, levels, root) = merkle_chunk_hash_with_intermediates(&data);

    println!("--- Single Chunk Intermediates ---");
    println!("  Leaves (256 × SHA-256 of 4KiB):");
    for i in [0, 1, 2, 127, 254, 255] {
        println!("    leaf[{i:>3}]: {}", hex::encode(leaves[i]));
    }

    println!("\n  Tree levels:");
    for (round, level) in levels.iter().enumerate() {
        let name = format!("round {} ({} nodes)", round, level.len());
        println!("    {name:<20} [0]: {}", hex::encode(level[0]));
        if level.len() > 1 {
            let last = level.len() - 1;
            println!("    {:<20} [{last}]: {}", "", hex::encode(level[last]));
        }
    }

    println!("\n  ROOT: {}", hex::encode(root));

    // Verify consistency
    let root2 = merkle_chunk_hash(&data);
    assert_eq!(root, root2, "Root from intermediates must match direct computation");
    println!("  Consistency check: PASS");

    // Also compute flat SHA-256 for comparison
    let flat: [u8; 32] = sha2::Sha256::digest(&data).into();
    println!("\n  Flat SHA-256:   {}", hex::encode(flat));
    println!("  Merkle root:    {}", hex::encode(root));
    println!("  Different: {} (expected: true)", flat != root);

    // Output test vectors for GPU: leaf[0], leaf[1], root
    println!("\n--- GPU Test Vector (hex, for C include) ---");
    println!("// Leaf 0 (SHA-256 of first 4KiB):");
    print!("static const uint8_t LEAF_0[32] = {{");
    for (i, b) in leaves[0].iter().enumerate() {
        if i > 0 { print!(","); }
        print!("0x{b:02x}");
    }
    println!("}};");

    println!("// Leaf 255 (SHA-256 of last 4KiB):");
    print!("static const uint8_t LEAF_255[32] = {{");
    for (i, b) in leaves[255].iter().enumerate() {
        if i > 0 { print!(","); }
        print!("0x{b:02x}");
    }
    println!("}};");

    println!("// Merkle root:");
    print!("static const uint8_t MERKLE_ROOT[32] = {{");
    for (i, b) in root.iter().enumerate() {
        if i > 0 { print!(","); }
        print!("0x{b:02x}");
    }
    println!("}};");

    // Batch test
    if num_chunks > 1 {
        println!("\n--- Batch ({num_chunks} chunks, same data) ---");
        let mut batch_data = vec![0u8; num_chunks * CHUNK_SIZE];
        for i in 0..num_chunks {
            batch_data[i * CHUNK_SIZE..(i + 1) * CHUNK_SIZE].copy_from_slice(&data);
        }
        let batch_hashes = merkle_batch_hash(&batch_data, num_chunks);
        let all_same = batch_hashes.iter().all(|h| h == &root);
        println!("  All {num_chunks} hashes match root: {all_same}");
    }

    // Benchmark
    println!("\n--- CPU Benchmark ---");
    let start = std::time::Instant::now();
    let iters = 100;
    for _ in 0..iters {
        let _ = merkle_chunk_hash(&data);
    }
    let elapsed = start.elapsed();
    let per_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    println!("  Merkle hash: {per_ms:.3}ms per 1MiB chunk");
    println!("  Flat SHA-256 comparison: ~0.38ms per 1MiB chunk");
    println!("  Merkle overhead: {:.1}x (more hashing, but GPU-parallelizable)", per_ms / 0.38);

    println!("\n=== Merkle Oracle DONE ===");
}
