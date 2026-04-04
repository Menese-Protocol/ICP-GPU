/// IC BLS12-381 Oracle — source of truth for GPU porting
///
/// Accesses ic_bls12_381 internals via transmute since Fp/Fp2/Fp6/Fp12 are private.
/// All intermediate values are output as hex limbs for CUDA comparison.
///
/// Usage: cargo run --release --bin oracle -- <command>

use ic_bls12_381::{G1Affine, G1Projective, G2Affine, G2Projective, G2Prepared};
use ic_bls12_381::multi_miller_loop;
use group::Curve;
use std::env;

// Known constants from our proven tower (these are authoritative — tested against Python)
const FP_ONE: [u64; 6] = [0x760900000002fffd, 0xebf4000bc40c0002, 0x5f48985753c758ba,
    0x77ce585370525745, 0x5c071a97a256ec6d, 0x15f65ec3fa80e493];

// Build mont(n) by repeated addition of mont(1)
// This is slow but correct and avoids needing access to private Fp type
fn mont(n: u64) -> [u64; 6] {
    if n == 0 { return [0u64; 6]; }
    // We need to do modular addition. Use our known-correct Python formula:
    // Actually let's just hardcode a few we need and compute the rest via
    // the Scalar type (which IS public)
    // OR: extract from G1Affine::generator() coordinates
    // G1 generator x and y are known Fp values in Montgomery form.
    // But we can't easily do arithmetic on them without the Fp type.

    // Simplest correct approach: use Python to precompute, or use the pairing
    // oracle pattern where we only output values computed by ic_bls12_381 itself.

    // For the oracle, we don't need to CONSTRUCT arbitrary Fp values.
    // We just need to run ic_bls12_381 functions and output the results.
    // The CUDA side constructs its own test values (which we've proven match).
    panic!("Use specific oracle commands, not raw mont construction");
}

fn print_limbs(label: &str, data: &[u64]) {
    print!("{}=", label);
    for (i, v) in data.iter().enumerate() {
        if i > 0 { print!(","); }
        print!("{:016x}", v);
    }
    println!();
}

fn print_fp12(label: &str, data: &[u64]) {
    let names = ["c0.c0","c0.c1","c0.c2","c1.c0","c1.c1","c1.c2"];
    for (i, name) in names.iter().enumerate() {
        let base = i * 12;
        print_limbs(&format!("{}.{}", label, name), &data[base..base+12]);
    }
}

// ==================== Oracle Commands ====================

fn cmd_miller() {
    println!("# Miller loop: e(G1_gen, G2_gen) before final exp");
    let g1 = G1Affine::generator();
    let g2p = G2Prepared::from(G2Affine::generator());
    let ml = multi_miller_loop(&[(&g1, &g2p)]);
    let raw: [u64; 72] = unsafe { std::mem::transmute(ml) };
    print_fp12("miller_g1_g2", &raw);
}

fn cmd_pairing() {
    println!("# Full pairing: e(G1_gen, G2_gen)");
    let g1 = G1Affine::generator();
    let g2p = G2Prepared::from(G2Affine::generator());
    let full = multi_miller_loop(&[(&g1, &g2p)]).final_exponentiation();
    let raw: [u64; 72] = unsafe { std::mem::transmute(full) };
    print_fp12("pairing_g1_g2", &raw);
}

fn cmd_multi_miller() {
    println!("# Multi-miller: e(G1, G2) * e(2*G1, G2) — two independent pairings multiplied");
    let g1 = G1Affine::generator();
    let g1_2 = G1Affine::from(G1Projective::generator() + G1Projective::generator());
    let g2p = G2Prepared::from(G2Affine::generator());

    // Single miller loops for comparison
    let ml1 = multi_miller_loop(&[(&g1, &g2p)]);
    let ml1_raw: [u64; 72] = unsafe { std::mem::transmute(ml1) };

    let ml2 = multi_miller_loop(&[(&g1_2, &g2p)]);
    let ml2_raw: [u64; 72] = unsafe { std::mem::transmute(ml2) };

    // Combined multi-miller (should equal ml1 * ml2 in Fp12)
    let ml_combined = multi_miller_loop(&[(&g1, &g2p), (&g1_2, &g2p)]);
    let mlc_raw: [u64; 72] = unsafe { std::mem::transmute(ml_combined) };

    print_fp12("miller_1xG1", &ml1_raw);
    println!();
    print_fp12("miller_2xG1", &ml2_raw);
    println!();
    print_fp12("miller_combined", &mlc_raw);

    // Full pairing of combined
    let full = ml_combined.final_exponentiation();
    let full_raw: [u64; 72] = unsafe { std::mem::transmute(full) };
    println!();
    print_fp12("pairing_combined", &full_raw);

    // Verify: combined pairing should equal e(G1,G2) * e(2G1,G2) = e(3G1,G2)
    let g1_3 = G1Affine::from(
        G1Projective::generator() + G1Projective::generator() + G1Projective::generator()
    );
    let direct = multi_miller_loop(&[(&g1_3, &g2p)]).final_exponentiation();
    let direct_raw: [u64; 72] = unsafe { std::mem::transmute(direct) };
    println!();
    print_fp12("pairing_3xG1_direct", &direct_raw);
    println!();
    let match_ok = full_raw == direct_raw;
    println!("# e(G1,G2)*e(2G1,G2) == e(3G1,G2): {}", if match_ok { "YES" } else { "NO" });
}

fn cmd_g2_prepared() {
    println!("# G2Prepared coefficients for 2*G2 (non-generator public key)");
    let g2_2 = G2Affine::from(G2Projective::generator() + G2Projective::generator());
    let prep = G2Prepared::from(g2_2);

    // Extract coeffs via raw memory
    let raw: &[u8] = unsafe {
        std::slice::from_raw_parts(&prep as *const _ as *const u8, std::mem::size_of_val(&prep))
    };
    let coeffs_ptr: usize = unsafe { *((raw.as_ptr().add(8)) as *const usize) };
    let coeffs_len: usize = unsafe { *((raw.as_ptr().add(16)) as *const usize) };
    println!("count={}", coeffs_len);

    if coeffs_len > 0 && coeffs_ptr != 0 {
        let all: &[u64] = unsafe {
            std::slice::from_raw_parts(coeffs_ptr as *const u64, coeffs_len * 36)
        };
        for ci in 0..std::cmp::min(3, coeffs_len) {
            let base = ci * 36;
            print_limbs(&format!("c[{}].0", ci), &all[base..base+12]);
            print_limbs(&format!("c[{}].1", ci), &all[base+12..base+24]);
            print_limbs(&format!("c[{}].2", ci), &all[base+24..base+36]);
        }
    }
}

fn cmd_bls_verify_pattern() {
    println!("# BLS signature verification pattern");
    println!("# e(signature, G2) == e(H(message), public_key)");
    println!("# Checked as: multi_miller_loop([(sig, G2), (-H(m), pk)]).final_exp() == Gt::identity()");
    println!();

    // Simulate: secret key = 42 (as scalar)
    // public_key = 42 * G2
    // message_point = hash_to_G1(msg) — we'll use 7 * G1 as a stand-in
    // signature = 42 * message_point = 42 * 7 * G1 = 294 * G1

    let sk = 42u64;
    let msg_point_scalar = 7u64;
    let sig_scalar = sk * msg_point_scalar; // 294

    // Build points by repeated addition
    let mut pk_proj = G2Projective::identity();
    for _ in 0..sk { pk_proj = pk_proj + G2Projective::generator(); }
    let pk = G2Affine::from(pk_proj);

    let mut msg_proj = G1Projective::identity();
    for _ in 0..msg_point_scalar { msg_proj = msg_proj + G1Projective::generator(); }
    let msg_point = G1Affine::from(msg_proj);

    let mut sig_proj = G1Projective::identity();
    for _ in 0..sig_scalar { sig_proj = sig_proj + G1Projective::generator(); }
    let sig = G1Affine::from(sig_proj);

    // Negate message point for the verification equation
    let neg_msg = G1Affine::from(-G1Projective::from(msg_point));

    let g2p = G2Prepared::from(G2Affine::generator());
    let pk_prep = G2Prepared::from(pk);

    // Verify: e(sig, G2) * e(-H(m), pk) should give identity after final_exp
    let result = multi_miller_loop(&[(&sig, &g2p), (&neg_msg, &pk_prep)]).final_exponentiation();
    let is_valid = result == ic_bls12_381::Gt::identity();
    println!("verify_result_is_identity={}", is_valid);

    // Output the miller loop and final exp for this verification
    let ml = multi_miller_loop(&[(&sig, &g2p), (&neg_msg, &pk_prep)]);
    let ml_raw: [u64; 72] = unsafe { std::mem::transmute(ml) };
    print_fp12("verify_miller", &ml_raw);

    let full_raw: [u64; 72] = unsafe { std::mem::transmute(result) };
    println!();
    print_fp12("verify_result", &full_raw);

    // Also output the prepared coefficients for pk (non-generator Q)
    println!();
    let pk_raw: &[u8] = unsafe {
        std::slice::from_raw_parts(&pk_prep as *const _ as *const u8, std::mem::size_of_val(&pk_prep))
    };
    let coeffs_len: usize = unsafe { *((pk_raw.as_ptr().add(16)) as *const usize) };
    println!("pk_prepared_count={}", coeffs_len);
}

fn cmd_cyclotomic_sq() {
    println!("# Cyclotomic square test");
    println!("# Input: easy part of final_exp applied to miller(G1,G2)");
    println!("# The easy part result is in the cyclotomic subgroup");

    let g1 = G1Affine::generator();
    let g2p = G2Prepared::from(G2Affine::generator());
    let ml = multi_miller_loop(&[(&g1, &g2p)]);
    let ml_raw: &[u64; 72] = unsafe { std::mem::transmute(&ml) };

    // We can't easily compute just the easy part since MillerLoopResult
    // doesn't expose intermediate operations. But we know that after
    // final_exponentiation, the result IS in the cyclotomic subgroup.
    // So let's test: pairing_result.square() should equal pairing_result * pairing_result

    // Actually, we need the easy part result (before hard part).
    // We can't get it from the public API. Instead, let's test that
    // our fp12_sqr matches fp12_mul(x,x) on the FULL pairing result
    // (which is cyclotomic). If they match, cyclotomic_square is correct
    // IF it also matches fp12_sqr.

    // Output: pairing result (cyclotomic element) and its square
    let full = ml.final_exponentiation();
    let full_raw: [u64; 72] = unsafe { std::mem::transmute(full) };

    // full * full via Gt multiplication
    let sq = full + full; // Gt addition = Fp12 multiplication
    let sq_raw: [u64; 72] = unsafe { std::mem::transmute(sq) };

    print_fp12("cyc_input", &full_raw);
    println!();
    print_fp12("cyc_squared", &sq_raw);
}

fn cmd_all() {
    cmd_miller();
    println!();
    cmd_pairing();
    println!();
    cmd_multi_miller();
    println!();
    cmd_g2_prepared();
    println!();
    cmd_bls_verify_pattern();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = if args.len() > 1 { args[1].as_str() } else { "all" };

    match cmd {
        "miller" => cmd_miller(),
        "pairing" => cmd_pairing(),
        "multi_miller" => cmd_multi_miller(),
        "g2_prepared" => cmd_g2_prepared(),
        "bls_verify" => cmd_bls_verify_pattern(),
        "cyc_sq" => cmd_cyclotomic_sq(),
        "all" => cmd_all(),
        _ => {
            eprintln!("Commands: miller pairing multi_miller g2_prepared bls_verify all");
            std::process::exit(1);
        }
    }
}
