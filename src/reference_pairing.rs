/// Extract easy part result from ic_bls12_381

use ic_bls12_381::*;

fn main() {
    let g1 = G1Affine::generator();
    let g2p = G2Prepared::from(G2Affine::generator());
    let ml = multi_miller_loop(&[(&g1, &g2p)]);
    let ml_raw: &[u64; 72] = unsafe { std::mem::transmute(&ml) };

    // Manually compute easy part using Fp12 operations accessible via MillerLoopResult
    // MillerLoopResult addition is Fp12 multiplication
    // We need: f^(p^6-1) * f^(p^2+1)
    // f^(p^6) = conjugate, accessible via negating MillerLoopResult? No.
    // Actually we can't easily extract intermediate values from ic_bls12_381.

    // Instead, let's just output the miller loop raw value and the final result.
    // We know miller loop matches. If we can narrow down where the hard part diverges,
    // that's enough.

    // Output both
    println!("Miller loop (input to final exp):");
    for i in 0..12 {
        let base = i * 6;
        println!("  [{:016x},{:016x},{:016x},{:016x},{:016x},{:016x}]",
                 ml_raw[base],ml_raw[base+1],ml_raw[base+2],
                 ml_raw[base+3],ml_raw[base+4],ml_raw[base+5]);
    }

    let full = ml.final_exponentiation();
    let full_raw: &[u64; 72] = unsafe { std::mem::transmute(&full) };
    println!("\nFull pairing (output):");
    for i in 0..12 {
        let base = i * 6;
        println!("  [{:016x},{:016x},{:016x},{:016x},{:016x},{:016x}]",
                 full_raw[base],full_raw[base+1],full_raw[base+2],
                 full_raw[base+3],full_raw[base+4],full_raw[base+5]);
    }
}
