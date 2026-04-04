/// CPU Baseline + Reference Value Extraction
/// Outputs raw Fp12 pairing result from blst for GPU comparison

use std::time::Instant;

// Use blst's low-level FFI to get raw pairing Fp12
extern "C" {
    fn blst_p1_affine_generator() -> *const blst::blst_p1_affine;
    fn blst_p2_affine_generator() -> *const blst::blst_p2_affine;
    fn blst_miller_loop(ret: *mut blst::blst_fp12, q: *const blst::blst_p2_affine, p: *const blst::blst_p1_affine);
    fn blst_final_exp(ret: *mut blst::blst_fp12, f: *const blst::blst_fp12);
    fn blst_fp12_is_one(a: *const blst::blst_fp12) -> bool;
}

fn print_fp12_limbs(label: &str, fp12: &blst::blst_fp12) {
    // blst_fp12 is { fp6[2] } where fp6 is { fp2[3] } where fp2 is { fp[2] } where fp is { limb_t[6] }
    // Access via raw bytes
    let bytes: &[u64] = unsafe {
        std::slice::from_raw_parts(fp12 as *const _ as *const u64, 72) // 12 Fp elements * 6 limbs = 72
    };

    println!("{}:", label);
    let names = ["c0.c0.c0", "c0.c0.c1", "c0.c1.c0", "c0.c1.c1", "c0.c2.c0", "c0.c2.c1",
                 "c1.c0.c0", "c1.c0.c1", "c1.c1.c0", "c1.c1.c1", "c1.c2.c0", "c1.c2.c1"];
    for (i, name) in names.iter().enumerate() {
        let base = i * 6;
        println!("  {} = [{:016x}, {:016x}, {:016x}, {:016x}, {:016x}, {:016x}]",
                 name,
                 bytes[base], bytes[base+1], bytes[base+2],
                 bytes[base+3], bytes[base+4], bytes[base+5]);
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  blst Reference: Raw Pairing Fp12 for GPU Comparison   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    unsafe {
        let g1 = blst_p1_affine_generator();
        let g2 = blst_p2_affine_generator();

        // Step 1: Miller loop
        let mut miller_result: blst::blst_fp12 = std::mem::zeroed();
        blst_miller_loop(&mut miller_result, g2, g1);

        print_fp12_limbs("MILLER_LOOP e(G1,G2)", &miller_result);
        println!();

        // Step 2: Final exponentiation
        let mut pairing_result: blst::blst_fp12 = std::mem::zeroed();
        blst_final_exp(&mut pairing_result, &miller_result);

        print_fp12_limbs("FULL_PAIRING e(G1,G2)", &pairing_result);
        println!();

        // Verify it's not 1 (generator pairing should not be trivial)
        let is_one = blst_fp12_is_one(&pairing_result);
        println!("Is ONE: {} (expect: false)", is_one);

        // Benchmark
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            blst_miller_loop(&mut miller_result, g2, g1);
        }
        let miller_elapsed = start.elapsed();
        println!("\nMiller loop:     {:.1}µs/op", miller_elapsed.as_micros() as f64 / iterations as f64);

        let start = Instant::now();
        for _ in 0..iterations {
            blst_final_exp(&mut pairing_result, &miller_result);
        }
        let fexp_elapsed = start.elapsed();
        println!("Final exp:       {:.1}µs/op", fexp_elapsed.as_micros() as f64 / iterations as f64);

        let start = Instant::now();
        for _ in 0..iterations {
            blst_miller_loop(&mut miller_result, g2, g1);
            blst_final_exp(&mut pairing_result, &miller_result);
        }
        let total_elapsed = start.elapsed();
        println!("Full pairing:    {:.1}µs/op", total_elapsed.as_micros() as f64 / iterations as f64);
    }
}
