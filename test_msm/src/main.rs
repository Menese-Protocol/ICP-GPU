// Oracle test: GPU G1 point decompression vs ic_bls12_381
use ic_bls12_381::*;
use rand::RngCore;
use std::time::Instant;

extern "C" {
    fn gpu_g1_decompress(compressed: *const u8, out_point: *mut u8) -> i32;
    fn gpu_g1_decompress_batch(compressed: *const u8, out_points: *mut u8, n: i32) -> i32;
}

fn raw<T>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn main() {
    println!("=== GPU G1 Decompression Oracle Test ===\n");

    let mut rng = rand::thread_rng();

    // Generate test points
    let gen = G1Affine::generator();
    let test_points: Vec<G1Affine> = (0..100).map(|i| {
        if i == 0 { return gen; }
        let s = Scalar::from_bytes_wide(&{ let mut b = [0u8; 64]; rng.fill_bytes(&mut b); b });
        (G1Affine::generator() * s).into()
    }).collect();

    // Unit 1: Single point (generator)
    println!("--- Unit 1: Decompress generator ---");
    let compressed = gen.to_compressed();
    let mut gpu_result = [0u8; 96];
    let ret = unsafe { gpu_g1_decompress(compressed.as_ptr(), gpu_result.as_mut_ptr()) };
    println!("  GPU ret: {ret}");

    let cpu_bytes = &raw(&gen)[..96];
    let match1 = &gpu_result[..96] == cpu_bytes;
    println!("  x match: {}", if &gpu_result[..48] == &cpu_bytes[..48] { "✓" } else { "✗" });
    println!("  y match: {}", if &gpu_result[48..96] == &cpu_bytes[48..96] { "✓" } else { "✗" });

    // Check if GPU y = -CPU y (negation = p - y, which in Montgomery = negate all limbs mod p)
    // If so, the sqrt is correct but sign is flipped
    let gpu_y_neg: G1Affine = {
        // Negate GPU y: compute -y via G1 negation
        let mut fake_pt_bytes = [0u8; 104];
        fake_pt_bytes[..96].copy_from_slice(&gpu_result[..96]);
        let fake_pt: G1Affine = unsafe { std::ptr::read(fake_pt_bytes.as_ptr() as *const G1Affine) };
        let neg_pt = -G1Projective::from(&fake_pt);
        let neg_aff = G1Affine::from(neg_pt);
        neg_aff
    };
    let neg_y_bytes = &raw(&gpu_y_neg)[48..96];
    let y_negated_match = neg_y_bytes == &cpu_bytes[48..96];
    println!("  -GPU_y matches CPU_y: {}", if y_negated_match { "✓ (sign flip)" } else { "✗" });
    println!("  full match: {}", if match1 { "✓" } else { "✗" });

    // Unit 2: Multiple known points
    println!("\n--- Unit 2: 10 random points ---");
    let mut all_pass = true;
    for (i, pt) in test_points[..10].iter().enumerate() {
        let compressed = pt.to_compressed();
        let mut gpu_out = [0u8; 96];
        unsafe { gpu_g1_decompress(compressed.as_ptr(), gpu_out.as_mut_ptr()) };
        let cpu_bytes = &raw(pt)[..96];
        let ok = &gpu_out[..96] == cpu_bytes;
        if !ok {
            println!("  Point {i}: ✗");
            println!("    GPU x[0..8]: {:02x?}", &gpu_out[..8]);
            println!("    CPU x[0..8]: {:02x?}", &cpu_bytes[..8]);
            all_pass = false;
        }
    }
    println!("  10/10 match: {}", if all_pass { "✓" } else { "✗" });

    // Unit 3: Batch decompression
    println!("\n--- Unit 3: Batch decompression (100 points) ---");
    let mut all_compressed = Vec::with_capacity(100 * 48);
    for pt in &test_points {
        all_compressed.extend_from_slice(&pt.to_compressed());
    }
    let mut all_gpu_out = vec![0u8; 100 * 96];
    let t0 = Instant::now();
    let ret = unsafe {
        gpu_g1_decompress_batch(all_compressed.as_ptr(), all_gpu_out.as_mut_ptr(), 100)
    };
    let gpu_ms = t0.elapsed().as_micros() as f64 / 1000.0;
    println!("  GPU batch ret: {ret}, time: {gpu_ms:.3}ms");

    let t1 = Instant::now();
    let mut cpu_mismatches = 0;
    for (i, pt) in test_points.iter().enumerate() {
        let cpu_bytes = &raw(pt)[..96];
        let gpu_bytes = &all_gpu_out[i * 96..(i + 1) * 96];
        if gpu_bytes != cpu_bytes {
            cpu_mismatches += 1;
            if cpu_mismatches <= 3 {
                println!("  Mismatch at {i}: GPU x[0..8]={:02x?} CPU x[0..8]={:02x?}",
                    &gpu_bytes[..8], &cpu_bytes[..8]);
            }
        }
    }
    println!("  Matches: {}/100 {}", 100 - cpu_mismatches,
        if cpu_mismatches == 0 { "✓" } else { "✗" });

    // Unit 4: Benchmark at DKG-relevant sizes
    println!("\n--- Unit 4: Benchmark ---");
    for n in [28, 100, 448, 1000, 5000] {
        let points: Vec<G1Affine> = (0..n).map(|_| {
            let s = Scalar::from_bytes_wide(&{ let mut b = [0u8; 64]; rng.fill_bytes(&mut b); b });
            (G1Affine::generator() * s).into()
        }).collect();

        // Compress all
        let mut compressed = Vec::with_capacity(n * 48);
        for pt in &points { compressed.extend_from_slice(&pt.to_compressed()); }

        // CPU decompress
        let t0 = Instant::now();
        let cpu_results: Vec<G1Affine> = (0..n).map(|i| {
            let mut buf = [0u8; 48];
            buf.copy_from_slice(&compressed[i*48..(i+1)*48]);
            G1Affine::from_compressed(&buf).unwrap()
        }).collect();
        let cpu_ms = t0.elapsed().as_micros() as f64 / 1000.0;

        // GPU decompress
        let mut gpu_out = vec![0u8; n * 96];
        let t1 = Instant::now();
        unsafe { gpu_g1_decompress_batch(compressed.as_ptr(), gpu_out.as_mut_ptr(), n as i32) };
        let gpu_ms = t1.elapsed().as_micros() as f64 / 1000.0;

        // Verify
        let mut ok = true;
        for (i, pt) in cpu_results.iter().enumerate() {
            if &gpu_out[i*96..(i+1)*96] != &raw(pt)[..96] { ok = false; break; }
        }

        let sp = if gpu_ms > 0.001 { cpu_ms / gpu_ms } else { 0.0 };
        println!("  n={n:5}: CPU={cpu_ms:8.2}ms GPU={gpu_ms:8.2}ms {sp:6.1}x {}",
            if ok { "✓" } else { "✗" });
    }

    println!("\n=== Done ===");
}
