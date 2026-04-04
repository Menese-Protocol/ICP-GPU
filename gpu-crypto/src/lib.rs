//! GPU-accelerated BLS12-381 cryptography for IC consensus
//!
//! Drop-in acceleration for ic_bls12_381 pairing operations.
//! All results are bit-exact with ic_bls12_381 v0.10.1.

use ic_bls12_381::{G1Affine, G2Affine, G2Prepared, Gt};
use ic_bls12_381::multi_miller_loop;

// ==================== FFI bindings ====================
extern "C" {
    fn gpu_crypto_init() -> i32;
    fn gpu_batch_sha256(
        chunks: *const u8,
        hashes: *mut u8,
        n: i32,
        chunk_size: i32,
    ) -> i32;
    fn gpu_batch_bls_verify(
        sigs: *const u64,       // M × 12 u64
        neg_hms: *const u64,    // M × 12 u64
        g2_coeffs: *const u64,  // 2448 u64
        pk_coeffs: *const u64,  // M × 2448 u64
        results: *mut i32,      // M × 1
        m: i32,
    ) -> i32;
}

// ==================== Constants ====================
const G2_COEFF_U64S: usize = 68 * 36; // 2448
const G1_U64S: usize = 12; // 6 limbs × 2 (x, y)

// ==================== Public API ====================

/// Initialize GPU. Call once at startup.
/// Returns Ok(()) if GPU is available, Err if not.
pub fn init() -> Result<(), &'static str> {
    let ret = unsafe { gpu_crypto_init() };
    if ret == 0 { Ok(()) } else { Err("No CUDA GPU found") }
}

/// Extract raw u64 limbs from G1Affine (x and y, each 6 u64)
/// G1Affine layout: Fp x (48 bytes) + Fp y (48 bytes) + Choice infinity (8 bytes)
fn g1_to_u64(p: &G1Affine) -> [u64; 12] {
    let raw: &[u64] = unsafe {
        std::slice::from_raw_parts(p as *const _ as *const u64, 13) // 104 bytes / 8
    };
    let mut out = [0u64; 12];
    out[..6].copy_from_slice(&raw[..6]);   // x
    out[6..12].copy_from_slice(&raw[6..12]); // y
    out
}

/// Extract raw u64 limbs from G2Prepared coefficients
fn g2_prepared_to_u64(prep: &G2Prepared) -> Vec<u64> {
    let raw: &[u8] = unsafe {
        std::slice::from_raw_parts(prep as *const _ as *const u8, std::mem::size_of_val(prep))
    };
    let coeffs_ptr: usize = unsafe { *((raw.as_ptr().add(8)) as *const usize) };
    let coeffs_len: usize = unsafe { *((raw.as_ptr().add(16)) as *const usize) };

    assert_eq!(coeffs_len, 68, "G2Prepared should have 68 coefficients");

    let all: &[u64] = unsafe {
        std::slice::from_raw_parts(coeffs_ptr as *const u64, G2_COEFF_U64S)
    };
    all.to_vec()
}

/// Batch BLS signature verification on GPU.
///
/// Verifies M signatures in parallel:
///   e(sig_i, G2) * e(-H(m_i), pk_i) == Gt::identity
///
/// # Arguments
/// * `sigs` - M signature points (G1Affine)
/// * `neg_hm_points` - M negated hash-to-curve points (G1Affine)
/// * `g2_prepared` - G2 generator prepared coefficients (shared for all)
/// * `pk_prepared` - M public key prepared coefficients
///
/// # Returns
/// Vec of bools, one per verification (true = valid signature)
pub fn batch_bls_verify(
    sigs: &[G1Affine],
    neg_hm_points: &[G1Affine],
    g2_prepared: &G2Prepared,
    pk_prepared: &[G2Prepared],
) -> Result<Vec<bool>, &'static str> {
    let m = sigs.len();
    assert_eq!(m, neg_hm_points.len());
    assert_eq!(m, pk_prepared.len());

    if m == 0 { return Ok(vec![]); }

    // Convert G1 points to raw u64 arrays
    let mut sig_u64 = Vec::with_capacity(m * G1_U64S);
    let mut nhm_u64 = Vec::with_capacity(m * G1_U64S);
    for i in 0..m {
        sig_u64.extend_from_slice(&g1_to_u64(&sigs[i]));
        nhm_u64.extend_from_slice(&g1_to_u64(&neg_hm_points[i]));
    }

    // Convert G2Prepared to raw coefficients
    let g2_coeffs = g2_prepared_to_u64(g2_prepared);
    let mut pk_coeffs = Vec::with_capacity(m * G2_COEFF_U64S);
    for i in 0..m {
        pk_coeffs.extend_from_slice(&g2_prepared_to_u64(&pk_prepared[i]));
    }

    // Call GPU
    let mut results = vec![0i32; m];
    let ret = unsafe {
        gpu_batch_bls_verify(
            sig_u64.as_ptr(),
            nhm_u64.as_ptr(),
            g2_coeffs.as_ptr(),
            pk_coeffs.as_ptr(),
            results.as_mut_ptr(),
            m as i32,
        )
    };

    if ret != 0 { return Err("GPU kernel failed"); }

    Ok(results.iter().map(|&r| r == 1).collect())
}

/// Verify a single BLS signature on GPU.
/// Falls back to CPU if GPU fails.
pub fn bls_verify(
    sig: &G1Affine,
    neg_hm: &G1Affine,
    g2_prepared: &G2Prepared,
    pk_prepared: &G2Prepared,
) -> bool {
    match batch_bls_verify(
        std::slice::from_ref(sig),
        std::slice::from_ref(neg_hm),
        g2_prepared,
        std::slice::from_ref(pk_prepared),
    ) {
        Ok(results) => results[0],
        Err(_) => {
            // CPU fallback via ic_bls12_381
            let ml = multi_miller_loop(&[(sig, g2_prepared), (neg_hm, pk_prepared)]);
            ml.final_exponentiation() == Gt::identity()
        }
    }
}

/// Batch SHA-256 hashing on GPU.
/// Hashes N independent chunks of equal size.
///
/// # Arguments
/// * `chunks` - Concatenated chunk data (N × chunk_size bytes)
/// * `chunk_size` - Size of each chunk in bytes
///
/// # Returns
/// Vec of 32-byte SHA-256 hashes
pub fn batch_sha256(chunks: &[u8], chunk_size: usize) -> Result<Vec<[u8; 32]>, &'static str> {
    if chunk_size == 0 { return Err("chunk_size must be > 0"); }
    let n = chunks.len() / chunk_size;
    if n == 0 { return Ok(vec![]); }
    if chunks.len() != n * chunk_size { return Err("chunks length must be multiple of chunk_size"); }

    let mut raw_hashes = vec![0u8; n * 32];
    let ret = unsafe {
        gpu_batch_sha256(
            chunks.as_ptr(),
            raw_hashes.as_mut_ptr(),
            n as i32,
            chunk_size as i32,
        )
    };
    if ret != 0 { return Err("GPU SHA-256 kernel failed"); }

    let mut hashes = Vec::with_capacity(n);
    for i in 0..n {
        let mut h = [0u8; 32];
        h.copy_from_slice(&raw_hashes[i*32..(i+1)*32]);
        hashes.push(h);
    }
    Ok(hashes)
}

// ==================== Tests ====================
#[cfg(test)]
mod tests {
    use super::*;
    use ic_bls12_381::{G1Projective, G2Projective};
    use group::Curve;

    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }

    #[test]
    fn test_single_verify() {
        init().unwrap();

        // sk=42, H(m)=7*G1, sig=42*7*G1=294*G1, pk=42*G2
        let mut sig_proj = G1Projective::identity();
        for _ in 0..294 { sig_proj = sig_proj + G1Projective::generator(); }
        let sig = G1Affine::from(sig_proj);

        let mut hm_proj = G1Projective::identity();
        for _ in 0..7 { hm_proj = hm_proj + G1Projective::generator(); }
        let neg_hm = G1Affine::from(-hm_proj);

        let g2_prep = G2Prepared::from(G2Affine::generator());

        let mut pk_proj = G2Projective::identity();
        for _ in 0..42 { pk_proj = pk_proj + G2Projective::generator(); }
        let pk_prep = G2Prepared::from(G2Affine::from(pk_proj));

        // GPU verify
        let result = bls_verify(&sig, &neg_hm, &g2_prep, &pk_prep);
        assert!(result, "GPU BLS verify should return true for valid signature");

        // CPU verify (oracle)
        let ml = multi_miller_loop(&[(&sig, &g2_prep), (&neg_hm, &pk_prep)]);
        let cpu_result = ml.final_exponentiation() == Gt::identity();
        assert_eq!(result, cpu_result, "GPU and CPU must agree");
    }

    #[test]
    fn test_invalid_signature() {
        init().unwrap();

        // Use wrong signature (G1 generator instead of 294*G1)
        let sig = G1Affine::generator();
        let mut hm_proj = G1Projective::identity();
        for _ in 0..7 { hm_proj = hm_proj + G1Projective::generator(); }
        let neg_hm = G1Affine::from(-hm_proj);
        let g2_prep = G2Prepared::from(G2Affine::generator());
        let mut pk_proj = G2Projective::identity();
        for _ in 0..42 { pk_proj = pk_proj + G2Projective::generator(); }
        let pk_prep = G2Prepared::from(G2Affine::from(pk_proj));

        let result = bls_verify(&sig, &neg_hm, &g2_prep, &pk_prep);
        assert!(!result, "Invalid signature should return false");
    }

    #[test]
    fn test_sha256_oracle() {
        use sha2::{Sha256, Digest};
        init().unwrap();

        // Test 1: "abc"
        let data = b"abc";
        let gpu_hashes = batch_sha256(data, 3).unwrap();
        let mut cpu_hasher = Sha256::new();
        cpu_hasher.update(data);
        let cpu_hash: [u8; 32] = cpu_hasher.finalize().into();
        assert_eq!(gpu_hashes[0], cpu_hash, "SHA256('abc') mismatch");

        // Test 2: Batch of different-content same-size chunks
        let chunk_size = 1000;
        let n = 50;
        let mut all_chunks = vec![0u8; n * chunk_size];
        for i in 0..n {
            for j in 0..chunk_size {
                all_chunks[i * chunk_size + j] = ((i * 7 + j * 13) & 0xFF) as u8;
            }
        }

        let gpu_hashes = batch_sha256(&all_chunks, chunk_size).unwrap();
        assert_eq!(gpu_hashes.len(), n);

        // Oracle: compute each with CPU sha2
        for i in 0..n {
            let chunk = &all_chunks[i*chunk_size..(i+1)*chunk_size];
            let mut h = Sha256::new();
            h.update(chunk);
            let expected: [u8; 32] = h.finalize().into();
            assert_eq!(gpu_hashes[i], expected, "SHA256 mismatch at chunk {}", i);
        }
    }

    #[test]
    fn test_batch_verify() {
        init().unwrap();

        let mut sig_proj = G1Projective::identity();
        for _ in 0..294 { sig_proj = sig_proj + G1Projective::generator(); }
        let sig = G1Affine::from(sig_proj);

        let mut hm_proj = G1Projective::identity();
        for _ in 0..7 { hm_proj = hm_proj + G1Projective::generator(); }
        let neg_hm = G1Affine::from(-hm_proj);

        let g2_prep = G2Prepared::from(G2Affine::generator());
        let mut pk_proj = G2Projective::identity();
        for _ in 0..42 { pk_proj = pk_proj + G2Projective::generator(); }
        let pk_prep = G2Prepared::from(G2Affine::from(pk_proj));

        // Batch of 10 identical valid signatures
        let sigs = vec![sig; 10];
        let nhms = vec![neg_hm; 10];
        let pks = vec![pk_prep; 10];

        let results = batch_bls_verify(&sigs, &nhms, &g2_prep, &pks).unwrap();
        assert_eq!(results.len(), 10);
        for (i, &valid) in results.iter().enumerate() {
            assert!(valid, "Signature {} should be valid", i);
        }
    }
}
