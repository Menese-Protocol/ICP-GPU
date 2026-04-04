use ic_bls12_381::{G1Affine, G1Projective};
use group::Curve;
fn main() {
    let g1_2 = G1Affine::from(G1Projective::generator() + G1Projective::generator());
    let raw: &[u64; 12] = unsafe { std::mem::transmute(&g1_2) };
    // G1Affine is (Fp x, Fp y, Choice infinity) = 12 u64 + 1 byte
    // Actually G1Affine might have different layout. Let's check size
    println!("size={}", std::mem::size_of::<G1Affine>());
    // x is first 6 u64, y is next 6 u64
    print!("x=");
    for i in 0..6 { print!("{:016x},", raw[i]); }
    println!();
    print!("y=");
    for i in 6..12 { print!("{:016x},", raw[i]); }
    println!();

    // Also get -7*G1 and 294*G1 for BLS verify test
    let mut p7 = G1Projective::identity();
    for _ in 0..7 { p7 = p7 + G1Projective::generator(); }
    let neg_7g1 = G1Affine::from(-p7);
    let raw7: &[u64; 12] = unsafe { std::mem::transmute(&neg_7g1) };
    print!("neg7g1_x="); for i in 0..6 { print!("{:016x},", raw7[i]); } println!();
    print!("neg7g1_y="); for i in 6..12 { print!("{:016x},", raw7[i]); } println!();

    let mut p294 = G1Projective::identity();
    for _ in 0..294 { p294 = p294 + G1Projective::generator(); }
    let sig = G1Affine::from(p294);
    let raws: &[u64; 12] = unsafe { std::mem::transmute(&sig) };
    print!("sig294_x="); for i in 0..6 { print!("{:016x},", raws[i]); } println!();
    print!("sig294_y="); for i in 6..12 { print!("{:016x},", raws[i]); } println!();

    // 42*G2 prepared coefficients
    let mut pk42 = ic_bls12_381::G2Projective::identity();
    for _ in 0..42 { pk42 = pk42 + ic_bls12_381::G2Projective::generator(); }
    let pk42_affine = ic_bls12_381::G2Affine::from(pk42);
    let pk42_prep = ic_bls12_381::G2Prepared::from(pk42_affine);
    let raw_prep: &[u8] = unsafe {
        std::slice::from_raw_parts(&pk42_prep as *const _ as *const u8, std::mem::size_of_val(&pk42_prep))
    };
    let coeffs_ptr: usize = unsafe { *((raw_prep.as_ptr().add(8)) as *const usize) };
    let coeffs_len: usize = unsafe { *((raw_prep.as_ptr().add(16)) as *const usize) };
    println!("pk42_prep_count={}", coeffs_len);

    // Dump all 68 coefficients as one big hex blob
    if coeffs_len > 0 && coeffs_ptr != 0 {
        let all: &[u64] = unsafe {
            std::slice::from_raw_parts(coeffs_ptr as *const u64, coeffs_len * 36)
        };
        print!("pk42_coeffs=");
        for i in 0..coeffs_len*36 {
            print!("{:016x}", all[i]);
            if i < coeffs_len*36-1 { print!(","); }
        }
        println!();
    }
}
