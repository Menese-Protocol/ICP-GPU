// C API for GPU BLS12-381 crypto library
// Links into Rust via FFI
//
// All types use plain uint64_t arrays to avoid C++/Rust struct layout issues.
// Fp = 6 × u64, Fp2 = 12 × u64, G1Affine = 12 × u64 (x,y)
// G2Prepared coefficients = 2448 × u64

// Skip sppark's host-side blst dependency — we only use device-side mont_t
#define GPU_CRYPTO_DEVICE_ONLY
#include "field_sppark.cuh"
#include <cstdio>

#define BLS_X_VAL 0xd201000000010000ULL

// ==================== Device helpers ====================
__device__ __host__ fp_t fp_from_u64(const uint64_t v[6]) {
    fp_t r; uint32_t* w = (uint32_t*)&r;
    for (int i = 0; i < 6; i++) { w[2*i] = (uint32_t)v[i]; w[2*i+1] = (uint32_t)(v[i]>>32); }
    return r;
}

__device__ void lc(const uint64_t* co, int idx, Fp2& c0, Fp2& c1, Fp2& c2) {
    int b = idx * 36;
    c0.c0 = fp_from_u64(co+b); c0.c1 = fp_from_u64(co+b+6);
    c1.c0 = fp_from_u64(co+b+12); c1.c1 = fp_from_u64(co+b+18);
    c2.c0 = fp_from_u64(co+b+24); c2.c1 = fp_from_u64(co+b+30);
}

// ==================== Device pairing functions ====================
__device__ __noinline__ Fp12 multi_miller_2(const G1Affine& p1, const uint64_t* q1,
                                             const G1Affine& p2, const uint64_t* q2) {
    Fp12 f = fp12_one(); int ci = 0; bool found = false;
    for (int b = 63; b >= 0; b--) {
        bool bit = (((BLS_X_VAL >> 1) >> b) & 1) == 1;
        if (!found) { found = bit; continue; }
        Fp2 c0,c1,c2;
        lc(q1,ci,c0,c1,c2); f = fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p1.x),fp2_mul_fp(c0,p1.y));
        lc(q2,ci,c0,c1,c2); f = fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p2.x),fp2_mul_fp(c0,p2.y));
        ci++;
        if (bit) {
            lc(q1,ci,c0,c1,c2); f = fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p1.x),fp2_mul_fp(c0,p1.y));
            lc(q2,ci,c0,c1,c2); f = fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p2.x),fp2_mul_fp(c0,p2.y));
            ci++;
        }
        f = fp12_sqr(f);
    }
    Fp2 c0,c1,c2;
    lc(q1,ci,c0,c1,c2); f = fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p1.x),fp2_mul_fp(c0,p1.y));
    lc(q2,ci,c0,c1,c2); f = fp12_mul_by_014(f,c2,fp2_mul_fp(c1,p2.x),fp2_mul_fp(c0,p2.y));
    f = fp12_conj(f); // BLS_X is negative
    return f;
}

__device__ __noinline__ Fp12 cyc_exp(const Fp12& f) {
    Fp12 t = fp12_one(); bool fd = false;
    for (int i = 63; i >= 0; i--) {
        if (fd) t = cyclotomic_square(t);
        bool bit = ((BLS_X_VAL >> i) & 1) == 1;
        if (!fd) { fd = bit; if (!bit) continue; }
        if (bit) t = fp12_mul(t, f);
    }
    return fp12_conj(t);
}

__device__ __noinline__ Fp12 final_exp(const Fp12& fi) {
    Fp12 f=fi, t0=fp12_conj(f), t1=fp12_inv(f), t2=fp12_mul(t0,t1);
    t1=t2; t2=fp12_mul(fp12_frob(fp12_frob(t2)),t1);
    f=t2; t1=fp12_conj(cyclotomic_square(t2));
    Fp12 t3=cyc_exp(t2), t4=fp12_mul(t1,t3); t1=cyc_exp(t4);
    t4=fp12_conj(t4); f=fp12_mul(f,t4);
    t4=cyclotomic_square(t3); t0=cyc_exp(t1); t3=fp12_mul(t3,t0);
    t3=fp12_frob(fp12_frob(t3)); f=fp12_mul(f,t3);
    t4=fp12_mul(t4,cyc_exp(t0)); f=fp12_mul(f,cyc_exp(t4));
    t4=fp12_mul(t4,fp12_conj(t2)); t2=fp12_mul(t2,t1);
    t2=fp12_frob(fp12_frob(fp12_frob(t2))); f=fp12_mul(f,t2);
    t4=fp12_frob(t4); f=fp12_mul(f,t4);
    return f;
}

__device__ bool bls_verify_impl(const G1Affine& sig, const G1Affine& nhm,
                                 const uint64_t* g2c, const uint64_t* pkc) {
    Fp12 ml = multi_miller_2(sig, g2c, nhm, pkc);
    Fp12 r = final_exp(ml);
    uint64_t v[6]; fp_to_u64(r.c0.c0.c0, v);
    uint64_t one[6] = {0x760900000002fffdULL,0xebf4000bc40c0002ULL,0x5f48985753c758baULL,
                       0x77ce585370525745ULL,0x5c071a97a256ec6dULL,0x15f65ec3fa80e493ULL};
    for (int i = 0; i < 6; i++) if (v[i] != one[i]) return false;
    fp_t* fps = (fp_t*)&r;
    for (int i = 1; i < 12; i++) if (!fps[i].is_zero()) return false;
    return true;
}

// ==================== Kernels ====================
__global__ void kernel_batch_bls_verify(
    const uint64_t* sigs,      // M × 12 u64 (G1Affine: x[6], y[6])
    const uint64_t* neg_hms,   // M × 12 u64
    const uint64_t* g2_coeffs, // 2448 u64 (shared)
    const uint64_t* pk_coeffs, // M × 2448 u64
    int32_t* results,          // M × 1 (1=valid, 0=invalid)
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    const uint64_t* s = sigs + idx * 12;
    const uint64_t* n = neg_hms + idx * 12;
    G1Affine sig = {fp_from_u64(s), fp_from_u64(s+6)};
    G1Affine nhm = {fp_from_u64(n), fp_from_u64(n+6)};
    results[idx] = bls_verify_impl(sig, nhm, g2_coeffs, pk_coeffs + (uint64_t)idx * 2448) ? 1 : 0;
}

// SHA-256 (from sha256.cuh)
#include "sha256.cuh"

// ==================== C API ====================
extern "C" {

// Initialize GPU (call once)
int gpu_crypto_init() {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count == 0) return -1;
    cudaSetDevice(0);
    return 0;
}

// Batch BLS signature verification
// Returns 0 on success, fills results[i] = 1 (valid) or 0 (invalid)
int gpu_batch_bls_verify(
    const uint64_t* h_sigs,      // host: M × 12 u64
    const uint64_t* h_neg_hms,   // host: M × 12 u64
    const uint64_t* h_g2_coeffs, // host: 2448 u64
    const uint64_t* h_pk_coeffs, // host: M × 2448 u64
    int32_t* h_results,          // host: M × 1
    int M
) {
    uint64_t *d_sigs, *d_nhms, *d_g2, *d_pk;
    int32_t *d_res;

    cudaMalloc(&d_sigs, (uint64_t)M * 12 * 8);
    cudaMalloc(&d_nhms, (uint64_t)M * 12 * 8);
    cudaMalloc(&d_g2, 2448 * 8);
    cudaMalloc(&d_pk, (uint64_t)M * 2448 * 8);
    cudaMalloc(&d_res, M * sizeof(int32_t));

    cudaMemcpy(d_sigs, h_sigs, (uint64_t)M * 12 * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nhms, h_neg_hms, (uint64_t)M * 12 * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g2, h_g2_coeffs, 2448 * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pk, h_pk_coeffs, (uint64_t)M * 2448 * 8, cudaMemcpyHostToDevice);

    int thr = 32, blk = (M + thr - 1) / thr;
    kernel_batch_bls_verify<<<blk, thr>>>(d_sigs, d_nhms, d_g2, d_pk, d_res, M);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_res, M * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_sigs); cudaFree(d_nhms); cudaFree(d_g2); cudaFree(d_pk); cudaFree(d_res);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

// Batch SHA-256 hashing
// Each chunk is chunk_size bytes, N chunks total
// Output: N × 32 bytes (SHA-256 hashes)
int gpu_batch_sha256(
    const uint8_t* h_chunks,   // host: N × chunk_size bytes
    uint8_t* h_hashes,         // host: N × 32 bytes output
    int N,
    int chunk_size
) {
    uint64_t total = (uint64_t)N * chunk_size;
    uint8_t *d_chunks, *d_hashes;
    cudaMalloc(&d_chunks, total);
    cudaMalloc(&d_hashes, (uint64_t)N * 32);
    cudaMemcpy(d_chunks, h_chunks, total, cudaMemcpyHostToDevice);

    int thr = 256, blk = (N + thr - 1) / thr;
    kernel_batch_sha256<<<blk, thr>>>(d_chunks, d_hashes, N, chunk_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_hashes, d_hashes, (uint64_t)N * 32, cudaMemcpyDeviceToHost);
    cudaFree(d_chunks);
    cudaFree(d_hashes);

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

} // extern "C"
