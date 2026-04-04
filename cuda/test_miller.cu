// Step 5: Miller Loop + Line Functions on GPU
// Tests against known generator points, outputs Fp12 result for blst comparison

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>

struct Fp { uint64_t v[6]; };
struct Fp2 { Fp c0, c1; };
struct Fp6 { Fp2 c0, c1, c2; };
struct Fp12 { Fp6 c0, c1; };
struct G1Affine { Fp x, y; };
struct G2Affine { Fp2 x, y; };
struct G2Proj { Fp2 x, y, z; };

// ========== Constants ==========
__device__ __constant__ uint64_t FP_P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL, 0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};
__device__ __constant__ uint64_t FP_ONE[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL,
    0x5f48985753c758baULL, 0x77ce585370525745ULL,
    0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL
};
#define M0 0x89f3fffcfffcfffdULL
#define BLS_X 0xd201000000010000ULL

// ========== Fp (proven) ==========
__device__ Fp fp_zero(){Fp r={};return r;}
__device__ Fp fp_one(){Fp r;for(int i=0;i<6;i++)r.v[i]=FP_ONE[i];return r;}
__device__ bool fp_is_zero(const Fp&a){uint64_t acc=0;for(int i=0;i<6;i++)acc|=a.v[i];return acc==0;}
__device__ bool fp_eq(const Fp&a,const Fp&b){for(int i=0;i<6;i++)if(a.v[i]!=b.v[i])return false;return true;}

__device__ Fp fp_add(const Fp&a,const Fp&b){
    Fp r;unsigned __int128 carry=0;
    for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)a.v[i]+b.v[i]+carry;r.v[i]=(uint64_t)s;carry=s>>64;}
    Fp t;unsigned __int128 borrow=0;
    for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-borrow;t.v[i]=(uint64_t)d;borrow=(d>>127)&1;}
    return(borrow==0)?t:r;
}
__device__ Fp fp_sub(const Fp&a,const Fp&b){
    Fp r;unsigned __int128 borrow=0;
    for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)a.v[i]-b.v[i]-borrow;r.v[i]=(uint64_t)d;borrow=(d>>127)&1;}
    if(borrow){unsigned __int128 carry=0;for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)r.v[i]+FP_P[i]+carry;r.v[i]=(uint64_t)s;carry=s>>64;}}
    return r;
}
__device__ Fp fp_neg(const Fp&a){if(fp_is_zero(a))return a;return fp_sub(fp_zero(),a);}
__device__ Fp fp_mul(const Fp&a,const Fp&b){
    uint64_t t[7]={0};
    for(int i=0;i<6;i++){
        uint64_t carry=0;
        for(int j=0;j<6;j++){unsigned __int128 prod=(unsigned __int128)a.v[j]*b.v[i]+t[j]+carry;t[j]=(uint64_t)prod;carry=(uint64_t)(prod>>64);}
        t[6]=carry;uint64_t m=t[0]*M0;unsigned __int128 red=(unsigned __int128)m*FP_P[0]+t[0];carry=(uint64_t)(red>>64);
        for(int j=1;j<6;j++){red=(unsigned __int128)m*FP_P[j]+t[j]+carry;t[j-1]=(uint64_t)red;carry=(uint64_t)(red>>64);}
        t[5]=t[6]+carry;t[6]=(t[5]<carry)?1:0;
    }
    Fp r;for(int i=0;i<6;i++)r.v[i]=t[i];
    Fp s;unsigned __int128 borrow=0;
    for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-borrow;s.v[i]=(uint64_t)d;borrow=(d>>127)&1;}
    return(borrow==0)?s:r;
}

// ========== Fp2 (proven) ==========
__device__ Fp2 fp2_zero(){return{fp_zero(),fp_zero()};}
__device__ Fp2 fp2_one(){return{fp_one(),fp_zero()};}
__device__ bool fp2_is_zero(const Fp2&a){return fp_is_zero(a.c0)&&fp_is_zero(a.c1);}
__device__ Fp2 fp2_add(const Fp2&a,const Fp2&b){return{fp_add(a.c0,b.c0),fp_add(a.c1,b.c1)};}
__device__ Fp2 fp2_sub(const Fp2&a,const Fp2&b){return{fp_sub(a.c0,b.c0),fp_sub(a.c1,b.c1)};}
__device__ Fp2 fp2_neg(const Fp2&a){return{fp_neg(a.c0),fp_neg(a.c1)};}
__device__ Fp2 fp2_mul(const Fp2&a,const Fp2&b){
    Fp t0=fp_mul(a.c0,b.c0),t1=fp_mul(a.c1,b.c1);
    return{fp_sub(t0,t1),fp_sub(fp_sub(fp_mul(fp_add(a.c0,a.c1),fp_add(b.c0,b.c1)),t0),t1)};
}
__device__ Fp2 fp2_sqr(const Fp2&a){
    Fp t=fp_mul(a.c0,a.c1);
    return{fp_mul(fp_add(a.c0,a.c1),fp_sub(a.c0,a.c1)),fp_add(t,t)};
}
__device__ Fp2 fp2_mul_nr(const Fp2&a){return{fp_sub(a.c0,a.c1),fp_add(a.c0,a.c1)};}
__device__ Fp2 fp2_mul_fp(const Fp2&a,const Fp&s){return{fp_mul(a.c0,s),fp_mul(a.c1,s)};}

// ========== Fp6 (proven) ==========
__device__ Fp6 fp6_zero(){return{fp2_zero(),fp2_zero(),fp2_zero()};}
__device__ Fp6 fp6_one(){return{fp2_one(),fp2_zero(),fp2_zero()};}
__device__ Fp6 fp6_add(const Fp6&a,const Fp6&b){return{fp2_add(a.c0,b.c0),fp2_add(a.c1,b.c1),fp2_add(a.c2,b.c2)};}
__device__ Fp6 fp6_sub(const Fp6&a,const Fp6&b){return{fp2_sub(a.c0,b.c0),fp2_sub(a.c1,b.c1),fp2_sub(a.c2,b.c2)};}
__device__ Fp6 fp6_neg(const Fp6&a){return{fp2_neg(a.c0),fp2_neg(a.c1),fp2_neg(a.c2)};}
__device__ Fp6 fp6_mul(const Fp6&a,const Fp6&b){
    Fp2 a_a=fp2_mul(a.c0,b.c0),b_b=fp2_mul(a.c1,b.c1),c_c=fp2_mul(a.c2,b.c2);
    return{
        fp2_add(a_a,fp2_mul_nr(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c1,a.c2),fp2_add(b.c1,b.c2)),b_b),c_c))),
        fp2_add(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c1),fp2_add(b.c0,b.c1)),a_a),b_b),fp2_mul_nr(c_c)),
        fp2_add(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c2),fp2_add(b.c0,b.c2)),a_a),c_c),b_b)
    };
}
__device__ Fp6 fp6_mul_by_v(const Fp6&a){return{fp2_mul_nr(a.c2),a.c0,a.c1};}

// ========== Fp12 (proven) ==========
__device__ Fp12 fp12_one(){return{fp6_one(),fp6_zero()};}
__device__ Fp12 fp12_mul(const Fp12&a,const Fp12&b){
    Fp6 aa=fp6_mul(a.c0,b.c0),bb=fp6_mul(a.c1,b.c1);
    return{fp6_add(aa,fp6_mul_by_v(bb)),fp6_sub(fp6_sub(fp6_mul(fp6_add(a.c0,a.c1),fp6_add(b.c0,b.c1)),aa),bb)};
}
__device__ Fp12 fp12_sqr(const Fp12&a){
    Fp6 ab=fp6_mul(a.c0,a.c1),c0c1=fp6_add(a.c0,a.c1),c0v=fp6_add(a.c0,fp6_mul_by_v(a.c1));
    return{fp6_sub(fp6_sub(fp6_mul(c0v,c0c1),ab),fp6_mul_by_v(ab)),fp6_add(ab,ab)};
}
__device__ Fp12 fp12_conj(const Fp12&a){return{a.c0,fp6_neg(a.c1)};}

// ========== Sparse Fp12 multiply (for line functions) ==========
// mul_by_01: Fp6 *= (c0 + c1*v)
__device__ void fp6_mul_by_01(Fp6& r, const Fp2& c0, const Fp2& c1) {
    Fp2 a_a = fp2_mul(r.c0, c0);
    Fp2 b_b = fp2_mul(r.c1, c1);
    Fp2 t1 = fp2_add(fp2_mul_nr(fp2_sub(fp2_mul(fp2_add(r.c1,r.c2),c1),b_b)), a_a);
    Fp2 t3 = fp2_add(fp2_sub(fp2_mul(fp2_add(r.c0,r.c2),c0),a_a), b_b);
    Fp2 t2 = fp2_sub(fp2_sub(fp2_mul(fp2_add(r.c0,r.c1),fp2_add(c0,c1)),a_a),b_b);
    r.c0=t1; r.c1=t2; r.c2=t3;
}

// mul_by_1: Fp6 *= (0 + c1*v)
__device__ void fp6_mul_by_1(Fp6& r, const Fp2& c1) {
    Fp2 b_b = fp2_mul(r.c1, c1);
    Fp2 t1 = fp2_mul_nr(fp2_sub(fp2_mul(fp2_add(r.c1,r.c2),c1),b_b));
    Fp2 t2 = fp2_sub(fp2_mul(fp2_add(r.c0,r.c1),c1),b_b);
    r.c0=t1; r.c1=t2; r.c2=b_b;
}

// mul_by_014: Fp12 *= sparse (c0, c1, 0, 0, c4, 0)
__device__ void fp12_mul_by_014(Fp12& f, const Fp2& c0, const Fp2& c1, const Fp2& c4) {
    Fp6 aa = f.c0; fp6_mul_by_01(aa, c0, c1);
    Fp6 bb = f.c1; fp6_mul_by_1(bb, c4);
    Fp2 o = fp2_add(c1, c4);
    f.c1 = fp6_add(f.c1, f.c0);
    fp6_mul_by_01(f.c1, c0, o);
    f.c1 = fp6_sub(fp6_sub(f.c1, aa), bb);
    f.c0 = fp6_add(fp6_mul_by_v(bb), aa);
}

// ========== Line functions ==========
__device__ Fp6 line_double(G2Proj& r, const Fp& two_inv_fp) {
    Fp2 two_inv = {two_inv_fp, fp_zero()};
    Fp2 a = fp2_mul(fp2_mul(r.x, r.y), two_inv);
    Fp2 b = fp2_sqr(r.y);
    Fp2 c = fp2_sqr(r.z);

    // e = b' * (c+c+c) where b' = 4(1+u) for BLS12-381 M-twist
    Fp2 c3 = fp2_add(fp2_add(c, c), c);
    Fp2 e = fp2_mul_nr(c3);  // (1+u)*c3
    e = fp2_add(e, e); e = fp2_add(e, e); // 4*(1+u)*c3

    Fp2 f = fp2_add(fp2_add(e, e), e); // 3*e
    Fp2 g = fp2_mul(fp2_add(b, f), two_inv);
    Fp2 h = fp2_sqr(fp2_add(r.y, r.z));
    h = fp2_sub(h, fp2_add(b, c));
    Fp2 i = fp2_sub(e, b);
    Fp2 j = fp2_sqr(r.x);
    Fp2 e_sq = fp2_sqr(e);

    r.x = fp2_mul(a, fp2_sub(b, f));
    r.y = fp2_sub(fp2_sqr(g), fp2_add(fp2_add(e_sq, e_sq), e_sq));
    r.z = fp2_mul(b, h);

    return {i, fp2_add(fp2_add(j,j),j), fp2_neg(h)}; // M-twist
}

__device__ Fp6 line_add(G2Proj& r, const G2Affine& q) {
    Fp2 theta = fp2_sub(r.y, fp2_mul(q.y, r.z));
    Fp2 lambda = fp2_sub(r.x, fp2_mul(q.x, r.z));
    Fp2 c = fp2_sqr(theta);
    Fp2 d = fp2_sqr(lambda);
    Fp2 e = fp2_mul(lambda, d);
    Fp2 f = fp2_mul(r.z, c);
    Fp2 g = fp2_mul(r.x, d);
    Fp2 h = fp2_sub(fp2_add(e, f), fp2_add(g, g));
    r.x = fp2_mul(lambda, h);
    r.y = fp2_sub(fp2_mul(theta, fp2_sub(g, h)), fp2_mul(e, r.y));
    r.z = fp2_mul(r.z, e);
    Fp2 j = fp2_sub(fp2_mul(theta, q.x), fp2_mul(lambda, q.y));
    return {j, fp2_neg(theta), lambda}; // M-twist
}

// ========== Evaluate line at P ==========
__device__ void ell(Fp12& f, const Fp6& coeffs, const G1Affine& p) {
    // M-twist: c0 stays, c1 *= p.x, c2 *= p.y
    Fp2 c0 = coeffs.c0;
    Fp2 c1 = fp2_mul_fp(coeffs.c1, p.x);
    Fp2 c2 = fp2_mul_fp(coeffs.c2, p.y);
    fp12_mul_by_014(f, c0, c1, c2);
}

// ========== Miller Loop ==========
__device__ Fp12 miller_loop(const G1Affine& p, const G2Affine& q) {
    // two_inv in Montgomery form
    Fp two_inv;
    two_inv.v[0]=0x1804000000015554ULL; two_inv.v[1]=0x855000053ab00001ULL;
    two_inv.v[2]=0x633cb57c253c276fULL; two_inv.v[3]=0x6e22d1ec31ebb502ULL;
    two_inv.v[4]=0xd3916126f2d14ca2ULL; two_inv.v[5]=0x17fbb8571a006596ULL;

    G2Proj r = {q.x, q.y, fp2_one()};
    Fp12 f = fp12_one();

    // Iterate bits of |x| from MSB-1 down to 0
    for (int j = 62; j >= 0; j--) {
        f = fp12_sqr(f);
        Fp6 coeffs = line_double(r, two_inv);
        ell(f, coeffs, p);
        if ((BLS_X >> j) & 1) {
            coeffs = line_add(r, q);
            ell(f, coeffs, p);
        }
    }

    // x is negative: conjugate
    f = fp12_conj(f);

    return f;
}

// ========== Output Fp12 for comparison ==========
__device__ void print_fp(const char* label, const Fp& a) {
    printf("%s = [%016llx, %016llx, %016llx, %016llx, %016llx, %016llx]\n",
           label,
           (unsigned long long)a.v[0], (unsigned long long)a.v[1],
           (unsigned long long)a.v[2], (unsigned long long)a.v[3],
           (unsigned long long)a.v[4], (unsigned long long)a.v[5]);
}

__global__ void test_miller_loop() {
    // G1 generator (Montgomery form)
    G1Affine g1;
    g1.x.v[0]=0x5cb38790fd530c16ULL; g1.x.v[1]=0x7817fc679976fff5ULL;
    g1.x.v[2]=0x154f95c7143ba1c1ULL; g1.x.v[3]=0xf0ae6acdf3d0e747ULL;
    g1.x.v[4]=0xedce6ecc21dbf440ULL; g1.x.v[5]=0x120177419e0bfb75ULL;
    g1.y.v[0]=0xbaac93d50ce72271ULL; g1.y.v[1]=0x8c22631a7918fd8eULL;
    g1.y.v[2]=0xdd595f13570725ceULL; g1.y.v[3]=0x51ac582950405194ULL;
    g1.y.v[4]=0x0e1c8c3fad0059c0ULL; g1.y.v[5]=0x0bbc3efc5008a26aULL;

    // G2 generator (Montgomery form)
    G2Affine g2;
    g2.x.c0.v[0]=0xf5f28fa202940a10ULL; g2.x.c0.v[1]=0xb3f5fb2687b4961aULL;
    g2.x.c0.v[2]=0xa1a893b53e2ae580ULL; g2.x.c0.v[3]=0x9894999d1a3caee9ULL;
    g2.x.c0.v[4]=0x6f67b7631863366bULL; g2.x.c0.v[5]=0x058191924350bcd7ULL;
    g2.x.c1.v[0]=0xa5a9c0759e23f606ULL; g2.x.c1.v[1]=0xaaa0c59dbccd60c3ULL;
    g2.x.c1.v[2]=0x3bb17e18e2867806ULL; g2.x.c1.v[3]=0x1b1ab6cc8541b367ULL;
    g2.x.c1.v[4]=0xc2b6ed0ef2158547ULL; g2.x.c1.v[5]=0x11922a097360edf3ULL;
    g2.y.c0.v[0]=0xc997377402cae928ULL; g2.y.c0.v[1]=0x46fd5fb6bcc56372ULL;
    g2.y.c0.v[2]=0x8487900d5dda8f19ULL; g2.y.c0.v[3]=0x1e1957f8bedfb7b8ULL;
    g2.y.c0.v[4]=0xf3df76f877f3179fULL; g2.y.c0.v[5]=0x09efcd879b152b26ULL;
    g2.y.c1.v[0]=0xadc0fc92df64b05dULL; g2.y.c1.v[1]=0x18aa270a2b1461dcULL;
    g2.y.c1.v[2]=0x86adac6a3be4eba0ULL; g2.y.c1.v[3]=0x79495c4ec93da33aULL;
    g2.y.c1.v[4]=0xe7175850a43ccaedULL; g2.y.c1.v[5]=0x0b2bc2a163de1bf2ULL;

    printf("Computing Miller loop: e(G1, G2)...\n");

    Fp12 result = miller_loop(g1, g2);

    // Print first component for cross-check with blst
    printf("\nMiller loop result (before final exp):\n");
    print_fp("f.c0.c0.c0", result.c0.c0.c0);
    print_fp("f.c0.c0.c1", result.c0.c0.c1);
    print_fp("f.c0.c1.c0", result.c0.c1.c0);
    print_fp("f.c0.c1.c1", result.c0.c1.c1);
    print_fp("f.c1.c0.c0", result.c1.c0.c0);
    print_fp("f.c1.c0.c1", result.c1.c0.c1);

    // Quick sanity: result should NOT be zero or one
    bool is_one = fp_eq(result.c0.c0.c0, fp_one()) && fp2_is_zero(result.c0.c1);
    printf("\nResult is ONE: %s (expect: NO)\n", is_one ? "YES - BAD!" : "NO - good");

    // Quick sanity: c1 should be non-zero (it's a full Fp12 element)
    bool c1_zero = fp2_is_zero(result.c1.c0) && fp2_is_zero(result.c1.c1) && fp2_is_zero(result.c1.c2);
    printf("Result c1 is zero: %s (expect: NO)\n", c1_zero ? "YES - BAD!" : "NO - good");
}

// Batch benchmark kernel
__global__ void bench_miller_batch(
    const G1Affine* P, const G2Affine* Q,
    Fp12* results, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    results[idx] = miller_loop(P[idx], Q[idx]);
}

int main() {
    printf("=== Miller Loop GPU Test ===\n\n");

    // Run single test
    test_miller_loop<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){printf("CUDA ERROR: %s\n",cudaGetErrorString(err));return 1;}

    // ===== Batch timing benchmark =====
    printf("\n=== Batch Miller Loop Benchmark ===\n");

    // Set up G1 and G2 generator points on host
    G1Affine h_g1;
    h_g1.x.v[0]=0x5cb38790fd530c16ULL; h_g1.x.v[1]=0x7817fc679976fff5ULL;
    h_g1.x.v[2]=0x154f95c7143ba1c1ULL; h_g1.x.v[3]=0xf0ae6acdf3d0e747ULL;
    h_g1.x.v[4]=0xedce6ecc21dbf440ULL; h_g1.x.v[5]=0x120177419e0bfb75ULL;
    h_g1.y.v[0]=0xbaac93d50ce72271ULL; h_g1.y.v[1]=0x8c22631a7918fd8eULL;
    h_g1.y.v[2]=0xdd595f13570725ceULL; h_g1.y.v[3]=0x51ac582950405194ULL;
    h_g1.y.v[4]=0x0e1c8c3fad0059c0ULL; h_g1.y.v[5]=0x0bbc3efc5008a26aULL;

    G2Affine h_g2;
    h_g2.x.c0.v[0]=0xf5f28fa202940a10ULL; h_g2.x.c0.v[1]=0xb3f5fb2687b4961aULL;
    h_g2.x.c0.v[2]=0xa1a893b53e2ae580ULL; h_g2.x.c0.v[3]=0x9894999d1a3caee9ULL;
    h_g2.x.c0.v[4]=0x6f67b7631863366bULL; h_g2.x.c0.v[5]=0x058191924350bcd7ULL;
    h_g2.x.c1.v[0]=0xa5a9c0759e23f606ULL; h_g2.x.c1.v[1]=0xaaa0c59dbccd60c3ULL;
    h_g2.x.c1.v[2]=0x3bb17e18e2867806ULL; h_g2.x.c1.v[3]=0x1b1ab6cc8541b367ULL;
    h_g2.x.c1.v[4]=0xc2b6ed0ef2158547ULL; h_g2.x.c1.v[5]=0x11922a097360edf3ULL;
    h_g2.y.c0.v[0]=0xc997377402cae928ULL; h_g2.y.c0.v[1]=0x46fd5fb6bcc56372ULL;
    h_g2.y.c0.v[2]=0x8487900d5dda8f19ULL; h_g2.y.c0.v[3]=0x1e1957f8bedfb7b8ULL;
    h_g2.y.c0.v[4]=0xf3df76f877f3179fULL; h_g2.y.c0.v[5]=0x09efcd879b152b26ULL;
    h_g2.y.c1.v[0]=0xadc0fc92df64b05dULL; h_g2.y.c1.v[1]=0x18aa270a2b1461dcULL;
    h_g2.y.c1.v[2]=0x86adac6a3be4eba0ULL; h_g2.y.c1.v[3]=0x79495c4ec93da33aULL;
    h_g2.y.c1.v[4]=0xe7175850a43ccaedULL; h_g2.y.c1.v[5]=0x0b2bc2a163de1bf2ULL;

    int batch_sizes[] = {1, 4, 13, 28, 34, 40, 100, 500, 1000};
    int num_batches = sizeof(batch_sizes)/sizeof(batch_sizes[0]);

    for (int bi = 0; bi < num_batches; bi++) {
        int N = batch_sizes[bi];

        // Allocate and fill arrays (all same point for now — different points later)
        G1Affine* d_P; G2Affine* d_Q; Fp12* d_R;
        cudaMalloc(&d_P, N * sizeof(G1Affine));
        cudaMalloc(&d_Q, N * sizeof(G2Affine));
        cudaMalloc(&d_R, N * sizeof(Fp12));

        // Fill with copies of generator
        G1Affine* h_P = new G1Affine[N];
        G2Affine* h_Q = new G2Affine[N];
        for(int i=0;i<N;i++){h_P[i]=h_g1; h_Q[i]=h_g2;}
        cudaMemcpy(d_P, h_P, N*sizeof(G1Affine), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Q, h_Q, N*sizeof(G2Affine), cudaMemcpyHostToDevice);

        // Warm up
        int threads = 128;
        int blocks = (N + threads - 1) / threads;
        bench_miller_batch<<<blocks,threads>>>(d_P, d_Q, d_R, N);
        cudaDeviceSynchronize();

        // Benchmark
        int rounds = (N <= 40) ? 100 : 20;
        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < rounds; r++) {
            bench_miller_batch<<<blocks,threads>>>(d_P, d_Q, d_R, N);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count() / rounds;
        double per_pairing = ms / N;
        double throughput = N / (ms / 1000.0);

        printf("  n=%-5d  batch=%.3fms  per_pairing=%.1fµs  throughput=%.0f/sec\n",
               N, ms, per_pairing * 1000.0, throughput);

        delete[] h_P; delete[] h_Q;
        cudaFree(d_P); cudaFree(d_Q); cudaFree(d_R);
    }

    printf("\n=== Done ===\n");
    return 0;
}
