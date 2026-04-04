// Miller Loop V2: matches ic_bls12_381 (DFINITY's library) exactly
// Uses Algorithm 26/27 from eprint 2010/354
// Line evaluation: ell(f, coeffs, p) with mul_by_014

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
#define BLS_X_IS_NEG true

// === Proven field arithmetic (unchanged) ===
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

// === Fp2 ===
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
// mul_by_nonresidue for Fp2: multiply by (u+1) i.e. β = u+1
// (c0+c1*u)(1+u) = (c0-c1) + (c0+c1)u
__device__ Fp2 fp2_mul_nr(const Fp2&a){return{fp_sub(a.c0,a.c1),fp_add(a.c0,a.c1)};}
__device__ Fp2 fp2_mul_fp(const Fp2&a,const Fp&s){return{fp_mul(a.c0,s),fp_mul(a.c1,s)};}

// === Fp6 ===
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

// === Fp12 ===
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

// === mul_by_014: matches ic_bls12_381's Fp12::mul_by_014 ===
// ic_bls12_381: f.mul_by_014(&c2_coeff, &c1_scaled, &c0_scaled)
// where c0_scaled = coeffs.0 * p.y, c1_scaled = coeffs.1 * p.x
// The 014 means non-zero at positions 0, 1, 4 in the Fp12 tower
// Direct port from ic_bls12_381 — no Karatsuba tricks, exact match
__device__ Fp6 fp6_mul_by_01(const Fp6& self, const Fp2& c0, const Fp2& c1) {
    Fp2 a_a = fp2_mul(self.c0, c0);
    Fp2 b_b = fp2_mul(self.c1, c1);
    Fp2 t1 = fp2_add(fp2_mul_nr(fp2_mul(self.c2, c1)), a_a);
    Fp2 t2 = fp2_sub(fp2_sub(fp2_mul(fp2_add(c0,c1), fp2_add(self.c0,self.c1)), a_a), b_b);
    Fp2 t3 = fp2_add(fp2_mul(self.c2, c0), b_b);
    return {t1, t2, t3};
}

__device__ Fp6 fp6_mul_by_1(const Fp6& self, const Fp2& c1) {
    return {
        fp2_mul_nr(fp2_mul(self.c2, c1)),
        fp2_mul(self.c0, c1),
        fp2_mul(self.c1, c1)
    };
}

// mul_by_014(c0_014, c1_014, c4_014)
// This matches ic_bls12_381's Fp12::mul_by_014 signature exactly
__device__ Fp12 fp12_mul_by_014(const Fp12& f, const Fp2& c0, const Fp2& c1, const Fp2& c4) {
    Fp6 aa = fp6_mul_by_01(f.c0, c0, c1);
    Fp6 bb = fp6_mul_by_1(f.c1, c4);
    Fp2 o = fp2_add(c1, c4);
    Fp6 c1_new = fp6_mul_by_01(fp6_add(f.c1, f.c0), c0, o);
    c1_new = fp6_sub(fp6_sub(c1_new, aa), bb);
    Fp6 c0_new = fp6_add(fp6_mul_by_v(bb), aa);
    return {c0_new, c1_new};
}

// === Doubling step: Algorithm 26, eprint 2010/354 ===
// Returns (c0, c1, c2) coefficients
struct LineCoeffs { Fp2 c0, c1, c2; };

__device__ LineCoeffs doubling_step(G2Proj& r) {
    Fp2 tmp0 = fp2_sqr(r.x);            // X^2
    Fp2 tmp1 = fp2_sqr(r.y);            // Y^2
    Fp2 tmp2 = fp2_sqr(tmp1);           // Y^4

    Fp2 tmp3 = fp2_sub(fp2_sqr(fp2_add(tmp1, r.x)), fp2_add(tmp0, tmp2));
    tmp3 = fp2_add(tmp3, tmp3);          // 2*((Y^2+X)^2 - X^2 - Y^4)

    Fp2 tmp4 = fp2_add(fp2_add(tmp0, tmp0), tmp0); // 3*X^2

    Fp2 tmp6 = fp2_add(r.x, tmp4);

    Fp2 tmp5 = fp2_sqr(tmp4);           // (3*X^2)^2 = 9*X^4

    Fp2 zsquared = fp2_sqr(r.z);        // Z^2

    r.x = fp2_sub(fp2_sub(tmp5, tmp3), tmp3); // 9*X^4 - 2*S

    r.z = fp2_sub(fp2_sqr(fp2_add(r.z, r.y)), fp2_add(tmp1, zsquared));
    // Z3 = (Z+Y)^2 - Y^2 - Z^2 = 2*Y*Z

    r.y = fp2_mul(fp2_sub(tmp3, r.x), tmp4);  // (S - X3) * 3*X^2
    tmp2 = fp2_add(tmp2, tmp2);
    tmp2 = fp2_add(tmp2, tmp2);
    tmp2 = fp2_add(tmp2, tmp2);           // 8*Y^4
    r.y = fp2_sub(r.y, tmp2);

    // Line coefficients
    tmp3 = fp2_mul(tmp4, zsquared);       // 3*X^2 * Z^2
    tmp3 = fp2_add(tmp3, tmp3);
    tmp3 = fp2_neg(tmp3);                 // -2 * 3*X^2 * Z^2

    tmp6 = fp2_sub(fp2_sqr(tmp6), fp2_add(tmp0, tmp5)); // (X+3X^2)^2 - X^2 - 9X^4
    tmp1 = fp2_add(tmp1, tmp1);
    tmp1 = fp2_add(tmp1, tmp1);           // 4*Y^2
    tmp6 = fp2_sub(tmp6, tmp1);           // line coeff c2

    Fp2 tmp0_out = fp2_mul(r.z, zsquared);
    tmp0_out = fp2_add(tmp0_out, tmp0_out); // 2*Z3*Z^2 = line coeff c0

    return {tmp0_out, tmp3, tmp6};
}

// === Addition step: Algorithm 27, eprint 2010/354 ===
__device__ LineCoeffs addition_step(G2Proj& r, const G2Affine& q) {
    Fp2 zsquared = fp2_sqr(r.z);
    Fp2 ysquared = fp2_sqr(q.y);
    Fp2 t0 = fp2_mul(zsquared, q.x);
    Fp2 t1 = fp2_mul(fp2_sub(fp2_sqr(fp2_add(q.y, r.z)), fp2_add(ysquared, zsquared)), zsquared);
    Fp2 t2 = fp2_sub(t0, r.x);
    Fp2 t3 = fp2_sqr(t2);
    Fp2 t4 = fp2_add(t3, t3);
    t4 = fp2_add(t4, t4);
    Fp2 t5 = fp2_mul(t4, t2);
    Fp2 t6 = fp2_sub(fp2_sub(t1, r.y), r.y);
    Fp2 t9 = fp2_mul(t6, q.x);
    Fp2 t7 = fp2_mul(t4, r.x);

    r.x = fp2_sub(fp2_sub(fp2_sub(fp2_sqr(t6), t5), t7), t7);
    r.z = fp2_sub(fp2_sqr(fp2_add(r.z, t2)), fp2_add(zsquared, t3));

    Fp2 t10 = fp2_add(q.y, r.z);
    Fp2 t8 = fp2_mul(fp2_sub(t7, r.x), t6);
    Fp2 t0b = fp2_mul(r.y, t5);
    t0b = fp2_add(t0b, t0b);
    r.y = fp2_sub(t8, t0b);

    t10 = fp2_sub(fp2_sqr(t10), ysquared);
    Fp2 ztsquared = fp2_sqr(r.z);
    t10 = fp2_sub(t10, ztsquared);

    t9 = fp2_sub(fp2_add(t9, t9), t10);
    Fp2 t10_out = fp2_add(r.z, r.z);
    t6 = fp2_neg(t6);
    Fp2 t1_out = fp2_add(t6, t6);

    return {t10_out, t1_out, t9};
}

// === ell: evaluate line at P, matching ic_bls12_381 exactly ===
__device__ Fp12 ell(const Fp12& f, const LineCoeffs& coeffs, const G1Affine& p) {
    Fp2 c0 = fp2_mul_fp(coeffs.c0, p.y);  // coeffs.0 * p.y
    Fp2 c1 = fp2_mul_fp(coeffs.c1, p.x);  // coeffs.1 * p.x
    return fp12_mul_by_014(f, coeffs.c2, c1, c0);
}

// === Miller loop: matches ic_bls12_381 exactly ===
__device__ Fp12 miller_loop(const G1Affine& p, const G2Affine& q) {
    G2Proj r = {q.x, q.y, fp2_one()};
    Fp12 f = fp12_one();
    bool found_one = false;

    // ic_bls12_381: for i in (0..64).rev().map(|b| (((BLS_X >> 1) >> b) & 1) == 1)
    for (int b = 63; b >= 0; b--) {
        bool i = (((BLS_X >> 1) >> b) & 1) == 1;
        if (!found_one) {
            found_one = i;
            continue;
        }

        LineCoeffs coeffs = doubling_step(r);
        f = ell(f, coeffs, p);

        if (i) {
            coeffs = addition_step(r, q);
            f = ell(f, coeffs, p);
        }

        f = fp12_sqr(f);  // square AFTER double+add
    }

    // Final doubling (after the loop)
    LineCoeffs coeffs = doubling_step(r);
    f = ell(f, coeffs, p);

    if (BLS_X_IS_NEG) {
        f = fp12_conj(f);
    }

    return f;
}

// === Output ===
__device__ void print_fp(const char* label, const Fp& a) {
    printf("  %s = [%016llx, %016llx, %016llx, %016llx, %016llx, %016llx]\n", label,
           (unsigned long long)a.v[0],(unsigned long long)a.v[1],(unsigned long long)a.v[2],
           (unsigned long long)a.v[3],(unsigned long long)a.v[4],(unsigned long long)a.v[5]);
}

__global__ void test_pairing() {
    G1Affine g1;
    g1.x.v[0]=0x5cb38790fd530c16ULL;g1.x.v[1]=0x7817fc679976fff5ULL;
    g1.x.v[2]=0x154f95c7143ba1c1ULL;g1.x.v[3]=0xf0ae6acdf3d0e747ULL;
    g1.x.v[4]=0xedce6ecc21dbf440ULL;g1.x.v[5]=0x120177419e0bfb75ULL;
    g1.y.v[0]=0xbaac93d50ce72271ULL;g1.y.v[1]=0x8c22631a7918fd8eULL;
    g1.y.v[2]=0xdd595f13570725ceULL;g1.y.v[3]=0x51ac582950405194ULL;
    g1.y.v[4]=0x0e1c8c3fad0059c0ULL;g1.y.v[5]=0x0bbc3efc5008a26aULL;

    G2Affine g2;
    g2.x.c0.v[0]=0xf5f28fa202940a10ULL;g2.x.c0.v[1]=0xb3f5fb2687b4961aULL;
    g2.x.c0.v[2]=0xa1a893b53e2ae580ULL;g2.x.c0.v[3]=0x9894999d1a3caee9ULL;
    g2.x.c0.v[4]=0x6f67b7631863366bULL;g2.x.c0.v[5]=0x058191924350bcd7ULL;
    g2.x.c1.v[0]=0xa5a9c0759e23f606ULL;g2.x.c1.v[1]=0xaaa0c59dbccd60c3ULL;
    g2.x.c1.v[2]=0x3bb17e18e2867806ULL;g2.x.c1.v[3]=0x1b1ab6cc8541b367ULL;
    g2.x.c1.v[4]=0xc2b6ed0ef2158547ULL;g2.x.c1.v[5]=0x11922a097360edf3ULL;
    g2.y.c0.v[0]=0xc997377402cae928ULL;g2.y.c0.v[1]=0x46fd5fb6bcc56372ULL;
    g2.y.c0.v[2]=0x8487900d5dda8f19ULL;g2.y.c0.v[3]=0x1e1957f8bedfb7b8ULL;
    g2.y.c0.v[4]=0xf3df76f877f3179fULL;g2.y.c0.v[5]=0x09efcd879b152b26ULL;
    g2.y.c1.v[0]=0xadc0fc92df64b05dULL;g2.y.c1.v[1]=0x18aa270a2b1461dcULL;
    g2.y.c1.v[2]=0x86adac6a3be4eba0ULL;g2.y.c1.v[3]=0x79495c4ec93da33aULL;
    g2.y.c1.v[4]=0xe7175850a43ccaedULL;g2.y.c1.v[5]=0x0b2bc2a163de1bf2ULL;

    printf("Computing Miller loop V2 (ic_bls12_381 compatible)...\n");
    Fp12 result = miller_loop(g1, g2);

    printf("\nMILLER_LOOP result:\n");
    const char* names[] = {"c0.c0.c0","c0.c0.c1","c0.c1.c0","c0.c1.c1","c0.c2.c0","c0.c2.c1",
                           "c1.c0.c0","c1.c0.c1","c1.c1.c0","c1.c1.c1","c1.c2.c0","c1.c2.c1"};
    Fp* fps = (Fp*)&result;
    for(int i=0;i<12;i++) print_fp(names[i], fps[i]);
}

// Batch benchmark
__global__ void bench_miller_v2(const G1Affine* P, const G2Affine* Q, Fp12* R, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    R[idx] = miller_loop(P[idx], Q[idx]);
}

int main() {
    printf("=== Miller Loop V2 (ic_bls12_381 compatible) ===\n\n");
    test_pairing<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){printf("CUDA ERROR: %s\n",cudaGetErrorString(err));return 1;}

    printf("\n=== Expected (from ic_bls12_381) ===\n");
    printf("  c0.c0.c0 = [a067a4e38dd6fea0, ce174a6ce348e8ca, 53e964dbf67fa93e, 5e14ad533455a788, be11f86e0de6770d, 03a22e046e708d71]\n");
    printf("  c0.c0.c1 = [a95c3278104a0731, 5d1858603d8f2f77, 1528757fa73ed1fe, 10631b692e7a1696, 18b9c8640c65f1cf, 00b1ac642909bf97]\n");

    // Batch benchmark
    printf("\n=== Batch Benchmark ===\n");
    G1Affine h_g1;
    h_g1.x.v[0]=0x5cb38790fd530c16ULL;h_g1.x.v[1]=0x7817fc679976fff5ULL;
    h_g1.x.v[2]=0x154f95c7143ba1c1ULL;h_g1.x.v[3]=0xf0ae6acdf3d0e747ULL;
    h_g1.x.v[4]=0xedce6ecc21dbf440ULL;h_g1.x.v[5]=0x120177419e0bfb75ULL;
    h_g1.y.v[0]=0xbaac93d50ce72271ULL;h_g1.y.v[1]=0x8c22631a7918fd8eULL;
    h_g1.y.v[2]=0xdd595f13570725ceULL;h_g1.y.v[3]=0x51ac582950405194ULL;
    h_g1.y.v[4]=0x0e1c8c3fad0059c0ULL;h_g1.y.v[5]=0x0bbc3efc5008a26aULL;
    G2Affine h_g2;
    h_g2.x.c0.v[0]=0xf5f28fa202940a10ULL;h_g2.x.c0.v[1]=0xb3f5fb2687b4961aULL;
    h_g2.x.c0.v[2]=0xa1a893b53e2ae580ULL;h_g2.x.c0.v[3]=0x9894999d1a3caee9ULL;
    h_g2.x.c0.v[4]=0x6f67b7631863366bULL;h_g2.x.c0.v[5]=0x058191924350bcd7ULL;
    h_g2.x.c1.v[0]=0xa5a9c0759e23f606ULL;h_g2.x.c1.v[1]=0xaaa0c59dbccd60c3ULL;
    h_g2.x.c1.v[2]=0x3bb17e18e2867806ULL;h_g2.x.c1.v[3]=0x1b1ab6cc8541b367ULL;
    h_g2.x.c1.v[4]=0xc2b6ed0ef2158547ULL;h_g2.x.c1.v[5]=0x11922a097360edf3ULL;
    h_g2.y.c0.v[0]=0xc997377402cae928ULL;h_g2.y.c0.v[1]=0x46fd5fb6bcc56372ULL;
    h_g2.y.c0.v[2]=0x8487900d5dda8f19ULL;h_g2.y.c0.v[3]=0x1e1957f8bedfb7b8ULL;
    h_g2.y.c0.v[4]=0xf3df76f877f3179fULL;h_g2.y.c0.v[5]=0x09efcd879b152b26ULL;
    h_g2.y.c1.v[0]=0xadc0fc92df64b05dULL;h_g2.y.c1.v[1]=0x18aa270a2b1461dcULL;
    h_g2.y.c1.v[2]=0x86adac6a3be4eba0ULL;h_g2.y.c1.v[3]=0x79495c4ec93da33aULL;
    h_g2.y.c1.v[4]=0xe7175850a43ccaedULL;h_g2.y.c1.v[5]=0x0b2bc2a163de1bf2ULL;

    int batch_sizes[] = {1, 13, 40, 100, 500, 1000};
    for(int bi=0;bi<6;bi++){
        int N=batch_sizes[bi];
        G1Affine*dP;G2Affine*dQ;Fp12*dR;
        cudaMalloc(&dP,N*sizeof(G1Affine));cudaMalloc(&dQ,N*sizeof(G2Affine));cudaMalloc(&dR,N*sizeof(Fp12));
        G1Affine*hP=new G1Affine[N];G2Affine*hQ=new G2Affine[N];
        for(int i=0;i<N;i++){hP[i]=h_g1;hQ[i]=h_g2;}
        cudaMemcpy(dP,hP,N*sizeof(G1Affine),cudaMemcpyHostToDevice);
        cudaMemcpy(dQ,hQ,N*sizeof(G2Affine),cudaMemcpyHostToDevice);
        int thr=128,blk=(N+thr-1)/thr;
        bench_miller_v2<<<blk,thr>>>(dP,dQ,dR,N);cudaDeviceSynchronize();
        int rounds=(N<=40)?100:20;
        auto start=std::chrono::high_resolution_clock::now();
        for(int r=0;r<rounds;r++) bench_miller_v2<<<blk,thr>>>(dP,dQ,dR,N);
        cudaDeviceSynchronize();
        auto end=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(end-start).count()/rounds;
        printf("  n=%-5d  batch=%.3fms  per=%.1fµs  throughput=%.0f/sec\n",N,ms,ms/N*1000,N/(ms/1000));
        delete[]hP;delete[]hQ;cudaFree(dP);cudaFree(dQ);cudaFree(dR);
    }
    printf("\n=== Done ===\n");
    return 0;
}
