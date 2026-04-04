// Miller loop using precomputed G2 coefficients from ic_bls12_381
// This verifies our ell() + mul_by_014 + loop structure
// bypassing the doubling_step formula (debug that separately)

#include <cstdint>
#include <cstdio>
#include <chrono>

struct Fp { uint64_t v[6]; };
struct Fp2 { Fp c0, c1; };
struct Fp6 { Fp2 c0, c1, c2; };
struct Fp12 { Fp6 c0, c1; };
struct G1Affine { Fp x, y; };

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

// Precomputed coefficients from ic_bls12_381
#include "g2_coeffs.h"

// === Proven Fp/Fp2/Fp6/Fp12 ===
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

__device__ Fp2 fp2_zero(){return{fp_zero(),fp_zero()};}
__device__ Fp2 fp2_one(){return{fp_one(),fp_zero()};}
__device__ Fp2 fp2_add(const Fp2&a,const Fp2&b){return{fp_add(a.c0,b.c0),fp_add(a.c1,b.c1)};}
__device__ Fp2 fp2_sub(const Fp2&a,const Fp2&b){return{fp_sub(a.c0,b.c0),fp_sub(a.c1,b.c1)};}
__device__ Fp2 fp2_neg(const Fp2&a){return{fp_neg(a.c0),fp_neg(a.c1)};}
__device__ Fp2 fp2_mul(const Fp2&a,const Fp2&b){
    Fp t0=fp_mul(a.c0,b.c0),t1=fp_mul(a.c1,b.c1);
    return{fp_sub(t0,t1),fp_sub(fp_sub(fp_mul(fp_add(a.c0,a.c1),fp_add(b.c0,b.c1)),t0),t1)};
}
__device__ Fp2 fp2_mul_nr(const Fp2&a){return{fp_sub(a.c0,a.c1),fp_add(a.c0,a.c1)};}
__device__ Fp2 fp2_mul_fp(const Fp2&a,const Fp&s){return{fp_mul(a.c0,s),fp_mul(a.c1,s)};}

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

// === mul_by_014 — direct from ic_bls12_381 ===
__device__ Fp6 fp6_mul_by_01(const Fp6& self, const Fp2& c0, const Fp2& c1) {
    Fp2 a_a = fp2_mul(self.c0, c0);
    Fp2 b_b = fp2_mul(self.c1, c1);
    return {
        fp2_add(fp2_mul_nr(fp2_mul(self.c2, c1)), a_a),
        fp2_sub(fp2_sub(fp2_mul(fp2_add(c0,c1), fp2_add(self.c0,self.c1)), a_a), b_b),
        fp2_add(fp2_mul(self.c2, c0), b_b)
    };
}
__device__ Fp6 fp6_mul_by_1(const Fp6& self, const Fp2& c1) {
    return {
        fp2_mul_nr(fp2_mul(self.c2, c1)),
        fp2_mul(self.c0, c1),
        fp2_mul(self.c1, c1)
    };
}
__device__ Fp12 fp12_mul_by_014(const Fp12& f, const Fp2& c0, const Fp2& c1, const Fp2& c4) {
    Fp6 aa = fp6_mul_by_01(f.c0, c0, c1);
    Fp6 bb = fp6_mul_by_1(f.c1, c4);
    Fp2 o = fp2_add(c1, c4);
    Fp6 c1_new = fp6_mul_by_01(fp6_add(f.c1, f.c0), c0, o);
    c1_new = fp6_sub(fp6_sub(c1_new, aa), bb);
    Fp6 c0_new = fp6_add(fp6_mul_by_v(bb), aa);
    return {c0_new, c1_new};
}

// === ell: matches ic_bls12_381 ===
__device__ Fp12 ell(const Fp12& f, const Fp2& coeff0, const Fp2& coeff1, const Fp2& coeff2, const G1Affine& p) {
    Fp2 c0 = fp2_mul_fp(coeff0, p.y);
    Fp2 c1 = fp2_mul_fp(coeff1, p.x);
    return fp12_mul_by_014(f, coeff2, c1, c0);
}

// Load precomputed coefficient
__device__ void load_coeff(int idx, Fp2& c0, Fp2& c1, Fp2& c2) {
    int base = idx * 36;
    for(int i=0;i<6;i++){c0.c0.v[i]=G2_COEFFS[base+i];c0.c1.v[i]=G2_COEFFS[base+6+i];}
    for(int i=0;i<6;i++){c1.c0.v[i]=G2_COEFFS[base+12+i];c1.c1.v[i]=G2_COEFFS[base+18+i];}
    for(int i=0;i<6;i++){c2.c0.v[i]=G2_COEFFS[base+24+i];c2.c1.v[i]=G2_COEFFS[base+30+i];}
}

// === Miller loop with precomputed coefficients ===
__device__ Fp12 miller_loop_precomp(const G1Affine& p) {
    Fp12 f = fp12_one();
    int coeff_idx = 0;
    bool found_one = false;

    for (int b = 63; b >= 0; b--) {
        bool bit = (((BLS_X >> 1) >> b) & 1) == 1;
        if (!found_one) { found_one = bit; continue; }

        Fp2 c0, c1, c2;
        load_coeff(coeff_idx++, c0, c1, c2);
        f = ell(f, c0, c1, c2, p);

        if (bit) {
            load_coeff(coeff_idx++, c0, c1, c2);
            f = ell(f, c0, c1, c2, p);
        }

        f = fp12_sqr(f);
    }

    // Final doubling
    Fp2 c0, c1, c2;
    load_coeff(coeff_idx++, c0, c1, c2);
    f = ell(f, c0, c1, c2, p);

    if (BLS_X_IS_NEG) f = fp12_conj(f);
    return f;
}

__global__ void test_miller_precomp() {
    G1Affine g1;
    g1.x.v[0]=0x5cb38790fd530c16ULL;g1.x.v[1]=0x7817fc679976fff5ULL;
    g1.x.v[2]=0x154f95c7143ba1c1ULL;g1.x.v[3]=0xf0ae6acdf3d0e747ULL;
    g1.x.v[4]=0xedce6ecc21dbf440ULL;g1.x.v[5]=0x120177419e0bfb75ULL;
    g1.y.v[0]=0xbaac93d50ce72271ULL;g1.y.v[1]=0x8c22631a7918fd8eULL;
    g1.y.v[2]=0xdd595f13570725ceULL;g1.y.v[3]=0x51ac582950405194ULL;
    g1.y.v[4]=0x0e1c8c3fad0059c0ULL;g1.y.v[5]=0x0bbc3efc5008a26aULL;

    Fp12 result = miller_loop_precomp(g1);

    printf("GPU Miller loop (precomputed coeffs):\n");
    const char* names[] = {"c0.c0.c0","c0.c0.c1","c0.c1.c0","c0.c1.c1","c0.c2.c0","c0.c2.c1",
                           "c1.c0.c0","c1.c0.c1","c1.c1.c0","c1.c1.c1","c1.c2.c0","c1.c2.c1"};
    Fp* fps = (Fp*)&result;
    for(int i=0;i<12;i++){
        printf("  %s = [%016llx, %016llx, %016llx, %016llx, %016llx, %016llx]\n",
               names[i],
               (unsigned long long)fps[i].v[0],(unsigned long long)fps[i].v[1],
               (unsigned long long)fps[i].v[2],(unsigned long long)fps[i].v[3],
               (unsigned long long)fps[i].v[4],(unsigned long long)fps[i].v[5]);
    }

    // Check match
    bool match = (fps[0].v[0] == 0xa067a4e38dd6fea0ULL) && (fps[0].v[5] == 0x03a22e046e708d71ULL);
    printf("\nFirst limb match: %s\n", match ? "YES - MATCH!" : "NO - mismatch");
}

int main() {
    printf("=== Miller Loop with Precomputed Coefficients ===\n\n");
    test_miller_precomp<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){printf("CUDA ERROR: %s\n",cudaGetErrorString(err));return 1;}

    printf("\nExpected:\n");
    printf("  c0.c0.c0 = [a067a4e38dd6fea0, ce174a6ce348e8ca, 53e964dbf67fa93e, 5e14ad533455a788, be11f86e0de6770d, 03a22e046e708d71]\n");
    printf("\n=== Done ===\n");
    return 0;
}
