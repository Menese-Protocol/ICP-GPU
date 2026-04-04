// Step 4: Fp12 = Fp6[w]/(w^2 - v) on GPU
// Last tower layer before Miller loop

#include <cstdint>
#include <cstdio>

struct Fp { uint64_t v[6]; };
struct Fp2 { Fp c0, c1; };
struct Fp6 { Fp2 c0, c1, c2; };
struct Fp12 { Fp6 c0, c1; };

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

// --- Fp (proven) ---
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

// --- Fp2 (proven) ---
__device__ Fp2 fp2_zero(){return{fp_zero(),fp_zero()};}
__device__ Fp2 fp2_one(){return{fp_one(),fp_zero()};}
__device__ bool fp2_is_zero(const Fp2&a){return fp_is_zero(a.c0)&&fp_is_zero(a.c1);}
__device__ bool fp2_eq(const Fp2&a,const Fp2&b){return fp_eq(a.c0,b.c0)&&fp_eq(a.c1,b.c1);}
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

// --- Fp6 (proven) ---
__device__ Fp6 fp6_zero(){return{fp2_zero(),fp2_zero(),fp2_zero()};}
__device__ Fp6 fp6_one(){return{fp2_one(),fp2_zero(),fp2_zero()};}
__device__ bool fp6_is_zero(const Fp6&a){return fp2_is_zero(a.c0)&&fp2_is_zero(a.c1)&&fp2_is_zero(a.c2);}
__device__ Fp6 fp6_add(const Fp6&a,const Fp6&b){return{fp2_add(a.c0,b.c0),fp2_add(a.c1,b.c1),fp2_add(a.c2,b.c2)};}
__device__ Fp6 fp6_sub(const Fp6&a,const Fp6&b){return{fp2_sub(a.c0,b.c0),fp2_sub(a.c1,b.c1),fp2_sub(a.c2,b.c2)};}
__device__ Fp6 fp6_neg(const Fp6&a){return{fp2_neg(a.c0),fp2_neg(a.c1),fp2_neg(a.c2)};}
__device__ Fp6 fp6_mul(const Fp6&a,const Fp6&b){
    Fp2 a_a=fp2_mul(a.c0,b.c0),b_b=fp2_mul(a.c1,b.c1),c_c=fp2_mul(a.c2,b.c2);
    Fp2 t1=fp2_add(a_a,fp2_mul_nr(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c1,a.c2),fp2_add(b.c1,b.c2)),b_b),c_c)));
    Fp2 t2=fp2_add(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c1),fp2_add(b.c0,b.c1)),a_a),b_b),fp2_mul_nr(c_c));
    Fp2 t3=fp2_add(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c2),fp2_add(b.c0,b.c2)),a_a),c_c),b_b);
    return{t1,t2,t3};
}
// v*(c0,c1,c2) = (β*c2, c0, c1)
__device__ Fp6 fp6_mul_by_v(const Fp6&a){return{fp2_mul_nr(a.c2),a.c0,a.c1};}

// --- Fp12 (testing now) ---
__device__ Fp12 fp12_one(){return{fp6_one(),fp6_zero()};}

__device__ Fp12 fp12_mul(const Fp12&a,const Fp12&b){
    Fp6 aa=fp6_mul(a.c0,b.c0);
    Fp6 bb=fp6_mul(a.c1,b.c1);
    return{
        fp6_add(aa,fp6_mul_by_v(bb)),
        fp6_sub(fp6_sub(fp6_mul(fp6_add(a.c0,a.c1),fp6_add(b.c0,b.c1)),aa),bb)
    };
}

__device__ Fp12 fp12_sqr(const Fp12&a){
    Fp6 ab=fp6_mul(a.c0,a.c1);
    Fp6 c0c1=fp6_add(a.c0,a.c1);
    Fp6 c0_vc1=fp6_add(a.c0,fp6_mul_by_v(a.c1));
    return{
        fp6_sub(fp6_sub(fp6_mul(c0_vc1,c0c1),ab),fp6_mul_by_v(ab)),
        fp6_add(ab,ab)
    };
}

__device__ Fp12 fp12_conj(const Fp12&a){return{a.c0,fp6_neg(a.c1)};}

__device__ Fp fp_from_int(int n){
    Fp r=fp_zero();if(n==0)return r;
    Fp one=fp_one();r=one;for(int i=1;i<n;i++)r=fp_add(r,one);return r;
}

__global__ void test_fp12() {
    // Build test elements matching Python:
    // a = Fp12(Fp6(Fp2(1,2),Fp2(3,4),Fp2(5,6)), Fp6(Fp2(7,8),Fp2(9,10),Fp2(11,12)))
    // b = Fp12(Fp6(Fp2(13,14),Fp2(15,16),Fp2(17,18)), Fp6(Fp2(19,20),Fp2(21,22),Fp2(23,24)))
    Fp12 a = {
        {{fp_from_int(1),fp_from_int(2)},{fp_from_int(3),fp_from_int(4)},{fp_from_int(5),fp_from_int(6)}},
        {{fp_from_int(7),fp_from_int(8)},{fp_from_int(9),fp_from_int(10)},{fp_from_int(11),fp_from_int(12)}}
    };
    Fp12 b = {
        {{fp_from_int(13),fp_from_int(14)},{fp_from_int(15),fp_from_int(16)},{fp_from_int(17),fp_from_int(18)}},
        {{fp_from_int(19),fp_from_int(20)},{fp_from_int(21),fp_from_int(22)},{fp_from_int(23),fp_from_int(24)}}
    };

    // Test 1: 1 * 1 = 1
    Fp12 one12 = fp12_one();
    Fp12 r1 = fp12_mul(one12, one12);
    bool is_one = fp2_eq(r1.c0.c0, fp2_one()) && fp2_is_zero(r1.c0.c1) && fp2_is_zero(r1.c0.c2)
               && fp6_is_zero(r1.c1);
    printf("TEST 1 (Fp12 1*1=1): %s\n", is_one ? "PASS" : "FAIL");

    // Test 2: a * 1 = a
    Fp12 r2 = fp12_mul(a, one12);
    bool eq_a = fp2_eq(r2.c0.c0, a.c0.c0) && fp2_eq(r2.c0.c1, a.c0.c1) && fp2_eq(r2.c0.c2, a.c0.c2)
             && fp2_eq(r2.c1.c0, a.c1.c0) && fp2_eq(r2.c1.c1, a.c1.c1) && fp2_eq(r2.c1.c2, a.c1.c2);
    printf("TEST 2 (a*1=a): %s\n", eq_a ? "PASS" : "FAIL");

    // Test 3: a*b cross-check with Python
    // Python: c0.c0 = Fp2(P-1650, 1405)
    Fp12 ab = fp12_mul(a, b);
    Fp neg1650 = fp_neg(fp_from_int(1650));
    Fp pos1405 = fp_from_int(1405);
    printf("TEST 3 (a*b c0.c0.c0=-1650): %s\n", fp_eq(ab.c0.c0.c0, neg1650) ? "PASS" : "FAIL");
    printf("TEST 4 (a*b c0.c0.c1=1405):  %s\n", fp_eq(ab.c0.c0.c1, pos1405) ? "PASS" : "FAIL");

    // Python: c0.c1 = Fp2(P-1282, 1475)
    Fp neg1282 = fp_neg(fp_from_int(1282));
    Fp pos1475 = fp_from_int(1475);
    printf("TEST 5 (a*b c0.c1.c0=-1282): %s\n", fp_eq(ab.c0.c1.c0, neg1282) ? "PASS" : "FAIL");
    printf("TEST 6 (a*b c0.c1.c1=1475):  %s\n", fp_eq(ab.c0.c1.c1, pos1475) ? "PASS" : "FAIL");

    // Python: c1.c0 = Fp2(P-1238, 1240)
    Fp neg1238 = fp_neg(fp_from_int(1238));
    Fp pos1240 = fp_from_int(1240);
    printf("TEST 7 (a*b c1.c0.c0=-1238): %s\n", fp_eq(ab.c1.c0.c0, neg1238) ? "PASS" : "FAIL");
    printf("TEST 8 (a*b c1.c0.c1=1240):  %s\n", fp_eq(ab.c1.c0.c1, pos1240) ? "PASS" : "FAIL");

    // Test 9: sqr(a) == mul(a,a)
    Fp12 a_sq = fp12_sqr(a);
    Fp12 a_mul = fp12_mul(a, a);
    bool sq_ok = fp2_eq(a_sq.c0.c0, a_mul.c0.c0) && fp2_eq(a_sq.c0.c1, a_mul.c0.c1)
              && fp2_eq(a_sq.c0.c2, a_mul.c0.c2) && fp2_eq(a_sq.c1.c0, a_mul.c1.c0)
              && fp2_eq(a_sq.c1.c1, a_mul.c1.c1) && fp2_eq(a_sq.c1.c2, a_mul.c1.c2);
    printf("TEST 9 (sqr==mul): %s\n", sq_ok ? "PASS" : "FAIL");

    // Test 10: conjugate: conj(a)*a should have c1=0 (it's the norm)
    // Actually conj(a+bw) = a-bw, and (a+bw)(a-bw) = a^2 - b^2*v which is Fp6 (c1=0 in Fp12)
    Fp12 ac = fp12_conj(a);
    Fp12 norm = fp12_mul(a, ac);
    printf("TEST 10 (a*conj(a) c1=0): %s\n", fp6_is_zero(norm.c1) ? "PASS" : "FAIL");
}

int main() {
    printf("=== Fp12 Arithmetic GPU Test ===\n");
    test_fp12<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){printf("CUDA ERROR: %s\n",cudaGetErrorString(err));return 1;}
    printf("=== Done ===\n");
    return 0;
}
