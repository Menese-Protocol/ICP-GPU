// Test easy part of final exponentiation in isolation
// Easy part: f^(p^6-1) * f^(p^2+1) = conjugate(f)*f^(-1) then frob^2 * that

#include <cstdint>
#include <cstdio>

struct Fp { uint64_t v[6]; };
struct Fp2 { Fp c0, c1; };
struct Fp6 { Fp2 c0, c1, c2; };
struct Fp12 { Fp6 c0, c1; };

__device__ __constant__ uint64_t FP_P[6]={0xb9feffffffffaaabULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};
__device__ __constant__ uint64_t FP_ONE[6]={0x760900000002fffdULL,0xebf4000bc40c0002ULL,0x5f48985753c758baULL,0x77ce585370525745ULL,0x5c071a97a256ec6dULL,0x15f65ec3fa80e493ULL};
#define M0 0x89f3fffcfffcfffdULL

__device__ Fp fp_zero(){Fp r={};return r;}
__device__ Fp fp_one(){Fp r;for(int i=0;i<6;i++)r.v[i]=FP_ONE[i];return r;}
__device__ bool fp_is_zero(const Fp&a){uint64_t ac=0;for(int i=0;i<6;i++)ac|=a.v[i];return ac==0;}
__device__ bool fp_eq(const Fp&a,const Fp&b){for(int i=0;i<6;i++)if(a.v[i]!=b.v[i])return false;return true;}
__device__ Fp fp_add(const Fp&a,const Fp&b){Fp r;unsigned __int128 c=0;for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)a.v[i]+b.v[i]+c;r.v[i]=(uint64_t)s;c=s>>64;}Fp t;unsigned __int128 bw=0;for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw;t.v[i]=(uint64_t)d;bw=(d>>127)&1;}return(bw==0)?t:r;}
__device__ Fp fp_sub(const Fp&a,const Fp&b){Fp r;unsigned __int128 bw=0;for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)a.v[i]-b.v[i]-bw;r.v[i]=(uint64_t)d;bw=(d>>127)&1;}if(bw){unsigned __int128 c=0;for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)r.v[i]+FP_P[i]+c;r.v[i]=(uint64_t)s;c=s>>64;}}return r;}
__device__ Fp fp_neg(const Fp&a){if(fp_is_zero(a))return a;return fp_sub(fp_zero(),a);}
__device__ __noinline__ Fp fp_mul(const Fp&a,const Fp&b){uint64_t t[7]={0};for(int i=0;i<6;i++){uint64_t carry=0;for(int j=0;j<6;j++){unsigned __int128 p=(unsigned __int128)a.v[j]*b.v[i]+t[j]+carry;t[j]=(uint64_t)p;carry=(uint64_t)(p>>64);}t[6]=carry;uint64_t m=t[0]*M0;unsigned __int128 rd=(unsigned __int128)m*FP_P[0]+t[0];carry=(uint64_t)(rd>>64);for(int j=1;j<6;j++){rd=(unsigned __int128)m*FP_P[j]+t[j]+carry;t[j-1]=(uint64_t)rd;carry=(uint64_t)(rd>>64);}t[5]=t[6]+carry;t[6]=(t[5]<carry)?1:0;}Fp r;for(int i=0;i<6;i++)r.v[i]=t[i];Fp s;unsigned __int128 bw=0;for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw;s.v[i]=(uint64_t)d;bw=(d>>127)&1;}return(bw==0)?s:r;}
__device__ Fp fp_sqr(const Fp&a){return fp_mul(a,a);}
__device__ __noinline__ Fp fp_inv(const Fp&a){Fp r=fp_one(),base=a;uint64_t exp[6]={0xb9feffffffffaaa9ULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};for(int w=0;w<6;w++)for(int bit=0;bit<64;bit++){if(w==5&&bit>=61)break;if((exp[w]>>bit)&1)r=fp_mul(r,base);base=fp_sqr(base);}return r;}

__device__ Fp2 fp2_zero(){return{fp_zero(),fp_zero()};}
__device__ Fp2 fp2_one(){return{fp_one(),fp_zero()};}
__device__ bool fp2_eq(const Fp2&a,const Fp2&b){return fp_eq(a.c0,b.c0)&&fp_eq(a.c1,b.c1);}
__device__ Fp2 fp2_add(const Fp2&a,const Fp2&b){return{fp_add(a.c0,b.c0),fp_add(a.c1,b.c1)};}
__device__ Fp2 fp2_sub(const Fp2&a,const Fp2&b){return{fp_sub(a.c0,b.c0),fp_sub(a.c1,b.c1)};}
__device__ Fp2 fp2_neg(const Fp2&a){return{fp_neg(a.c0),fp_neg(a.c1)};}
__device__ Fp2 fp2_mul(const Fp2&a,const Fp2&b){Fp t0=fp_mul(a.c0,b.c0),t1=fp_mul(a.c1,b.c1);return{fp_sub(t0,t1),fp_sub(fp_sub(fp_mul(fp_add(a.c0,a.c1),fp_add(b.c0,b.c1)),t0),t1)};}
__device__ Fp2 fp2_sqr(const Fp2&a){Fp t=fp_mul(a.c0,a.c1);return{fp_mul(fp_add(a.c0,a.c1),fp_sub(a.c0,a.c1)),fp_add(t,t)};}
__device__ Fp2 fp2_mul_nr(const Fp2&a){return{fp_sub(a.c0,a.c1),fp_add(a.c0,a.c1)};}
__device__ Fp2 fp2_conj(const Fp2&a){return{a.c0,fp_neg(a.c1)};}
__device__ Fp2 fp2_inv(const Fp2&a){Fp norm=fp_add(fp_sqr(a.c0),fp_sqr(a.c1));Fp inv=fp_inv(norm);return{fp_mul(a.c0,inv),fp_neg(fp_mul(a.c1,inv))};}

__device__ Fp6 fp6_zero(){return{fp2_zero(),fp2_zero(),fp2_zero()};}
__device__ Fp6 fp6_one(){return{fp2_one(),fp2_zero(),fp2_zero()};}
__device__ Fp6 fp6_add(const Fp6&a,const Fp6&b){return{fp2_add(a.c0,b.c0),fp2_add(a.c1,b.c1),fp2_add(a.c2,b.c2)};}
__device__ Fp6 fp6_sub(const Fp6&a,const Fp6&b){return{fp2_sub(a.c0,b.c0),fp2_sub(a.c1,b.c1),fp2_sub(a.c2,b.c2)};}
__device__ Fp6 fp6_neg(const Fp6&a){return{fp2_neg(a.c0),fp2_neg(a.c1),fp2_neg(a.c2)};}
__device__ __noinline__ Fp6 fp6_mul(const Fp6&a,const Fp6&b){Fp2 a_a=fp2_mul(a.c0,b.c0),b_b=fp2_mul(a.c1,b.c1),c_c=fp2_mul(a.c2,b.c2);return{fp2_add(a_a,fp2_mul_nr(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c1,a.c2),fp2_add(b.c1,b.c2)),b_b),c_c))),fp2_add(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c1),fp2_add(b.c0,b.c1)),a_a),b_b),fp2_mul_nr(c_c)),fp2_add(fp2_sub(fp2_sub(fp2_mul(fp2_add(a.c0,a.c2),fp2_add(b.c0,b.c2)),a_a),c_c),b_b)};}
__device__ Fp6 fp6_mul_by_v(const Fp6&a){return{fp2_mul_nr(a.c2),a.c0,a.c1};}
__device__ __noinline__ Fp6 fp6_inv(const Fp6&f){Fp2 c0s=fp2_sqr(f.c0),c1s=fp2_sqr(f.c1),c2s=fp2_sqr(f.c2);Fp2 c01=fp2_mul(f.c0,f.c1),c02=fp2_mul(f.c0,f.c2),c12=fp2_mul(f.c1,f.c2);Fp2 t0=fp2_sub(c0s,fp2_mul_nr(c12));Fp2 t1=fp2_sub(fp2_mul_nr(c2s),c01);Fp2 t2=fp2_sub(c1s,c02);Fp2 scalar=fp2_add(fp2_mul(f.c0,t0),fp2_mul_nr(fp2_add(fp2_mul(f.c2,t1),fp2_mul(f.c1,t2))));Fp2 si=fp2_inv(scalar);return{fp2_mul(t0,si),fp2_mul(t1,si),fp2_mul(t2,si)};}

__device__ Fp6 fp6_frobenius(const Fp6&f){Fp2 c0=fp2_conj(f.c0);Fp2 c1=fp2_conj(f.c1);Fp2 c2=fp2_conj(f.c2);Fp2 fc1={fp_zero(),{0xcd03c9e48671f071ULL,0x5dab22461fcda5d2ULL,0x587042afd3851b95ULL,0x8eb60ebe01bacb9eULL,0x03f97d6e83d050d2ULL,0x18f0206554638741ULL}};c1=fp2_mul(c1,fc1);Fp2 fc2={{0x890dc9e4867545c3ULL,0x2af322533285a5d5ULL,0x50880866309b7e2cULL,0xa20d1b8c7e881024ULL,0x14e4f04fe2db9068ULL,0x14e56d3f1564853aULL},fp_zero()};c2=fp2_mul(c2,fc2);return{c0,c1,c2};}

__device__ Fp12 fp12_one(){return{fp6_one(),fp6_zero()};}
__device__ __noinline__ Fp12 fp12_mul(const Fp12&a,const Fp12&b){Fp6 aa=fp6_mul(a.c0,b.c0),bb=fp6_mul(a.c1,b.c1);return{fp6_add(aa,fp6_mul_by_v(bb)),fp6_sub(fp6_sub(fp6_mul(fp6_add(a.c0,a.c1),fp6_add(b.c0,b.c1)),aa),bb)};}
__device__ Fp12 fp12_conj(const Fp12&a){return{a.c0,fp6_neg(a.c1)};}
__device__ __noinline__ Fp12 fp12_inv(const Fp12&f){Fp6 t0=fp6_sub(fp6_mul(f.c0,f.c0),fp6_mul_by_v(fp6_mul(f.c1,f.c1)));Fp6 t0i=fp6_inv(t0);return{fp6_mul(f.c0,t0i),fp6_neg(fp6_mul(f.c1,t0i))};}
__device__ Fp12 fp12_frobenius(const Fp12&f){Fp6 c0=fp6_frobenius(f.c0);Fp6 c1=fp6_frobenius(f.c1);Fp2 coeff={{0x07089552b319d465ULL,0xc6695f92b50a8313ULL,0x97e83cccd117228fULL,0xa35baecab2dc29eeULL,0x1ce393ea5daace4dULL,0x08f2220fb0fb66ebULL},{0xb2f66aad4ce5d646ULL,0x5842a06bfc497cecULL,0xcf4895d42599d394ULL,0xc11b9cba40a8e8d0ULL,0x2e3813cbe5a0de89ULL,0x110eefda88847fafULL}};c1={fp2_mul(c1.c0,coeff),fp2_mul(c1.c1,coeff),fp2_mul(c1.c2,coeff)};return{c0,c1};}

__global__ void test_easy_part() {
    // Load known miller loop result
    Fp12 f;
    uint64_t data[72] = {
        0xa067a4e38dd6fea0ULL,0xce174a6ce348e8caULL,0x53e964dbf67fa93eULL,0x5e14ad533455a788ULL,0xbe11f86e0de6770dULL,0x03a22e046e708d71ULL,
        0xa95c3278104a0731ULL,0x5d1858603d8f2f77ULL,0x1528757fa73ed1feULL,0x10631b692e7a1696ULL,0x18b9c8640c65f1cfULL,0x00b1ac642909bf97ULL,
        0x1b084ffe1b41e437ULL,0x08b48d1be8e95430ULL,0x3438cf0c35312aeaULL,0x7a82969f0868335cULL,0xfbd19692f51c6ccaULL,0x06a7cb9b68aa7756ULL,
        0x5f3731f203b80d89ULL,0x9743a51a787cefe3ULL,0x2190fcbea5c7d10eULL,0x48da57ee6bd4a781ULL,0x526aef8aa252e463ULL,0x1461ae5ff2690e00ULL,
        0x064eba2edf2c68fbULL,0x2c8a17202fb5dcf9ULL,0xaec2b98cc1e58ab4ULL,0xd6436f9303a8557aULL,0xb33c6bd2bf897e0fULL,0x0fd24b214c1104caULL,
        0xc4fc89decd238f6aULL,0x8674bcb58aa33344ULL,0xa32380b4307d3d97ULL,0x546dba137ec053ffULL,0x0c91a8d36cd6bf53ULL,0x03310475fa5d50f1ULL,
        0x35951d4dfba91c51ULL,0xaa3c7f10216b5537ULL,0x0a19a78c93509e31ULL,0x527adb08066714b3ULL,0x894d9f074d9cf8faULL,0x073e9b5713188876ULL,
        0x14af0d003f992928ULL,0xb82ff5aa4e6d5792ULL,0x7e566da54e41c7a8ULL,0xc70460323745c572ULL,0x0a7fed49daf8b3c1ULL,0x03b9688a4370961fULL,
        0x3c5623f3f74f2a64ULL,0x8ebf0af77cc62c49ULL,0xa2294263eb2361b9ULL,0x34b65f76f91ecea7ULL,0x595a846df0084bcbULL,0x1271b1eb2f6b10e3ULL,
        0x9870232f453bec18ULL,0xf9893ee829d81d36ULL,0x1b134a659272d960ULL,0x8f256ef89af28b11ULL,0xa5306576198bff9cULL,0x156cb2b28f1b5d00ULL,
        0x3d5b7f3227cab8bfULL,0x6f64c24b8eb0b7fbULL,0x8cd4667412f80d51ULL,0x7bc4e5240b0fbe43ULL,0x55247ba8f7f91e74ULL,0x0b6be53a49b410e3ULL,
        0x12468332df82d2dcULL,0xb2f532d0a25a5f9aULL,0x1a7fff370ea7e58aULL,0x06c8bb791513a71dULL,0x7c10ed472bb3c3c4ULL,0x0ed44c77f58001ecULL};
    memcpy(&f, data, sizeof(f));

    // Step 1: t0 = f^(p^6) = conjugate(f)
    Fp12 t0 = fp12_conj(f);

    // Step 2: t1 = f^(-1)
    Fp12 t1 = fp12_inv(f);

    // Verify: f * f^-1 = 1
    Fp12 check = fp12_mul(f, t1);
    Fp* cp = (Fp*)&check;
    bool inv_ok = fp_eq(cp[0], fp_one()); // c0.c0.c0 should be mont(1)
    printf("TEST inv(f)*f c0.c0.c0==ONE: %s\n", inv_ok ? "PASS" : "FAIL");
    if(!inv_ok){
        printf("  got=[%016llx,%016llx,%016llx,%016llx,%016llx,%016llx]\n",
               (unsigned long long)cp[0].v[0],(unsigned long long)cp[0].v[1],
               (unsigned long long)cp[0].v[2],(unsigned long long)cp[0].v[3],
               (unsigned long long)cp[0].v[4],(unsigned long long)cp[0].v[5]);
    }

    // Step 3: t2 = t0 * t1 = f^(p^6 - 1)
    Fp12 t2 = fp12_mul(t0, t1);

    // Step 4: t1 = t2 (save)
    t1 = t2;

    // Step 5: t2 = frob^2(t2) * t1 = t2^(p^2+1) = f^((p^6-1)(p^2+1))
    t2 = fp12_mul(fp12_frobenius(fp12_frobenius(t2)), t1);

    // Output easy part result for comparison
    Fp* rp = (Fp*)&t2;
    printf("\nEasy part result:\n");
    printf("  c0.c0.c0=[%016llx,%016llx,%016llx,%016llx,%016llx,%016llx]\n",
           (unsigned long long)rp[0].v[0],(unsigned long long)rp[0].v[1],
           (unsigned long long)rp[0].v[2],(unsigned long long)rp[0].v[3],
           (unsigned long long)rp[0].v[4],(unsigned long long)rp[0].v[5]);
}

int main() {
    printf("=== Easy Part Test ===\n");
    test_easy_part<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){printf("CUDA ERROR: %s\n",cudaGetErrorString(err));return 1;}
    printf("=== Done ===\n");
    return 0;
}
