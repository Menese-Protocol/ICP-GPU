// Test Frobenius map in isolation
// Property: frobenius^6 = conjugate, frobenius^12 = identity

#include <cstdint>
#include <cstdio>

struct Fp { uint64_t v[6]; };
struct Fp2 { Fp c0, c1; };
struct Fp6 { Fp2 c0, c1, c2; };
struct Fp12 { Fp6 c0, c1; };

__device__ __constant__ uint64_t FP_P[6] = {
    0xb9feffffffffaaabULL,0x1eabfffeb153ffffULL,0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL,0x4b1ba7b6434bacd7ULL,0x1a0111ea397fe69aULL};
__device__ __constant__ uint64_t FP_ONE[6] = {
    0x760900000002fffdULL,0xebf4000bc40c0002ULL,0x5f48985753c758baULL,
    0x77ce585370525745ULL,0x5c071a97a256ec6dULL,0x15f65ec3fa80e493ULL};
#define M0 0x89f3fffcfffcfffdULL

__device__ Fp fp_zero(){Fp r={};return r;}
__device__ Fp fp_one(){Fp r;for(int i=0;i<6;i++)r.v[i]=FP_ONE[i];return r;}
__device__ bool fp_is_zero(const Fp&a){uint64_t acc=0;for(int i=0;i<6;i++)acc|=a.v[i];return acc==0;}
__device__ bool fp_eq(const Fp&a,const Fp&b){for(int i=0;i<6;i++)if(a.v[i]!=b.v[i])return false;return true;}
__device__ Fp fp_add(const Fp&a,const Fp&b){
    Fp r;unsigned __int128 c=0;
    for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)a.v[i]+b.v[i]+c;r.v[i]=(uint64_t)s;c=s>>64;}
    Fp t;unsigned __int128 bw=0;
    for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw;t.v[i]=(uint64_t)d;bw=(d>>127)&1;}
    return(bw==0)?t:r;
}
__device__ Fp fp_sub(const Fp&a,const Fp&b){
    Fp r;unsigned __int128 bw=0;
    for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)a.v[i]-b.v[i]-bw;r.v[i]=(uint64_t)d;bw=(d>>127)&1;}
    if(bw){unsigned __int128 c=0;for(int i=0;i<6;i++){unsigned __int128 s=(unsigned __int128)r.v[i]+FP_P[i]+c;r.v[i]=(uint64_t)s;c=s>>64;}}
    return r;
}
__device__ Fp fp_neg(const Fp&a){if(fp_is_zero(a))return a;return fp_sub(fp_zero(),a);}
__device__ __noinline__ Fp fp_mul(const Fp&a,const Fp&b){
    uint64_t t[7]={0};
    for(int i=0;i<6;i++){
        uint64_t carry=0;
        for(int j=0;j<6;j++){unsigned __int128 p=(unsigned __int128)a.v[j]*b.v[i]+t[j]+carry;t[j]=(uint64_t)p;carry=(uint64_t)(p>>64);}
        t[6]=carry;uint64_t m=t[0]*M0;unsigned __int128 rd=(unsigned __int128)m*FP_P[0]+t[0];carry=(uint64_t)(rd>>64);
        for(int j=1;j<6;j++){rd=(unsigned __int128)m*FP_P[j]+t[j]+carry;t[j-1]=(uint64_t)rd;carry=(uint64_t)(rd>>64);}
        t[5]=t[6]+carry;t[6]=(t[5]<carry)?1:0;
    }
    Fp r;for(int i=0;i<6;i++)r.v[i]=t[i];
    Fp s;unsigned __int128 bw=0;
    for(int i=0;i<6;i++){unsigned __int128 d=(unsigned __int128)r.v[i]-FP_P[i]-bw;s.v[i]=(uint64_t)d;bw=(d>>127)&1;}
    return(bw==0)?s:r;
}

__device__ Fp2 fp2_zero(){return{fp_zero(),fp_zero()};}
__device__ Fp2 fp2_one(){return{fp_one(),fp_zero()};}
__device__ bool fp2_eq(const Fp2&a,const Fp2&b){return fp_eq(a.c0,b.c0)&&fp_eq(a.c1,b.c1);}
__device__ Fp2 fp2_neg(const Fp2&a){return{fp_neg(a.c0),fp_neg(a.c1)};}
__device__ Fp2 fp2_mul(const Fp2&a,const Fp2&b){
    Fp t0=fp_mul(a.c0,b.c0),t1=fp_mul(a.c1,b.c1);
    return{fp_sub(t0,t1),fp_sub(fp_sub(fp_mul(fp_add(a.c0,a.c1),fp_add(b.c0,b.c1)),t0),t1)};
}
__device__ Fp2 fp2_conj(const Fp2&a){return{a.c0,fp_neg(a.c1)};}
__device__ Fp2 fp2_mul_nr(const Fp2&a){return{fp_sub(a.c0,a.c1),fp_add(a.c0,a.c1)};}

// Fp6 frobenius — exact from ic_bls12_381
__device__ Fp6 fp6_frobenius(const Fp6& f) {
    Fp2 c0 = fp2_conj(f.c0);
    Fp2 c1 = fp2_conj(f.c1);
    Fp2 c2 = fp2_conj(f.c2);
    Fp2 fc1 = {fp_zero(),
        {0xcd03c9e48671f071ULL,0x5dab22461fcda5d2ULL,0x587042afd3851b95ULL,
         0x8eb60ebe01bacb9eULL,0x03f97d6e83d050d2ULL,0x18f0206554638741ULL}};
    c1 = fp2_mul(c1, fc1);
    Fp2 fc2 = {{0x890dc9e4867545c3ULL,0x2af322533285a5d5ULL,0x50880866309b7e2cULL,
                 0xa20d1b8c7e881024ULL,0x14e4f04fe2db9068ULL,0x14e56d3f1564853aULL},
                fp_zero()};
    c2 = fp2_mul(c2, fc2);
    return {c0, c1, c2};
}

// Fp12 frobenius — exact from ic_bls12_381
__device__ Fp12 fp12_frobenius(const Fp12& f) {
    Fp6 c0 = fp6_frobenius(f.c0);
    Fp6 c1 = fp6_frobenius(f.c1);
    Fp2 coeff = {
        {0x07089552b319d465ULL,0xc6695f92b50a8313ULL,0x97e83cccd117228fULL,
         0xa35baecab2dc29eeULL,0x1ce393ea5daace4dULL,0x08f2220fb0fb66ebULL},
        {0xb2f66aad4ce5d646ULL,0x5842a06bfc497cecULL,0xcf4895d42599d394ULL,
         0xc11b9cba40a8e8d0ULL,0x2e3813cbe5a0de89ULL,0x110eefda88847fafULL}};
    // c1 *= Fp6::from(coeff) = Fp6(coeff, 0, 0), so each component *= coeff
    c1 = {fp2_mul(c1.c0, coeff), fp2_mul(c1.c1, coeff), fp2_mul(c1.c2, coeff)};
    return {c0, c1};
}

__device__ Fp12 fp12_conj(const Fp12&a){
    return{a.c0, {fp2_neg(a.c1.c0),fp2_neg(a.c1.c1),fp2_neg(a.c1.c2)}};
}

__device__ Fp fp_from_int(int n){Fp r=fp_zero();Fp one=fp_one();for(int i=0;i<n;i++)r=fp_add(r,one);return r;}

__global__ void test_frobenius() {
    // Build test Fp12: (Fp2(1,2), Fp2(3,4), Fp2(5,6)) / (Fp2(7,8), Fp2(9,10), Fp2(11,12))
    Fp12 a = {
        {{fp_from_int(1),fp_from_int(2)},{fp_from_int(3),fp_from_int(4)},{fp_from_int(5),fp_from_int(6)}},
        {{fp_from_int(7),fp_from_int(8)},{fp_from_int(9),fp_from_int(10)},{fp_from_int(11),fp_from_int(12)}}
    };

    // Test 1: frobenius^6 should equal conjugate
    Fp12 f6 = a;
    for(int i=0;i<6;i++) f6 = fp12_frobenius(f6);
    Fp12 conj = fp12_conj(a);

    bool match6 = true;
    Fp* f6p = (Fp*)&f6;
    Fp* cp = (Fp*)&conj;
    for(int i=0;i<12;i++) if(!fp_eq(f6p[i],cp[i])){match6=false;break;}
    printf("TEST 1 (frob^6 == conjugate): %s\n", match6 ? "PASS" : "FAIL");

    // Test 2: frobenius^12 should equal identity
    Fp12 f12 = a;
    for(int i=0;i<12;i++) f12 = fp12_frobenius(f12);
    Fp* f12p = (Fp*)&f12;
    Fp* ap = (Fp*)&a;
    bool match12 = true;
    for(int i=0;i<12;i++) if(!fp_eq(f12p[i],ap[i])){match12=false;break;}
    printf("TEST 2 (frob^12 == identity): %s\n", match12 ? "PASS" : "FAIL");

    if(!match6) {
        printf("\nDEBUG frob^6 c0.c0:\n");
        printf("  got =[%016llx,%016llx,%016llx,%016llx,%016llx,%016llx]\n",
               (unsigned long long)f6p[0].v[0],(unsigned long long)f6p[0].v[1],
               (unsigned long long)f6p[0].v[2],(unsigned long long)f6p[0].v[3],
               (unsigned long long)f6p[0].v[4],(unsigned long long)f6p[0].v[5]);
        printf("  conj=[%016llx,%016llx,%016llx,%016llx,%016llx,%016llx]\n",
               (unsigned long long)cp[0].v[0],(unsigned long long)cp[0].v[1],
               (unsigned long long)cp[0].v[2],(unsigned long long)cp[0].v[3],
               (unsigned long long)cp[0].v[4],(unsigned long long)cp[0].v[5]);
    }
}

int main() {
    printf("=== Frobenius Map Test ===\n");
    test_frobenius<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){printf("CUDA ERROR: %s\n",cudaGetErrorString(err));return 1;}
    printf("=== Done ===\n");
    return 0;
}
