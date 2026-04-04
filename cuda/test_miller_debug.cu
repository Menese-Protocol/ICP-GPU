// Debug: output first few line coefficients and intermediate f values
// to pinpoint where GPU diverges from ic_bls12_381

#include <cstdint>
#include <cstdio>

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

// === Proven Fp/Fp2/Fp6/Fp12 (same as test_miller.cu) ===
__device__ Fp fp_zero(){Fp r={};return r;}
__device__ Fp fp_one(){Fp r;for(int i=0;i<6;i++)r.v[i]=FP_ONE[i];return r;}
__device__ bool fp_is_zero(const Fp&a){uint64_t acc=0;for(int i=0;i<6;i++)acc|=a.v[i];return acc==0;}
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

__device__ void print_fp2(const char* label, const Fp2& a) {
    printf("  %s.c0 = [%016llx, %016llx, %016llx, %016llx, %016llx, %016llx]\n", label,
           (unsigned long long)a.c0.v[0],(unsigned long long)a.c0.v[1],(unsigned long long)a.c0.v[2],
           (unsigned long long)a.c0.v[3],(unsigned long long)a.c0.v[4],(unsigned long long)a.c0.v[5]);
    printf("  %s.c1 = [%016llx, %016llx, %016llx, %016llx, %016llx, %016llx]\n", label,
           (unsigned long long)a.c1.v[0],(unsigned long long)a.c1.v[1],(unsigned long long)a.c1.v[2],
           (unsigned long long)a.c1.v[3],(unsigned long long)a.c1.v[4],(unsigned long long)a.c1.v[5]);
}

// Line double — same as test_miller.cu
__device__ Fp6 line_double(G2Proj& r, const Fp& two_inv_fp) {
    Fp2 two_inv = {two_inv_fp, fp_zero()};
    Fp2 a = fp2_mul(fp2_mul(r.x, r.y), two_inv);
    Fp2 b = fp2_sqr(r.y);
    Fp2 c = fp2_sqr(r.z);
    Fp2 c3 = fp2_add(fp2_add(c, c), c);
    Fp2 e = fp2_mul_nr(c3);
    e = fp2_add(e, e); e = fp2_add(e, e);
    Fp2 f = fp2_add(fp2_add(e, e), e);
    Fp2 g = fp2_mul(fp2_add(b, f), two_inv);
    Fp2 h = fp2_sqr(fp2_add(r.y, r.z));
    h = fp2_sub(h, fp2_add(b, c));
    Fp2 i = fp2_sub(e, b);
    Fp2 j = fp2_sqr(r.x);
    Fp2 e_sq = fp2_sqr(e);
    r.x = fp2_mul(a, fp2_sub(b, f));
    r.y = fp2_sub(fp2_sqr(g), fp2_add(fp2_add(e_sq, e_sq), e_sq));
    r.z = fp2_mul(b, h);
    return {i, fp2_add(fp2_add(j,j),j), fp2_neg(h)};
}

__global__ void debug_first_step() {
    // G1 generator
    G1Affine g1;
    g1.x.v[0]=0x5cb38790fd530c16ULL; g1.x.v[1]=0x7817fc679976fff5ULL;
    g1.x.v[2]=0x154f95c7143ba1c1ULL; g1.x.v[3]=0xf0ae6acdf3d0e747ULL;
    g1.x.v[4]=0xedce6ecc21dbf440ULL; g1.x.v[5]=0x120177419e0bfb75ULL;
    g1.y.v[0]=0xbaac93d50ce72271ULL; g1.y.v[1]=0x8c22631a7918fd8eULL;
    g1.y.v[2]=0xdd595f13570725ceULL; g1.y.v[3]=0x51ac582950405194ULL;
    g1.y.v[4]=0x0e1c8c3fad0059c0ULL; g1.y.v[5]=0x0bbc3efc5008a26aULL;

    // G2 generator
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

    Fp two_inv;
    two_inv.v[0]=0x1804000000015554ULL; two_inv.v[1]=0x855000053ab00001ULL;
    two_inv.v[2]=0x633cb57c253c276fULL; two_inv.v[3]=0x6e22d1ec31ebb502ULL;
    two_inv.v[4]=0xd3916126f2d14ca2ULL; two_inv.v[5]=0x17fbb8571a006596ULL;

    // First step: initialize R = Q (projective)
    G2Proj R = {g2.x, g2.y, fp2_one()};

    printf("=== Initial R (should be G2 generator) ===\n");
    print_fp2("R.x", R.x);
    print_fp2("R.y", R.y);
    print_fp2("R.z", R.z);

    // First doubling (bit 62 of BLS_X = 1, so first iteration is double)
    printf("\n=== First line_double ===\n");
    Fp6 coeffs = line_double(R, two_inv);
    printf("Coefficients:\n");
    print_fp2("coeff[0] (i)", coeffs.c0);
    print_fp2("coeff[1] (3j)", coeffs.c1);
    print_fp2("coeff[2] (-h)", coeffs.c2);

    printf("\nR after double:\n");
    print_fp2("R.x", R.x);
    print_fp2("R.y", R.y);
    print_fp2("R.z", R.z);
}

int main() {
    printf("=== Miller Loop Debug: First Doubling Step ===\n\n");
    debug_first_step<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){printf("CUDA ERROR: %s\n",cudaGetErrorString(err));return 1;}
    return 0;
}
