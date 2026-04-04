// Symbol aliases: sppark calls non-ADX names, blst exports ADX names
typedef unsigned long long limb_t;
typedef unsigned long size_t;
extern void mulx_mont_384(limb_t[],const limb_t[],const limb_t[],const limb_t[],limb_t);
extern void sqrx_mont_384(limb_t[],const limb_t[],const limb_t[],limb_t);
extern void mulx_mont_384x(limb_t[],const limb_t[],const limb_t[],const limb_t[],limb_t);
extern void fromx_mont_384(limb_t[],const limb_t[],const limb_t[],limb_t);
extern void sqrx_n_mul_mont_384(limb_t[],const limb_t[],size_t,const limb_t[],limb_t,const limb_t[]);
extern void ctx_inverse_mod_384(limb_t[],const limb_t[],const limb_t[],const limb_t[]);
extern void redcx_mont_384(limb_t[],const limb_t[],const limb_t[],limb_t);
extern void fromx_mont_256(limb_t[],const limb_t[],const limb_t[],limb_t);
extern void mulx_mont_sparse_256(limb_t[],const limb_t[],const limb_t[],const limb_t[],limb_t);
void mul_mont_384(limb_t r[],const limb_t a[],const limb_t b[],const limb_t p[],limb_t n){mulx_mont_384(r,a,b,p,n);}
void sqr_mont_384(limb_t r[],const limb_t a[],const limb_t p[],limb_t n){sqrx_mont_384(r,a,p,n);}
void mul_mont_384x(limb_t r[],const limb_t a[],const limb_t b[],const limb_t p[],limb_t n){mulx_mont_384x(r,a,b,p,n);}
void from_mont_384(limb_t r[],const limb_t a[],const limb_t p[],limb_t n){fromx_mont_384(r,a,p,n);}
void sqr_n_mul_mont_384(limb_t r[],const limb_t a[],size_t c,const limb_t p[],limb_t n,const limb_t b[]){sqrx_n_mul_mont_384(r,a,c,p,n,b);}
void ct_inverse_mod_384(limb_t r[],const limb_t i[],const limb_t m[],const limb_t x[]){ctx_inverse_mod_384(r,i,m,x);}
void redc_mont_384(limb_t r[],const limb_t a[],const limb_t p[],limb_t n){redcx_mont_384(r,a,p,n);}
void from_mont_256(limb_t r[],const limb_t a[],const limb_t p[],limb_t n){fromx_mont_256(r,a,p,n);}
void mul_mont_sparse_256(limb_t r[],const limb_t a[],const limb_t b[],const limb_t p[],limb_t n){mulx_mont_sparse_256(r,a,b,p,n);}
