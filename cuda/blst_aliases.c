// Symbol aliases: sppark calls mul_mont_384, blst exports mulx_mont_384
// This bridges the naming gap when blst is compiled with ADX support

#include <stddef.h>

typedef unsigned long long limb_t;

// Declare the ADX versions that blst exports
extern void mulx_mont_384(limb_t ret[], const limb_t a[], const limb_t b[],
                           const limb_t p[], limb_t n0);
extern void sqrx_mont_384(limb_t ret[], const limb_t a[],
                           const limb_t p[], limb_t n0);
extern void mulx_mont_384x(limb_t ret[], const limb_t a[], const limb_t b[],
                            const limb_t p[], limb_t n0);
extern void fromx_mont_384(limb_t ret[], const limb_t a[],
                            const limb_t p[], limb_t n0);
extern void sqrx_n_mul_mont_384(limb_t ret[], const limb_t a[], size_t count,
                                const limb_t p[], limb_t n0, const limb_t b[]);

// Alias the non-ADX names to the ADX implementations
void mul_mont_384(limb_t ret[], const limb_t a[], const limb_t b[],
                  const limb_t p[], limb_t n0) {
    mulx_mont_384(ret, a, b, p, n0);
}

void sqr_mont_384(limb_t ret[], const limb_t a[],
                  const limb_t p[], limb_t n0) {
    sqrx_mont_384(ret, a, p, n0);
}

void mul_mont_384x(limb_t ret[], const limb_t a[], const limb_t b[],
                   const limb_t p[], limb_t n0) {
    mulx_mont_384x(ret, a, b, p, n0);
}

void from_mont_384(limb_t ret[], const limb_t a[],
                   const limb_t p[], limb_t n0) {
    fromx_mont_384(ret, a, p, n0);
}

void sqr_n_mul_mont_384(limb_t ret[], const limb_t a[], size_t count,
                        const limb_t p[], limb_t n0, const limb_t b[]) {
    sqrx_n_mul_mont_384(ret, a, count, p, n0, b);
}

// 256-bit aliases (for scalar/fr_t operations)
extern void fromx_mont_256(limb_t ret[], const limb_t a[],
                           const limb_t p[], limb_t n0);
extern void mulx_mont_sparse_256(limb_t ret[], const limb_t a[], const limb_t b[],
                                  const limb_t p[], limb_t n0);

void from_mont_256(limb_t ret[], const limb_t a[],
                   const limb_t p[], limb_t n0) {
    fromx_mont_256(ret, a, p, n0);
}

void mul_mont_sparse_256(limb_t ret[], const limb_t a[], const limb_t b[],
                          const limb_t p[], limb_t n0) {
    mulx_mont_sparse_256(ret, a, b, p, n0);
}

// Inverse and reduction aliases
extern void ctx_inverse_mod_384(limb_t ret[], const limb_t inp[],
                                 const limb_t mod[], const limb_t modx[]);
extern void redcx_mont_384(limb_t ret[], const limb_t a[],
                            const limb_t p[], limb_t n0);

void ct_inverse_mod_384(limb_t ret[], const limb_t inp[],
                         const limb_t mod[], const limb_t modx[]) {
    ctx_inverse_mod_384(ret, inp, mod, modx);
}

void redc_mont_384(limb_t ret[], const limb_t a[],
                    const limb_t p[], limb_t n0) {
    redcx_mont_384(ret, a, p, n0);
}
