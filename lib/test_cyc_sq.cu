// Test cyclotomic_square against oracle
// Input: pairing result e(G1,G2) — known to be cyclotomic
// Expected output from oracle: cyc_squared values

#include "field.cuh"
#include <cstdio>

// Algorithm 5.5.4 from Guide to Pairing-Based Cryptography
// fp4_square(a, b) -> (c0, c1) where c0 = a²+β·b², c1 = 2ab
// with β = non-residue of Fp2 (which is (1+u) in our tower)
__device__ void fp4_square(const Fp2& a, const Fp2& b, Fp2& out0, Fp2& out1) {
    Fp2 t0 = fp2_sqr(a);
    Fp2 t1 = fp2_sqr(b);
    Fp2 t2 = fp2_mul_nr(t1);   // β * b²
    out0 = fp2_add(t2, t0);     // a² + β·b²
    t2 = fp2_add(a, b);
    t2 = fp2_sqr(t2);
    t2 = fp2_sub(t2, t0);
    out1 = fp2_sub(t2, t1);     // (a+b)² - a² - b² = 2ab
}

__device__ __noinline__ Fp12 cyclotomic_square(const Fp12& f) {
    Fp2 z0=f.c0.c0, z4=f.c0.c1, z3=f.c0.c2;
    Fp2 z2=f.c1.c0, z1=f.c1.c1, z5=f.c1.c2;

    Fp2 t0, t1;
    fp4_square(z0, z1, t0, t1);   // fp4_square(z0, z1)
    // For A
    z0 = fp2_sub(t0, z0); z0 = fp2_add(fp2_add(z0, z0), t0);
    z1 = fp2_add(t1, z1); z1 = fp2_add(fp2_add(z1, z1), t1);

    Fp2 t0b, t1b;
    fp4_square(z2, z3, t0b, t1b);  // fp4_square(z2, z3)

    Fp2 t2, t3;
    fp4_square(z4, z5, t2, t3);    // fp4_square(z4, z5)

    // For C
    z4 = fp2_sub(t2, z4); z4 = fp2_add(fp2_add(z4, z4), t2);
    z5 = fp2_add(t3, z5); z5 = fp2_add(fp2_add(z5, z5), t3);

    // For B: t0 = t3.mul_by_nonresidue() in ic_bls12_381
    // But which t3? In ic_bls12_381, (t0,t1) from fp4(z2,z3) was reused as (t0,t1)
    // then (t2,t3) from fp4(z4,z5). So t3 = from fp4(z4,z5) = our t3.
    // And the reused t0 = from fp4(z2,z3) = our t0b.
    // ic_bls12_381: t0 = t3.mul_by_nonresidue(); z2 = t0+z2; z2 = z2+z2+t0;
    //              z3 = t2 - z3; z3 = z3+z3+t2;
    // Wait — ic_bls12_381 says z3 = t2 - z3 where t2 is from fp4_square(z2,z3)
    // which is our t0b! Not t2 from fp4_square(z4,z5).
    // Let me re-read ic_bls12_381 VERY carefully...

    // ic_bls12_381 code (line by line):
    //   let (t0, t1) = fp4_square(z0, z1);
    //   z0 = t0-z0; z0 = z0+z0+t0;
    //   z1 = t1+z1; z1 = z1+z1+t1;
    //   let (mut t0, t1) = fp4_square(z2, z3);   <-- reuses t0!
    //   let (t2, t3) = fp4_square(z4, z5);
    //   z4 = t0-z4; z4 = z4+z4+t0;   <-- WAIT: this uses t0 from fp4(z2,z3)!
    //   z5 = t1+z5; z5 = z5+z5+t1;   <-- t1 from fp4(z2,z3)!

    // NO! Let me re-read. In the Rust code:
    //   let (t0, t1) = fp4_square(z0, z1);  // A
    //   // apply A
    //   let (mut t0, t1) = fp4_square(z2, z3);  // shadows t0, t1
    //   let (t2, t3) = fp4_square(z4, z5);
    //   // For C: z4 = t0-z4 ... z5 = t1+z5 ...
    //   // BUT t0 and t1 are from fp4(z2,z3), not fp4(z4,z5)!

    // Wait, that can't be right. Let me look again at the ACTUAL ic_bls12_381 source:
    // Line 86: let (mut t0, t1) = fp4_square(z2, z3);
    // Line 87: let (t2, t3) = fp4_square(z4, z5);
    // Line 90: z4 = t0 - z4;   <-- t0 is from fp4(z2,z3)!!
    // Line 93: z5 = t1 + z5;   <-- t1 is from fp4(z2,z3)!!

    // Hmm, but the comment says "For C" which should involve z4,z5.
    // Let me check if this is correct by looking at the algorithm...
    // Algorithm 5.5.4: A = Sq(a0,a1), B = Sq(a2,a3), C = Sq(a4,a5)
    // where a0=z0, a1=z1, a2=z2, a3=z3, a4=z4, a5=z5
    // Then: z0' = 3A0 - 2a0, z1' = 3A1 + 2a1
    //       z2' = 3β·C1 + 2a2, z3' = 3B0 - 2a3
    //       z4' = 3B1 + 2a4... wait let me just look at the paper

    // Actually the ic_bls12_381 code is:
    // For C:
    //   z4 = t0 - z4; z4 = z4 + z4 + t0;    // where t0 is from fp4(z2,z3)!
    //   z5 = t1 + z5; z5 = z5 + z5 + t1;
    //
    // For B:
    //   t0 = t3.mul_by_nonresidue();
    //   z2 = t0 + z2; z2 = z2 + z2 + t0;
    //   z3 = t2 - z3; z3 = z3 + z3 + t2;

    // So the mapping is:
    // A = fp4_square(z0,z1) → first (t0,t1)
    // B_data = fp4_square(z2,z3) → second (t0,t1) which shadows first
    // C_data = fp4_square(z4,z5) → (t2,t3)
    //
    // For C output: uses B_data (t0,t1 from z2,z3)
    // For B output: uses C_data (t2,t3 from z4,z5)

    // This is a CROSS-APPLICATION: C's output uses B's square, B's output uses C's square.
    // That's the algorithm — it's intentional!

    // So our original code was WRONG because we used the wrong mapping.
    // Let me redo with correct mapping:

    // Reset z values
    z0=f.c0.c0; z4=f.c0.c1; z3=f.c0.c2; z2=f.c1.c0; z1=f.c1.c1; z5=f.c1.c2;

    // A = fp4_square(z0, z1)
    Fp2 A0, A1;
    fp4_square(z0, z1, A0, A1);
    z0 = fp2_sub(A0, z0); z0 = fp2_add(fp2_add(z0, z0), A0);
    z1 = fp2_add(A1, z1); z1 = fp2_add(fp2_add(z1, z1), A1);

    // B_data = fp4_square(z2, z3)
    Fp2 B0, B1;
    fp4_square(z2, z3, B0, B1);

    // C_data = fp4_square(z4, z5)
    Fp2 C0, C1;
    fp4_square(z4, z5, C0, C1);

    // For C output: uses B_data!
    z4 = fp2_sub(B0, z4); z4 = fp2_add(fp2_add(z4, z4), B0);
    z5 = fp2_add(B1, z5); z5 = fp2_add(fp2_add(z5, z5), B1);

    // For B output: uses C_data!
    // t0 = C1.mul_by_nonresidue(); z2 = t0+z2; z2 = z2+z2+t0;
    // z3 = C0 - z3; z3 = z3+z3+C0;
    Fp2 nr_C1 = fp2_mul_nr(C1);
    z2 = fp2_add(nr_C1, z2); z2 = fp2_add(fp2_add(z2, z2), nr_C1);
    z3 = fp2_sub(C0, z3); z3 = fp2_add(fp2_add(z3, z3), C0);

    return {{z0,z4,z3},{z2,z1,z5}};
}

__global__ void test() {
    // Load pairing result (cyclotomic element)
    Fp12 inp;
    uint64_t data[72] = {
        0x1972e433a01f85c5ULL,0x97d32b76fd772538ULL,0xc8ce546fc96bcdf9ULL,0xcef63e7366d40614ULL,0xa611342781843780ULL,0x13f3448a3fc6d825ULL,
        0xd26331b02e9d6995ULL,0x9d68a482f7797e7dULL,0x9c9b29248d39ea92ULL,0xf4801ca2e13107aaULL,0xa16c0732bdbcb066ULL,0x083ca4afba360478ULL,
        0x59e261db0916b641ULL,0x2716b6f4b23e960dULL,0xc8e55b10a0bd9c45ULL,0x0bdb0bd99c4deda8ULL,0x8cf89ebf57fdaac5ULL,0x12d6b7929e777a5eULL,
        0x5fc85188b0e15f35ULL,0x34a06e3a8f096365ULL,0xdb3126a6e02ad62cULL,0xfc6f5aa97d9a990bULL,0xa12f55f5eb89c210ULL,0x1723703a926f8889ULL,
        0x93588f2971828778ULL,0x43f65b8611ab7585ULL,0x3183aaf5ec279fdfULL,0xfa73d7e18ac99df6ULL,0x64e176a6a64c99b0ULL,0x179fa78c58388f1fULL,
        0x672a0a11ca2aef12ULL,0x0d11b9b52aa3f16bULL,0xa44412d0699d056eULL,0xc01d0177221a5ba5ULL,0x66e0cede6c735529ULL,0x05f5a71e9fddc339ULL,
        0xd30a88a1b062c679ULL,0x5ac56a5d35fc8304ULL,0xd0c834a6a81f290dULL,0xcd5430c2da3707c7ULL,0xf0c27ff780500af0ULL,0x09245da6e2d72eaeULL,
        0x9f2e0676791b5156ULL,0xe2d1c8234918fe13ULL,0x4c9e459f3c561bf4ULL,0xa3e85e53b9d3e3c1ULL,0x820a121e21a70020ULL,0x15af618341c59accULL,
        0x7c95658c24993ab1ULL,0x73eb38721ca886b9ULL,0x5256d749477434bcULL,0x8ba41902ea504a8bULL,0x04a3d3f80c86ce6dULL,0x18a64a87fb686eaaULL,
        0xbb83e71bb920cf26ULL,0x2a5277ac92a73945ULL,0xfc0ee59f94f046a0ULL,0x7158cdf3786058f7ULL,0x7cc1061b82f945f6ULL,0x03f847aa9fdbe567ULL,
        0x8078dba56134e657ULL,0x1cd7ec9a43998a6eULL,0xb1aa599a1a993766ULL,0xc9a0f62f0842ee44ULL,0x8e159be3b605dffaULL,0x0c86ba0d4af13fc2ULL,
        0xe80ff2a06a52ffb1ULL,0x7694ca48721a906cULL,0x7583183e03b08514ULL,0xf567afdd40cee4e2ULL,0x9a6d96d2e526a5fcULL,0x197e9f49861f2242ULL};
    memcpy(&inp, data, sizeof(inp));

    // Compute cyclotomic square
    Fp12 csq = cyclotomic_square(inp);
    // Compute regular square for comparison
    Fp12 rsq = fp12_sqr(inp);
    // Compute mul(x,x)
    Fp12 msq = fp12_mul(inp, inp);

    // Oracle expected c0.c0 first limb
    bool oracle_match = ((Fp*)&csq)[0].v[0] == 0x610b7774a0972440ULL;
    bool rsq_match = ((Fp*)&rsq)[0].v[0] == 0x610b7774a0972440ULL;
    bool msq_match = ((Fp*)&msq)[0].v[0] == 0x610b7774a0972440ULL;

    printf("cyclotomic_square matches oracle: %s\n", oracle_match ? "PASS" : "FAIL");
    printf("fp12_sqr matches oracle:          %s\n", rsq_match ? "PASS" : "FAIL");
    printf("fp12_mul(x,x) matches oracle:     %s\n", msq_match ? "PASS" : "FAIL");

    // Check full match for cyclotomic
    if (oracle_match) {
        Fp* cp = (Fp*)&csq;
        uint64_t expected_c0[6] = {0x610b7774a0972440ULL,0x20c3c7ca479c2a02ULL,0xa7d777057bc36b2eULL,
                                    0x90828a4a5dd716dbULL,0xeaf10d1745442603ULL,0x08cea6a6710aed6fULL};
        bool full = true;
        for(int i=0;i<6;i++) if(cp[0].v[i]!=expected_c0[i]) full=false;
        printf("cyclotomic_square full c0.c0.c0: %s\n", full ? "PASS" : "FAIL");
    }

    if (!oracle_match) {
        printf("cyc[0]=%016llx  oracle=610b7774a0972440\n", (unsigned long long)((Fp*)&csq)[0].v[0]);
        printf("rsq[0]=%016llx\n", (unsigned long long)((Fp*)&rsq)[0].v[0]);
    }
}

int main() {
    printf("=== Cyclotomic Square Test ===\n");
    test<<<1,1>>>();
    cudaDeviceSynchronize();
    printf("=== Done ===\n");
    return 0;
}
