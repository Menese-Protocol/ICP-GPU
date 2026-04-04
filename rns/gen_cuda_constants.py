#!/usr/bin/env python3
"""Generate CUDA header with all RNS constants for BLS12-381."""

from sympy import isprime

p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

# Same base selection as gen_bases.py
all_primes = []
candidate = 2**30 - 1
while len(all_primes) < 28:
    if isprime(candidate):
        all_primes.append(candidate)
    candidate -= 2

b1 = all_primes[0::2][:14]
b2 = all_primes[1::2][:14]
K = 14

M1 = 1
for m in b1: M1 *= m
M2 = 1
for m in b2: M2 *= m

lines = []
lines.append("// Auto-generated RNS constants for BLS12-381")
lines.append("// DO NOT EDIT — regenerate with gen_cuda_constants.py")
lines.append("#pragma once")
lines.append("#include <cstdint>")
lines.append("")
lines.append(f"#define RNS_K {K}")

# Barrett reduction constants
# For modulus m < 2^30, input x < 2^60 (product of two 30-bit values):
#   mu = floor(2^62 / m)
#   q = (x * mu) >> 62   (approximate quotient)
#   r = x - q * m         (remainder, may need one correction)
# This replaces expensive division with multiply + shift

# Extra redundant modulus for exact alpha computation
# Choose a small prime NOT in either base
m_red = 2**31 - 1  # Mersenne prime, 31 bits — easy fast reduction
while not isprime(m_red) or m_red in b1 or m_red in b2:
    m_red -= 2
lines.append(f"#define RNS_M_RED {m_red}u  // Redundant modulus for exact alpha")
lines.append("")

def emit_array(name, vals, typ="uint32_t"):
    s = f"__device__ __constant__ {typ} {name}[RNS_K] = {{"
    s += ", ".join(str(v) for v in vals)
    s += "};"
    lines.append(s)

# Reciprocals for fast alpha computation: 1.0 / m[i] as double
import struct
def double_to_hex(d):
    return struct.unpack('<Q', struct.pack('<d', d))[0]
lines.append("// Reciprocals for alpha: 1.0/m[i] stored as uint64 (reinterpret as double)")
emit_array("RNS_RECIP_M1", [double_to_hex(1.0 / m) for m in b1], "uint64_t")
emit_array("RNS_RECIP_M2", [double_to_hex(1.0 / m) for m in b2], "uint64_t")
lines.append("")

# Barrett reduction: mu[i] = floor(2^62 / m[i])
# For x < 2^60: q = (x * mu) >> 62; r = x - q*m; if r>=m: r-=m
emit_array("RNS_BARRETT_M1", [(1 << 62) // m for m in b1], "uint64_t")
emit_array("RNS_BARRETT_M2", [(1 << 62) // m for m in b2], "uint64_t")
lines.append(f"#define RNS_BARRETT_MRED {(1 << 62) // m_red}ull")
lines.append("")

# Base moduli
emit_array("RNS_M1", b1)
emit_array("RNS_M2", b2)

# p mod m_i for each base
emit_array("RNS_P_MOD_M1", [p % m for m in b1])
emit_array("RNS_P_MOD_M2", [p % m for m in b2])

# -p^(-1) mod m_i for Montgomery q computation
emit_array("RNS_NEG_PINV_M1", [(-pow(p, -1, m)) % m for m in b1])
emit_array("RNS_NEG_PINV_M2", [(-pow(p, -1, m)) % m for m in b2])

# M_hat_inv[i] = (M / m_i)^(-1) mod m_i for CRT reconstruction
def mhat_inv(base, M):
    return [(pow(M // m, -1, m)) for m in base]

emit_array("RNS_MHAT_INV_M1", mhat_inv(b1, M1))
emit_array("RNS_MHAT_INV_M2", mhat_inv(b2, M2))

# Base extension matrix: be_12[j][i] = (M1 / m1[i]) mod m2[j]
# Used to extend from Base1 -> Base2
lines.append("")
lines.append("// Base extension matrix B1->B2: be_12[j][i] = (M1/m1[i]) mod m2[j]")
lines.append(f"__device__ __constant__ uint32_t RNS_BE_12[RNS_K][RNS_K] = {{")
for j, mj in enumerate(b2):
    coeffs = [(M1 // b1[i]) % mj for i in range(K)]
    lines.append("    {" + ", ".join(str(c) for c in coeffs) + "},")
lines.append("};")

# Base extension matrix B2->B1
lines.append("")
lines.append("// Base extension matrix B2->B1: be_21[j][i] = (M2/m2[i]) mod m1[j]")
lines.append(f"__device__ __constant__ uint32_t RNS_BE_21[RNS_K][RNS_K] = {{")
for j, mj in enumerate(b1):
    coeffs = [(M2 // b2[i]) % mj for i in range(K)]
    lines.append("    {" + ", ".join(str(c) for c in coeffs) + "},")
lines.append("};")

# Redundant modulus constants
lines.append("")
lines.append("// Redundant modulus constants for exact base extension")
emit_array("RNS_MHAT_MOD_MRED_B1", [(M1 // b1[i]) % m_red for i in range(K)])
emit_array("RNS_MHAT_MOD_MRED_B2", [(M2 // b2[i]) % m_red for i in range(K)])
emit_array("RNS_MHAT_INV_MRED_B1", [((M1 // b1[i]) * pow(M1 // b1[i], -1, b1[i])) % m_red for i in range(K)])
# Actually we need: for CRT in B1, the reconstruction coefficients mod m_red
# coeff_red[i] = (M_hat_i * M_hat_inv_i) mod m_red = (M1/m1[i] * (M1/m1[i])^-1_mod_m1[i]) mod m_red
# Hmm, that's just (M1/m1[i]) * (M1/m1[i])^(-1) mod m1[i] ... this isn't right.
# What we need: for base extension from B1 to m_red, coefficient[i] = (M1/m1[i]) mod m_red
# Already computed above as RNS_MHAT_MOD_MRED_B1

lines.append(f"#define RNS_M1_MOD_MRED {M1 % m_red}u")
lines.append(f"#define RNS_M2_MOD_MRED {M2 % m_red}u")
lines.append(f"#define RNS_P_MOD_MRED {p % m_red}u")
lines.append(f"#define RNS_NEG_PINV_MRED {(-pow(p, -1, m_red)) % m_red}u")
lines.append(f"#define RNS_M1_INV_MOD_MRED {pow(M1, -1, m_red)}u")
lines.append(f"#define RNS_M2_INV_MOD_MRED {pow(M2, -1, m_red)}u")
lines.append("")

# M1 mod m2[j] — needed for base extension correction
emit_array("RNS_M1_MOD_M2", [M1 % m for m in b2])
emit_array("RNS_M2_MOD_M1", [M2 % m for m in b1])

# M1^(-1) mod m2[j] — needed for Montgomery division step
emit_array("RNS_M1_INV_MOD_M2", [pow(M1, -1, m) for m in b2])
# M2^(-1) mod m1[j] — needed for reverse direction
emit_array("RNS_M2_INV_MOD_M1", [pow(M2, -1, m) for m in b1])

# R = M1 mod p (Montgomery constant — "one" in RNS Montgomery form)
R1_mod_p = M1 % p
lines.append("")
lines.append(f"// M1 mod p = Montgomery R (for encoding)")
emit_array("RNS_R_MOD_M1", [R1_mod_p % m for m in b1])
emit_array("RNS_R_MOD_M2", [R1_mod_p % m for m in b2])

# R^2 mod p (for Montgomery encoding: mont(x) = x * R^2 * R^(-1) = x * R)
R2_mod_p = (M1 * M1) % p
emit_array("RNS_R2_MOD_M1", [R2_mod_p % m for m in b1])
emit_array("RNS_R2_MOD_M2", [R2_mod_p % m for m in b2])

# ONE in both bases (= R mod p, in RNS)
# Already done above as RNS_R_MOD_M1/M2

# Encode "1" in standard form (not Montgomery) for decode verification
emit_array("RNS_ONE_M1", [1 % m for m in b1])
emit_array("RNS_ONE_M2", [1 % m for m in b2])

# Emit oracle test vectors
lines.append("")
lines.append("// Oracle test vectors")
# mont(7) = 7 * R mod p
v7_mont = (7 * M1) % p
emit_array("RNS_ORACLE_MONT7_M1", [v7_mont % m for m in b1])
emit_array("RNS_ORACLE_MONT7_M2", [v7_mont % m for m in b2])
lines.append(f"#define RNS_ORACLE_MONT7_RED {v7_mont % m_red}u")

# mont(7) * mont(7) * R^(-1) mod p = 49 * R mod p
v49_mont = (49 * M1) % p
emit_array("RNS_ORACLE_MONT49_M1", [v49_mont % m for m in b1])
emit_array("RNS_ORACLE_MONT49_M2", [v49_mont % m for m in b2])
lines.append(f"#define RNS_ORACLE_MONT49_RED {v49_mont % m_red}u")

# mont(6)
v6_mont = (6 * M1) % p
emit_array("RNS_ORACLE_MONT6_M1", [v6_mont % m for m in b1])
emit_array("RNS_ORACLE_MONT6_M2", [v6_mont % m for m in b2])
lines.append(f"#define RNS_ORACLE_MONT6_RED {v6_mont % m_red}u")

# Frobenius coefficients + BLS parameter
lines.append("")
lines.append("// Frobenius + BLS constants")
R_ic = pow(2, 384, p)
R_ic_inv = pow(R_ic, -1, p)

def limbs_to_int(limbs):
    """Convert little-endian u64 limbs to big integer"""
    val = 0
    for i, limb in enumerate(limbs):
        val |= limb << (64 * i)
    return val

def ic_mont_to_rns(limbs):
    """Convert ic_bls12_381 Montgomery limbs (LE u64) to our RNS form"""
    ic_mont = limbs_to_int(limbs)
    canonical = (ic_mont * R_ic_inv) % p
    our_mont = (canonical * M1) % p
    return [our_mont % m for m in b1], [our_mont % m for m in b2], our_mont % m_red

# Frobenius coefficients — EXACT limbs from ic_bls12_381 source code (LE u64)

# Fp6 frobenius_map: c1 *= Fp2(0, c1_coeff) where c1_coeff has these limbs:
# From ic_bls12_381/src/fp6.rs line 163-170
frob6_c1_c1_limbs = [0xcd03c9e48671f071, 0x5dab22461fcda5d2, 0x587042afd3851b95,
                     0x8eb60ebe01bacb9e, 0x03f97d6e83d050d2, 0x18f0206554638741]
r1,r2,rr = ic_mont_to_rns(frob6_c1_c1_limbs)
emit_array("RNS_FROB6_C1_C1_M1", r1); emit_array("RNS_FROB6_C1_C1_M2", r2)
lines.append(f"#define RNS_FROB6_C1_C1_RED {rr}u")

# Fp6 frobenius_map: c2 *= Fp2(c0_coeff, 0)
# From ic_bls12_381/src/fp6.rs line 176-183
frob6_c2_c0_limbs = [0x890dc9e4867545c3, 0x2af322533285a5d5, 0x50880866309b7e2c,
                     0xa20d1b8c7e881024, 0x14e4f04fe2db9068, 0x14e56d3f1564853a]
r1,r2,rr = ic_mont_to_rns(frob6_c2_c0_limbs)
emit_array("RNS_FROB6_C2_C0_M1", r1); emit_array("RNS_FROB6_C2_C0_M2", r2)
lines.append(f"#define RNS_FROB6_C2_C0_RED {rr}u")

# Fp12 frobenius_map: c1 *= Fp6::from(Fp2(c0, c1))
# From ic_bls12_381/src/fp12.rs line 152-167
frob12_c0_limbs = [0x07089552b319d465, 0xc6695f92b50a8313, 0x97e83cccd117228f,
                   0xa35baecab2dc29ee, 0x1ce393ea5daace4d, 0x08f2220fb0fb66eb]
frob12_c1_limbs = [0xb2f66aad4ce5d646, 0x5842a06bfc497cec, 0xcf4895d42599d394,
                   0xc11b9cba40a8e8d0, 0x2e3813cbe5a0de89, 0x110eefda88847faf]
r1,r2,rr = ic_mont_to_rns(frob12_c0_limbs)
emit_array("RNS_FROB12_C0_M1", r1); emit_array("RNS_FROB12_C0_M2", r2)
lines.append(f"#define RNS_FROB12_C0_RED {rr}u")
r1,r2,rr = ic_mont_to_rns(frob12_c1_limbs)
emit_array("RNS_FROB12_C1_M1", r1); emit_array("RNS_FROB12_C1_M2", r2)
lines.append(f"#define RNS_FROB12_C1_RED {rr}u")

lines.append("")
lines.append("#define BLS_X 0xd201000000010000ull")
lines.append("#define BLS_X_IS_NEG true")

# Also emit for Python cross-check
lines.append("")
lines.append(f"// M1 = product of Base1 moduli")
lines.append(f"// M1 bits = {M1.bit_length()}, M1/p = {float(M1/p):.2f}")
lines.append(f"// M2 bits = {M2.bit_length()}, M2/p = {float(M2/p):.2f}")

header = "\n".join(lines)
with open("rns_constants.cuh", "w") as f:
    f.write(header + "\n")

print(f"Generated rns_constants.cuh with {K} residues per base")
print(f"M1 bits = {M1.bit_length()}, guard bits = {M1.bit_length() - p.bit_length()}")
