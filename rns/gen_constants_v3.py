#!/usr/bin/env python3
"""
Generate RNS constants for BLS12-381 — Production Architecture (Kawamura Cox-Rower)

B1 = 15 moduli (14 primes near 2^30 + Mersenne prime 2^31-1)
B2 = 14 primes near 2^30 (disjoint from B1)
M1 = product(B1), M2 = product(B2)

Base extension B1→B2: exact alpha (all 15 B1 residues known)
Base extension B2→B1: exact alpha (r mod m_red computed independently)
No approximate alpha, no heuristics, no floats.
"""

from sympy import isprime
from math import gcd

p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

# Collect primes near 2^30
all_primes = []
candidate = 2**30 - 1
while len(all_primes) < 30:
    if isprime(candidate):
        all_primes.append(candidate)
    candidate -= 2

# B1 core: 14 primes (even-indexed from all_primes)
b1_core = all_primes[0::2][:14]
# B2: 14 primes (odd-indexed, disjoint from B1 core)
b2 = all_primes[1::2][:14]
# m_red: Mersenne prime 2^31-1
m_red = 2**31 - 1
assert isprime(m_red)
assert m_red not in b1_core and m_red not in b2

# B1 = B1_core + [m_red] — 15 moduli total
b1 = b1_core + [m_red]
K1 = len(b1)   # 15
K2 = len(b2)   # 14

# Verify coprimality
for i in range(K1):
    for j in range(i+1, K1):
        assert gcd(b1[i], b1[j]) == 1, f"B1[{i}]={b1[i]} and B1[{j}]={b1[j]} not coprime"
for i in range(K2):
    for j in range(i+1, K2):
        assert gcd(b2[i], b2[j]) == 1
for i in range(K1):
    for j in range(K2):
        assert gcd(b1[i], b2[j]) == 1, f"B1[{i}]={b1[i]} and B2[{j}]={b2[j]} not coprime"

M1 = 1
for m in b1: M1 *= m
M2 = 1
for m in b2: M2 *= m

assert M1 > p, f"M1 must be > p"
assert M2 > p, f"M2 must be > p"

print(f"BLS12-381 p: {p.bit_length()} bits")
print(f"B1: {K1} moduli, M1: {M1.bit_length()} bits, M1/p = {float(M1/p):.2e}")
print(f"B2: {K2} moduli, M2: {M2.bit_length()} bits, M2/p = {float(M2/p):.2e}")

# ============================================================
# Generate CUDA header
# ============================================================

lines = []
lines.append("// Auto-generated RNS constants for BLS12-381 — Production Kawamura Architecture")
lines.append("// B1 = 15 moduli (14 primes + Mersenne), B2 = 14 primes. Exact alpha everywhere.")
lines.append("// DO NOT EDIT — regenerate with gen_constants_v3.py")
lines.append("#pragma once")
lines.append("#include <cstdint>")
lines.append("")
lines.append(f"#define RNS_K1 {K1}  // Base1 moduli count (includes redundant)")
lines.append(f"#define RNS_K2 {K2}  // Base2 moduli count")
lines.append(f"#define RNS_MRED_IDX {K1-1}  // Index of m_red in B1")
lines.append("")

def emit(name, vals, typ="uint32_t"):
    s = f"__device__ __constant__ {typ} {name}[] = {{"
    s += ", ".join(str(v) + ("ull" if typ == "uint64_t" else "u") for v in vals)
    s += "};"
    lines.append(s)

# Moduli
emit("RNS_M1", b1)
emit("RNS_M2", b2)

# Barrett constants: mu = floor(2^62 / m)
emit("RNS_BARRETT_M1", [(1 << 62) // m for m in b1], "uint64_t")
emit("RNS_BARRETT_M2", [(1 << 62) // m for m in b2], "uint64_t")
lines.append("")

# p mod m_i
emit("RNS_P_MOD_M1", [p % m for m in b1])
emit("RNS_P_MOD_M2", [p % m for m in b2])

# -p^(-1) mod m_i (for Montgomery q = a*b*(-p^-1) mod M1)
emit("RNS_NEG_PINV_M1", [(-pow(p, -1, m)) % m for m in b1])

# M_hat_inv: (M/m_i)^(-1) mod m_i for CRT
def mhat_inv(base, M):
    return [pow(M // m, -1, m) for m in base]
emit("RNS_MHAT_INV_M1", mhat_inv(b1, M1))
emit("RNS_MHAT_INV_M2", mhat_inv(b2, M2))
lines.append("")

# Base extension matrix B1→B2: be[j][i] = (M1/m1[i]) mod m2[j]
lines.append(f"// Base extension B1({K1})→B2({K2}): be_12[j][i] = (M1/m1[i]) mod m2[j]")
lines.append(f"__device__ __constant__ uint32_t RNS_BE_12[{K2}][{K1}] = {{")
for j, mj in enumerate(b2):
    coeffs = [(M1 // b1[i]) % mj for i in range(K1)]
    lines.append("    {" + ", ".join(str(c)+"u" for c in coeffs) + "},")
lines.append("};")

# Base extension B2→B1: be[j][i] = (M2/m2[i]) mod m1[j]
lines.append(f"// Base extension B2({K2})→B1({K1}): be_21[j][i] = (M2/m2[i]) mod m1[j]")
lines.append(f"__device__ __constant__ uint32_t RNS_BE_21[{K1}][{K2}] = {{")
for j, mj in enumerate(b1):
    coeffs = [(M2 // b2[i]) % mj for i in range(K2)]
    lines.append("    {" + ", ".join(str(c)+"u" for c in coeffs) + "},")
lines.append("};")
lines.append("")

# M1 mod m2[j], M2 mod m1[j] — for alpha correction in base extension
emit("RNS_M1_MOD_M2", [M1 % m for m in b2])
emit("RNS_M2_MOD_M1", [M2 % m for m in b1])

# M1^(-1) mod m2[j] — for Montgomery step 4
emit("RNS_M1_INV_MOD_M2", [pow(M1, -1, m) for m in b2])
# M2^(-1) mod m1[j] — for exact alpha in B2→B1
emit("RNS_M2_INV_MOD_M1", [pow(M2, -1, m) for m in b1])

# M1^(-1) mod m_red (m_red is b1[14], so M1_inv_mod_mred = already in M2_INV? No.)
# Actually m_red is IN B1, so M1 mod m_red = 0. M1^(-1) mod m_red doesn't exist!
# This is fine because in B1→B2 extension, we use all 15 B1 residues for exact CRT.
# For B2→B1, we need M2^(-1) mod m1[j] including m_red, which is provided above.
lines.append("")

# Montgomery R = M1 mod p (in both bases)
R = M1 % p
emit("RNS_R_MOD_M1", [R % m for m in b1])
emit("RNS_R_MOD_M2", [R % m for m in b2])

# R^2 mod p
R2 = (M1 * M1) % p
emit("RNS_R2_MOD_M1", [R2 % m for m in b1])
emit("RNS_R2_MOD_M2", [R2 % m for m in b2])
lines.append("")

# Oracle test vectors
lines.append("// Oracle test vectors")
for val, name in [(7, "7"), (49, "49"), (6, "6"), (42, "42")]:
    v = (val * M1) % p
    emit(f"RNS_ORACLE_MONT{name}_M1", [v % m for m in b1])
    emit(f"RNS_ORACLE_MONT{name}_M2", [v % m for m in b2])

lines.append("")
# Frobenius coefficients for Fp6/Fp12 (exact from ic_bls12_381)
# These are specific Fp2 constants used in the Frobenius endomorphism
lines.append("")
lines.append("// Frobenius coefficients (from ic_bls12_381)")

# fc1 for Fp6 frobenius: ξ^((p-1)/3) — c0=0, c1=known
fc1_c1 = 0x1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaac
# Actually these are the MONTGOMERY form values from ic_bls12_381
# Let me use the exact hex values from our field_sppark.cuh

# Fp6 frobenius coefficient 1: (0, c1) where c1 is:
frob6_c1_c1 = 0x18f0206554638741cd03c9e48671f0715dab22461fcda5d2587042afd3851b958eb60ebe01bacb9e03f97d6e83d050d2
# Fp6 frobenius coefficient 2: (c0, 0) where c0 is:
frob6_c2_c0 = 0x14e56d3f1564853a890dc9e4867545c32af322533285a5d550880866309b7e2ca20d1b8c7e88102414e4f04fe2db9068

# Fp12 frobenius coefficient: (c0, c1)
frob12_c0 = 0x08f2220fb0fb66eb07089552b319d465c6695f92b50a831397e83cccd117228fa35baecab2dc29ee1ce393ea5daace4d
frob12_c1 = 0x110eefda88847fafb2f66aad4ce5d6465842a06bfc497ceccf4895d42599d394c11b9cba40a8e8d02e3813cbe5a0de89

# These values are in ic_bls12_381's Montgomery form (R = 2^384 mod p)
# We need them in OUR Montgomery form (R = M mod p)
# To convert: our_mont(x) = x * M * R_ic^(-1) mod p
# Where R_ic = 2^384 mod p
# Or simpler: get the canonical value first, then encode in our RNS

# The canonical values (non-Montgomery) can be extracted from to_bytes()
# For now, encode the CANONICAL values directly
# Canonical = ic_mont_value * R_ic^(-1) mod p

R_ic = pow(2, 384, p)
R_ic_inv = pow(R_ic, -1, p)

def ic_mont_to_canonical(ic_mont):
    return (ic_mont * R_ic_inv) % p

def canonical_to_rns(val):
    """Encode canonical value into our Montgomery form: val * M mod p, then residues"""
    mont = (val * M1) % p
    r1 = [mont % m for m in b1]
    r2 = [mont % m for m in b2]
    return r1, r2

# Frobenius Fp6 coefficient 1: Fp2(0, c1)
fc1_c1_canonical = ic_mont_to_canonical(frob6_c1_c1)
fc1_c1_r1, fc1_c1_r2 = canonical_to_rns(fc1_c1_canonical)
emit("RNS_FROB6_C1_C1_M1", fc1_c1_r1)
emit("RNS_FROB6_C1_C1_M2", fc1_c1_r2)

# Frobenius Fp6 coefficient 2: Fp2(c0, 0)
fc2_c0_canonical = ic_mont_to_canonical(frob6_c2_c0)
fc2_c0_r1, fc2_c0_r2 = canonical_to_rns(fc2_c0_canonical)
emit("RNS_FROB6_C2_C0_M1", fc2_c0_r1)
emit("RNS_FROB6_C2_C0_M2", fc2_c0_r2)

# Frobenius Fp12 coefficient: Fp2(c0, c1)
f12_c0_canonical = ic_mont_to_canonical(frob12_c0)
f12_c1_canonical = ic_mont_to_canonical(frob12_c1)
f12_c0_r1, f12_c0_r2 = canonical_to_rns(f12_c0_canonical)
f12_c1_r1, f12_c1_r2 = canonical_to_rns(f12_c1_canonical)
emit("RNS_FROB12_C0_M1", f12_c0_r1)
emit("RNS_FROB12_C0_M2", f12_c0_r2)
emit("RNS_FROB12_C1_M1", f12_c1_r1)
emit("RNS_FROB12_C1_M2", f12_c1_r2)

# BLS parameter x
lines.append("")
lines.append("#define BLS_X 0xd201000000010000ull")
lines.append("#define BLS_X_IS_NEG true")

lines.append(f"\n// M1 bits = {M1.bit_length()}, guard bits = {M1.bit_length() - p.bit_length()}")
lines.append(f"// M2 bits = {M2.bit_length()}, guard bits = {M2.bit_length() - p.bit_length()}")

# Write
with open("rns_constants_v3.cuh", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Generated rns_constants_v3.cuh: B1={K1} moduli, B2={K2} moduli")
print(f"M1 guard bits = {M1.bit_length() - p.bit_length()}")
print(f"M2 guard bits = {M2.bit_length() - p.bit_length()}")

# Verify encode-decode
print("\nVerification:")
for v in [0, 1, 7, 42, p-1]:
    v = v % p
    r1 = [v % m for m in b1]
    # CRT decode
    result = sum(r1[i] * pow(M1//b1[i], -1, b1[i]) * (M1//b1[i]) for i in range(K1)) % M1
    print(f"  v={v%1000:>5d}... round-trip: {'PASS' if result == v else 'FAIL'}")
