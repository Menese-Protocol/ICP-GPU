#!/usr/bin/env python3
"""
Generate RNS bases for BLS12-381 field arithmetic.

We need two co-prime bases B1 and B2, each with k moduli,
such that product(B1) > p and product(B2) > p.

For 384-bit p, we need k moduli of ~30 bits each: k*30 >= 384+guard_bits
k = 14 gives 420 bits of dynamic range (36 bits of guard).

Each modulus must be:
- Co-prime to all others
- Fit in 30 bits (so products fit in 64-bit during mac)
- Ideally of form 2^30 - c for small c (fast reduction)
"""

from sympy import isprime, nextprime

# BLS12-381 base field modulus
p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

print(f"p = {p}")
print(f"p bits = {p.bit_length()}")  # 381

# Strategy: pick 28 primes near 2^30 (but not too close to avoid overflow)
# For CUDA: we want mᵢ < 2^30 so that a_i * b_i < 2^60 fits in uint64
# And mᵢ * mⱼ fits in uint64 for base extension

# Find primes just below 2^30
def find_primes_near(target, count, direction='below'):
    primes = []
    if direction == 'below':
        candidate = target - 1
        while len(primes) < count:
            if isprime(candidate):
                primes.append(candidate)
            candidate -= 2  # skip evens
    else:
        candidate = target + 1
        while len(primes) < count:
            if isprime(candidate):
                primes.append(candidate)
            candidate += 2
    return primes

# Base1: 14 primes just below 2^30
b1_primes = find_primes_near(2**30, 14, 'below')
# Base2: 14 primes just below 2^29 + 2^30 (different range to ensure coprimality)
b2_primes = find_primes_near(2**30 - 200, 14, 'below')

# Make sure no overlap
b2_start = max(b1_primes) + 100  # start well above base1
b2_primes = find_primes_near(2**30 - 500, 14, 'below')

# Actually, better approach: interleave from a wider range
all_primes = []
candidate = 2**30 - 1
while len(all_primes) < 28:
    if isprime(candidate):
        all_primes.append(candidate)
    candidate -= 2

b1 = all_primes[0::2][:14]  # even-indexed
b2 = all_primes[1::2][:14]  # odd-indexed

# Verify coprimality
from math import gcd
for i, a in enumerate(b1):
    for j, b in enumerate(b2):
        assert gcd(a, b) == 1, f"b1[{i}]={a} and b2[{j}]={b} share factor"
    for j, b in enumerate(b1):
        if i != j:
            assert gcd(a, b) == 1, f"b1[{i}]={a} and b1[{j}]={b} share factor"

print(f"\nBase1 ({len(b1)} primes, each ~{b1[0].bit_length()} bits):")
M1 = 1
for i, m in enumerate(b1):
    M1 *= m
    print(f"  m1[{i:2d}] = {m:>12d}  (0x{m:08x})  {m.bit_length()} bits")
print(f"  Product M1 bits = {M1.bit_length()}")

print(f"\nBase2 ({len(b2)} primes, each ~{b2[0].bit_length()} bits):")
M2 = 1
for i, m in enumerate(b2):
    M2 *= m
    print(f"  m2[{i:2d}] = {m:>12d}  (0x{m:08x})  {m.bit_length()} bits")
print(f"  Product M2 bits = {M2.bit_length()}")

print(f"\np bits = {p.bit_length()}")
print(f"M1 > p: {M1 > p}  (M1/p = {M1/p:.2f})")
print(f"M2 > p: {M2 > p}  (M2/p = {M2/p:.2f})")

# Precompute constants needed for RNS Montgomery
print("\n=== Precomputed Constants ===")

# For each base: M_hat_i = M / m_i (partial product without m_i)
# And M_hat_i_inv = M_hat_i^(-1) mod m_i
def precompute_base(base, name):
    M = 1
    for m in base:
        M *= m

    print(f"\n{name}:")
    for i, mi in enumerate(base):
        M_hat = M // mi
        M_hat_inv = pow(M_hat, -1, mi)  # modular inverse
        M_hat_mod_mi = M_hat % mi  # for verification
        print(f"  {name}[{i:2d}]: m={mi}, M_hat_inv={M_hat_inv}, M_hat%m={M_hat_mod_mi}")

    # p^(-1) mod m_i for Montgomery reduction
    print(f"\n  -p^(-1) mod m_i (for Montgomery q computation):")
    for i, mi in enumerate(base):
        p_inv = pow(p, -1, mi)
        neg_p_inv = (-p_inv) % mi
        print(f"    neg_p_inv[{i:2d}] = {neg_p_inv}")

    # p mod m_i
    print(f"\n  p mod m_i:")
    for i, mi in enumerate(base):
        print(f"    p_mod[{i:2d}] = {p % mi}")

    return M

M1 = precompute_base(b1, "Base1")
M2 = precompute_base(b2, "Base2")

# M1^(-1) mod p and M2^(-1) mod p (for final conversion)
M1_inv_p = pow(M1, -1, p)
M2_inv_p = pow(M2, -1, p)
print(f"\nM1^(-1) mod p = 0x{M1_inv_p:096x}")
print(f"M2^(-1) mod p = 0x{M2_inv_p:096x}")

# Base extension constants: for extending from B1 to B2
# To extend x from B1 to B2:
#   x = sum_i (x_i * M_hat_i_inv mod m_i) * M_hat_i  (CRT reconstruction)
#   Then reduce mod each m'_j in B2
print("\n=== Base Extension Constants (B1 -> B2) ===")
for j, mj in enumerate(b2):
    coeffs = []
    for i, mi in enumerate(b1):
        M_hat = M1 // mi
        M_hat_inv = pow(M_hat, -1, mi)
        # (M_hat * M_hat_inv) mod mj = M_hat mod mj (since M_hat_inv only used with xi)
        # Actually: xi_contrib = (xi * M_hat_inv mod mi) * M_hat mod mj
        # = xi * (M_hat * M_hat_inv mod mi ... no, let me think again)
        # CRT: x = sum_i xi_tilde * M_hat_i where xi_tilde = xi * M_hat_i_inv mod mi
        # So x mod mj = sum_i (xi_tilde * (M_hat_i mod mj)) mod mj
        coeff = (M1 // mi) % mj
        coeffs.append(coeff)
    if j == 0:
        print(f"  B1->B2 coefficients for m'[0]={mj}:")
        for i, c in enumerate(coeffs):
            print(f"    coeff[{i}] = {c}")
        print(f"  (similar for other m'[j])")

# Verify encode-decode round trip
print("\n=== Verification: encode-decode round trip ===")
test_vals = [0, 1, 7, 42, p-1, 0xdeadbeef, 2**380]
for v in test_vals:
    v = v % p
    # Encode in B1
    residues = [v % m for m in b1]
    # Decode via CRT
    result = 0
    for i, mi in enumerate(b1):
        M_hat = M1 // mi
        M_hat_inv = pow(M_hat, -1, mi)
        result += residues[i] * M_hat_inv * M_hat
    result = result % M1

    ok = "PASS" if result == v else "FAIL"
    print(f"  v={v:>50d} -> {ok}")

print("\n=== Done ===")
