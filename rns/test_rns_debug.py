#!/usr/bin/env python3
"""Debug the RNS Montgomery mul step by step in Python."""

from sympy import isprime

p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

# Same bases
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

# mont(7) = 7 * M1 mod p
v7 = (7 * M1) % p
v49 = (49 * M1) % p

print(f"mont(7) = {v7}")
print(f"mont(49) = {v49}")

# Encode mont(7) in both bases
a1 = [v7 % m for m in b1]
a2 = [v7 % m for m in b2]

print(f"\nmont(7) in B1: {a1}")
print(f"mont(7) in B2: {a2}")

# Step 1: q = a * a in both bases
q1 = [(a1[i] * a1[i]) % b1[i] for i in range(K)]
q2 = [(a2[i] * a2[i]) % b2[i] for i in range(K)]

print(f"\nq = a*a in B1: {q1}")
print(f"q = a*a in B2: {q2}")

# Step 2: t = q * (-p^(-1)) mod m_i in Base1
neg_pinv = [(-pow(p, -1, m)) % m for m in b1]
t1 = [(q1[i] * neg_pinv[i]) % b1[i] for i in range(K)]

print(f"\nt = q * (-p^-1) in B1: {t1}")

# Step 3: base extend t from B1 to B2
# Using exact CRT (no alpha approximation needed in Python)
mhat_inv_b1 = [pow(M1 // m, -1, m) for m in b1]
xt = [(t1[i] * mhat_inv_b1[i]) % b1[i] for i in range(K)]

print(f"\nxt (CRT coefficients): {xt}")

# Exact CRT reconstruction of t
t_value = 0
for i in range(K):
    t_value += xt[i] * (M1 // b1[i])
t_value = t_value % M1

print(f"\nt (exact value): {t_value}")
print(f"t mod p: {t_value % p}")

# Now extend to B2
t2_exact = [t_value % m for m in b2]
print(f"\nt in B2 (exact): {t2_exact}")

# Step 4: r = (q + t*p) / M1 in B2
# Check that q + t*p is divisible by M1
# First compute q and t as actual values
q_value = (v7 * v7)  # NOT mod p — full product
print(f"\nq (full product, not mod): {q_value}")
print(f"q mod p: {q_value % p}")

# Actually q_value needs to be the value that q1/q2 represent
# q1[i] = (v7^2) mod m1[i] — so q represents (v7^2 mod M1)
# But v7^2 might be > M1, so q actually represents v7^2 mod M1
# The issue: RNS represents values mod M, so q = v7^2 mod M1

q_full = v7 * v7  # actual product
q_mod_M1 = q_full % M1
print(f"q = v7^2 mod M1: {q_mod_M1}")

# Verify q1 encodes q_mod_M1
for i in range(K):
    assert q1[i] == q_mod_M1 % b1[i], f"q1 mismatch at {i}"
print("q1 encoding verified")

# The Montgomery reduction:
# We want: (q + t*p) / M1 mod p
# where t = q * (-p^-1) mod M1

# t should satisfy: q + t*p ≡ 0 (mod M1)
# Check: (q_mod_M1 + t_value * p) % M1
check = (q_mod_M1 + t_value * p) % M1
print(f"\n(q + t*p) mod M1 = {check}")
if check == 0:
    print("GOOD: divisible by M1")
else:
    print(f"BAD: not divisible by M1! This is the bug.")
    # The issue might be that t should be computed differently
    # t = q * (-p^-1) mod M1 means:
    # t_value should satisfy t_value ≡ q_mod_M1 * (-p^-1) (mod M1)
    neg_pinv_M1 = (-pow(p, -1, M1)) % M1
    t_correct = (q_mod_M1 * neg_pinv_M1) % M1
    print(f"t (computed per-residue): {t_value}")
    print(f"t (correct via big int):  {t_correct}")
    if t_value != t_correct:
        print("MISMATCH — the per-residue computation of -p^-1 is wrong!")
        print("Per-residue: t_i = q_i * (-p^-1 mod m_i) mod m_i")
        print("This is CRT of t where t = q * (-p^-1 mod M1)")
        print("But (-p^-1 mod m_i) != ((-p^-1 mod M1) mod m_i)!")
        print("We need (-p^-1 mod M1) decomposed into residues, not (-p^-1 mod m_i)")

        # The correct neg_p_inv for RNS Montgomery:
        # We need a single value N = -p^-1 mod M1
        # Then decompose N into residues: N mod m_i
        N = neg_pinv_M1
        N_residues = [N % m for m in b1]
        print(f"\nCorrect neg_p_inv residues: {N_residues}")
        print(f"Current neg_p_inv residues: {neg_pinv}")
        print("These are DIFFERENT because (-p^-1 mod m_i) != ((-p^-1 mod M1) mod m_i)")

# The expected result
r_expected = (q_full * pow(M1, -1, p)) % p
print(f"\nExpected result: {r_expected}")
print(f"Expected = mont(49)? {r_expected == v49}")
