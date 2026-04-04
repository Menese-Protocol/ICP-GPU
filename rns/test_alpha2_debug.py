#!/usr/bin/env python3
"""Debug alpha for mont(2) * mont(2)."""
from sympy import isprime

p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

all_primes = []
candidate = 2**30 - 1
while len(all_primes) < 28:
    if isprime(candidate):
        all_primes.append(candidate)
    candidate -= 2

b1 = all_primes[0::2][:14]
K = 14
M1 = 1
for m in b1: M1 *= m

# mont(2)
v2 = (2 * M1) % p
print(f"mont(2) = {v2}")

a1 = [v2 % m for m in b1]

# q = a * a in B1
q1 = [(a1[i] * a1[i]) % b1[i] for i in range(K)]

# t = q * (-p^-1) in B1
neg_pinv = [(-pow(p, -1, m)) % m for m in b1]
t1 = [(q1[i] * neg_pinv[i]) % b1[i] for i in range(K)]

# CRT coefficients
mhat_inv = [pow(M1 // m, -1, m) for m in b1]
xt = [(t1[i] * mhat_inv[i]) % b1[i] for i in range(K)]

# Alpha
alpha_exact = sum(xt[i] / b1[i] for i in range(K))
print(f"alpha (exact float64) = {alpha_exact:.15f}")
print(f"alpha (floor)        = {int(alpha_exact)}")
print(f"fractional part      = {alpha_exact - int(alpha_exact):.15f}")

# True alpha
t_value = sum(xt[i] * (M1 // b1[i]) for i in range(K))
true_alpha = t_value // M1
print(f"true alpha           = {true_alpha}")
print(f"MATCH: {int(alpha_exact) == true_alpha}")
