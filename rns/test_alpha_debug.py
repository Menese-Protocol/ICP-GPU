#!/usr/bin/env python3
"""Debug the alpha computation for base extension."""
from sympy import isprime

p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

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

# The t1 values from Step 2 (confirmed correct)
t1 = [97649642, 242164948, 211185841, 1067946485, 1021763924, 702423266, 165058719, 210617947, 33464787, 84608615, 93319377, 612166296, 667961219, 277774837]

# Compute xt (CRT coefficients)
mhat_inv = [pow(M1 // m, -1, m) for m in b1]
xt = [(t1[i] * mhat_inv[i]) % b1[i] for i in range(K)]

print("xt values:")
for i, x in enumerate(xt):
    print(f"  xt[{i}] = {x}, xt/m = {x / b1[i]:.6f}")

# Exact alpha
alpha_exact = sum(xt[i] / b1[i] for i in range(K))
print(f"\nalpha (exact float64) = {alpha_exact:.6f}")
print(f"alpha (rounded)      = {round(alpha_exact)}")

# Float32 alpha (what CUDA computes)
import struct
def to_float32(x):
    return struct.unpack('f', struct.pack('f', float(x)))[0]

alpha_f32 = 0.0
for i in range(K):
    alpha_f32 = to_float32(alpha_f32 + to_float32(to_float32(xt[i]) / to_float32(b1[i])))
print(f"alpha (float32 sim)  = {alpha_f32:.6f}")
print(f"alpha (f32 rounded)  = {int(alpha_f32 + 0.5)}")

# What if alpha is off by 1?
t_value_exact = sum(xt[i] * (M1 // b1[i]) for i in range(K))
# True alpha = floor(sum / M1) where sum = sum(xt[i] * M_hat[i])
true_alpha = t_value_exact // M1
remainder = t_value_exact % M1
print(f"\ntrue alpha = {true_alpha}")
print(f"remainder  = {remainder}")
print(f"t = remainder = the actual value we want")

# With correct alpha, compute extension
print("\nWith correct alpha, extension to B2:")
for j in range(3):  # just first 3
    mj = b2[j]
    acc = sum(xt[i] * ((M1 // b1[i]) % mj) for i in range(K))
    val = acc % mj
    correction = (true_alpha * (M1 % mj)) % mj
    result = (val - correction) % mj

    # Expected from Python
    expected_t2 = [615773812, 914326032, 826202594]
    print(f"  B2[{j}]: val={val}, correction={correction}, result={result}, expected={expected_t2[j]}, match={result==expected_t2[j]}")
