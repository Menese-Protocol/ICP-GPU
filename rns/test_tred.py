#!/usr/bin/env python3
"""Test whether q_red * NEG_PINV_MRED gives useful t_red for exact alpha."""
from sympy import isprime
p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
all_primes = []
candidate = 2**30 - 1
while len(all_primes) < 28:
    if isprime(candidate): all_primes.append(candidate)
    candidate -= 2
b1 = all_primes[0::2][:14]
K = 14
M1 = 1
for m in b1: M1 *= m
m_red = 2**31 - 1

# mont(7)
v7 = (7 * M1) % p
q_full = v7 * v7  # full product (could be > M1)
q_mod_M1 = q_full % M1

neg_pinv_M1 = (-pow(p, -1, M1)) % M1
t = (q_mod_M1 * neg_pinv_M1) % M1  # true t in [0, M1)
t_red_true = t % m_red

# q_red * NEG_PINV_MRED
q_red = q_full % m_red  # NOTE: q_full, not q_mod_M1
neg_pinv_mred = (-pow(p, -1, m_red)) % m_red
t_red_from_q = (q_red * neg_pinv_mred) % m_red

print(f"t_red_true      = {t_red_true}")
print(f"t_red_from_q    = {t_red_from_q}")
print(f"match: {t_red_true == t_red_from_q}")

# q_mod_M1 in m_red
q_mod_M1_mred = q_mod_M1 % m_red
t_red_from_qM1 = (q_mod_M1_mred * neg_pinv_mred) % m_red
print(f"t_red_from_qM1  = {t_red_from_qM1}")
print(f"match: {t_red_true == t_red_from_qM1}")

# The difference
diff = (t_red_from_q - t_red_true) % m_red
print(f"diff (q_full) = {diff}")
diff2 = (t_red_from_qM1 - t_red_true) % m_red
print(f"diff (q_M1)   = {diff2}")

# q_red in rns_mul context: q_red = a.rr * b.rr mod m_red
# a.rr = v7 mod m_red. So q_red = (v7 mod m_red)^2 mod m_red
# = v7^2 mod m_red = q_full mod m_red. YES!
# So q_red = q_full mod m_red, NOT q_mod_M1 mod m_red.
# The difference: q_full = q_mod_M1 + k1 * M1 for some k1.
# t_from_q = (q_full * neg_pinv) mod m_red = ((q_mod_M1 + k1*M1) * neg_pinv) mod m_red
# = (t + alpha*M1 + k1*M1*neg_pinv) mod m_red (where t + alpha*M1 = q_mod_M1 * neg_pinv mod m_red)
# Hmm, not quite. Let me be more careful.
#
# t = (q_mod_M1 * NEG_PINV_M1) % M1  ← definition
# recon_red = CRT(t in B1) mod m_red = (t + alpha_crt * M1) mod m_red
# q_red * NEG_PINV_MRED = (q_full * neg_pinv_mred) mod m_red
#
# q_full = q_mod_M1 + k * M1
# q_mod_M1 * neg_pinv_mred = t * (neg_pinv_mred / neg_pinv_M1_mod_mred) mod m_red?
# No, this is getting circular.
#
# But empirically: if diff = (recon_red - q_red * NEG_PINV_MRED) is a known multiple of M1,
# we can extract alpha_crt.

# CRT recon of t
mhat_inv = [pow(M1 // m, -1, m) for m in b1]
t1 = [(q_mod_M1 % m * ((-pow(p, -1, m)) % m)) % m for m in b1]
xt = [(t1[i] * mhat_inv[i]) % b1[i] for i in range(K)]
recon = sum(xt[i] * (M1 // b1[i]) for i in range(K))
recon_red = recon % m_red
alpha_crt = recon // M1

print(f"\nalpha_crt       = {alpha_crt}")
print(f"recon_red       = {recon_red}")
print(f"q_red*npinv_red = {t_red_from_q}")
print(f"(recon - q*npinv) mod m_red = {(recon_red - t_red_from_q) % m_red}")
print(f"alpha_crt * M1 mod m_red    = {(alpha_crt * (M1 % m_red)) % m_red}")

# Check if the difference is alpha_crt * M1
diff_check = (recon_red - t_red_from_q) % m_red
expected_diff = (alpha_crt * (M1 % m_red)) % m_red
# But wait, there's also the k factor from q_full = q_mod_M1 + k*M1
k_q = q_full // M1
extra = (k_q * (M1 % m_red) * neg_pinv_mred) % m_red
adjusted = (recon_red - t_red_from_q + extra) % m_red
print(f"k_q = {k_q}")
print(f"extra = {extra}")
print(f"adjusted diff = {adjusted}")
print(f"alpha_crt * M1_mod_mred = {expected_diff}")
print(f"MATCH: {adjusted == expected_diff}")
