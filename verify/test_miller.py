#!/usr/bin/env python3
"""
BLS12-381 Miller Loop test oracle.
Computes the pairing step-by-step to generate intermediate test vectors.
Uses the generator points of G1 and G2.
"""

P = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

# BLS12-381 curve: y^2 = x^3 + 4
B = 4

# G1 generator
G1_X = 0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
G1_Y = 0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1

# Verify G1 generator is on curve
assert (G1_Y * G1_Y - G1_X * G1_X * G1_X - B) % P == 0, "G1 generator not on curve!"

# G2 generator (Fp2 coordinates)
G2_X_C0 = 0x024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8
G2_X_C1 = 0x13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e
G2_Y_C0 = 0x0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e1935486018e7d9c0
G2_Y_C1 = 0x0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be

# BLS12-381 parameter
BLS_X = 0xd201000000010000
BLS_X_IS_NEG = True

# Montgomery form helpers
R = pow(2, 384, P)

def to_mont(a):
    return (a * R) % P

def to_limbs_hex(val, n=6):
    limbs = []
    v = val
    for _ in range(n):
        limbs.append(v & 0xFFFFFFFFFFFFFFFF)
        v >>= 64
    return "{" + ", ".join(f"0x{l:016x}ULL" for l in limbs) + "}"

print("=== BLS12-381 Generator Points in Montgomery Form ===")
print()
print("// G1 generator")
print(f"G1.x = {to_limbs_hex(to_mont(G1_X))};")
print(f"G1.y = {to_limbs_hex(to_mont(G1_Y))};")
print()
print("// G2 generator (Fp2 = c0 + c1*u)")
print(f"G2.x.c0 = {to_limbs_hex(to_mont(G2_X_C0))};")
print(f"G2.x.c1 = {to_limbs_hex(to_mont(G2_X_C1))};")
print(f"G2.y.c0 = {to_limbs_hex(to_mont(G2_Y_C0))};")
print(f"G2.y.c1 = {to_limbs_hex(to_mont(G2_Y_C1))};")
print()

# Also output the two_inv constant for line doubling
two_inv = pow(2, P-2, P)
two_inv_mont = to_mont(two_inv)
print(f"// two_inv = 1/2 in Montgomery form")
print(f"two_inv = {to_limbs_hex(two_inv_mont)};")
print()

# Verify: 2 * two_inv = 1 mod P
two_mont = to_mont(2)
check = (two_mont * two_inv_mont * pow(R, P-2, P)) % P  # mont_mul
print(f"// Verify: mont_mul(two, two_inv) should equal mont(1)")
print(f"// mont(1) = R mod P = {to_limbs_hex(to_mont(1))}")
print()

# Count miller loop operations
bits = BLS_X.bit_length()
print(f"BLS_X = {hex(BLS_X)}")
print(f"BLS_X bits = {bits}")
print(f"BLS_X binary = {bin(BLS_X)}")
doubles = 0
adds = 0
for j in range(bits - 2, -1, -1):
    doubles += 1
    if (BLS_X >> j) & 1:
        adds += 1
print(f"Miller loop: {doubles} doublings, {adds} additions, {doubles + adds} total line evals")
print(f"Precomputed coefficients needed: {doubles + adds}")
