#!/usr/bin/env python3
"""Convert Rust oracle G2 coefficients + pairing result to RNS form."""

import subprocess, sys
from sympy import isprime

p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

# Same bases as gen_cuda_constants.py
all_primes = []
candidate = 2**30 - 1
while len(all_primes) < 28:
    if isprime(candidate): all_primes.append(candidate)
    candidate -= 2
b1 = all_primes[0::2][:14]
b2 = all_primes[1::2][:14]
K = 14
M1 = 1
for m in b1: M1 *= m
m_red = 2**31 - 1

R_ic = pow(2, 384, p)
R_ic_inv = pow(R_ic, -1, p)

def limbs_to_int(hex_str):
    """Parse 6 LE u64 limbs from hex string to big int"""
    # hex_str is 6*16=96 hex chars = 6 u64 values concatenated
    limbs = []
    for i in range(6):
        limb_hex = hex_str[i*16:(i+1)*16]
        limbs.append(int(limb_hex, 16))
    val = 0
    for i, limb in enumerate(limbs):
        val |= limb << (64 * i)
    return val

def ic_mont_to_our_mont(ic_mont_int):
    """Convert ic_bls12_381 Montgomery value to our RNS Montgomery form"""
    canonical = (ic_mont_int * R_ic_inv) % p
    return (canonical * M1) % p

def to_rns(our_mont):
    """Convert our Montgomery value to residues"""
    return [our_mont % m for m in b1], [our_mont % m for m in b2], our_mont % m_red

# Run Rust oracle
print("Running Rust oracle...", file=sys.stderr)
result = subprocess.run(
    ['/workspace/gpu_pairing/oracle/target/release/oracle', 'g2-coeffs'],
    capture_output=True, text=True
)
g2_lines = result.stdout.strip().split('\n')

result2 = subprocess.run(
    ['/workspace/gpu_pairing/oracle/target/release/oracle', 'pairing-oracle'],
    capture_output=True, text=True
)
pair_lines = result2.stdout.strip().split('\n')

# Parse G2 coefficients
num_coeffs = int(g2_lines[0].split('=')[1])
print(f"Parsing {num_coeffs} G2 coefficients...", file=sys.stderr)

lines = []
lines.append("// Auto-generated pairing constants for RNS BLS12-381")
lines.append("// G2 prepared coefficients + e(G1,G2) oracle values")
lines.append("// DO NOT EDIT — regenerate with gen_pairing_constants.py")
lines.append("#pragma once")
lines.append("#include <cstdint>")
lines.append(f"")
lines.append(f"#define G2_COEFFS_COUNT {num_coeffs}")

# Each coefficient has 6 Fp values (3 Fp2 = 6 Fp)
# Each Fp in RNS form needs K residues for B1 and K for B2 + 1 for m_red
# Total per Fp: K*2 + 1 = 29 uint32 values
# Total: 68 * 6 * 29 = 11,832 uint32 values — too much for __constant__
# Better: store as flat arrays of residues

# For each coefficient, store 6 Fp values, each as (r1[K], r2[K], rr)
# Flatten: g2_coeffs_r1[68][6][K], g2_coeffs_r2[68][6][K], g2_coeffs_rr[68][6]

all_r1 = []  # [coeff_idx][fp_idx][residue_idx]
all_r2 = []
all_rr = []

for ci in range(num_coeffs):
    hex_data = g2_lines[1 + ci].split('=')[1]
    # 36 u64 values = 6 Fp values (each 6 u64)
    coeff_r1 = []
    coeff_r2 = []
    coeff_rr = []
    for fi in range(6):
        fp_hex = hex_data[fi*96:(fi+1)*96]  # 6*16=96 hex chars per Fp
        ic_mont = limbs_to_int(fp_hex)
        our_mont = ic_mont_to_our_mont(ic_mont)
        r1, r2, rr = to_rns(our_mont)
        coeff_r1.append(r1)
        coeff_r2.append(r2)
        coeff_rr.append(rr)
    all_r1.append(coeff_r1)
    all_r2.append(coeff_r2)
    all_rr.append(coeff_rr)

# Emit as flat arrays
# g2_r1[68*6*14] — indexed as [coeff*6*K + fp*K + residue]
lines.append(f"")
lines.append(f"// G2 coefficients in RNS form: {num_coeffs} coeffs × 6 Fp values × {K} residues")
flat_r1 = []
flat_r2 = []
flat_rr = []
for ci in range(num_coeffs):
    for fi in range(6):
        flat_r1.extend(all_r1[ci][fi])
        flat_r2.extend(all_r2[ci][fi])
        flat_rr.append(all_rr[ci][fi])

lines.append(f"__device__ __constant__ uint32_t G2_COEFFS_R1[{len(flat_r1)}] = {{")
for i in range(0, len(flat_r1), 14):
    chunk = flat_r1[i:i+14]
    lines.append("    " + ", ".join(f"{v}u" for v in chunk) + ",")
lines.append("};")

lines.append(f"__device__ __constant__ uint32_t G2_COEFFS_R2[{len(flat_r2)}] = {{")
for i in range(0, len(flat_r2), 14):
    chunk = flat_r2[i:i+14]
    lines.append("    " + ", ".join(f"{v}u" for v in chunk) + ",")
lines.append("};")

lines.append(f"__device__ __constant__ uint32_t G2_COEFFS_RR[{len(flat_rr)}] = {{")
for i in range(0, len(flat_rr), 14):
    chunk = flat_rr[i:i+14]
    lines.append("    " + ", ".join(f"{v}u" for v in chunk) + ",")
lines.append("};")

# Parse G1 generator coordinates
g1x_hex = [l for l in pair_lines if l.startswith('G1_X=')][0].split('=')[1]
g1y_hex = [l for l in pair_lines if l.startswith('G1_Y=')][0].split('=')[1]

g1x_ic = limbs_to_int(g1x_hex)
g1y_ic = limbs_to_int(g1y_hex)
g1x_our = ic_mont_to_our_mont(g1x_ic)
g1y_our = ic_mont_to_our_mont(g1y_ic)

g1x_r1, g1x_r2, g1x_rr = to_rns(g1x_our)
g1y_r1, g1y_r2, g1y_rr = to_rns(g1y_our)

lines.append("")
lines.append("// G1 generator in RNS form")
lines.append(f"__device__ __constant__ uint32_t G1_GEN_X_R1[{K}] = {{{', '.join(f'{v}u' for v in g1x_r1)}}};")
lines.append(f"__device__ __constant__ uint32_t G1_GEN_X_R2[{K}] = {{{', '.join(f'{v}u' for v in g1x_r2)}}};")
lines.append(f"#define G1_GEN_X_RR {g1x_rr}u")
lines.append(f"__device__ __constant__ uint32_t G1_GEN_Y_R1[{K}] = {{{', '.join(f'{v}u' for v in g1y_r1)}}};")
lines.append(f"__device__ __constant__ uint32_t G1_GEN_Y_R2[{K}] = {{{', '.join(f'{v}u' for v in g1y_r2)}}};")
lines.append(f"#define G1_GEN_Y_RR {g1y_rr}u")

# Parse pairing result e(G1,G2) as oracle value
lines.append("")
lines.append("// Oracle: e(G1, G2) in RNS form (12 Fp values)")
fp_names = ["c0c0c0", "c0c0c1", "c0c1c0", "c0c1c1",
            "c0c2c0", "c0c2c1", "c1c0c0", "c1c0c1",
            "c1c1c0", "c1c1c1", "c1c2c0", "c1c2c1"]
for name in fp_names:
    line = [l for l in pair_lines if l.strip().startswith(f"{name}=")][0]
    hex_val = line.strip().split('=')[1]
    ic_mont = limbs_to_int(hex_val)
    our_mont = ic_mont_to_our_mont(ic_mont)
    r1, r2, rr = to_rns(our_mont)
    lines.append(f"__device__ __constant__ uint32_t ORACLE_GT_{name.upper()}_R1[{K}] = {{{', '.join(f'{v}u' for v in r1)}}};")
    lines.append(f"__device__ __constant__ uint32_t ORACLE_GT_{name.upper()}_R2[{K}] = {{{', '.join(f'{v}u' for v in r2)}}};")
    lines.append(f"#define ORACLE_GT_{name.upper()}_RR {rr}u")

with open("rns_pairing_constants.cuh", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Generated rns_pairing_constants.cuh: {num_coeffs} coefficients, {len(flat_r1)} R1 values", file=sys.stderr)
