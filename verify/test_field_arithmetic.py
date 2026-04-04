#!/usr/bin/env python3
"""
BLS12-381 Field Arithmetic Test Oracle
Generates test vectors for GPU kernel verification at each layer:
  Fp → Fp2 → Fp6 → Fp12 → Miller Loop → Final Exp

Uses pure Python (no external deps except hashlib) to compute
reference values. Each function prints hex test vectors that the
CUDA code must match exactly.
"""

# BLS12-381 prime
P = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

# Montgomery R = 2^384 mod p
R = pow(2, 384, P)
# R^2 mod p
R_SQR = pow(R, 2, P)
# R^{-1} mod p
R_INV = pow(R, P - 2, P)
# M0 = -p^{-1} mod 2^64
M0 = (-pow(P, -1, 2**64)) % (2**64)

def to_mont(a):
    """Convert to Montgomery form: a * R mod p"""
    return (a * R) % P

def from_mont(a):
    """Convert from Montgomery form: a * R^{-1} mod p"""
    return (a * R_INV) % P

def mont_mul(a, b):
    """Montgomery multiplication: a * b * R^{-1} mod p (both inputs in Montgomery form)"""
    return (a * b * R_INV) % P

def to_limbs(val, n=6):
    """Convert integer to n uint64 limbs (little-endian)"""
    limbs = []
    for _ in range(n):
        limbs.append(val & 0xFFFFFFFFFFFFFFFF)
        val >>= 64
    return limbs

def limbs_hex(val, n=6):
    """Convert to hex limb representation for C comparison"""
    limbs = to_limbs(val, n)
    return ", ".join(f"0x{l:016x}" for l in limbs)

# ============================================================
# Fp2 = Fp[u] / (u^2 + 1)   where u^2 = -1
# Element: a + b*u
# ============================================================

class Fp2:
    def __init__(self, c0, c1):
        self.c0 = c0 % P  # real
        self.c1 = c1 % P  # imaginary

    def __add__(self, other):
        return Fp2((self.c0 + other.c0) % P, (self.c1 + other.c1) % P)

    def __sub__(self, other):
        return Fp2((self.c0 - other.c0) % P, (self.c1 - other.c1) % P)

    def __neg__(self):
        return Fp2((-self.c0) % P, (-self.c1) % P)

    def __mul__(self, other):
        if isinstance(other, int):
            return Fp2((self.c0 * other) % P, (self.c1 * other) % P)
        # (a + bu)(c + du) = (ac - bd) + (ad + bc)u
        ac = (self.c0 * other.c0) % P
        bd = (self.c1 * other.c1) % P
        ad = (self.c0 * other.c1) % P
        bc = (self.c1 * other.c0) % P
        return Fp2((ac - bd) % P, (ad + bc) % P)

    def sqr(self):
        # (a + bu)^2 = (a^2 - b^2) + 2ab*u = (a+b)(a-b) + 2ab*u
        a, b = self.c0, self.c1
        t = (a * b) % P
        return Fp2(((a + b) * (a - b)) % P, (2 * t) % P)

    def conjugate(self):
        return Fp2(self.c0, (-self.c1) % P)

    def norm(self):
        """Norm: c0^2 + c1^2"""
        return (self.c0 * self.c0 + self.c1 * self.c1) % P

    def inverse(self):
        n = pow(self.norm(), P - 2, P)
        return Fp2((self.c0 * n) % P, ((-self.c1) * n) % P)

    @staticmethod
    def zero():
        return Fp2(0, 0)

    @staticmethod
    def one():
        return Fp2(1, 0)

    def __eq__(self, other):
        return self.c0 == other.c0 and self.c1 == other.c1

    def __repr__(self):
        return f"Fp2({self.c0}, {self.c1})"

# ============================================================
# Fp6 = Fp2[v] / (v^3 - β) where β = u + 1  (the cubic nonresidue in Fp2)
# Element: c0 + c1*v + c2*v^2
# ============================================================

# Cubic non-residue: β = u + 1 = Fp2(1, 1)
BETA = Fp2(1, 1)

def mul_by_nonresidue_fp2(a):
    """Multiply Fp2 element by β = (1 + u): (c0 + c1*u)(1 + u) = (c0 - c1) + (c0 + c1)*u"""
    return Fp2((a.c0 - a.c1) % P, (a.c0 + a.c1) % P)

class Fp6:
    def __init__(self, c0, c1, c2):
        self.c0 = c0  # Fp2
        self.c1 = c1  # Fp2
        self.c2 = c2  # Fp2

    def __add__(self, other):
        return Fp6(self.c0 + other.c0, self.c1 + other.c1, self.c2 + other.c2)

    def __sub__(self, other):
        return Fp6(self.c0 - other.c0, self.c1 - other.c1, self.c2 - other.c2)

    def __neg__(self):
        return Fp6(-self.c0, -self.c1, -self.c2)

    def __mul__(self, other):
        if isinstance(other, Fp2):
            return Fp6(self.c0 * other, self.c1 * other, self.c2 * other)
        # Karatsuba over Fp2
        a_a = self.c0 * other.c0
        b_b = self.c1 * other.c1
        c_c = self.c2 * other.c2

        t1 = (self.c1 + self.c2) * (other.c1 + other.c2) - b_b - c_c
        t1 = a_a + mul_by_nonresidue_fp2(t1)

        t2 = (self.c0 + self.c1) * (other.c0 + other.c1) - a_a - b_b
        t2 = t2 + mul_by_nonresidue_fp2(c_c)

        t3 = (self.c0 + self.c2) * (other.c0 + other.c2) - a_a - c_c + b_b

        return Fp6(t1, t2, t3)

    def sqr(self):
        return self * self

    def inverse(self):
        c0s = self.c0.sqr()
        c1s = self.c1.sqr()
        c2s = self.c2.sqr()
        c01 = self.c0 * self.c1
        c02 = self.c0 * self.c2
        c12 = self.c1 * self.c2

        # t0 = c0^2 - β * c1 * c2
        t0 = c0s - mul_by_nonresidue_fp2(c12)
        # t1 = β * c2^2 - c0 * c1
        t1 = mul_by_nonresidue_fp2(c2s) - c01
        # t2 = c1^2 - c0 * c2
        t2 = c1s - c02

        inv = (self.c0 * t0 + mul_by_nonresidue_fp2(self.c2 * t1 + self.c1 * t2)).inverse()

        return Fp6(t0 * inv, t1 * inv, t2 * inv)

    @staticmethod
    def zero():
        return Fp6(Fp2.zero(), Fp2.zero(), Fp2.zero())

    @staticmethod
    def one():
        return Fp6(Fp2.one(), Fp2.zero(), Fp2.zero())

    def __eq__(self, other):
        return self.c0 == other.c0 and self.c1 == other.c1 and self.c2 == other.c2

    def __repr__(self):
        return f"Fp6({self.c0}, {self.c1}, {self.c2})"

# ============================================================
# Fp12 = Fp6[w] / (w^2 - v)
# Element: c0 + c1*w   where c0, c1 are Fp6
# ============================================================

def mul_fp6_by_nonresidue(a):
    """Multiply Fp6 by v (the variable in Fp6): shift components and multiply by β"""
    # v * (c0 + c1*v + c2*v^2) = β*c2 + c0*v + c1*v^2
    return Fp6(mul_by_nonresidue_fp2(a.c2), a.c0, a.c1)

class Fp12:
    def __init__(self, c0, c1):
        self.c0 = c0  # Fp6
        self.c1 = c1  # Fp6

    def __add__(self, other):
        return Fp12(self.c0 + other.c0, self.c1 + other.c1)

    def __sub__(self, other):
        return Fp12(self.c0 - other.c0, self.c1 - other.c1)

    def __neg__(self):
        return Fp12(-self.c0, -self.c1)

    def __mul__(self, other):
        # (a + bw)(c + dw) = (ac + bd*v) + (ad + bc)w
        # where v is the Fp6 nonresidue
        aa = self.c0 * other.c0
        bb = self.c1 * other.c1
        return Fp12(
            aa + mul_fp6_by_nonresidue(bb),
            (self.c0 + self.c1) * (other.c0 + other.c1) - aa - bb
        )

    def sqr(self):
        ab = self.c0 * self.c1
        c0c1 = self.c0 + self.c1
        c0_plus_vc1 = self.c0 + mul_fp6_by_nonresidue(self.c1)
        return Fp12(
            c0_plus_vc1 * c0c1 - ab - mul_fp6_by_nonresidue(ab),
            ab + ab
        )

    def conjugate(self):
        """Unitary inverse for elements on cyclotomic subgroup: conj(a + bw) = a - bw"""
        return Fp12(self.c0, -self.c1)

    def inverse(self):
        t0 = self.c0 * self.c0
        t1 = self.c1 * self.c1
        t0 = t0 - mul_fp6_by_nonresidue(t1)
        t0_inv = t0.inverse()
        return Fp12(self.c0 * t0_inv, -(self.c1 * t0_inv))

    @staticmethod
    def one():
        return Fp12(Fp6.one(), Fp6.zero())

    def __eq__(self, other):
        return self.c0 == other.c0 and self.c1 == other.c1

    def __repr__(self):
        return f"Fp12({self.c0}, {self.c1})"

# ============================================================
# Test vector generation
# ============================================================

def test_fp():
    print("=" * 60)
    print("Fp Test Vectors")
    print("=" * 60)

    a = 42
    b = 77
    a_mont = to_mont(a)
    b_mont = to_mont(b)

    print(f"a = {a}, b = {b}")
    print(f"a_mont = {limbs_hex(a_mont)}")
    print(f"b_mont = {limbs_hex(b_mont)}")

    # a + b
    ab_add = (a_mont + b_mont) % P
    print(f"a + b (mont) = {limbs_hex(ab_add)}")
    print(f"a + b (val)  = {from_mont(ab_add)}")

    # a * b
    ab_mul = mont_mul(a_mont, b_mont)
    print(f"a * b (mont) = {limbs_hex(ab_mul)}")
    print(f"a * b (val)  = {from_mont(ab_mul)}")

    # a^2
    a_sqr = mont_mul(a_mont, a_mont)
    print(f"a^2 (mont)   = {limbs_hex(a_sqr)}")
    print(f"a^2 (val)    = {from_mont(a_sqr)}")
    print()

def test_fp2():
    print("=" * 60)
    print("Fp2 Test Vectors (non-Montgomery, pure field)")
    print("=" * 60)

    a = Fp2(3, 7)
    b = Fp2(11, 5)

    print(f"a = {a}")
    print(f"b = {b}")

    c = a * b
    print(f"a * b = {c}")
    print(f"  c0 = {c.c0}")
    print(f"  c1 = {c.c1}")

    d = a.sqr()
    print(f"a^2 = {d}")
    print(f"  c0 = {d.c0}")
    print(f"  c1 = {d.c1}")

    e = a.inverse()
    check = a * e
    print(f"a^-1 = {e}")
    print(f"a * a^-1 = {check} (should be (1, 0))")

    nr = mul_by_nonresidue_fp2(a)
    print(f"β * a = {nr} (β = 1+u)")
    print()

def test_fp6():
    print("=" * 60)
    print("Fp6 Test Vectors")
    print("=" * 60)

    a = Fp6(Fp2(1, 2), Fp2(3, 4), Fp2(5, 6))
    b = Fp6(Fp2(7, 8), Fp2(9, 10), Fp2(11, 12))

    c = a * b
    print(f"Fp6 mul c0 = {c.c0}")
    print(f"Fp6 mul c1 = {c.c1}")
    print(f"Fp6 mul c2 = {c.c2}")

    d = a.inverse()
    check = a * d
    print(f"Fp6 a * a^-1 c0 = {check.c0} (should be (1,0))")
    print(f"Fp6 a * a^-1 c1 = {check.c1} (should be (0,0))")
    print(f"Fp6 a * a^-1 c2 = {check.c2} (should be (0,0))")
    print()

def test_fp12():
    print("=" * 60)
    print("Fp12 Test Vectors")
    print("=" * 60)

    a = Fp12(
        Fp6(Fp2(1, 2), Fp2(3, 4), Fp2(5, 6)),
        Fp6(Fp2(7, 8), Fp2(9, 10), Fp2(11, 12))
    )
    b = Fp12(
        Fp6(Fp2(13, 14), Fp2(15, 16), Fp2(17, 18)),
        Fp6(Fp2(19, 20), Fp2(21, 22), Fp2(23, 24))
    )

    c = a * b
    print(f"Fp12 mul c0.c0 = {c.c0.c0}")
    print(f"Fp12 mul c0.c1 = {c.c0.c1}")
    print(f"Fp12 mul c0.c2 = {c.c0.c2}")
    print(f"Fp12 mul c1.c0 = {c.c1.c0}")
    print(f"Fp12 mul c1.c1 = {c.c1.c1}")
    print(f"Fp12 mul c1.c2 = {c.c1.c2}")

    d = a.sqr()
    print(f"Fp12 sqr c0.c0 = {d.c0.c0}")

    e = a.inverse()
    check = a * e
    print(f"Fp12 a * a^-1 c0.c0 = {check.c0.c0} (should be (1,0))")
    print(f"Fp12 a * a^-1 c1.c0 = {check.c1.c0} (should be (0,0))")

    conj = a.conjugate()
    print(f"Fp12 conjugate c1.c0 = {conj.c1.c0}")
    print()

# ============================================================
# BLS12-381 curve parameters for pairing
# ============================================================

# The parameter x (also called u or z) for BLS12-381
# x = -0xd201000000010000
BLS_X = 0xd201000000010000
BLS_X_IS_NEG = True

def test_pairing_constants():
    print("=" * 60)
    print("BLS12-381 Pairing Constants")
    print("=" * 60)
    print(f"P = {hex(P)}")
    print(f"R (Montgomery) = {hex(R)}")
    print(f"R^2 mod P = {hex(R_SQR)}")
    print(f"M0 = {hex(M0)}")
    print(f"BLS x param = {hex(BLS_X)}")
    print(f"x is negative = {BLS_X_IS_NEG}")
    print(f"x bit length = {BLS_X.bit_length()}")
    print(f"x binary = {bin(BLS_X)}")

    # Count operations in miller loop
    bits = BLS_X.bit_length()
    doubles = bits - 1  # Skip MSB
    adds = bin(BLS_X).count('1') - 1  # Non-MSB set bits
    print(f"Miller loop: {doubles} doublings + {adds} additions = {doubles + adds} line evals")
    print()

if __name__ == "__main__":
    test_pairing_constants()
    test_fp()
    test_fp2()
    test_fp6()
    test_fp12()

    print("=" * 60)
    print("ALL TEST VECTORS GENERATED")
    print("Each layer must match before proceeding to the next.")
    print("=" * 60)
