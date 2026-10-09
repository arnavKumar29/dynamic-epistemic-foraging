"""
sdes.py — Complete operational Simplified DES (S-DES) cipher.

Used for EDUCATIONAL demonstration only ("break S-DES in class, can't break AES").
Operates on 8-bit data blocks using a 10-bit key.

NOT used for actual security — AES-256-GCM is the production cipher.
"""
from __future__ import annotations


class SDES:
    """
    Byte-level S-DES implementation.
    Permutation tables use 1-indexed positions (as in standard S-DES spec).
    """

    # --- Permutation tables (1-indexed bit positions) ---
    P10 = (3, 5, 2, 7, 4, 10, 1, 9, 8, 6)
    P8  = (6, 3, 7, 4, 8, 5, 10, 9)
    IP  = (2, 6, 3, 1, 4, 8, 5, 7)
    IP_INV = (4, 1, 3, 5, 7, 2, 8, 6)
    EP  = (4, 1, 2, 3, 2, 3, 4, 1)
    P4  = (2, 4, 3, 1)

    S0 = [
        [1, 0, 3, 2],
        [3, 2, 1, 0],
        [0, 2, 1, 3],
        [3, 1, 3, 2],
    ]
    S1 = [
        [0, 1, 2, 3],
        [2, 0, 1, 3],
        [3, 0, 1, 0],
        [2, 1, 0, 3],
    ]

    def __init__(self, key_10bit: int):
        self.k1, self.k2 = self._generate_keys(key_10bit & 0x3FF)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    def _permute(self, bits: int, table: tuple, length: int) -> int:
        res = 0
        for i, pos in enumerate(table):
            bit = (bits >> (length - pos)) & 1
            res |= bit << (len(table) - 1 - i)
        return res

    def _left_shift(self, bits: int, count: int, length: int = 5) -> int:
        mask = (1 << length) - 1
        return ((bits << count) | (bits >> (length - count))) & mask

    def _generate_keys(self, key: int):
        p10 = self._permute(key, self.P10, 10)
        left  = (p10 >> 5) & 0x1F
        right = p10 & 0x1F

        left  = self._left_shift(left,  1)
        right = self._left_shift(right, 1)
        k1 = self._permute((left << 5) | right, self.P8, 10)

        left  = self._left_shift(left,  2)
        right = self._left_shift(right, 2)
        k2 = self._permute((left << 5) | right, self.P8, 10)

        return k1, k2

    def _fk(self, bits: int, key: int) -> int:
        left  = (bits >> 4) & 0x0F
        right = bits & 0x0F
        ep    = self._permute(right, self.EP, 4)
        xored = ep ^ key

        l_xor = (xored >> 4) & 0x0F
        r_xor = xored & 0x0F

        r0 = ((l_xor & 0x08) >> 2) | (l_xor & 0x01)
        c0 = (l_xor >> 1) & 0x03
        r1 = ((r_xor & 0x08) >> 2) | (r_xor & 0x01)
        c1 = (r_xor >> 1) & 0x03

        p4 = self._permute((self.S0[r0][c0] << 2) | self.S1[r1][c1], self.P4, 4)
        return ((left ^ p4) << 4) | right

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def encrypt_byte(self, byte: int) -> int:
        ip      = self._permute(byte & 0xFF, self.IP, 8)
        f1      = self._fk(ip, self.k1)
        swapped = ((f1 & 0x0F) << 4) | ((f1 >> 4) & 0x0F)
        f2      = self._fk(swapped, self.k2)
        return self._permute(f2, self.IP_INV, 8)

    def decrypt_byte(self, byte: int) -> int:
        ip      = self._permute(byte & 0xFF, self.IP, 8)
        f1      = self._fk(ip, self.k2)   # keys reversed for decryption
        swapped = ((f1 & 0x0F) << 4) | ((f1 >> 4) & 0x0F)
        f2      = self._fk(swapped, self.k1)
        return self._permute(f2, self.IP_INV, 8)

    def encrypt(self, data: bytes) -> bytes:
        return bytes([self.encrypt_byte(b) for b in data])

    def decrypt(self, data: bytes) -> bytes:
        return bytes([self.decrypt_byte(b) for b in data])
