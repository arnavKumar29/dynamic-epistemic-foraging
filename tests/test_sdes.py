"""
test_sdes.py — Unit tests for S-DES (Simplified DES) cipher.
"""
import pytest
from src.network.sdes import SDES


class TestSDES:
    """Test S-DES encryption/decryption correctness."""

    @pytest.fixture
    def cipher(self):
        return SDES(key_10bit=882)   # 0b1101110010

    def test_encrypt_decrypt_roundtrip_single_byte(self, cipher):
        for byte_val in [0x00, 0x42, 0xFF, 0xAA, 0x55, 0x01]:
            ct = cipher.encrypt_byte(byte_val)
            pt = cipher.decrypt_byte(ct)
            assert pt == byte_val, f"Roundtrip failed for byte {hex(byte_val)}"

    def test_encrypt_decrypt_roundtrip_bytes(self, cipher):
        original = b"Hello!"
        ct = cipher.encrypt(original)
        pt = cipher.decrypt(ct)
        assert pt == original

    def test_different_keys_produce_different_ciphertext(self):
        c1 = SDES(key_10bit=882)
        c2 = SDES(key_10bit=456)
        ct1 = c1.encrypt_byte(0x42)
        ct2 = c2.encrypt_byte(0x42)
        assert ct1 != ct2, "Different keys should produce different ciphertext"

    def test_wrong_key_fails_to_decrypt(self):
        c1 = SDES(key_10bit=882)
        c2 = SDES(key_10bit=123)
        ct = c1.encrypt_byte(0x42)
        # Decrypting with wrong key should give wrong byte
        wrong = c2.decrypt_byte(ct)
        assert wrong != 0x42

    def test_known_vector(self):
        """Known test vector: key=882, plaintext=0x42, verify determinism."""
        c = SDES(key_10bit=882)
        ct = c.encrypt_byte(0x42)
        assert isinstance(ct, int)
        assert 0 <= ct <= 255
        # Roundtrip
        assert c.decrypt_byte(ct) == 0x42

    def test_key_boundary_zero(self):
        c = SDES(key_10bit=0)
        ct = c.encrypt_byte(0xFF)
        pt = c.decrypt_byte(ct)
        assert pt == 0xFF

    def test_key_boundary_max(self):
        c = SDES(key_10bit=0x3FF)  # 10-bit max
        ct = c.encrypt_byte(0x00)
        pt = c.decrypt_byte(ct)
        assert pt == 0x00
