"""
test_network.py — Unit tests for secure comms (AES-256-GCM + replay guard).
"""
import pytest
from src.network.secure_comms import ECDHSession, SecureChannel


class TestECDHSession:

    def test_key_derivation_deterministic(self):
        """Two ECDH sessions with each other's public key must derive the same secret."""
        a = ECDHSession()
        b = ECDHSession()
        ka = a.derive_shared(b.pub_bytes())
        kb = b.derive_shared(a.pub_bytes())
        assert ka == kb, "ECDH shared secrets must be equal"

    def test_derived_key_length(self):
        a = ECDHSession()
        b = ECDHSession()
        key = a.derive_shared(b.pub_bytes())
        assert len(key) == 32, "HKDF should produce 32-byte (256-bit) key"

    def test_different_sessions_different_keys(self):
        a1, b1 = ECDHSession(), ECDHSession()
        a2, b2 = ECDHSession(), ECDHSession()
        k1 = a1.derive_shared(b1.pub_bytes())
        k2 = a2.derive_shared(b2.pub_bytes())
        assert k1 != k2, "Fresh ECDH sessions should yield different keys"

    def test_pub_bytes_length(self):
        a = ECDHSession()
        assert len(a.pub_bytes()) == 65, "Uncompressed SECP256R1 point = 65 bytes"


class TestSecureChannel:

    @pytest.fixture
    def key(self):
        import os
        return os.urandom(32)

    @pytest.fixture
    def channel(self, key):
        return SecureChannel(key)

    def test_encrypt_decrypt_roundtrip(self, channel):
        payload = {"sender_id": 1, "tick": 5, "x": 100.0, "y": 200.0}
        channel_rx = SecureChannel.__new__(SecureChannel)
        channel_rx._aesgcm      = channel._aesgcm
        channel_rx._seen_nonces = set()
        channel_rx._last_tick   = {}
        packet = channel.encrypt_state(payload)
        result = channel_rx.decrypt_state(packet)
        assert result["x"]         == 100.0
        assert result["sender_id"] == 1

    def test_wrong_key_raises(self, key):
        import os
        ch1 = SecureChannel(key)
        ch2 = SecureChannel(os.urandom(32))   # different key
        payload = {"sender_id": 0, "tick": 1}
        packet  = ch1.encrypt_state(payload)
        with pytest.raises(ValueError, match="Authentication tag"):
            ch2.decrypt_state(packet)

    def test_replay_attack_blocked(self, channel):
        payload = {"sender_id": 0, "tick": 1}
        packet  = channel.encrypt_state(payload)
        # First decryption OK
        channel.decrypt_state(packet)
        # Second decryption (replay) must fail
        with pytest.raises(ValueError, match="[Rr]eplay|[Nn]once"):
            channel.decrypt_state(packet)

    def test_tick_regression_blocked(self, channel):
        p1 = channel.encrypt_state({"sender_id": "A", "tick": 10})
        p2 = channel.encrypt_state({"sender_id": "A", "tick": 5})   # lower tick
        channel.decrypt_state(p1)   # accept tick=10
        with pytest.raises(ValueError, match="[Tt]ick|[Rr]eplay"):
            channel.decrypt_state(p2)

    def test_tampered_packet_rejected(self, channel):
        payload = {"sender_id": 0, "tick": 1, "x": 100.0}
        packet  = channel.encrypt_state(payload)
        # Flip a byte in the ciphertext
        tampered        = bytearray(packet)
        tampered[-5]   ^= 0xFF
        with pytest.raises(ValueError):
            channel.decrypt_state(bytes(tampered))

    def test_invalid_key_length_raises(self):
        with pytest.raises(ValueError, match="32-byte"):
            SecureChannel(b"tooshort")
