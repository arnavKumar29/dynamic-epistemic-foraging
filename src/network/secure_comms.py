"""
secure_comms.py — Rev. 02 Cryptographic Protocol Layer.

Provides:
  - ECDHSession   : ECDH (SECP256R1) key agreement, no pre-shared secret.
  - SecureChannel : AES-256-GCM encrypt/decrypt with nonce-replay and
                    monotonic-tick guard (blocks replay & delay attacks).

S-DES is kept in sdes.py for educational purposes only.
All production traffic uses SecureChannel.
"""
from __future__ import annotations

import os
import json
from typing import Dict

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag


class ECDHSession:
    """
    One-time ECDH session for pairwise key derivation.

    Usage (conceptual fleet handshake):
        a = ECDHSession()
        b = ECDHSession()
        # Exchange public bytes over plaintext channel:
        key_a = a.derive_shared(b.pub_bytes())
        key_b = b.derive_shared(a.pub_bytes())
        assert key_a == key_b  # True
    """

    def __init__(self):
        self._priv = ec.generate_private_key(ec.SECP256R1())
        self._pub  = self._priv.public_key()

    def pub_bytes(self) -> bytes:
        """Serialized uncompressed public key (65 bytes)."""
        return self._pub.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )

    def derive_shared(self, peer_pub_bytes: bytes) -> bytes:
        """
        Derive a 32-byte AES key from peer's public key via ECDH + HKDF-SHA256.
        Forward-secret: ephemeral private key never reused.
        """
        peer_pub = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), peer_pub_bytes
        )
        raw_shared = self._priv.exchange(ec.ECDH(), peer_pub)
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"epistemic-foraging-fleet-v2",
        ).derive(raw_shared)


class SecureChannel:
    """
    AES-256-GCM encrypted channel with two layers of replay protection:

    1. Nonce uniqueness: each packet uses os.urandom(12); seen nonces
       are tracked (ring-buffer cleared at 10k entries to cap memory).
    2. Monotonic tick guard: each sender's tick must be strictly greater
       than the last accepted tick for that sender. This blocks:
         - Pure replay attacks (same packet retransmitted).
         - Delay attacks (old packets sent later with smaller ticks).
         - Clock rollback attacks.

    Tag authentication (built into AES-GCM) blocks:
        - Ciphertext forgery (MITM modifying payload bytes).
        - Key mismatch (attacker using wrong key).
    """

    NONCE_CACHE_LIMIT = 10_000

    def __init__(self, aes_key_32: bytes):
        if len(aes_key_32) != 32:
            raise ValueError(f"AES-256 requires 32-byte key, got {len(aes_key_32)}")
        self._aesgcm = AESGCM(aes_key_32)
        self._seen_nonces: set[bytes] = set()
        self._last_tick: Dict[str, int] = {}   # sender_id -> last accepted tick

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def encrypt_state(self, payload: dict) -> bytes:
        """
        Serialize payload dict to JSON, encrypt with AES-256-GCM.
        Returns: nonce (12 B) || ciphertext+tag bytes.
        """
        raw   = json.dumps(payload).encode("utf-8")
        nonce = os.urandom(12)
        ct    = self._aesgcm.encrypt(nonce, raw, None)
        return nonce + ct

    def decrypt_state(self, packet: bytes) -> dict:
        """
        Decrypt, authenticate, and validate monotonic tick.
        Raises ValueError on any cryptographic or replay violation.
        """
        if len(packet) < 28:   # 12 nonce + 1 plaintext + 16 tag = 29 min
            raise ValueError("Packet too short — likely truncated or garbage.")

        nonce = packet[:12]
        ct    = packet[12:]

        # --- Replay check (nonce uniqueness) ---
        if nonce in self._seen_nonces:
            raise ValueError("Replay attack detected: nonce already seen.")

        # --- Decrypt + tag verification ---
        try:
            raw = self._aesgcm.decrypt(nonce, ct, None)
        except InvalidTag:
            raise ValueError(
                "Authentication tag mismatch — packet forged, corrupted, or wrong key."
            )

        msg = json.loads(raw.decode("utf-8"))

        # --- Monotonic tick guard ---
        sender = str(msg.get("sender_id", "unknown"))
        tick   = int(msg.get("tick", 0))
        last   = self._last_tick.get(sender, -1)
        if tick <= last:
            raise ValueError(
                f"Tick regression: got {tick}, last accepted {last} from '{sender}'."
                " This is a replay or delay attack."
            )

        # Accept packet
        self._seen_nonces.add(nonce)
        if len(self._seen_nonces) > self.NONCE_CACHE_LIMIT:
            self._seen_nonces.clear()   # reset ring buffer
        self._last_tick[sender] = tick

        return msg
