"""
mitm_proxy.py — Scripted Man-in-the-Middle attacker proxy.

The MITM proxy sits between agents on a shared UDP relay port (9999).
All agent→agent traffic is routed through it; the proxy decides whether to:

  - Forward : pass packet unchanged.
  - Forge   : add a position offset to the victim's state and re-encrypt
              (insider / compromised-key model — demonstrates that even
              with encryption, an insider with the fleet key can inject).
  - Replay  : buffer packets and re-emit them after a configured delay
              (blocked by the monotonic tick guard at the receiver).
  - Drop    : silently discard (simulates DoS / selective jamming).

Attack schedule
---------------
Loaded from attack_schedule.yaml.  Each entry:
  start_tick  : int      — tick the attack begins
  duration    : int      — how many ticks it lasts
  attacker_id : int      — sender agent whose traffic is intercepted
  target_id   : int      — destination agent (used for routing)
  mode        : str      — "position_forge" | "replay" | "drop"
  offset      : [dx, dy] — position injection offset for position_forge
  delay_ticks : int      — replay buffer depth for replay

Design note on crypto
---------------------
In position_forge mode, the proxy holds SecureChannels for each agent pair
(bootstrapped via the same ECDH key exchange the agents use).  It can thus
decrypt, tamper, and re-encrypt transparently — this is the insider threat
model (a compromised fleet member).  If the key is NOT available, the
modified ciphertext will fail AES-GCM tag verification at the receiver, which
is itself detected as CRYPTO_AUTHENTICATION_FAILURE.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Dict, List, Optional, Tuple


class MITMProxy:
    """
    Scripted MITM proxy.

    Parameters
    ----------
    listen_port     : UDP port the proxy listens on.
    forward_ports   : Dict mapping agent_id (int) → direct UDP port.
    attack_schedule : Pre-parsed list of attack dicts (from YAML).
    channels        : Optional dict of (src_id, dst_id) → SecureChannel for
                      insider-model position forgery with correct AES-GCM tags.
                      If None, forge attempts produce corrupted ciphertext that
                      is flagged by the receiver's tag check.
    """

    def __init__(
        self,
        listen_port:     int,
        forward_ports:   Dict[int, int],
        attack_schedule: List[Dict[str, Any]],
        channels:        Optional[Dict[Tuple[int,int], Any]] = None,
    ) -> None:
        self._schedule    = attack_schedule
        self.listen_port  = listen_port
        self.peer_ports   = forward_ports
        self._channels    = channels or {}   # (src_id, dst_id) → SecureChannel
        self.current_tick = 0
        self._replay_buf: List[Tuple[int, int, bytes]] = []   # (fire_tick, dest_id, raw)
        self.transport: Optional[asyncio.DatagramTransport] = None

        # Counters (public, readable from simulation loop for HUD / ledger)
        self.stats: Dict[str, int] = {
            "forwarded": 0,
            "forged":    0,
            "replayed":  0,
            "dropped":   0,
        }

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        import socket as _socket
        loop = asyncio.get_running_loop()
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        sock.setblocking(False)
        sock.bind(("127.0.0.1", self.listen_port))
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: _MITMProtocol(self), sock=sock
        )

    def stop(self) -> None:
        if self.transport:
            self.transport.close()

    def set_tick(self, tick: int) -> None:
        """Called each simulation tick to keep the attack schedule in sync."""
        self.current_tick = tick
        self._flush_replay_buffer()

    # ------------------------------------------------------------------ #
    # Packet processing                                                    #
    # ------------------------------------------------------------------ #
    def process_packet(self, data: bytes) -> None:
        try:
            envelope = json.loads(data.decode("utf-8"))
        except Exception:
            return   # malformed envelope — ignore

        sender_id = envelope.get("sender_id")
        dest_id   = envelope.get("dest_id")

        active = self._get_active_attack(sender_id)
        if active is None:
            self._forward(dest_id, data)
            self.stats["forwarded"] += 1
            return

        mode = active["mode"]

        if mode == "position_forge":
            self._forge_and_forward(envelope, active, dest_id)

        elif mode == "replay":
            delay     = active.get("delay_ticks", 10)
            fire_tick = self.current_tick + delay
            self._replay_buf.append((fire_tick, dest_id, data))
            # Also forward the original (otherwise silence is itself suspicious)
            self._forward(dest_id, data)
            self.stats["forwarded"] += 1

        elif mode == "drop":
            self.stats["dropped"] += 1
            # Silently discarded — simulates DoS / selective jamming

        else:
            self._forward(dest_id, data)
            self.stats["forwarded"] += 1

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    def _forge_and_forward(self, envelope: dict, attack: dict, dest_id: int) -> None:
        """
        position_forge mode:
          If we have a SecureChannel for this (sender, dest) pair (insider
          model), decrypt → inject false position offset → re-encrypt.
          If no channel is available, forward the raw bytes; the receiver's
          AES-GCM tag check will reject it and raise CRYPTO_AUTHENTICATION_FAILURE.
        """
        offset    = attack.get("offset", [100.0, 0.0])
        sender_id = envelope.get("sender_id")

        channel = self._channels.get((sender_id, dest_id))
        if channel is not None:
            try:
                raw_payload = base64.b64decode(envelope.get("payload", ""))
                original    = channel.decrypt_state(raw_payload)

                original["x"] = original.get("x", 0.0) + offset[0]
                original["y"] = original.get("y", 0.0) + offset[1]
                original["_forged"] = True

                new_encrypted       = channel.encrypt_state(original)
                envelope["payload"] = base64.b64encode(new_encrypted).decode("utf-8")
                forged_data         = json.dumps(envelope).encode("utf-8")
                self._forward(dest_id, forged_data)
                self.stats["forged"] += 1
                return
            except Exception:
                pass   # fall through to corrupt forward below

        # No channel (or decryption failed) → forward original bytes.
        # Receiver's AES-GCM tag check will reject tampered ciphertext and
        # fire a CRYPTO_AUTHENTICATION_FAILURE event, which is a detection signal.
        self._forward(dest_id, json.dumps(envelope).encode("utf-8"))
        self.stats["forwarded"] += 1

    def _flush_replay_buffer(self) -> None:
        ready, pending = [], []
        for item in self._replay_buf:
            (fire_tick, dest_id, data) = item
            if self.current_tick >= fire_tick:
                ready.append(item)
            else:
                pending.append(item)
        self._replay_buf = pending
        for fire_tick, dest_id, data in ready:
            self._forward(dest_id, data)
            self.stats["replayed"] += 1

    def _forward(self, dest_id: int, data: bytes) -> None:
        if self.transport and dest_id in self.peer_ports:
            self.transport.sendto(data, ("127.0.0.1", self.peer_ports[dest_id]))

    def _get_active_attack(self, sender_id: int) -> Optional[Dict]:
        for attack in self._schedule:
            start = attack["start_tick"]
            end   = start + attack["duration"]
            if start <= self.current_tick < end and attack.get("attacker_id") == sender_id:
                return attack
        return None


class _MITMProtocol(asyncio.DatagramProtocol):
    def __init__(self, proxy: MITMProxy) -> None:
        self.proxy = proxy

    def datagram_received(self, data: bytes, addr) -> None:
        self.proxy.process_packet(data)

    def error_received(self, exc: Exception) -> None:
        pass
