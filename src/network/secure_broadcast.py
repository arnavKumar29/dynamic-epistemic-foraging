"""
secure_broadcast.py — Async UDP broadcast layer for inter-agent communication.

Each agent owns one SecureBroadcastLayer.  It:
  - Listens on a dedicated UDP port (9000 + agent_id).
  - Sends encrypted packets to all peers (or through the MITM proxy).
  - Decrypts received packets and calls an on_receive callback.
  - Uses SecureChannel (AES-256-GCM) for all production traffic.
  - Optionally uses SDES for educational demonstrations.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Callable, Dict, Optional

from src.network.secure_comms import SecureChannel
from src.network.sdes import SDES


class SecureBroadcastLayer:
    """
    Asyncio-based UDP transport for one agent.

    Args:
        agent_id   : Integer agent id.
        channel    : Initialized SecureChannel (shared key must match peers).
        peer_ports : Dict mapping peer_id -> UDP port.
        listen_port: The port this agent listens on.
        use_sdes   : If True, use S-DES instead of AES-GCM (educational only).
        sdes_key   : 10-bit integer key for S-DES.
    """

    def __init__(
        self,
        agent_id: int,
        channel: SecureChannel,
        peer_ports: Dict[int, int],
        listen_port: int,
        use_sdes: bool = False,
        sdes_key: int = 882,
    ):
        self.agent_id   = agent_id
        self.channel    = channel
        self.peer_ports = peer_ports
        self.listen_port = listen_port
        self.use_sdes   = use_sdes
        self.sdes       = SDES(sdes_key) if use_sdes else None
        self.transport: Optional[asyncio.DatagramTransport] = None
        self._on_receive: Optional[Callable[[dict], None]] = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #
    async def start(self, on_receive: Callable[[dict], None]) -> None:
        """Bind UDP socket and register receive callback."""
        import socket as _socket
        self._on_receive = on_receive
        loop = asyncio.get_running_loop()
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        sock.setblocking(False)
        sock.bind(("127.0.0.1", self.listen_port))
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(self),
            sock=sock,
        )

    def stop(self) -> None:
        if self.transport:
            self.transport.close()
            self.transport = None

    # ------------------------------------------------------------------ #
    # Sending                                                              #
    # ------------------------------------------------------------------ #
    async def broadcast(
        self, payload: dict, proxy_port: Optional[int] = None
    ) -> None:
        """
        Encrypt and send payload to all peers.
        If proxy_port is set, route through MITM proxy instead of direct.
        """
        for peer_id in self.peer_ports:
            if peer_id == self.agent_id:
                continue
            await self.send_to(peer_id, payload, proxy_port=proxy_port)

    async def send_to(
        self, peer_id: int, payload: dict, proxy_port: Optional[int] = None
    ) -> None:
        """Encrypt and send to a specific peer (or through proxy)."""
        if self.transport is None:
            return

        if self.use_sdes:
            raw      = json.dumps(payload).encode("utf-8")
            ct       = self.sdes.encrypt(raw)
            envelope = {
                "sender_id": self.agent_id,
                "dest_id":   peer_id,
                "cipher":    "SDES",
                "payload":   base64.b64encode(ct).decode("utf-8"),
            }
        else:
            encrypted = self.channel.encrypt_state(payload)
            envelope  = {
                "sender_id": self.agent_id,
                "dest_id":   peer_id,
                "cipher":    "AES-256-GCM",
                "payload":   base64.b64encode(encrypted).decode("utf-8"),
            }

        data        = json.dumps(envelope).encode("utf-8")
        target_port = proxy_port if proxy_port is not None else self.peer_ports[peer_id]
        self.transport.sendto(data, ("127.0.0.1", target_port))

    # ------------------------------------------------------------------ #
    # Receiving (called from _UDPProtocol)                                #
    # ------------------------------------------------------------------ #
    def _handle_raw(self, data: bytes) -> None:
        try:
            envelope    = json.loads(data.decode("utf-8"))
            raw_payload = base64.b64decode(envelope["payload"])
            cipher      = envelope.get("cipher", "AES-256-GCM")

            if cipher == "SDES" and self.sdes:
                decrypted = self.sdes.decrypt(raw_payload)
                msg       = json.loads(decrypted.decode("utf-8"))
            else:
                msg = self.channel.decrypt_state(raw_payload)

            if self._on_receive:
                self._on_receive(msg)

        except Exception as e:
            # Crypto failure / auth mismatch / replay → signal corruption
            sender = -1
            try:
                envelope = json.loads(data.decode("utf-8"))
                sender   = envelope.get("sender_id", -1)
            except Exception:
                pass
            if self._on_receive:
                self._on_receive({
                    "CORRUPTED": True,
                    "sender_id": sender,
                    "error": str(e),
                })


class _UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, parent: SecureBroadcastLayer):
        self.parent = parent

    def datagram_received(self, data: bytes, addr) -> None:
        self.parent._handle_raw(data)

    def error_received(self, exc: Exception) -> None:
        pass   # suppress noisy ICMP unreachable on localhost
