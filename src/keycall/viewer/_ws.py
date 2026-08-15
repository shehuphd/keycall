"""Minimal RFC 6455 WebSocket framing, hand-rolled because the viewer adds
no dependency to the base package (see _server.py's module docstring) and
the standard library ships an HTTP server but no WebSocket one.

Server-side only, and narrow on purpose: every frame a browser sends is
masked, per the spec, and every frame this module sends stays unmasked,
per the same spec. Binary frames are read (a ping is answered, anything
else is skipped) but never sent: the realtime bridge's protocol is JSON
text frames in both directions, audio included, base64-encoded.
"""

from __future__ import annotations

import base64
import hashlib
import struct
from typing import Protocol

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_OP_TEXT = 0x1
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA

# A peer that claims a multi-gigabyte frame is lying or broken; refuse
# before trying to read that many bytes rather than after.
_MAX_FRAME_BYTES = 2 * 1024 * 1024

__all__ = ["WebSocketConnection", "accept_key"]


class _Readable(Protocol):
    def read(self, n: int, /) -> bytes: ...


class _Writable(Protocol):
    def write(self, data: bytes, /) -> object: ...
    def flush(self) -> object: ...


def accept_key(sec_websocket_key: str) -> str:
    """The Sec-WebSocket-Accept value for a given Sec-WebSocket-Key, per
    RFC 6455 section 1.3: SHA-1 the key concatenated with the protocol's
    fixed GUID, then base64 the digest."""
    digest = hashlib.sha1((sec_websocket_key + _GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


class WebSocketConnection:
    """A hijacked HTTP connection, now speaking WebSocket frames over the
    same underlying socket. One reader and one writer may use this
    concurrently: the socket's read and write directions are
    independent, but two writers (or two readers) racing each other
    would interleave frames and corrupt the stream."""

    def __init__(self, rfile: _Readable, wfile: _Writable) -> None:
        self._rfile = rfile
        self._wfile = wfile

    def send_text(self, text: str) -> None:
        self._send_frame(_OP_TEXT, text.encode("utf-8"))

    def close(self) -> None:
        try:
            self._send_frame(_OP_CLOSE, b"")
        except OSError:
            pass

    def recv(self) -> str | None:
        """The next text message, or None once the peer closed the
        connection (a close frame, or the socket just ending). A ping is
        answered and skipped; anything else this bridge's protocol
        doesn't use (binary, pong, continuation) is skipped too."""
        while True:
            header = self._read_exact(2)
            if header is None:
                return None
            b0, b1 = header
            opcode = b0 & 0x0F
            masked = bool(b1 & 0x80)
            length = b1 & 0x7F
            if length == 126:
                ext = self._read_exact(2)
                if ext is None:
                    return None
                length = struct.unpack("!H", ext)[0]
            elif length == 127:
                ext = self._read_exact(8)
                if ext is None:
                    return None
                length = struct.unpack("!Q", ext)[0]
            if length > _MAX_FRAME_BYTES:
                return None
            mask = self._read_exact(4) if masked else b""
            payload = self._read_exact(length) if length else b""
            if payload is None:
                return None
            if masked and mask:
                payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
            if opcode == _OP_CLOSE:
                return None
            if opcode == _OP_PING:
                self._send_frame(_OP_PONG, payload)
                continue
            if opcode == _OP_TEXT:
                return payload.decode("utf-8", errors="replace")
            # Pong, binary, continuation: not part of this bridge's
            # protocol. Skip rather than fail the whole connection.

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        header = bytes([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header += bytes([length])
        elif length < 65536:
            header += bytes([126]) + struct.pack("!H", length)
        else:
            header += bytes([127]) + struct.pack("!Q", length)
        self._wfile.write(header + payload)
        self._wfile.flush()

    def _read_exact(self, n: int) -> bytes | None:
        if n == 0:
            return b""
        data = self._rfile.read(n)
        if not data or len(data) < n:
            return None
        return data
