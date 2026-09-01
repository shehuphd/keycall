"""Streaming transcription sessions: live speech-to-text over a WebSocket.

A session is a context manager. Entering it connects and authenticates
(headers only — the credential never enters a URL; the session's audio
parameters ride the path as query parameters, which carry no secrets).
The caller pushes raw 16-bit mono PCM with ``send_audio``, signals the
end of audio with ``finish``, and reads normalized ``TranscriptionEvent``s
from ``events()``. Leaving the context closes the socket.

Reconnection is the caller's: a dropped connection surfaces as the
session ending (TranscriptionSessionEnded with the close reason and no
audio-duration summary), and resuming means opening a new session and
continuing to send audio from the point of the last FinalTranscript —
words since then were interim-only and are re-recognized from the
resent audio. KeyCall does not buffer or replay audio itself.

The wire and the credential live in the transport; the frame dialects
live in the adapters. This module only sequences them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from ._types import TranscriptionEvent, TranscriptionSessionEnded

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._transport import AsyncTransport, Transport


class TranscriptionSession:
    """Synchronous streaming transcription session. Use as a context
    manager."""

    def __init__(
        self,
        transport: Transport,
        *,
        path: str,
        translator: Any,
    ) -> None:
        self._transport = transport
        self._path = path
        self._translator = translator
        self._cm: Any = None
        self._wire: Any = None
        self._ended = False

    def __enter__(self) -> Self:
        self._cm = self._transport.realtime_connect(self._path)
        self._wire = self._cm.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> None:
        cm, self._cm, self._wire = self._cm, None, None
        if cm is not None:
            cm.__exit__(*exc_info)

    def _require_open(self) -> Any:
        if self._wire is None:
            raise RuntimeError(
                "transcription session is not open; use it as a context manager"
            )
        return self._wire

    def send_audio(self, pcm: bytes) -> None:
        """A chunk of caller audio: raw 16-bit mono PCM at the session's
        sample rate. The provider's translator owns the wire encoding —
        binary frames for most, a JSON text message where the dialect
        wraps audio (ElevenLabs)."""
        wire = self._require_open()
        frame = self._translator.encode_audio(pcm)
        if isinstance(frame, bytes):
            wire.send_bytes(frame)
        else:
            wire.send(frame)

    def finish(self) -> None:
        """No more audio is coming: ask the provider to finalize what it
        holds and close the session. Keep reading events() — the last
        finals and the session-ended event arrive after this."""
        wire = self._require_open()
        for message in self._translator.finish_messages():
            wire.send(message)

    def events(self, *, timeout: float | None = None) -> Iterator[TranscriptionEvent]:
        """Normalized events, in arrival order, until the peer closes the
        connection (the final event is always TranscriptionSessionEnded,
        carrying the provider's billable-audio-seconds count when its
        session summary arrived). ``timeout`` bounds the wait for each
        frame."""
        wire = self._require_open()
        while not self._ended:
            payload = wire.receive(timeout)
            if payload is None:
                self._ended = True
                yield TranscriptionSessionEnded(
                    reason=wire.close_reason,
                    audio_duration_seconds=self._translator.audio_duration_seconds,
                )
                return
            yield from self._translator.events_for_frame(payload)
            if getattr(self._translator, "session_over", False):
                # The provider holds the socket open after answering
                # finish() (ElevenLabs); the final that answers it is the
                # honest end of the session, so end it here rather than
                # waiting on a close that never comes.
                self._ended = True
                yield TranscriptionSessionEnded(
                    reason="client finished",
                    audio_duration_seconds=self._translator.audio_duration_seconds,
                )
                return


class AsyncTranscriptionSession:
    """Asynchronous twin of TranscriptionSession."""

    def __init__(
        self,
        transport: AsyncTransport,
        *,
        path: str,
        translator: Any,
    ) -> None:
        self._transport = transport
        self._path = path
        self._translator = translator
        self._cm: Any = None
        self._wire: Any = None
        self._ended = False

    async def __aenter__(self) -> Self:
        self._cm = self._transport.realtime_connect(self._path)
        self._wire = await self._cm.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        cm, self._cm, self._wire = self._cm, None, None
        if cm is not None:
            await cm.__aexit__(*exc_info)

    def _require_open(self) -> Any:
        if self._wire is None:
            raise RuntimeError(
                "transcription session is not open; use it as a context manager"
            )
        return self._wire

    async def send_audio(self, pcm: bytes) -> None:
        wire = self._require_open()
        frame = self._translator.encode_audio(pcm)
        if isinstance(frame, bytes):
            await wire.send_bytes(frame)
        else:
            await wire.send(frame)

    async def finish(self) -> None:
        wire = self._require_open()
        for message in self._translator.finish_messages():
            await wire.send(message)

    async def events(
        self, *, timeout: float | None = None
    ) -> AsyncIterator[TranscriptionEvent]:
        wire = self._require_open()
        while not self._ended:
            payload = await wire.receive(timeout)
            if payload is None:
                self._ended = True
                yield TranscriptionSessionEnded(
                    reason=wire.close_reason,
                    audio_duration_seconds=self._translator.audio_duration_seconds,
                )
                return
            for event in self._translator.events_for_frame(payload):
                yield event
            if getattr(self._translator, "session_over", False):
                self._ended = True
                yield TranscriptionSessionEnded(
                    reason="client finished",
                    audio_duration_seconds=self._translator.audio_duration_seconds,
                )
                return
