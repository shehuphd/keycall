"""Realtime sessions: a live WebSocket conversation with a voice model.

A session is a context manager. Entering it connects, authenticates
(headers only — the credential never enters a URL), and sends the
provider its session configuration. The caller then pushes turns up with
``send_text`` / ``send_audio`` / ``end_audio_turn`` and reads normalized
``RealtimeEvent``s back from ``events()``. Leaving the context closes
the socket.

The wire and the credential live in the transport; the frame dialects
live in the adapters. This module only sequences them.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from ._types import RealtimeConfig, RealtimeEvent, RealtimeSessionEnded

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._transport import AsyncTransport, Transport


def _warn_on_provider_config(config: RealtimeConfig, provider: str) -> None:
    if config.provider_config is not None:
        warnings.warn(
            f"keycall: realtime provider_config is passed through to {provider!r} "
            "verbatim and will not port to other providers",
            UserWarning,
            stacklevel=3,
        )


class RealtimeSession:
    """Synchronous realtime session. Use as a context manager."""

    def __init__(
        self,
        transport: Transport,
        *,
        path: str,
        translator: Any,
        provider: str,
        config: RealtimeConfig,
    ) -> None:
        self._transport = transport
        self._path = path
        self._translator = translator
        self._provider = provider
        self._config = config
        self._cm: Any = None
        self._wire: Any = None
        self._ended = False

    def __enter__(self) -> Self:
        _warn_on_provider_config(self._config, self._provider)
        self._cm = self._transport.realtime_connect(self._path)
        self._wire = self._cm.__enter__()
        for message in self._translator.setup_messages():
            self._wire.send(message)
        return self

    def __exit__(self, *exc_info: object) -> None:
        cm, self._cm, self._wire = self._cm, None, None
        if cm is not None:
            cm.__exit__(*exc_info)

    def _require_open(self) -> Any:
        if self._wire is None:
            raise RuntimeError("realtime session is not open; use it as a context manager")
        return self._wire

    def send_text(self, text: str) -> None:
        """A whole user text turn; the provider answers with a response."""
        wire = self._require_open()
        for message in self._translator.user_text_messages(text):
            wire.send(message)

    def send_audio(self, pcm: bytes) -> None:
        """A chunk of caller audio (16-bit PCM; 24 kHz on OpenAI and xAI,
        16 kHz on Gemini). Chunks accumulate until the provider's voice
        detection, or ``end_audio_turn``, closes the turn."""
        wire = self._require_open()
        for message in self._translator.audio_chunk_messages(pcm):
            wire.send(message)

    def end_audio_turn(self) -> None:
        """The caller's audio turn is over; ask for the response."""
        wire = self._require_open()
        for message in self._translator.end_audio_messages():
            wire.send(message)

    def events(self, *, timeout: float | None = None) -> Iterator[RealtimeEvent]:
        """Normalized events, in arrival order, until the peer closes the
        connection (the final event is always RealtimeSessionEnded).
        ``timeout`` bounds the wait for each frame."""
        wire = self._require_open()
        while not self._ended:
            payload = wire.receive(timeout)
            if payload is None:
                self._ended = True
                yield RealtimeSessionEnded(reason=wire.close_reason)
                return
            yield from self._translator.events_for_frame(payload)


class AsyncRealtimeSession:
    """Asynchronous twin of RealtimeSession."""

    def __init__(
        self,
        transport: AsyncTransport,
        *,
        path: str,
        translator: Any,
        provider: str,
        config: RealtimeConfig,
    ) -> None:
        self._transport = transport
        self._path = path
        self._translator = translator
        self._provider = provider
        self._config = config
        self._cm: Any = None
        self._wire: Any = None
        self._ended = False

    async def __aenter__(self) -> Self:
        _warn_on_provider_config(self._config, self._provider)
        self._cm = self._transport.realtime_connect(self._path)
        self._wire = await self._cm.__aenter__()
        for message in self._translator.setup_messages():
            await self._wire.send(message)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        cm, self._cm, self._wire = self._cm, None, None
        if cm is not None:
            await cm.__aexit__(*exc_info)

    def _require_open(self) -> Any:
        if self._wire is None:
            raise RuntimeError("realtime session is not open; use it as a context manager")
        return self._wire

    async def send_text(self, text: str) -> None:
        wire = self._require_open()
        for message in self._translator.user_text_messages(text):
            await wire.send(message)

    async def send_audio(self, pcm: bytes) -> None:
        wire = self._require_open()
        for message in self._translator.audio_chunk_messages(pcm):
            await wire.send(message)

    async def end_audio_turn(self) -> None:
        wire = self._require_open()
        for message in self._translator.end_audio_messages():
            await wire.send(message)

    async def events(self, *, timeout: float | None = None) -> AsyncIterator[RealtimeEvent]:
        wire = self._require_open()
        while not self._ended:
            payload = await wire.receive(timeout)
            if payload is None:
                self._ended = True
                yield RealtimeSessionEnded(reason=wire.close_reason)
                return
            for event in self._translator.events_for_frame(payload):
                yield event
