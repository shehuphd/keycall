"""Live sessions: a full-duplex voice conversation with a live model.

This is a sibling of the realtime session, not a replacement. It serves
OpenAI's gpt-live on its own ``v1/live/sessions`` WebSocket endpoint,
where the caller's audio and the model's audio overlap (full duplex) and
the model delegates reasoning and tool use to a separately-billed backend
model. The realtime session (``realtime()``, the Realtime API) is
untouched.

A session is a context manager. Entering it connects, authenticates
(headers only; the credential never enters a URL), and sends the session
configuration. The caller then streams audio up with ``send_audio`` and
reads normalized ``LiveEvent``s back from ``events()``. Because the model
does its own endpointing, no explicit turn boundary is required from the
caller; ``send_text`` and ``end_audio_turn`` remain for manual control.
Leaving the context closes the socket.

The wire and the credential live in the transport; the frame dialect
lives in the adapter's translator. This module only sequences them.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from ._types import LiveConfig, LiveEvent, LiveSessionEnded

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._transport import AsyncTransport, Transport


def _warn_on_provider_config(config: LiveConfig, provider: str) -> None:
    if config.provider_config is not None:
        warnings.warn(
            f"keycall: live provider_config is passed through to {provider!r} "
            "verbatim and will not port to other providers",
            UserWarning,
            stacklevel=3,
        )


class LiveSession:
    """Synchronous full-duplex live session. Use as a context manager."""

    def __init__(
        self,
        transport: Transport,
        *,
        path: str,
        translator: Any,
        provider: str,
        config: LiveConfig,
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
            raise RuntimeError("live session is not open; use it as a context manager")
        return self._wire

    def send_text(self, text: str) -> None:
        """A whole user text turn; the model answers with a response."""
        wire = self._require_open()
        for message in self._translator.user_text_messages(text):
            wire.send(message)

    def send_audio(self, pcm: bytes) -> None:
        """A chunk of caller audio (16-bit PCM). In a full-duplex session
        chunks stream continuously and the model endpoints the caller's
        turn itself; ``end_audio_turn`` forces a turn boundary by hand."""
        wire = self._require_open()
        for message in self._translator.audio_chunk_messages(pcm):
            wire.send(message)

    def end_audio_turn(self) -> None:
        """Close the caller's audio turn by hand, rather than leaving the
        model's own endpointing to decide."""
        wire = self._require_open()
        for message in self._translator.end_audio_messages():
            wire.send(message)

    def events(self, *, timeout: float | None = None) -> Iterator[LiveEvent]:
        """Normalized events, in arrival order, until the peer closes the
        connection (the final event is always LiveSessionEnded).
        ``timeout`` bounds the wait for each frame."""
        wire = self._require_open()
        while not self._ended:
            payload = wire.receive(timeout)
            if payload is None:
                self._ended = True
                yield LiveSessionEnded(
                    reason=wire.close_reason,
                    billed_seconds=self._translator.billed_seconds,
                )
                return
            yield from self._translator.events_for_frame(payload)


class AsyncLiveSession:
    """Asynchronous twin of LiveSession."""

    def __init__(
        self,
        transport: AsyncTransport,
        *,
        path: str,
        translator: Any,
        provider: str,
        config: LiveConfig,
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
            raise RuntimeError("live session is not open; use it as a context manager")
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

    async def events(self, *, timeout: float | None = None) -> AsyncIterator[LiveEvent]:
        wire = self._require_open()
        while not self._ended:
            payload = await wire.receive(timeout)
            if payload is None:
                self._ended = True
                yield LiveSessionEnded(
                    reason=wire.close_reason,
                    billed_seconds=self._translator.billed_seconds,
                )
                return
            for event in self._translator.events_for_frame(payload):
                yield event
