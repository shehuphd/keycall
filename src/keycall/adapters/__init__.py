"""Provider adapters and the protocol → adapter factory."""

from .._enums import ProviderProtocol
from .._registry import ResolvedProvider
from ._anthropic import AnthropicAdapter
from ._base import ProviderAdapter
from ._elevenlabs import ElevenLabsAdapter
from ._gemini import GeminiAdapter
from ._moonshot import MoonshotAdapter
from ._openai import OpenAIAdapter
from ._openai_compat import OpenAICompatibleAdapter
from ._perplexity import PerplexityAdapter
from ._stt import AssemblyAIAdapter, DeepgramAdapter
from ._xai import XAIAdapter

__all__ = ["ProviderAdapter", "adapter_for"]

_BY_PROTOCOL: dict[ProviderProtocol, type[ProviderAdapter]] = {
    ProviderProtocol.OPENAI: OpenAIAdapter,
    ProviderProtocol.ANTHROPIC: AnthropicAdapter,
    ProviderProtocol.GEMINI: GeminiAdapter,
    ProviderProtocol.OPENAI_COMPATIBLE: OpenAICompatibleAdapter,
    ProviderProtocol.ELEVENLABS: ElevenLabsAdapter,
    # No generic STT entry: each STT provider speaks its own dialect, and
    # resolve_provider already refuses custom targets on this protocol, so
    # the named overrides below are the only way to reach an STT adapter.
}

# A provider whose behavior diverges from its protocol's conventions gets a
# named override. Custom targets never match: they can't claim a name here.
_BY_PROVIDER: dict[str, type[ProviderAdapter]] = {
    "moonshot": MoonshotAdapter,
    "perplexity": PerplexityAdapter,
    "xai": XAIAdapter,
    "assemblyai": AssemblyAIAdapter,
    "deepgram": DeepgramAdapter,
}


def adapter_for(resolved: ResolvedProvider) -> ProviderAdapter:
    if not resolved.is_custom:
        override = _BY_PROVIDER.get(resolved.provider)
        if override is not None:
            return override(resolved)
    return _BY_PROTOCOL[resolved.protocol](resolved)
