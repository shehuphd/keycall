"""Provider adapters and the protocol → adapter factory."""

from .._enums import ProviderProtocol
from .._registry import ResolvedProvider
from ._anthropic import AnthropicAdapter
from ._base import ProviderAdapter
from ._gemini import GeminiAdapter
from ._openai import OpenAIAdapter
from ._openai_compat import OpenAICompatibleAdapter
from ._perplexity import PerplexityAdapter

__all__ = ["ProviderAdapter", "adapter_for"]

_BY_PROTOCOL: dict[ProviderProtocol, type[ProviderAdapter]] = {
    ProviderProtocol.OPENAI: OpenAIAdapter,
    ProviderProtocol.ANTHROPIC: AnthropicAdapter,
    ProviderProtocol.GEMINI: GeminiAdapter,
    ProviderProtocol.OPENAI_COMPATIBLE: OpenAICompatibleAdapter,
}

# A provider whose behavior diverges from its protocol's conventions gets a
# named override. Custom targets never match: they can't claim a name here.
_BY_PROVIDER: dict[str, type[ProviderAdapter]] = {
    "perplexity": PerplexityAdapter,
}


def adapter_for(resolved: ResolvedProvider) -> ProviderAdapter:
    if not resolved.is_custom:
        override = _BY_PROVIDER.get(resolved.provider)
        if override is not None:
            return override(resolved)
    return _BY_PROTOCOL[resolved.protocol](resolved)
