"""Maintained per-model capability evidence (PRD section 8).

Small and deliberately conservative: an entry here blocks a request before
it reaches the provider, so entries require confirmed evidence that the
provider rejects the parameter. Wrong entries block valid calls — when
unsure, leave the model out and let the provider's own 400 answer.
"""

from __future__ import annotations

import re

__all__ = ["rejects_sampling_params"]

# Model families whose APIs reject temperature/top_p outright.
# OpenAI reasoning families are documented: o-series and gpt-5 accept only
# the default temperature and error on an explicit value.
_SAMPLING_REJECTING = (
    re.compile(r"^o\d"),        # o1, o3, o4-mini, ...
    re.compile(r"^gpt-5"),      # gpt-5 family (Responses API)
    # Anthropic deprecated temperature/top_p/top_k for newer reasoning and
    # Opus models: sending them returns 400. Claude Opus 4.7+ and the newer
    # Sonnet generations are affected; behavior is now controlled by prompting.
    re.compile(r"^claude-opus-4-[7-9]"),
    re.compile(r"^claude-opus-[5-9]"),
    re.compile(r"^claude-sonnet-[5-9]"),
)


def rejects_sampling_params(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(pattern.match(lowered) for pattern in _SAMPLING_REJECTING)
