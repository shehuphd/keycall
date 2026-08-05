"""Local web viewer: dashboard, model catalog browser, playground, and
verify report over a loaded target source. Runs entirely on the standard
library — no new dependency on the base package.

A token is always required (unlike TraceAct's viewer, where auth is
opt-in): this server holds live credentials in memory and can trigger real
provider calls, a materially higher-stakes local surface than a read-only
trace viewer.
"""

from ._server import run
from .auth import Token

__all__ = ["Token", "run"]
