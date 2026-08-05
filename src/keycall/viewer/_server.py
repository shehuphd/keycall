"""The viewer's local web server. Standard library only — the viewer adds
no dependency to the base package.

Routes (all under the base "/"):
  GET  /                         the single-page app
  GET  /static/<file>            CSS and JS
  GET  /api/health               {"status":"ok","version":...,"targets":N}
  GET  /api/targets              keyless list of loaded targets
  GET  /api/models?target=&category=&refresh=   list/filter a target's models
  POST /api/verify               {target, generate, attempts} -> VerifyResult
  POST /api/generate             {target, model, prompt, ...} -> InvocationResult

Auth: a token is required on every /api/* request (X-KeyCall-Token header or
?token= query param). Unlike TraceAct's opt-in token, it is mandatory here:
this server holds live credentials and can make real provider calls. The
page shell and static assets are unauthenticated — they're the same bytes
`pip install keycall` ships, and every credential-touching path is behind
/api. The token is printed once to the terminal and embedded in the opened
URL; it is never written to disk.

Binds 127.0.0.1 by default. ThreadingHTTPServer so a slow provider call on
one request never blocks the static assets or another tab.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import __version__
from .._sources import Target
from . import _api
from ._registry import Registry
from .auth import Token

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_CONTENT_TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}
_MAX_BODY_BYTES = 64 * 1024


class _Handler(BaseHTTPRequestHandler):
    server_version = "keycall-viewer"

    # Quieten the default per-request stderr logging.
    def log_message(self, *args: Any) -> None:
        pass

    @property
    def _registry(self) -> Registry:
        return self.server.registry  # type: ignore[attr-defined]

    @property
    def _token(self) -> Token:
        return self.server.token  # type: ignore[attr-defined]

    def _authorised(self, parsed: Any) -> bool:
        header = self.headers.get("X-KeyCall-Token")
        if header is not None:
            return self._token.matches(header)
        supplied = parse_qs(parsed.query).get("token")
        return self._token.matches(supplied[0] if supplied else None)

    def _send_json(self, body: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        # This app makes no external requests; lock that down so a stray
        # citation URL or model string can never be fetched from the page.
        self.send_header("Content-Security-Policy", "default-src 'self'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_static(self, name: str) -> None:
        safe = os.path.normpath(name).lstrip(os.sep)
        path = os.path.join(_STATIC_DIR, safe)
        if not os.path.abspath(path).startswith(os.path.abspath(_STATIC_DIR)) or not os.path.isfile(
            path
        ):
            self.send_error(404, "Not found")
            return
        ext = os.path.splitext(path)[1]
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", _CONTENT_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Security-Policy", "default-src 'self'")
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > _MAX_BODY_BYTES:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            return None

    def _target_id(self, params: dict[str, list[str]]) -> int | None:
        raw = params.get("target")
        if not raw:
            return None
        try:
            return int(raw[0])
        except ValueError:
            return None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route.startswith("/api/") and not self._authorised(parsed):
            self._send_json({"error": {"code": "unauthorized", "message": "token required"}}, 403)
            return

        if route == "/":
            self._send_static("index.html")
        elif route.startswith("/static/"):
            self._send_static(route[len("/static/") :])
        elif route == "/api/health":
            self._send_json(
                {"status": "ok", "version": __version__, "targets": len(self._registry.views())}
            )
        elif route == "/api/targets":
            self._send_json(_api.list_targets(self._registry))
        elif route == "/api/models":
            params = parse_qs(parsed.query)
            target_id = self._target_id(params)
            if target_id is None:
                self._send_json({"error": {"code": "bad_request", "message": "target required"}}, 400)
                return
            category = params.get("category", [None])[0]
            refresh = params.get("refresh", ["0"])[0] in ("1", "true")
            self._send_json(
                _api.browse_models(
                    self._registry, target_id, category=category, refresh=refresh
                )
            )
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if not self._authorised(parsed):
            self._send_json({"error": {"code": "unauthorized", "message": "token required"}}, 403)
            return

        body = self._read_json_body()
        if body is None:
            self._send_json({"error": {"code": "bad_request", "message": "invalid JSON body"}}, 400)
            return
        target_id = body.get("target")
        if not isinstance(target_id, int):
            self._send_json({"error": {"code": "bad_request", "message": "target required"}}, 400)
            return

        if route == "/api/verify":
            self._send_json(
                _api.verify_target(
                    self._registry,
                    target_id,
                    generate=bool(body.get("generate", False)),
                    attempts=int(body.get("attempts", 8)),
                )
            )
        elif route == "/api/generate":
            self._send_json(_api.generate(self._registry, target_id, body))
        else:
            self.send_error(404, "Not found")


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], registry: Registry, token: Token) -> None:
        super().__init__(address, _Handler)
        self.registry = registry
        self.token = token


def run(
    targets: list[Target],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> int:
    registry = Registry(targets)
    token = Token()
    server = _Server((host, port), registry, token)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/?token={token.value}"

    print(f"KeyCall viewer running for {len(targets)} target(s)")
    print(f"  {url}")
    print("  (token required; Ctrl-C to stop)")

    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        server.shutdown()
        registry.close()
    return 0
