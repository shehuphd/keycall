"""The viewer's local web server. Standard library only — the viewer adds
no dependency to the base package.

Routes (all under the base "/"):
  GET  /                         the single-page app
  GET  /static/<file>            CSS and JS
  GET  /api/health               {"status":"ok","version":...,"targets":N}
  GET  /api/targets              keyless list of loaded targets
  GET  /api/models?target=&category=&refresh=   list/filter a target's models
  POST /api/source               {path} -> load a key file server-side
  POST /api/key                  {provider, key, ...} -> load one typed key
  POST /api/verify               {target, generate, attempts} -> VerifyResult
  POST /api/generate             {target, model, prompt, ...} -> InvocationResult
  POST /api/generate/stream      same body -> SSE events ending in a result or error
  POST /api/generate/image       {target, model, prompt} -> InvocationResult
  POST /api/generate/video       {target, model, prompt} -> InvocationResult
  GET  /api/realtime?target=&model=&voice=&instructions=   WebSocket upgrade;
                                  bridges the browser to a realtime session
  GET  /api/conversations        metadata for every saved Playground conversation
  GET  /api/conversations?id=    one conversation's full history and transcript
  POST /api/conversations        {id?, title, mode, target, model, history,
                                   transcript_html} -> create or overwrite one
  POST /api/conversations/clear  drop every saved conversation

Auth: a token is required on every /api/* request. Unlike TraceAct's opt-in
token, it is mandatory here: this server holds live credentials and can make
billable provider calls. It is printed once to the terminal, never written
to disk, and accepted three ways:

  X-KeyCall-Token header   scripts, curl, the test suite
  session cookie           the browser, after the handshake below
  ?token= query param      first open only, from the printed link

Opening the printed link is a handshake: the server sets an httpOnly,
SameSite=Strict cookie and redirects to the bare path. The token therefore
never reaches page script (which renders untrusted model output) and never
lands in browser history.

That cookie is why POSTs are CSRF-checked. A custom header cannot be set
cross-origin without a CORS preflight this server never answers, so header
auth was immune by construction; a cookie is not, because the browser
attaches it to whatever another site asks for. Every POST must therefore
carry Content-Type: application/json — which forces a preflight — and must
not carry a foreign Origin.

The page shell and static assets are unauthenticated: they're the same bytes
`pip install keycall` ships, and every credential-touching path is behind
/api.

Binds 127.0.0.1 by default. ThreadingHTTPServer so a slow provider call on
one request never blocks the static assets or another tab.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import webbrowser
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import __version__
from .._sources import Target
from . import _api, _realtime_bridge
from ._registry import Registry
from ._traces import TraceLog
from ._ws import WebSocketConnection, accept_key
from .auth import Token

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_CONTENT_TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}
# Scripts, styles, and fetches stay same-origin. Images additionally allow
# data: URIs so a generated picture can be shown from the bytes the page
# already holds, and media allows both blob: (a recording played back from
# memory before it is sent) and data: (a generated video, same reasoning
# as a generated picture). None of it reaches the network.
#
# The last three don't inherit from default-src and were therefore unset:
#   base-uri       a <base> tag can re-point every relative URL on the page
#   form-action    a form can post somewhere else entirely
#   frame-ancestors  nothing may embed this page, so it can't be clickjacked
# None of them are used by the viewer, so 'none' costs nothing and closes
# the gap. Relax frame-ancestors only if the viewer ever needs embedding.
#
# media-src also allows data:, the same way img-src does: a generated
# video arrives as a data: URI built from bytes the page already holds,
# the same as a generated picture, and blob: alone does not cover that.
_CSP = (
    "default-src 'self'; img-src 'self' data:; media-src 'self' blob: data:; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)

# The browser's copy of the token. httpOnly so page script can't read it:
# the viewer renders untrusted model output, and a token in reach of script
# is a token an injection could send elsewhere. SameSite=Strict so it never
# rides along on a request another site makes. Deliberately not Secure —
# this server is plain http on loopback, and Secure would stop the cookie
# being stored at all. No Max-Age, so it dies with the browser session,
# matching a token that was never persisted server-side either.
_COOKIE_NAME = "keycall_viewer_token"

# Large enough for a base64-encoded photo from the Playground's image
# picker (encoding costs about a third on top of the file size), small
# enough that a single request can't exhaust memory on a local server.
_MAX_BODY_BYTES = 8 * 1024 * 1024
_MAX_VERIFY_ATTEMPTS_DEFAULT = 8
_MAX_VERIFY_ATTEMPTS = 32
# Standing instructions for a realtime session; generous enough for a
# full system prompt, bounded so a malformed query string can't be used
# to push an unbounded string through a single request line.
_MAX_REALTIME_INSTRUCTIONS = 4000


class _Handler(BaseHTTPRequestHandler):
    server_version = "keycall-viewer"
    # The base class types this as BaseServer, which carries no registry or
    # token. This handler is only ever constructed by _Server, so naming
    # that type states the truth and lets the accessors below be checked
    # instead of silenced.
    server: _Server

    # Quieten the default per-request stderr logging.
    def log_message(self, *args: Any) -> None:
        pass

    @property
    def _registry(self) -> Registry:
        return self.server.registry

    @property
    def _token(self) -> Token:
        return self.server.token

    def _cookie_token(self) -> str | None:
        """The token from the session cookie, if the browser sent one."""
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        jar: SimpleCookie = SimpleCookie()
        try:
            jar.load(raw)
        except CookieError:
            return None
        morsel = jar.get(_COOKIE_NAME)
        return morsel.value if morsel else None

    def _authorised(self, parsed: Any) -> bool:
        # Header first: it is what a script, curl, or the test suite uses,
        # and it is immune to CSRF because a cross-origin request carrying a
        # custom header must pass a CORS preflight, which this server never
        # answers.
        header = self.headers.get("X-KeyCall-Token")
        if header is not None:
            return self._token.matches(header)
        # Then the cookie, which is how the browser authenticates once the
        # page has loaded. A cookie rides along on cross-site requests by
        # default, so everything that accepts one has to be CSRF-checked;
        # see _csrf_safe below.
        if self._token.matches(self._cookie_token()):
            return True
        # Finally the query string, which exists only so the link printed to
        # the terminal works on first open. That request is a top-level GET
        # that sets the cookie and redirects the token out of the URL.
        supplied = parse_qs(parsed.query).get("token")
        return self._token.matches(supplied[0] if supplied else None)

    def _csrf_safe(self) -> bool:
        """Whether a state-changing request may be honored.

        Only relevant for cookie-authenticated requests: the browser
        attaches the cookie to any request another site makes to this
        origin, so without a check, a page you visit while the viewer is
        open could spend your API credit or read a key file.

        Two gates, either of which is sufficient on its own:

        1. Content-Type must be JSON. A cross-origin POST can skip the CORS
           preflight only by staying a "simple request", which restricts it
           to form, plain-text, or multipart content types. Requiring JSON
           forces a preflight, and this server answers none.
        2. Origin, when the browser sends one, must be this server.

        The cookie is also SameSite=Strict, which browsers enforce
        independently. This is the belt to that pair of braces.
        """
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self._self_origins():
            return False
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        return content_type.lower() == "application/json"

    def _self_origins(self) -> set[str]:
        address = self.server.server_address
        # socketserver types the bound address loosely enough to be bytes,
        # and formatting bytes straight into a string yields "b'127.0.0.1'",
        # which would never match a browser's Origin and would quietly fail
        # the check open or shut depending on the branch. Decode it.
        raw_host = address[0]
        host = raw_host.decode() if isinstance(raw_host, (bytes, bytearray)) else str(raw_host)
        port = address[1]
        # The browser's Origin carries the host as the user typed it, so
        # accept the loopback spellings that reach this server.
        names = {host, "127.0.0.1", "localhost", "[::1]"}
        return {f"http://{name}:{port}" for name in names}

    def _provider_of(self, target_id: Any) -> str | None:
        if not isinstance(target_id, int):
            return None
        try:
            return self._registry.client(target_id).provider
        except Exception:  # noqa: BLE001, a nameless trace row beats a lost one
            return None

    def _record(
        self,
        *,
        route: str,
        method: str,
        started: float,
        body: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        events: int | None = None,
        status: str | None = None,
        detail: str | None = None,
    ) -> None:
        error = result.get("error") if isinstance(result, dict) else None
        if status is None:
            status = f"error: {error.get('code', '?')}" if isinstance(error, dict) else "ok"
        if detail is None and isinstance(error, dict):
            detail = str(error.get("message", ""))
        target = (body or {}).get("target")
        self.server.trace_log.record(
            route=route,
            method=method,
            duration_ms=(time.monotonic() - started) * 1000.0,
            status=status,
            target=target if isinstance(target, int) else None,
            provider=self._provider_of(target),
            model=(body or {}).get("model") if isinstance((body or {}).get("model"), str) else None,
            detail=detail,
            events=events,
        )

    def _traced_stream(self, route: str, body: dict[str, Any], events: Any) -> Any:
        """Pass stream events through while counting them, and write one
        trace row when the stream ends — however it ends. A browser
        navigating away closes the generator, and the row still records
        what happened up to that point."""
        started = time.monotonic()
        count = 0
        status = "ok"
        detail: str | None = None
        try:
            for event in events:
                count += 1
                if isinstance(event, dict) and isinstance(event.get("error"), dict):
                    status = f"error: {event['error'].get('code', '?')}"
                    detail = str(event["error"].get("message", ""))
                yield event
        finally:
            self._record(
                route=route,
                method="POST",
                started=started,
                body=body,
                events=count,
                status=status,
                detail=detail,
            )

    def _send_json(self, body: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        # This app makes no external requests; lock that down so a stray
        # citation URL or model string can never be fetched from the page.
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_sse(self, events: Any) -> None:
        """Relay JSON events as an SSE stream. HTTP/1.0 semantics: no
        Content-Length, the connection close delimits the stream. A browser
        navigating away mid-stream is not an error."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            for event in events:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            # Close the generator deterministically so a wrapper's trace
            # row is written now, not whenever collection happens.
            close = getattr(events, "close", None)
            if callable(close):
                close()

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
        # Without this the browser caches the page and its script
        # heuristically, so upgrading KeyCall leaves an open tab running the
        # previous version's JavaScript against the new server: the symptoms
        # are a control that does nothing or a status that never resolves,
        # and a plain reload doesn't clear it.
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", _CSP)
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return None
        if length <= 0 or length > _MAX_BODY_BYTES:
            return None
        try:
            body = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            return None
        return body if isinstance(body, dict) else None

    def _target_id(self, params: dict[str, list[str]]) -> int | None:
        raw = params.get("target")
        if not raw:
            return None
        try:
            return int(raw[0])
        except ValueError:
            return None

    def _adopt_token(self, parsed: Any) -> bool:
        """Turn the token in the opened link into a session cookie.

        The terminal prints a URL carrying the token, which is the only way
        the browser can learn it. Left in the address bar it would end up in
        history and in anything the user copies or bookmarks, so this trades
        it for an httpOnly cookie and redirects to the bare path. The page
        script never sees it at any point.
        """
        supplied = parse_qs(parsed.query).get("token")
        if not supplied or not self._token.matches(supplied[0]):
            return False
        cookie: SimpleCookie = SimpleCookie()
        cookie[_COOKIE_NAME] = self._token.value
        morsel = cookie[_COOKIE_NAME]
        morsel["path"] = "/"
        morsel["httponly"] = True
        morsel["samesite"] = "Strict"
        self.send_response(303)
        self.send_header("Location", parsed.path or "/")
        self.send_header("Set-Cookie", morsel.OutputString())
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def _handle_realtime(self, parsed: Any) -> None:
        """Upgrade to a WebSocket and bridge it to a realtime session for
        as long as the browser tab keeps it open. Token auth already ran
        in do_GET, same as every other /api/ route; this adds an Origin
        check on top because unlike a plain GET, this opens a live,
        billable connection that stays open until something closes it,
        the same reasoning _csrf_safe applies to a state-changing POST.
        """
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self._self_origins():
            self._send_json({"error": {"code": "forbidden", "message": "origin rejected"}}, 403)
            return

        if (self.headers.get("Upgrade") or "").lower() != "websocket":
            self._send_json(
                {"error": {"code": "bad_request", "message": "expected a WebSocket upgrade"}}, 400
            )
            return
        sec_key = self.headers.get("Sec-WebSocket-Key")
        if not sec_key:
            self._send_json(
                {"error": {"code": "bad_request", "message": "missing Sec-WebSocket-Key"}}, 400
            )
            return

        params = parse_qs(parsed.query)
        target_id = self._target_id(params)
        model = (params.get("model") or [None])[0]
        if target_id is None or not model:
            self._send_json(
                {"error": {"code": "bad_request", "message": "target and model required"}}, 400
            )
            return
        voice = (params.get("voice") or [None])[0]
        instructions = (params.get("instructions") or [None])[0]
        if instructions is not None and len(instructions) > _MAX_REALTIME_INSTRUCTIONS:
            self._send_json(
                {"error": {"code": "bad_request", "message": "instructions too long"}}, 400
            )
            return
        try:
            client = self._registry.client(target_id)
        except KeyError:
            self._send_json({"error": {"code": "bad_request", "message": "unknown target"}}, 400)
            return

        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept_key(sec_key))
        self.end_headers()
        # The base class must not try to read another HTTP request off
        # this socket once we start speaking WebSocket frames over it.
        self.close_connection = True

        wire = WebSocketConnection(self.rfile, self.wfile)
        started = time.monotonic()
        try:
            _realtime_bridge.run_bridge(
                client, wire, model=model, voice=voice, instructions=instructions
            )
            status = "ok"
        except Exception:  # noqa: BLE001, a nameless trace row beats a lost one
            status = "error: internal"
        finally:
            self._record(
                route="/api/realtime",
                method="GET",
                started=started,
                body={"target": target_id, "model": model},
                status=status,
            )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        # The opened link, before anything else: swap the token for a cookie
        # and get it out of the URL.
        if route == "/" and parse_qs(parsed.query).get("token") and self._adopt_token(parsed):
            return

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
        elif route == "/api/traces":
            self._send_json({"traces": self.server.trace_log.entries()})
        elif route == "/api/conversations":
            params = parse_qs(parsed.query)
            raw_id = params.get("id", [None])[0]
            if raw_id is None:
                self._send_json(_api.list_conversations(self._registry))
                return
            try:
                conversation_id = int(raw_id)
            except ValueError:
                self._send_json(
                    {"error": {"code": "bad_request", "message": "id must be an integer"}}, 400
                )
                return
            result = _api.get_conversation(self._registry, conversation_id)
            self._send_json(result, 404 if "error" in result else 200)
        elif route == "/api/realtime":
            self._handle_realtime(parsed)
        elif route == "/api/models":
            params = parse_qs(parsed.query)
            target_id = self._target_id(params)
            if target_id is None:
                self._send_json({"error": {"code": "bad_request", "message": "target required"}}, 400)
                return
            category = params.get("category", [None])[0]
            refresh = params.get("refresh", ["0"])[0] in ("1", "true")
            started = time.monotonic()
            result = _api.browse_models(
                self._registry, target_id, category=category, refresh=refresh
            )
            self._record(
                route=route,
                method="GET",
                started=started,
                body={"target": target_id},
                result=result,
            )
            self._send_json(result)
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if not self._authorised(parsed):
            self._send_json({"error": {"code": "unauthorized", "message": "token required"}}, 403)
            return

        # Every POST here spends money or reads a key file. Authorization
        # alone stopped being enough once a cookie could supply it, because
        # the browser attaches a cookie to whatever any other site asks for.
        if not self._csrf_safe():
            self._send_json(
                {
                    "error": {
                        "code": "forbidden",
                        "message": (
                            "request rejected: send Content-Type: application/json "
                            "from this origin"
                        ),
                    }
                },
                403,
            )
            return

        body = self._read_json_body()
        if body is None:
            self._send_json({"error": {"code": "bad_request", "message": "invalid JSON body"}}, 400)
            return

        if route == "/api/source":
            self._send_json(_api.add_source(self._registry, body))
            return

        if route == "/api/key":
            self._send_json(_api.add_key(self._registry, body))
            return

        if route == "/api/traces/clear":
            self.server.trace_log.clear()
            self._send_json({"cleared": True})
            return

        if route == "/api/settings":
            self._send_json(_api.set_settings(self._registry, body))
            return

        if route == "/api/conversations":
            self._send_json(_api.save_conversation(self._registry, body))
            return

        if route == "/api/conversations/clear":
            self._send_json(_api.clear_conversations(self._registry))
            return

        target_id = body.get("target")
        if not isinstance(target_id, int):
            self._send_json({"error": {"code": "bad_request", "message": "target required"}}, 400)
            return

        if route == "/api/verify":
            attempts = body.get("attempts", _MAX_VERIFY_ATTEMPTS_DEFAULT)
            if not isinstance(attempts, int) or isinstance(attempts, bool) or not (
                1 <= attempts <= _MAX_VERIFY_ATTEMPTS
            ):
                self._send_json(
                    {
                        "error": {
                            "code": "bad_request",
                            "message": f"attempts must be an integer from 1 to {_MAX_VERIFY_ATTEMPTS}",
                        }
                    },
                    400,
                )
                return
            started = time.monotonic()
            result = _api.verify_target(
                self._registry,
                target_id,
                generate=bool(body.get("generate", False)),
                attempts=attempts,
            )
            self._record(route=route, method="POST", started=started, body=body, result=result)
            self._send_json(result)
        elif route == "/api/generate":
            started = time.monotonic()
            result = _api.generate(self._registry, target_id, body)
            self._record(route=route, method="POST", started=started, body=body, result=result)
            self._send_json(result)
        elif route == "/api/generate/image":
            target_id = body.get("target")
            if not isinstance(target_id, int):
                self._send_json(
                    {"error": {"code": "bad_request", "message": "target required"}}, 400
                )
                return
            started = time.monotonic()
            result = _api.generate_image(self._registry, target_id, body)
            self._record(route=route, method="POST", started=started, body=body, result=result)
            self._send_json(result)
        elif route == "/api/generate/video":
            # Blocks this handler thread for the render duration (up to
            # VIDEO_JOB_TIMEOUT), unlike every other route: the local,
            # single-user ThreadingHTTPServer has a thread to spare.
            started = time.monotonic()
            result = _api.generate_video(self._registry, target_id, body)
            self._record(route=route, method="POST", started=started, body=body, result=result)
            self._send_json(result)
        elif route == "/api/generate/stream":
            self._send_sse(
                self._traced_stream(
                    route, body, _api.generate_stream_events(self._registry, target_id, body)
                )
            )
        else:
            self.send_error(404, "Not found")


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], registry: Registry, token: Token) -> None:
        super().__init__(address, _Handler)
        self.registry = registry
        self.token = token
        self.trace_log = TraceLog()
        self.reload_requested = False


# Dev reload passes the token and bound port to the restarted process
# through the environment, never argv: argv shows in `ps` for every user
# on the machine, the environment only for the process owner.
_RELOAD_TOKEN_ENV = "KEYCALL_VIEWER_RELOAD_TOKEN"
_RELOAD_PORT_ENV = "KEYCALL_VIEWER_RELOAD_PORT"


def _source_mtimes(root: Path) -> dict[str, float]:
    """Modification times for every file whose change needs a process
    restart: Python loads once, so an edited module is invisible to a
    running server. Static assets are excluded because the handler
    re-reads them from disk on every request already."""
    files = [*root.rglob("*.py"), root / "_catalog" / "catalog.json"]
    out: dict[str, float] = {}
    for path in files:
        try:
            out[str(path)] = path.stat().st_mtime
        except OSError:
            # A file mid-save can vanish between rglob and stat; the next
            # poll sees the settled state.
            continue
    return out


def _watch_sources(server: _Server, root: Path, interval: float = 0.5) -> None:
    baseline = _source_mtimes(root)
    while True:
        time.sleep(interval)
        current = _source_mtimes(root)
        if current != baseline:
            # Let a multi-file save settle before restarting once for all
            # of it.
            time.sleep(0.3)
            server.reload_requested = True
            server.shutdown()
            return


def run(
    targets: list[Target],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    reload: bool = False,
) -> int:
    registry = Registry(targets)
    resumed = os.environ.get(_RELOAD_TOKEN_ENV) if reload else None
    if resumed:
        token = Token(value=resumed)
        port = int(os.environ[_RELOAD_PORT_ENV])
        open_browser = False
    else:
        token = Token()

    server: _Server | None = None
    for attempt in range(20):
        try:
            server = _Server((host, port), registry, token)
            break
        except OSError:
            # On a reload restart the old process's port can linger for a
            # moment; a fresh start gets no such grace.
            if not resumed or attempt == 19:
                registry.close()
                raise
            time.sleep(0.1)
    assert server is not None
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/?token={token.value}"

    if resumed:
        print("KeyCall viewer reloaded: same address, same token")
    else:
        print(f"KeyCall viewer running for {len(targets)} target(s)")
    print(f"  {url}")
    print("  (token required; Ctrl-C to stop)")

    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()

    if reload:
        package_root = Path(__file__).resolve().parents[1]
        threading.Thread(
            target=_watch_sources, args=(server, package_root), daemon=True
        ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        server.shutdown()
        registry.close()

    if reload and server.reload_requested:
        print("source changed, restarting…")
        os.environ[_RELOAD_TOKEN_ENV] = token.value
        os.environ[_RELOAD_PORT_ENV] = str(actual_port)
        os.execv(sys.executable, [sys.executable, "-m", "keycall._cli", *sys.argv[1:]])
    return 0
