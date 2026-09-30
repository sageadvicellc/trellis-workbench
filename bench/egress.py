"""The replay harness egress proxy and loopback gate (issue #6).

Done-when 2 of trellis-workbench#6: network is open only to the model
endpoint. This module gives two layers, and names the one it does not.

1. The loopback gate. Until an OS-level network layer lands, a replay
   takes only a loopback endpoint (the Tech Lead's ruling on #6).
   loopback_endpoint checks the parsed URL: an http or https scheme, no
   userinfo, a port from 1 to 65535, and a host that is an IP literal in
   127.0.0.0/8 or exactly ::1, in canonical form. A name is never looked
   up, so localhost, 127.0.0.1.example.com, and every other name are
   refused, and so are an IPv4-mapped address, a zone id, and short or
   decimal IPv4 forms. A refusal says that live endpoints wait for the OS
   network layer.
2. The egress proxy. EgressProxy is a small in-process HTTP proxy on
   127.0.0.1 at a random port. It tunnels CONNECT and forwards
   absolute-form plain HTTP to exactly one host and port, the model
   endpoint's. Every other target gets 403 before any connection or name
   lookup is made, and each refusal is recorded by method, host, and
   port, never by path or body. deny_all() then refuses the endpoint too,
   for the gate commands that run after the harness. The harness child
   gets HTTP_PROXY, HTTPS_PROXY, and ALL_PROXY (upper and lower case)
   pointing at it, and no NO_PROXY.

The proxy's limit, stated plainly: a proxy binds only the clients that
honor the proxy variables. A client that ignores them, or opens a raw
socket, is not blocked yet. Codex's own sandbox turns network off for
the commands the model runs, but Codex's own model client is bound only
by this proxy. An OS-level egress sandbox is the stronger layer, and it
is NOT in this change.

Tokens. For plain HTTP the proxy reads each endpoint response's JSON
usage fields (input_tokens and output_tokens, or prompt_tokens and
completion_tokens) and sums them. Through a CONNECT tunnel it sees only
ciphertext, so a run that used a tunnel has unknown tokens, as does a
response with no usage fields or a streamed body it cannot parse.
Unknown is recorded as unknown, never guessed.
"""

from __future__ import annotations

import http.client
import http.server
import ipaddress
import json
import selectors
import socket
import socketserver
import threading
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

WAIT_MESSAGE = "live endpoints wait for the OS network layer"
MAX_BODY_BYTES = 16 * 1024 * 1024
_IPV4_LOOPBACK = ipaddress.ip_network("127.0.0.0/8")
_IPV6_LOOPBACK = ipaddress.IPv6Address("::1")
# Headers that belong to one connection, never forwarded.
_HOP_BY_HOP = frozenset({
    "connection", "proxy-connection", "keep-alive", "proxy-authorization", "proxy-authenticate", "te",
    "trailer", "transfer-encoding", "upgrade", "host", "content-length",
})


class EndpointRefused(ValueError):
    """The endpoint is not a loopback address, so the replay refuses it."""


@dataclass(frozen=True)
class Endpoint:
    scheme: str
    host: str
    port: int

    @property
    def authority(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{host}:{self.port}"


def loopback_endpoint(url) -> Endpoint:
    """The endpoint a replay may use, or EndpointRefused. See the module
    docstring for the rules."""
    def refused(why: str) -> EndpointRefused:
        return EndpointRefused(f"{WAIT_MESSAGE}; a replay takes only a loopback endpoint for now: {why}")

    if not isinstance(url, str) or not url:
        raise refused("there is no endpoint URL")
    if any(ch.isspace() or not ch.isprintable() or ch == "\\" for ch in url):
        raise refused("the URL holds white space, a control character, or a backslash")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise refused("the scheme is not http or https")
    if "@" in parts.netloc:
        raise refused("the URL holds userinfo")
    host = parts.hostname
    if not host:
        raise refused("the URL has no host")
    try:
        port = parts.port
    except ValueError:
        raise refused("the port is not a number from 1 to 65535") from None
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    if not 1 <= port <= 65535:
        raise refused("the port is not a number from 1 to 65535")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise refused("the host is a name, and a name is never looked up; use 127.0.0.1 or ::1") from None
    if str(address) != host:
        raise refused("the host is not a loopback address in canonical form")
    if address.version == 4 and address in _IPV4_LOOPBACK:
        return Endpoint(parts.scheme, host, port)
    if address.version == 6 and address == _IPV6_LOOPBACK:
        return Endpoint(parts.scheme, host, port)
    raise refused("the host is not in 127.0.0.0/8 and is not ::1")


@dataclass(frozen=True)
class Refusal:
    """One refused request: method, host, port, and why. No path, no body."""

    method: str
    host: str
    port: Optional[int]
    reason: str

    def as_record(self) -> dict:
        return {"method": self.method, "host": self.host, "port": self.port, "reason": self.reason}


@dataclass(frozen=True)
class TokenUsage:
    """The tokens the endpoint reported, summed. input_tokens and
    output_tokens are None when known is False; reason says why."""

    input_tokens: Optional[int]
    output_tokens: Optional[int]
    calls: int
    known: bool
    reason: Optional[str]


def _usage_of(body: bytes):
    """(input, output) from a JSON body's usage fields, or None."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return None
    counts = []
    for names in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
        value = next((usage[name] for name in names if name in usage), None)
        if type(value) is not int or value < 0:
            return None
        counts.append(value)
    return tuple(counts)


def _split_authority(authority: str):
    """(host, port) from a CONNECT target, or None."""
    host, sep, port_text = authority.rpartition(":")
    if not sep or not port_text.isdigit():
        return None
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    port = int(port_text)
    if not host or not 1 <= port <= 65535:
        return None
    return host.lower(), port


class _Server(socketserver.ThreadingTCPServer):
    # socketserver, not http.server.HTTPServer, whose bind looks up the
    # host's name.
    daemon_threads = True
    allow_reuse_address = False


class EgressProxy:
    """An HTTP CONNECT and plain-HTTP forward proxy on 127.0.0.1 that
    allows exactly one host and port. See the module docstring."""

    def __init__(self, allowed_host: str, allowed_port: int, *, upstream_timeout_s: float = 30.0):
        self.allowed = (str(allowed_host).lower(), int(allowed_port))
        self.upstream_timeout_s = upstream_timeout_s
        self._lock = threading.Lock()
        self._refusals = []
        self._denying = False
        self._closing = threading.Event()
        self._sockets = set()
        self._server = None
        self._thread = None
        self.allowed_requests = 0
        self.tunnels = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._calls = 0
        self._unknown_reason = None

    # -- lifecycle

    def start(self) -> str:
        """Bind 127.0.0.1 at a random port, serve on a daemon thread, and
        return the proxy URL."""
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = proxy.upstream_timeout_s

            def log_message(self, *_args):
                pass

            def do_CONNECT(self):
                proxy._connect(self)

            def _forward(self):
                proxy._forward(self)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _forward

        self._server = _Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="egress-proxy", daemon=True)
        self._thread.start()
        return self.url

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def deny_all(self) -> None:
        """Refuse every target from now on, the endpoint included."""
        with self._lock:
            self._denying = True

    def stop(self) -> None:
        """Stop serving and close every open tunnel."""
        self._closing.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        with self._lock:
            sockets = list(self._sockets)
        for sock in sockets:
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(5)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.stop()

    # -- what it saw

    @property
    def refusals(self) -> tuple:
        with self._lock:
            return tuple(self._refusals)

    def usage(self) -> TokenUsage:
        with self._lock:
            reason = self._unknown_reason
            if self.tunnels:
                reason = "the endpoint was reached through a CONNECT tunnel, which the proxy cannot read"
            if reason is not None:
                return TokenUsage(None, None, self._calls, False, reason)
            return TokenUsage(self._input_tokens, self._output_tokens, self._calls, True, None)

    # -- handling

    def _refuse(self, handler, method: str, host: str, port: Optional[int], reason: str, status: int = 403) -> None:
        with self._lock:
            self._refusals.append(Refusal(method, host, port, reason))
        self._status(handler, status)

    @staticmethod
    def _status(handler, status: int) -> None:
        handler.send_response(status)
        handler.send_header("Content-Length", "0")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True

    def _allows(self, host: str, port: int) -> bool:
        with self._lock:
            return not self._denying and (host, port) == self.allowed

    def _connect(self, handler) -> None:
        target = _split_authority(handler.path)
        if target is None:
            self._refuse(handler, "CONNECT", "", None, "the CONNECT target is not host:port")
            return
        host, port = target
        if not self._allows(host, port):
            self._refuse(handler, "CONNECT", host, port, "not the model endpoint")
            return
        try:
            upstream = socket.create_connection((host, port), timeout=self.upstream_timeout_s)
        except OSError:
            self._status(handler, 502)
            return
        with self._lock:
            self.tunnels += 1
            self.allowed_requests += 1
            self._sockets.update({upstream, handler.connection})
        try:
            handler.send_response(200, "Connection established")
            handler.end_headers()
            handler.wfile.flush()
            self._pump(handler.connection, upstream)
        finally:
            handler.close_connection = True
            with self._lock:
                self._sockets.discard(upstream)
                self._sockets.discard(handler.connection)
            upstream.close()

    def _pump(self, client: socket.socket, upstream: socket.socket) -> None:
        with selectors.DefaultSelector() as selector:
            selector.register(client, selectors.EVENT_READ, upstream)
            selector.register(upstream, selectors.EVENT_READ, client)
            while not self._closing.is_set():
                for key, _ in selector.select(timeout=0.5):
                    try:
                        data = key.fileobj.recv(65536)
                        if not data:
                            return
                        key.data.sendall(data)
                    except OSError:
                        return

    def _forward(self, handler) -> None:
        method = handler.command
        target = urlsplit(handler.path)
        if handler.path.startswith("/") or not target.scheme:
            self._refuse(handler, method, "", None, "not an absolute-form proxy request")
            return
        host = (target.hostname or "").lower()
        try:
            port = target.port
        except ValueError:
            self._refuse(handler, method, host, None, "the port is not a number")
            return
        if target.scheme != "http":
            self._refuse(handler, method, host, port, "only plain http is forwarded; https uses CONNECT")
            return
        port = port or 80
        if "@" in target.netloc:
            self._refuse(handler, method, host, port, "the target holds userinfo")
            return
        if not host or not self._allows(host, port):
            self._refuse(handler, method, host, port, "not the model endpoint")
            return
        if handler.headers.get("Transfer-Encoding"):
            self._refuse(handler, method, host, port, "a chunked request body is not forwarded", status=411)
            return
        try:
            length = int(handler.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY_BYTES:
            self._refuse(handler, method, host, port, "the request body length is not valid", status=413)
            return
        body = handler.rfile.read(length) if length else None
        drop = set(_HOP_BY_HOP)
        drop.update(name.strip().lower() for name in (handler.headers.get("Connection") or "").split(","))
        headers = {name: value for name, value in handler.headers.items() if name.lower() not in drop}
        path = (target.path or "/") + (f"?{target.query}" if target.query else "")
        with self._lock:
            self.allowed_requests += 1
        conn = http.client.HTTPConnection(host, port, timeout=self.upstream_timeout_s)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            data = response.read(MAX_BODY_BYTES + 1)
            status, reason, response_headers = response.status, response.reason, response.getheaders()
        except (OSError, http.client.HTTPException):
            self._status(handler, 502)
            return
        finally:
            conn.close()
        if len(data) > MAX_BODY_BYTES:
            self._status(handler, 502)
            return
        self._count(method, data)
        handler.send_response(status, reason)
        for name, value in response_headers:
            if name.lower() not in _HOP_BY_HOP:
                handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        if method != "HEAD":
            handler.wfile.write(data)
        handler.close_connection = True

    def _count(self, method: str, data: bytes) -> None:
        if method == "HEAD":
            return  # no body, so no model call and no usage
        usage = _usage_of(data)
        with self._lock:
            self._calls += 1
            if usage is None:
                self._unknown_reason = self._unknown_reason or "a response from the endpoint carried no usage fields"
                return
            self._input_tokens += usage[0]
            self._output_tokens += usage[1]
