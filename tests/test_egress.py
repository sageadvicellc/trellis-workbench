"""Tests for the replay harness egress proxy and loopback gate (issue #6).

Every server here listens on a loopback address at a random port. A
refused host is a reserved name (example.invalid) or a decoy listener on
another loopback port, which counts every connection it gets. Nothing
here reaches a real host.

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import http.client
import http.server
import json
import pathlib
import socket
import socketserver
import sys
import threading
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench import egress  # noqa: E402
from bench.egress import EgressProxy, EndpointRefused, loopback_endpoint  # noqa: E402

WAIT = "live endpoints wait for the OS network layer"


def ipv6_loopback_works() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
        return True
    except OSError:
        return False


class StubEndpoint:
    """A stub model endpoint on a loopback address. Every response carries
    usage fields; it records each request it gets."""

    def __init__(self, host="127.0.0.1", usage=None):
        self.requests = []
        self.usage = {"prompt_tokens": 11, "completion_tokens": 7} if usage is None else usage
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                outer.requests.append((self.command, self.path, self.rfile.read(length)))
                body = {"choices": [{"message": {"content": "42"}}]}
                if outer.usage is not False:
                    body["usage"] = outer.usage
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST

        class Server(socketserver.ThreadingTCPServer):
            address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
            daemon_threads = True

        self.host = host
        self.server = Server((host, 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Decoy:
    """A listener on another loopback port that counts every connection."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self.connections = 0
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def _accept(self):
        while not self.stopping.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            self.connections += 1
            conn.close()

    def close(self):
        self.stopping.set()
        self.thread.join(2)
        self.sock.close()


def proxy_request(proxy, method, target, body=b"", headers=None):
    """One request to the proxy, as a proxy client sends it."""
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    try:
        conn.request(method, target, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def tunnel(proxy, host, port):
    """A CONNECT through the proxy, then one GET inside the tunnel."""
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.set_tunnel(host, port)
    try:
        conn.request("GET", "/v1")
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


class LoopbackGateTests(unittest.TestCase):
    def test_a_loopback_literal_passes(self):
        cases = {
            "http://127.0.0.1:8080": ("127.0.0.1", 8080),
            "http://127.5.6.7:1/v1": ("127.5.6.7", 1),
            "http://[::1]:9000": ("::1", 9000),
            "https://127.0.0.1": ("127.0.0.1", 443),
            "http://127.0.0.1": ("127.0.0.1", 80),
        }
        for url, (host, port) in cases.items():
            with self.subTest(url=url):
                endpoint = loopback_endpoint(url)
                self.assertEqual((endpoint.host, endpoint.port), (host, port))

    def test_a_live_or_lookalike_endpoint_is_refused(self):
        for url in [
            "https://api.example.com",
            "http://127.0.0.1.example.com",
            "http://user@evil.example:80",
            "http://user@127.0.0.1:80",
            "http://127.0.0.1:80@evil.example",
            "http://[2001:db8::1]:80",
            "http://[::ffff:127.0.0.1]:80",
            "http://[0:0:0:0:0:0:0:1]:80",
            "http://[::1%25lo0]:80",
            "http://localhost:80",
            "http://localhost.example.com",
            "http://0.0.0.0:80",
            "http://10.0.0.1:80",
            "http://127.1:80",
            "http://2130706433:80",
            "http://127.0.0.1:0",
            "http://127.0.0.1:99999",
            "ftp://127.0.0.1",
            "http://127.0.0.1 :80",
            "http://127.0.0.1\\@evil.example",
            "",
            None,
        ]:
            with self.subTest(url=url):
                with self.assertRaises(EndpointRefused) as ctx:
                    loopback_endpoint(url)
                self.assertIn(WAIT, str(ctx.exception))


class ProxyTestCase(unittest.TestCase):
    def setUp(self):
        self.endpoint = StubEndpoint()
        self.addCleanup(self.endpoint.close)
        self.decoy = Decoy()
        self.addCleanup(self.decoy.close)
        self.proxy = EgressProxy("127.0.0.1", self.endpoint.port)
        self.proxy.start()
        self.addCleanup(self.proxy.stop)


class ForwardTests(ProxyTestCase):
    def test_the_proxy_listens_on_loopback_at_a_random_port(self):
        self.assertTrue(self.proxy.url.startswith("http://127.0.0.1:"))
        self.assertNotEqual(self.proxy.port, 0)

    def test_plain_http_to_the_endpoint_is_forwarded_and_its_usage_summed(self):
        for _ in range(2):
            status, body = proxy_request(self.proxy, "POST", self.endpoint.url + "/v1/chat", b'{"q": 1}',
                                         {"Content-Type": "application/json"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "42")
        self.assertEqual(len(self.endpoint.requests), 2)
        self.assertEqual(self.endpoint.requests[0][1], "/v1/chat")
        self.assertEqual(self.endpoint.requests[0][2], b'{"q": 1}')
        usage = self.proxy.usage()
        self.assertTrue(usage.known)
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.calls), (22, 14, 2))
        self.assertEqual(self.proxy.allowed_requests, 2)
        self.assertEqual(self.proxy.refusals, ())

    def test_responses_api_usage_names_are_read_too(self):
        self.endpoint.usage = {"input_tokens": 5, "output_tokens": 3}
        proxy_request(self.proxy, "POST", self.endpoint.url + "/v1/responses", b"{}")
        usage = self.proxy.usage()
        self.assertEqual((usage.input_tokens, usage.output_tokens), (5, 3))

    def test_a_response_with_no_usage_makes_the_tokens_unknown(self):
        self.endpoint.usage = False
        proxy_request(self.proxy, "POST", self.endpoint.url + "/v1/chat", b"{}")
        usage = self.proxy.usage()
        self.assertFalse(usage.known)
        self.assertIsNone(usage.input_tokens)
        self.assertIn("usage", usage.reason)

    def test_a_tunnel_to_the_endpoint_is_allowed_but_its_usage_is_unknown(self):
        status, _ = tunnel(self.proxy, "127.0.0.1", self.endpoint.port)
        self.assertEqual(status, 200)
        self.assertEqual(len(self.endpoint.requests), 1)
        usage = self.proxy.usage()
        self.assertFalse(usage.known)
        self.assertIn("tunnel", usage.reason)
        self.assertEqual(self.proxy.tunnels, 1)


class RefusalTests(ProxyTestCase):
    def assertRefused(self, method, host, port):
        self.assertIn((method, host, port), [(r.method, r.host, r.port) for r in self.proxy.refusals])

    def test_a_plain_request_to_another_host_is_refused_with_403_and_recorded(self):
        status, _ = proxy_request(self.proxy, "GET", "http://example.invalid/steal")
        self.assertEqual(status, 403)
        self.assertRefused("GET", "example.invalid", 80)

    def test_a_plain_request_to_another_loopback_port_is_refused_with_no_connection(self):
        status, _ = proxy_request(self.proxy, "POST", f"http://127.0.0.1:{self.decoy.port}/x", b"data")
        self.assertEqual(status, 403)
        self.assertRefused("POST", "127.0.0.1", self.decoy.port)
        self.assertEqual(self.decoy.connections, 0)
        self.assertEqual(self.endpoint.requests, [])

    def test_a_tunnel_to_another_host_or_port_is_refused_with_403_and_no_connection(self):
        for host, port in [("127.0.0.1", self.decoy.port), ("example.invalid", 443)]:
            with self.subTest(host=host):
                with self.assertRaises(OSError) as ctx:
                    tunnel(self.proxy, host, port)
                self.assertIn("403", str(ctx.exception))
                self.assertRefused("CONNECT", host, port)
        self.assertEqual(self.decoy.connections, 0)

    def test_https_through_a_plain_request_and_an_origin_form_request_are_refused(self):
        self.assertEqual(proxy_request(self.proxy, "GET", f"https://127.0.0.1:{self.endpoint.port}/")[0], 403)
        self.assertEqual(proxy_request(self.proxy, "GET", "/v1/chat")[0], 403)
        self.assertEqual(proxy_request(self.proxy, "GET", f"http://u:p@127.0.0.1:{self.endpoint.port}/")[0], 403)
        self.assertEqual(self.endpoint.requests, [])

    def test_deny_all_refuses_even_the_endpoint(self):
        self.proxy.deny_all()
        status, _ = proxy_request(self.proxy, "POST", self.endpoint.url + "/v1/chat", b"{}")
        self.assertEqual(status, 403)
        self.assertRefused("POST", "127.0.0.1", self.endpoint.port)
        self.assertEqual(self.endpoint.requests, [])

    def test_a_refusal_records_no_path_or_body(self):
        proxy_request(self.proxy, "POST", "http://example.invalid/secret-path?k=v", b"secret-body")
        text = json.dumps([r.as_record() for r in self.proxy.refusals])
        self.assertNotIn("secret", text)


class Ipv6Tests(unittest.TestCase):
    def test_an_ipv6_loopback_endpoint_is_forwarded(self):
        if not ipv6_loopback_works():
            self.skipTest("this host has no IPv6 loopback")
        endpoint = StubEndpoint(host="::1")
        self.addCleanup(endpoint.close)
        proxy = EgressProxy("::1", endpoint.port)
        proxy.start()
        self.addCleanup(proxy.stop)
        status, _ = proxy_request(proxy, "POST", endpoint.url + "/v1/chat", b"{}")
        self.assertEqual(status, 200)
        self.assertEqual(len(endpoint.requests), 1)


class DocumentationTests(unittest.TestCase):
    def test_the_docstring_states_the_proxy_limit(self):
        text = " ".join(egress.__doc__.split())
        self.assertIn("honor", text)
        self.assertIn("not blocked", text)
        self.assertIn("OS-level", text)


if __name__ == "__main__":
    unittest.main()
