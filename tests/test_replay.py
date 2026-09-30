"""Tests for the bench replay harness (issue #6).

Every test makes its own source repository, work root, record folder,
and temporary home inside one temporary directory, and removes them. The
model endpoint is a stub http.server on a loopback address. The harness
is a stub Python script written into the same temporary directory, run
in place of codex with the argv codex would get. No real model, codex,
or key is used, and nothing reaches a real host.

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import date
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench import egress, spend_guard  # noqa: E402
from bench.corpus import ManifestEntry, Task  # noqa: E402
from bench.egress import EndpointRefused  # noqa: E402
from bench.guards import GuardRefused  # noqa: E402
from bench.pin import PinMismatch, prompt_sha256, provider_snapshot_digest, require_pin  # noqa: E402
from bench.replay import ReplayRefused, ReplaySpend, run_replay  # noqa: E402
from bench.spend_guard import Estimate, PriceSource, RunStatus, TokenPrice  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_egress import Decoy, StubEndpoint, ipv6_loopback_works  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = "a" * 40
CORPUS = "b" * 40
PROVIDER = "example-provider"
SNAPSHOT = "example-model-20260930"
SNAPSHOT_DIGEST = provider_snapshot_digest(PROVIDER, SNAPSHOT)
PROMPT = "Write the answer to answer.txt."
WAIT = "live endpoints wait for the OS network layer"
SOURCE = PriceSource(name="stub price list", reference="tests/stub", read_on=date(2026, 9, 30))
# 1 micro-dollar per input token and 10 per output token: a run budget of
# 100 input and 90 output tokens is reserved at 1,000.
PRICE = TokenPrice(input_micros_per_token=1, output_micros_per_token=10, source=SOURCE)

CHECK_PY = """import pathlib, sys
path = pathlib.Path("answer.txt")
sys.exit(0 if path.exists() and path.read_text().strip() == "42" else 3)
"""

HARNESS_HEAD = """#!{python}
import http.client, json, os, pathlib, sys, time, urllib.error, urllib.parse, urllib.request
ENDPOINT = {endpoint!r}
REPORT = pathlib.Path({report!r})
DECOY = {decoy!r}

def report(**fields):
    REPORT.write_text(json.dumps(fields))

def ask():
    request = urllib.request.Request(ENDPOINT + "/v1/chat/completions", data=json.dumps({{"q": sys.argv[-1]}}).encode(),
                                     headers={{"Content-Type": "application/json"}})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)

"""

PASS_BODY = """
first = ask()
ask()
pathlib.Path("answer.txt").write_text(first["choices"][0]["message"]["content"] + "\\n")
pathlib.Path("version.txt").write_text("v1-edited\\n")
report(argv=sys.argv[1:], env=sorted(os.environ), version=open("version.txt").read(), cwd=os.getcwd(),
       head_version=open("version.txt").read())
"""

FAIL_BODY = """
ask()
pathlib.Path("answer.txt").write_text("41\\n")
report(argv=sys.argv[1:])
"""

BLOCKED_BODY = """
results = {}
for name, url in [("invalid", "http://example.invalid/steal"), ("decoy", f"http://127.0.0.1:{DECOY}/x")]:
    try:
        urllib.request.urlopen(url, timeout=10)
        results[name] = "open"
    except urllib.error.HTTPError as error:
        results[name] = error.code
    except Exception as error:
        results[name] = type(error).__name__
proxy = urllib.parse.urlsplit(os.environ["HTTPS_PROXY"])
conn = http.client.HTTPConnection(proxy.hostname, proxy.port, timeout=10)
conn.set_tunnel("127.0.0.1", DECOY)
try:
    conn.request("GET", "/")
    results["tunnel"] = "open"
except OSError as error:
    results["tunnel"] = str(error)
report(results=results)
"""

VERSION_BODY = """
report(version=open("version.txt").read())
"""

CRASH_BODY = """
report(started=True)
raise RuntimeError("the stub harness crashed")
"""

HANG_BODY = """
report(started=True)
time.sleep(30)
"""

REWRITE_GITFILE_BODY = """
# Point the gitfile at the source repository's own .git folder, a real
# repository, so git still sees this folder as a worktree top.
admin = pathlib.Path(pathlib.Path(".git").read_text().split("gitdir: ", 1)[1].strip())
pathlib.Path(".git").write_text("gitdir: " + str(admin.parent.parent) + "\\n")
pathlib.Path("answer.txt").write_text("42\\n")
report(started=True)
"""

TUNNEL_BODY = """
proxy = urllib.parse.urlsplit(os.environ["HTTPS_PROXY"])
endpoint = urllib.parse.urlsplit(ENDPOINT)
conn = http.client.HTTPConnection(proxy.hostname, proxy.port, timeout=10)
conn.set_tunnel(endpoint.hostname, endpoint.port)
conn.request("POST", "/v1/chat/completions", body=b"{}")
answer = json.load(conn.getresponse())
pathlib.Path("answer.txt").write_text(answer["choices"][0]["message"]["content"] + "\\n")
report(started=True)
"""

PLANT_NESTED_BODY = """
pathlib.Path("vendor/.git").mkdir(parents=True)
pathlib.Path("answer.txt").write_text("42\\n")
report(started=True)
"""


class FakeClock:
    """Returns each value once, then the last one again."""

    def __init__(self, *values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


class Fixture:
    """A source repository with two commits, a corpus task frozen at the
    first, a stub endpoint, a decoy listener, and a pin."""

    def __init__(self, endpoint_host="127.0.0.1"):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(os.path.realpath(self.tmp.name))
        self.home = self.root / "home"
        self.home.mkdir()
        self.work_root = self.root / "runs"
        self.work_root.mkdir()
        self.record_dir = self.root / "records"
        self.record_dir.mkdir()
        self.report = self.root / "harness-report.json"
        self.harness_path = self.root / "harness.py"
        self.source = self.root / "source"
        self.source.mkdir()
        self.git_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.home), "LANG": "C",
                        "GIT_CONFIG_NOSYSTEM": "1"}
        self.git("init", "-q")
        (self.source / "check.py").write_text(CHECK_PY)
        (self.source / "version.txt").write_text("v1\n")
        self.commit_all("first")
        self.frozen = self.git("rev-parse", "HEAD").strip()
        self.tree = self.git("rev-parse", "HEAD^{tree}").strip()
        (self.source / "version.txt").write_text("v2\n")
        self.commit_all("second")
        self.task = self.make_task()
        self.endpoint = StubEndpoint(host=endpoint_host)
        self.decoy = Decoy()
        # GIT_DIR, a secret, and NO_PROXY are in the caller's environment;
        # the harness must never see them, and git must ignore GIT_DIR.
        self.base_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.home), "LANG": "C",
                         "GIT_DIR": str(self.root / "nowhere"), "BENCH_SECRET": "do-not-pass", "NO_PROXY": "*",
                         "no_proxy": "*"}
        self.pin = require_pin(self.pin_fields())
        self.pin_facts = dict(harness_head=HARNESS, corpus_head=CORPUS, weights_digest=SNAPSHOT_DIGEST,
                              model_version=SNAPSHOT_DIGEST, runtime_version="0.6.3", client_version="1.51.0",
                              endpoint_kind="provider_snapshot", served_model_id=SNAPSHOT)
        self.write_harness(PASS_BODY)

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.source), *args], check=True, env=self.git_env,
                              capture_output=True).stdout.decode()

    def commit_all(self, message):
        self.git("add", "-A")
        self.git("-c", "user.name=Bench", "-c", "user.email=bench@example.invalid", "-c", "commit.gpgsign=false",
                 "commit", "-q", "-m", message)

    def make_task(self, prompt=PROMPT):
        task_dir = self.root / "corpus" / "tasks" / "answer-fix"
        task_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = task_dir / "prompt.md"
        prompt_path.write_text(prompt)
        data = prompt_path.read_bytes()
        entry = ManifestEntry("prompt.md", len(data), hashlib.sha256(data).hexdigest())
        return Task(task_id="answer-fix", repo="example-org/example-repo", commit=self.frozen, tree=self.tree,
                    prompt_path=prompt_path, prompt_id="tasks/answer-fix/prompt.md",
                    gates=(("python3", "check.py"),), task_manifest=(entry,),
                    snapshot_path=self.root / "unused", snapshot_manifest=())

    def pin_fields(self, **overrides):
        fields = {
            "model_id": SNAPSHOT, "model_version": SNAPSHOT_DIGEST, "weights_sha256": SNAPSHOT_DIGEST,
            "weights_kind": "provider_snapshot", "quantization": "none", "runtime_name": PROVIDER,
            "runtime_version": "0.6.3", "client_version": "1.51.0", "temperature": 0, "top_p": 1,
            "max_output_tokens": 90, "prompt_id": "tasks/answer-fix/prompt.md",
            "prompt_sha256": prompt_sha256(PROMPT), "seed": 42, "harness_commit": HARNESS,
            "corpus_commit": CORPUS,
        }
        fields.update(overrides)
        return fields

    def write_harness(self, body):
        head = HARNESS_HEAD.format(python=sys.executable, endpoint=self.endpoint.url, report=str(self.report),
                                   decoy=self.decoy.port)
        self.harness_path.write_text(head + body)
        self.harness_path.chmod(0o755)

    def spend(self, **overrides):
        fields = dict(estimate=Estimate(run_count=1, price_per_run_micros=5_000, source=SOURCE), token_price=PRICE,
                      cap_micros=5_000, input_token_budget=100, output_token_budget=90, request_timeout_s=60,
                      run_time_limit_s=120)
        fields.update(overrides)
        return ReplaySpend(**fields)

    def run(self, answer="yes", tty=True, **overrides):
        """Run the replay with the spend guard's terminal stubbed. Returns
        the result and the mock that stands in for the founder's typing."""
        args = dict(task=self.task, source_repo=self.source, endpoint_url=self.endpoint.url, pin=self.pin,
                    pin_facts=self.pin_facts, spend=self.spend(), out=io.StringIO(), work_root=self.work_root,
                    record_dir=self.record_dir, run_id="run-1", harness_binary=str(self.harness_path),
                    base_env=self.base_env, harness_timeout_s=30, gate_timeout_s=30,
                    clock=FakeClock(100.0, 112.5))
        args.update(overrides)
        read = mock.Mock(return_value=answer)
        with mock.patch.object(spend_guard, "_stdin_is_tty", return_value=tty), \
                mock.patch.object(spend_guard, "_out_is_tty", return_value=tty), \
                mock.patch.object(spend_guard, "_read_answer", read):
            return run_replay(**args), read

    def harness_report(self):
        return json.loads(self.report.read_text()) if self.report.exists() else None

    def worktrees(self):
        text = self.git("worktree", "list", "--porcelain")
        return [line for line in text.splitlines() if line.startswith("worktree ")]

    def close(self):
        self.endpoint.close()
        self.decoy.close()
        self.tmp.cleanup()


class ReplayTestCase(unittest.TestCase):
    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        if any(ch.isspace() for ch in sys.executable):
            self.skipTest("the stub harness needs a Python path with no white space for its #! line")
        self.fx = Fixture()
        self.addCleanup(self.fx.close)

    def assertCleanedUp(self):
        self.assertEqual(len(self.fx.worktrees()), 1, "only the source repository's own worktree remains")
        self.assertEqual(list(self.fx.work_root.iterdir()), [], "the run's temporary root is removed")

    def assertNothingRan(self):
        self.assertIsNone(self.fx.harness_report(), "the harness never ran")
        self.assertEqual(self.fx.endpoint.requests, [])
        self.assertEqual(list(self.fx.record_dir.iterdir()), [])
        self.assertCleanedUp()


class PassTests(ReplayTestCase):
    def test_a_passing_run_records_the_gate_tokens_wall_time_and_diff_beside_the_pin(self):
        result, _ = self.fx.run()
        self.assertEqual(result.spend.status, RunStatus.COMPLETED)
        record = result.record
        self.assertEqual(record["outcome"], "pass")
        self.assertEqual(record["gate"]["result"], "pass")
        self.assertEqual(record["gate"]["commands"], [{"argv": ["python3", "check.py"], "exit_code": 0,
                                                        "timed_out": False}])
        self.assertEqual(record["tokens"], {"input": 22, "output": 14, "model_calls": 2, "known": True,
                                            "reason": None})
        self.assertEqual(record["wall_time_s"], 12.5)
        self.assertEqual(record["diff"], {"files": 2, "added": 2, "deleted": 1, "binary_files": 0})
        self.assertEqual(record["harness"], {"exit_code": 0, "timed_out": False})
        self.assertEqual(len(self.fx.endpoint.requests), 2)
        self.assertEqual(record["egress"]["allowed_requests"], 2)
        self.assertEqual(record["egress"]["refused"], [])
        self.assertTrue(record["worktree_removed"])
        self.assertCleanedUp()

    def test_the_record_holds_the_pin_unchanged_and_its_digest_in_one_file(self):
        result, _ = self.fx.run()
        self.assertEqual(result.record_path, self.fx.record_dir / "run-run-1.json")
        on_disk = json.loads(result.record_path.read_text())
        self.assertEqual(on_disk, result.record)
        self.assertEqual(on_disk["pin"], self.fx.pin.record())
        canonical = json.dumps(on_disk["pin"], sort_keys=True, separators=(",", ":"))
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(), on_disk["pin_digest"])
        self.assertEqual(on_disk["pin_digest"], self.fx.pin.digest())
        self.assertEqual(on_disk["spend"]["status"], "completed")
        self.assertEqual(on_disk["task_id"], "answer-fix")
        self.assertEqual(on_disk["run_id"], "run-1")

    def test_the_harness_gets_the_codex_argv_and_an_allowlisted_environment_with_the_proxy(self):
        self.fx.run()
        seen = self.fx.harness_report()
        self.assertEqual(seen["argv"], ["exec", "--sandbox", "workspace-write", "-c",
                                        "sandbox_workspace_write.network_access=false", "-c",
                                        "sandbox_workspace_write.writable_roots=[]", "--", PROMPT])
        for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
            self.assertIn(name, seen["env"])
        for name in ["GIT_DIR", "BENCH_SECRET", "NO_PROXY", "no_proxy"]:
            self.assertNotIn(name, seen["env"])

    def test_the_spend_guard_charges_the_run_against_its_cap(self):
        result, read = self.fx.run()
        read.assert_called_once()
        self.assertEqual(result.spend.requests_sent, 1)
        self.assertEqual(result.spend.spent_micros, 1_000)

    def test_an_ipv6_loopback_endpoint_runs(self):
        if not ipv6_loopback_works():
            self.skipTest("this host has no IPv6 loopback")
        self.fx.endpoint.close()
        self.fx.endpoint = StubEndpoint(host="::1")
        self.fx.write_harness(PASS_BODY)
        result, _ = self.fx.run(endpoint_url=self.fx.endpoint.url)
        self.assertEqual(result.record["outcome"], "pass")
        self.assertEqual(result.record["egress"]["endpoint"], f"[::1]:{self.fx.endpoint.port}")


class TunnelTests(ReplayTestCase):
    def test_a_run_through_a_tunnel_has_unknown_tokens_and_the_spend_guard_fails_it(self):
        self.fx.write_harness(TUNNEL_BODY)
        result, _ = self.fx.run()
        self.assertEqual(len(self.fx.endpoint.requests), 1)
        self.assertEqual(result.record["gate"]["result"], "pass")
        self.assertFalse(result.record["tokens"]["known"])
        self.assertIsNone(result.record["tokens"]["input"])
        self.assertIn("tunnel", result.record["tokens"]["reason"])
        self.assertEqual(result.spend.status, RunStatus.FAILED)
        self.assertEqual(result.record["outcome"], "error")
        self.assertEqual(result.record["egress"]["tunnels"], 1)
        self.assertCleanedUp()


class FailTests(ReplayTestCase):
    def test_a_change_that_fails_the_gate_records_fail_with_its_exit_code(self):
        self.fx.write_harness(FAIL_BODY)
        result, _ = self.fx.run()
        record = result.record
        self.assertEqual(record["outcome"], "fail")
        self.assertEqual(record["gate"]["result"], "fail")
        self.assertEqual(record["gate"]["commands"][0]["exit_code"], 3)
        self.assertEqual(record["tokens"]["input"], 11)
        self.assertEqual(record["diff"], {"files": 1, "added": 1, "deleted": 0, "binary_files": 0})
        self.assertCleanedUp()


class BlockedNetworkTests(ReplayTestCase):
    def test_a_call_to_any_other_host_is_refused_with_403_recorded_and_never_connected(self):
        self.fx.write_harness(BLOCKED_BODY)
        result, _ = self.fx.run()
        seen = self.fx.harness_report()["results"]
        self.assertEqual(seen["invalid"], 403)
        self.assertEqual(seen["decoy"], 403)
        self.assertIn("403", seen["tunnel"])
        refused = [(r["method"], r["host"], r["port"]) for r in result.record["egress"]["refused"]]
        self.assertEqual(sorted(refused), sorted([("GET", "example.invalid", 80),
                                                   ("GET", "127.0.0.1", self.fx.decoy.port),
                                                   ("CONNECT", "127.0.0.1", self.fx.decoy.port)]))
        self.assertEqual(self.fx.decoy.connections, 0, "no connection reached the refused port")
        self.assertEqual(self.fx.endpoint.requests, [])
        self.assertEqual(result.record["outcome"], "fail")
        self.assertCleanedUp()


class LoopbackGateTests(ReplayTestCase):
    def test_a_live_endpoint_is_refused_before_anything_starts(self):
        for url in ["https://api.example.com", "http://127.0.0.1.example.com", "http://user@evil.example:80",
                    "http://[2001:db8::1]:80"]:
            with self.subTest(url=url):
                with mock.patch.object(egress.EgressProxy, "start") as start, \
                        mock.patch("subprocess.Popen") as popen, mock.patch("subprocess.run") as run:
                    with self.assertRaises(EndpointRefused) as ctx:
                        self.fx.run(endpoint_url=url)
                    start.assert_not_called()
                    popen.assert_not_called()
                    run.assert_not_called()
                self.assertIn(WAIT, str(ctx.exception))
                self.assertNothingRan()


class SpendGuardTests(ReplayTestCase):
    def test_a_refused_spend_runs_nothing_and_makes_no_worktree(self):
        for answer, tty in [("no", True), ("yes", False)]:
            with self.subTest(answer=answer, tty=tty):
                with mock.patch.object(egress.EgressProxy, "start") as start:
                    result, _ = self.fx.run(answer=answer, tty=tty)
                    start.assert_not_called()
                self.assertEqual(result.spend.status, RunStatus.REFUSED)
                self.assertIsNone(result.record)
                self.assertIsNone(result.record_path)
                self.assertNothingRan()

    def test_the_spend_guard_prompt_comes_after_the_pure_checks(self):
        with self.assertRaises(PinMismatch):
            self.fx.run(pin_facts=dict(self.fx.pin_facts, harness_head="c" * 40))
        with self.assertRaises(PinMismatch):
            self.fx.run(task=self.fx.make_task(prompt="Another prompt."))
        self.assertNothingRan()

    def test_a_run_past_the_time_limit_is_stopped_and_cleaned_up(self):
        self.fx.write_harness(HANG_BODY)
        started = time.monotonic()
        result, _ = self.fx.run(spend=self.fx.spend(run_time_limit_s=3, request_timeout_s=3), cleanup_wait_s=20)
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(result.spend.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(result.record["outcome"], "stopped_at_cap")
        self.assertTrue(result.record["worktree_removed"])
        self.assertCleanedUp()


class WorktreeTests(ReplayTestCase):
    def test_the_worktree_is_at_the_frozen_commit_not_the_source_head(self):
        self.fx.write_harness(VERSION_BODY)
        result, _ = self.fx.run()
        self.assertEqual(self.fx.harness_report()["version"], "v1\n")
        self.assertEqual(result.record["frozen_commit"], self.fx.frozen)
        self.assertEqual(result.record["head"], self.fx.frozen)
        self.assertCleanedUp()

    def test_the_worktree_is_removed_when_the_harness_crashes(self):
        self.fx.write_harness(CRASH_BODY)
        result, _ = self.fx.run()
        self.assertTrue(self.fx.harness_report()["started"])
        self.assertEqual(result.record["harness"]["exit_code"], 1)
        self.assertEqual(result.record["outcome"], "fail")
        self.assertCleanedUp()

    def test_the_worktree_is_removed_when_the_harness_cannot_start(self):
        result, _ = self.fx.run(harness_binary=str(self.fx.root / "no-such-harness"))
        self.assertEqual(result.spend.status, RunStatus.FAILED)
        self.assertEqual(result.record["outcome"], "error")
        self.assertCleanedUp()

    def test_a_harness_past_its_timeout_is_killed_and_cleaned_up(self):
        self.fx.write_harness(HANG_BODY)
        started = time.monotonic()
        result, _ = self.fx.run(harness_timeout_s=1)
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(result.record["harness"], {"exit_code": None, "timed_out": True})
        self.assertCleanedUp()

    def test_a_short_or_unknown_commit_is_refused(self):
        with self.assertRaises(ReplayRefused):
            self.fx.run(task=self._task(commit=self.fx.frozen[:12]))
        self.assertNothingRan()
        result, _ = self.fx.run(task=self._task(commit="0" * 40))
        self.assertEqual(result.record["outcome"], "refused")
        self.assertIsNone(self.fx.harness_report())
        self.assertCleanedUp()

    def test_a_tree_that_is_not_the_commits_tree_is_refused(self):
        result, _ = self.fx.run(task=self._task(tree="f" * 40))
        self.assertEqual(result.record["outcome"], "refused")
        self.assertIn("tree", result.record["reason"])
        self.assertIsNone(self.fx.harness_report())
        self.assertCleanedUp()

    def _task(self, **changes):
        fields = {name: getattr(self.fx.task, name) for name in self.fx.task.__dataclass_fields__}
        fields.update(changes)
        return Task(**fields)


class GuardRefusalTests(ReplayTestCase):
    def test_a_hooks_path_inside_the_worktree_refuses_before_the_harness(self):
        self.fx.git("config", "core.hooksPath", ".githooks")
        result, _ = self.fx.run()
        self.assertEqual(result.record["outcome"], "refused")
        self.assertIn("core.hooksPath", result.record["reason"])
        self.assertIsNone(self.fx.harness_report())
        self.assertIsNone(result.record["egress"])
        self.assertCleanedUp()

    def test_a_nested_repository_planted_by_the_harness_refuses_the_gate(self):
        self.fx.write_harness(PLANT_NESTED_BODY)
        result, _ = self.fx.run()
        self.assertEqual(result.record["outcome"], "refused")
        self.assertIn("another git repository", result.record["reason"])
        self.assertIsNone(result.record["gate"])
        self.assertCleanedUp()

    def test_a_gitfile_rewritten_by_the_harness_refuses_the_gate(self):
        self.fx.write_harness(REWRITE_GITFILE_BODY)
        result, _ = self.fx.run()
        self.assertTrue(self.fx.harness_report()["started"])
        self.assertEqual(result.record["outcome"], "refused")
        self.assertIn(".git", result.record["reason"])
        self.assertIsNone(result.record["gate"])
        self.assertCleanedUp()

    def test_a_smuggled_codex_flag_is_refused_before_the_spend_guard(self):
        for flags in (["--dangerously-bypass-approvals-and-sandbox"], ["--sandbox", "danger-full-access"],
                      ["-c", "sandbox_workspace_write.network_access=true"]):
            with self.subTest(flags=flags):
                with self.assertRaises(GuardRefused):
                    _, read = self.fx.run(harness_flags=flags)
                self.assertNothingRan()

    def test_a_harness_binary_that_looks_like_a_flag_is_refused(self):
        with self.assertRaises(ReplayRefused):
            self.fx.run(harness_binary="--yolo")
        self.assertNothingRan()


class ArgumentTests(ReplayTestCase):
    def test_an_existing_record_is_never_overwritten(self):
        (self.fx.record_dir / "run-run-1.json").write_text("{}")
        with self.assertRaises(ReplayRefused):
            self.fx.run()
        self.assertIsNone(self.fx.harness_report())
        self.assertEqual((self.fx.record_dir / "run-run-1.json").read_text(), "{}")

    def test_a_run_id_must_be_a_slug(self):
        for run_id in ["", "../x", "a/b", "A B"]:
            with self.subTest(run_id=run_id):
                with self.assertRaises(ReplayRefused):
                    self.fx.run(run_id=run_id)
        self.assertNothingRan()

    def test_pin_facts_must_be_exactly_the_verify_pin_inputs(self):
        with self.assertRaises(ReplayRefused):
            self.fx.run(pin_facts=dict(self.fx.pin_facts, extra="x"))
        with self.assertRaises(ReplayRefused):
            self.fx.run(pin_facts={k: v for k, v in self.fx.pin_facts.items() if k != "harness_head"})
        self.assertNothingRan()


class ReadmeTests(unittest.TestCase):
    def test_the_readme_describes_the_replay_harness_and_the_proxy_limit(self):
        readme = " ".join((REPO_ROOT / "README.md").read_text().split())
        self.assertIn("## Replay harness", (REPO_ROOT / "README.md").read_text())
        for part in ["spend guard", "worktree", "egress proxy", "honor", "not blocked", "OS-level"]:
            self.assertIn(part, readme)


if __name__ == "__main__":
    unittest.main()
