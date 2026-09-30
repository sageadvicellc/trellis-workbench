"""The bench replay harness (issue #6).

run_replay runs one task, one model, and one harness in a fresh worktree
and records the result. The order is fixed:

1. The loopback gate. The endpoint must be a loopback address, checked
   on the parsed URL by egress.loopback_endpoint, before anything else.
   Until an OS-level network layer lands, live endpoints are refused. A
   loopback endpoint is not proof of a local model: if a local relay
   forwards a loopback endpoint to a vendor API, that endpoint is live,
   and the operator must not point a run at such a relay.
2. Pure checks, which start no worktree, process, or proxy: the
   arguments, the gate_sandbox, the codex argv guard (every extra launch
   flag refused), and the pin. Until #9's OS layer lands, the gate runs
   no model-written code outside a sandbox, so a run with no
   gate_sandbox, or an empty one, is refused here. verify_pin runs here, before the spend guard, as pin.py
   says, so no spend is approved for inputs that differ from the pin.
   The facts it compares against are handed in as pin_facts.
3. The spend guard. run_guarded prints the estimate and waits for the
   founder's typed yes. The whole replay runs as its one request: the
   request's input_tokens and max_tokens are the run's token budget,
   summed over every model call. If the guard refuses, nothing else
   happens: no worktree, no process, no proxy, and no record file.
   The guard's request timeout and run limit bind the whole replay; when
   either passes, the harness is killed and the worktree removed.
4. A fresh worktree: `git worktree prune` on the source repository
   first, to clear a stale entry, then `git worktree add --detach
   --no-checkout` into the run's own temporary root, then the guards,
   then the checkout. HEAD must equal the frozen commit and its tree the
   task's tree, both in full. Every git call drops every GIT_ variable
   and sets GIT_ATTR_SOURCE to the empty tree, so git reads no
   .gitattributes. A git call that acts also runs with no hooks, no
   fsmonitor, and every configured filter's commands emptied: a git
   older than GIT_ATTR_SOURCE ignores the variable (Apple git 2.39.5
   does), and the emptied filters are the layer that holds there.
5. The guards (bench/guards.py, ported from trellis-crew #24) run on the
   empty worktree, again after the checkout, and again after the
   harness, before the gate: a worktree top, not home or /, no nested
   repository, and no git setting that runs a file inside it. After the
   harness, the worktree's .git gitfile must also be byte for byte the
   one git wrote, checked before any git call, so no later git call or
   gate is pointed at another repository.
6. The egress proxy (bench/egress.py) allows exactly the endpoint. The
   harness gets an allowlisted environment with the proxy variables set
   and no NO_PROXY. The proxy binds only the clients that honor it; a
   client that ignores it is not blocked yet.
7. The harness: the codex argv from guards.codex_exec_args, run from
   harness_binary in the worktree, under a timeout. Tests pass a stub
   script as harness_binary; real codex is never run by the tests.
8. The gate: the task's gate commands, run in the worktree with the proxy
   refusing every host. Each runs as gate_sandbox + its argv, because it
   runs model-written code. Every command runs; the gate passes only if
   each exits 0.
9. The record. The diff size is `git diff --numstat` against the frozen
   commit, taken before the gate, with new files counted through
   `git add --intent-to-add`. Tokens are the endpoint's usage fields as
   the proxy saw them. They are unknown through a CONNECT tunnel, and
   when the harness exits 0 with no call through the proxy ("no call
   went through the proxy"), since it may have reached a model some
   other way. Unknown tokens make the spend guard fail the run rather
   than guess. Wall time is the harness's start to exit, on the injected
   clock.

The record sits in the same file as the reproducibility pin: one JSON
document at record_dir/run-<run_id>.json holds pin.record() unchanged
under "pin", its digest under "pin_digest", and the result beside them.
An existing record is never overwritten.

The worktree and the run's temporary root are removed in a finally, on
every exit, with `git worktree remove --force` and a prune, and only
inside the run's own temporary root under work_root.

The harness and gate processes start on the spend guard's worker thread,
which blocks SIGTERM, SIGHUP, and SIGINT, and a child keeps that mask.
So every stop uses SIGKILL on the child's process group.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, TextIO

from bench import egress, guards, spend_guard
from bench.pin import PinMismatch, RunPin, verify_pin
from bench.spend_guard import Estimate, Request, Response, RunResult, RunStatus, TokenPrice

SCHEMA = 1
GIT_TIMEOUT_S = 60.0
_RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
PIN_FACTS = frozenset({"harness_head", "corpus_head", "weights_digest", "model_version", "runtime_version",
                       "client_version", "endpoint_kind"})
OPTIONAL_PIN_FACTS = frozenset({"served_model_id"})
# Options for every git call that acts: no hook and no fsmonitor runs.
_GIT_ACT = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false")


class ReplayRefused(Exception):
    """The replay cannot run, or cannot go on, as asked."""


class _Cancelled(Exception):
    """The spend guard stopped the run, so the replay stops too."""


def _is_whole(value) -> bool:
    return type(value) is int


def _is_positive(value) -> bool:
    return type(value) in (int, float) and value > 0 and value != float("inf")


@dataclass(frozen=True)
class ReplaySpend:
    """What the spend guard needs for one replay. The run's token budget
    is one guard request: input_token_budget as its input size and
    output_token_budget as its output limit, over every model call."""

    estimate: Estimate
    token_price: TokenPrice
    cap_micros: int
    input_token_budget: int
    output_token_budget: int
    request_timeout_s: float
    run_time_limit_s: float

    def __post_init__(self):
        if not _is_whole(self.input_token_budget) or self.input_token_budget < 0:
            raise ValueError("input_token_budget must be a whole number of 0 or more")
        if not _is_whole(self.output_token_budget) or self.output_token_budget < 1:
            raise ValueError("output_token_budget must be a whole number of 1 or more")


@dataclass
class ReplayResult:
    """The spend guard's result, and the record and its path. record is
    None only when the spend guard refused, so nothing ran."""

    spend: RunResult
    record: Optional[dict]
    record_path: Optional[Path]


@dataclass
class _Run:
    """What one replay saw, shared between the guard's worker thread and
    run_replay."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    done: threading.Event = field(default_factory=threading.Event)
    proc: Optional[subprocess.Popen] = None
    cancelled: bool = False
    entered: bool = False
    refused: Optional[str] = None
    error: Optional[str] = None
    head: Optional[str] = None
    harness: Optional[dict] = None
    wall_time_s: Optional[float] = None
    diff: Optional[dict] = None
    gate: Optional[dict] = None
    egress: Optional[dict] = None
    tokens: Optional[egress.TokenUsage] = None
    worktree_removed: Optional[bool] = None
    notes: list = field(default_factory=list)

    def set_proc(self, proc) -> None:
        with self.lock:
            self.proc = proc
            cancelled = self.cancelled
        if proc is not None and cancelled:
            _kill_group(proc)

    def cancel(self) -> None:
        with self.lock:
            self.cancelled = True
            proc = self.proc
        if proc is not None:
            _kill_group(proc)

    def check(self) -> None:
        with self.lock:
            if self.cancelled:
                raise _Cancelled("the spend guard stopped the run")


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's process group while the child is unreaped."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _git(cwd, args: Sequence[str], git_env: Mapping[str, str], refuse: Optional[str] = None,
         overrides: Sequence[str] = ()) -> str:
    """Run one git call that acts, with no hooks and no fsmonitor, plus
    any filter overrides."""
    try:
        done = subprocess.run(["git", "-C", str(cwd), *_GIT_ACT, *overrides, *args], env=dict(git_env),
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise ReplayRefused(f"git {args[0]} did not answer in time") from None
    if done.returncode != 0:
        raise ReplayRefused(refuse or f"git {args[0]} failed with exit code {done.returncode}")
    return done.stdout.decode("utf-8", "surrogateescape")


def _run_child(state: _Run, argv: Sequence[str], cwd: Path, env: Mapping[str, str], timeout_s: float,
               log_path: Path):
    """Run one child in its own process group, output to a log outside
    the worktree. Returns (exit code, timed out); a child past its
    timeout is killed and has no exit code."""
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(list(argv), cwd=str(cwd), env=dict(env), stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        state.set_proc(proc)
        try:
            try:
                return proc.wait(timeout=timeout_s), False
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.wait()
                return None, True
        finally:
            state.set_proc(None)


def _filter_overrides(cwd, git_env: Mapping[str, str]) -> list:
    """`-c` options that empty the clean, smudge, and process commands of
    every filter git's config defines, and mark each one not required. A
    git older than GIT_ATTR_SOURCE ignores that variable, so this is the
    layer that holds there: a .gitattributes can name a filter, but the
    filter has no command left to run. A filter name that cannot be
    written as a `-c` key refuses the run."""
    text = _git(cwd, ["config", "--list", "-z", "--includes"], git_env)
    names = set()
    for item in text.split("\0"):
        section, subsection, _ = guards._split_key(item.split("\n", 1)[0])
        if section == "filter" and subsection is not None:
            names.add(subsection)
    overrides = []
    for name in sorted(names):
        if "=" in name or _CONTROL.search(name):
            raise ReplayRefused("a filter in git's config has a name that cannot be emptied, so it could still run")
        for variable in ("clean", "smudge", "process"):
            overrides += ["-c", f"filter.{name}.{variable}="]
        overrides += ["-c", f"filter.{name}.required=false"]
    return overrides


def empty_tree(commit: str) -> str:
    """The id of git's empty tree in the commit's hash format: SHA-1 for a
    40-hex commit, SHA-256 for a 64-hex one."""
    algorithm = "sha1" if len(commit) == 40 else "sha256"
    return hashlib.new(algorithm, b"tree 0\0").hexdigest()


def _read_gitfile(worktree: Path) -> Optional[bytes]:
    """The bytes of the worktree's .git gitfile, or None when .git is not
    a regular file. A link is never followed."""
    path = worktree / ".git"
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
        with open(path, "rb") as handle:
            return handle.read(64 * 1024)
    except OSError:
        return None


def _diff_size(worktree: Path, frozen: str, git_env: Mapping[str, str], overrides: Sequence[str]) -> dict:
    """Files, lines added, and lines deleted against the frozen commit,
    new files included. A binary file counts as a file with no lines."""
    _git(worktree, ["add", "--intent-to-add", "--all"], git_env, overrides=overrides)
    text = _git(worktree, ["diff", "--numstat", "-z", "--no-renames", "--no-ext-diff", "--no-textconv", frozen],
                git_env, overrides=overrides)
    files = added = deleted = binary = 0
    for item in text.split("\0"):
        if not item:
            continue
        parts = item.split("\t", 2)
        if len(parts) != 3:
            raise ReplayRefused("git diff --numstat gave an answer that cannot be read")
        files += 1
        if parts[0] == "-" and parts[1] == "-":
            binary += 1
            continue
        if not (parts[0].isdigit() and parts[1].isdigit()):
            raise ReplayRefused("git diff --numstat gave an answer that cannot be read")
        added += int(parts[0])
        deleted += int(parts[1])
    return {"files": files, "added": added, "deleted": deleted, "binary_files": binary}


def _remove_run_root(source: Path, work_root: Path, run_root: Path, worktree: Path,
                     git_env: Mapping[str, str]) -> bool:
    """Remove the worktree and the run's temporary root, only inside
    work_root. Returns whether the run root is gone."""
    real_run = os.path.realpath(run_root)
    if os.path.dirname(real_run) != os.path.realpath(work_root):
        return False
    if os.path.lexists(worktree) and os.path.dirname(os.path.realpath(worktree)) == real_run:
        try:
            _git(source, ["worktree", "remove", "--force", str(worktree)], git_env)
        except (ReplayRefused, OSError):
            pass
        if os.path.lexists(worktree):
            shutil.rmtree(worktree, ignore_errors=True)
    try:
        _git(source, ["worktree", "prune"], git_env)
    except (ReplayRefused, OSError):
        pass
    shutil.rmtree(run_root, ignore_errors=True)
    return not os.path.lexists(run_root)


def _replay(state: _Run, *, task, source: Path, endpoint: egress.Endpoint, argv: Sequence[str],
            env: Mapping[str, str], home: str, work_root: Path, run_id: str, request: Request,
            harness_timeout_s: float, gate_timeout_s: float, clock: Callable[[], float],
            gate_sandbox: Sequence[str]) -> Response:
    # Every git call reads attributes from the empty tree, so no
    # .gitattributes, the frozen commit's or a model-written one, can
    # name a filter or driver git would run.
    attr_source = empty_tree(task.commit)
    git_env = dict(guards.without_git_vars(env), GIT_ATTR_SOURCE=attr_source)

    def check(folder) -> Optional[str]:
        return guards.check_workdir(folder, home, env, runner=lambda args, cwd, run_env, timeout_s: guards.run_git(
            args, cwd, dict(run_env, GIT_ATTR_SOURCE=attr_source), timeout_s))

    run_root = Path(tempfile.mkdtemp(prefix=f"replay-{run_id}-", dir=work_root))
    worktree = run_root / f"run-{run_id}"
    proxy = None
    try:
        state.check()
        # A stale entry left by an earlier run that could not clean up is
        # cleared first.
        _git(source, ["worktree", "prune"], git_env)
        _git(source, ["worktree", "add", "--detach", "--no-checkout", str(worktree), task.commit], git_env,
             refuse="the frozen commit cannot be added as a worktree from the source repository")
        problem = check(worktree)
        if problem is not None:
            raise ReplayRefused(problem)
        _git(worktree, ["checkout", "--force", "--detach", task.commit], git_env,
             overrides=_filter_overrides(worktree, git_env))
        head = _git(worktree, ["rev-parse", "--verify", "HEAD"], git_env).strip()
        if head != task.commit:
            raise ReplayRefused("the worktree HEAD is not the frozen commit")
        state.head = head
        tree = _git(worktree, ["rev-parse", "--verify", "HEAD^{tree}"], git_env).strip()
        if tree != task.tree:
            raise ReplayRefused("the frozen commit's tree is not the task's recorded tree")
        problem = check(worktree)
        if problem is not None:
            raise ReplayRefused(problem)
        gitfile = _read_gitfile(worktree)
        if gitfile is None:
            raise ReplayRefused("the worktree's .git is not a regular gitfile")

        state.check()
        proxy = egress.EgressProxy(endpoint.host, endpoint.port)
        child = guards.child_env(env, proxy.start())
        started = clock()
        code, timed_out = _run_child(state, argv, worktree, child, harness_timeout_s, run_root / "harness.log")
        state.wall_time_s = clock() - started
        state.harness = {"exit_code": code, "timed_out": timed_out}
        proxy.deny_all()

        state.check()
        # Before any git call: a changed gitfile would point every later
        # git call, and the gate's grading, at another repository.
        if _read_gitfile(worktree) != gitfile:
            raise ReplayRefused("after the harness ran: the worktree's .git file changed")
        problem = check(worktree)
        if problem is not None:
            raise ReplayRefused(f"after the harness ran: {problem}")
        # Read again after the harness, so a filter added meanwhile is
        # emptied too.
        state.diff = _diff_size(worktree, task.commit, git_env, _filter_overrides(worktree, git_env))
        commands = []
        for index, gate in enumerate(task.gates):
            state.check()
            # The gate runs model-written code, so it runs only under the
            # gate sandbox prefix.
            code, timed_out = _run_child(state, [*gate_sandbox, *gate], worktree, child, gate_timeout_s,
                                         run_root / f"gate-{index}.log")
            commands.append({"argv": list(gate), "exit_code": code, "timed_out": timed_out})
        passed = bool(commands) and all(command["exit_code"] == 0 for command in commands)
        state.gate = {"result": "pass" if passed else "fail", "sandbox": list(gate_sandbox), "commands": commands}
    finally:
        if proxy is not None:
            proxy.stop()
            usage = proxy.usage()
            if usage.known and usage.calls == 0 and proxy.tunnels == 0 \
                    and state.harness is not None and state.harness["exit_code"] == 0:
                # A harness that finished its task with no model call seen
                # may have reached a model some other way, so its tokens
                # are unknown, never a known 0.
                usage = egress.TokenUsage(None, None, 0, False, "no call went through the proxy")
            state.tokens = usage
            state.egress = {
                "endpoint": endpoint.authority,
                "allowed_requests": proxy.allowed_requests,
                "tunnels": proxy.tunnels,
                "refused": [refusal.as_record() for refusal in proxy.refusals],
            }
        state.worktree_removed = _remove_run_root(source, work_root, run_root, worktree, git_env)
    usage = state.tokens
    return Response(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, cost_micros=0,
                    attempts=1, max_tokens_sent=request.max_tokens)


def _tokens_record(usage: Optional[egress.TokenUsage]) -> dict:
    if usage is None:
        return {"input": None, "output": None, "model_calls": 0, "known": False,
                "reason": "the harness did not run"}
    return {"input": usage.input_tokens, "output": usage.output_tokens, "model_calls": usage.calls,
            "known": usage.known, "reason": usage.reason}


def _outcome(result: RunResult, state: _Run):
    if state.refused is not None:
        return "refused", state.refused
    if result.status is RunStatus.STOPPED_AT_CAP:
        return "stopped_at_cap", result.reason
    if result.status is RunStatus.FAILED or state.error is not None:
        return "error", state.error or result.reason
    if state.gate is None:
        return "error", "the gate did not run"
    if state.gate["result"] == "pass":
        return "pass", "every gate command exited 0"
    return "fail", "a gate command did not exit 0"


def _checked_gate_sandbox(gate_sandbox) -> list:
    """The gate sandbox prefix as a list, or ReplayRefused. It must be a
    non-empty list or tuple of non-empty strings with no control
    character, and its first word must not be a flag."""
    why = ("the gate runs model-written code, so until the OS layer (#9) lands a replay needs a gate_sandbox,"
           " an argv prefix every gate command runs under")
    if not isinstance(gate_sandbox, (list, tuple)) or not gate_sandbox:
        raise ReplayRefused(f"{why}; none was given")
    if not all(isinstance(word, str) and word and not _CONTROL.search(word) for word in gate_sandbox):
        raise ReplayRefused(f"{why}; each gate_sandbox word must be a non-empty string with no control character")
    if gate_sandbox[0].startswith("-"):
        raise ReplayRefused(f"{why}; the first gate_sandbox word must be a program, never a flag")
    return list(gate_sandbox)


def _check_arguments(*, task, source_repo, spend, work_root, record_dir, run_id, harness_binary,
                     harness_timeout_s, gate_timeout_s, cleanup_wait_s) -> None:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ReplayRefused("run_id must be a lowercase slug of 1 to 64 characters")
    commit, tree = getattr(task, "commit", None), getattr(task, "tree", None)
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise ReplayRefused("the task's frozen commit is not a full 40-hex or 64-hex commit id")
    if not isinstance(tree, str) or not _COMMIT_RE.fullmatch(tree) or len(tree) != len(commit):
        raise ReplayRefused("the task's tree is not a full tree id in the commit's hash format")
    if not getattr(task, "gates", None):
        raise ReplayRefused("the task names no gate command")
    if not isinstance(spend, ReplaySpend):
        raise ReplayRefused("a replay needs a ReplaySpend for the spend guard")
    for name, value in (("harness_timeout_s", harness_timeout_s), ("gate_timeout_s", gate_timeout_s),
                        ("cleanup_wait_s", cleanup_wait_s)):
        if not _is_positive(value):
            raise ReplayRefused(f"{name} must be a positive number")
    for name, value in (("source_repo", source_repo), ("work_root", work_root), ("record_dir", record_dir)):
        if not Path(value).is_dir():
            raise ReplayRefused(f"{name} is not a folder")
    if not isinstance(harness_binary, str) or not harness_binary or harness_binary.startswith("-") \
            or _CONTROL.search(harness_binary):
        raise ReplayRefused("harness_binary must be a path or a name, never a flag")


def run_replay(
    *,
    task,
    source_repo,
    endpoint_url: str,
    pin: RunPin,
    pin_facts: Mapping[str, object],
    spend: ReplaySpend,
    out: TextIO,
    work_root,
    record_dir,
    run_id: str,
    harness_binary: str,
    harness_flags: Sequence[str] = (),
    base_env: Optional[Mapping[str, str]] = None,
    harness_timeout_s: float = 1800.0,
    gate_timeout_s: float = 600.0,
    clock: Callable[[], float] = time.monotonic,
    cleanup_wait_s: float = 30.0,
    gate_sandbox: Optional[Sequence[str]] = None,
) -> ReplayResult:
    """Run one replay in the order the module docstring gives, and write
    its record beside the pin.

    task is a bench.corpus.Task, frozen at task.commit. source_repo is a
    local clone of task.repo; the corpus names the repository, not a
    clone, so the caller gives it. pin_facts are the verify_pin inputs
    other than the pin and the prompt text. base_env is the environment
    the git calls and the child allowlist read from; it defaults to this
    process's environment.

    Raises EndpointRefused, ReplayRefused, GuardRefused, or PinMismatch
    before the spend guard asks, with nothing started. Otherwise returns a
    ReplayResult; record is None only when the spend guard refused.
    """
    endpoint = egress.loopback_endpoint(endpoint_url)
    _check_arguments(task=task, source_repo=source_repo, spend=spend, work_root=work_root, record_dir=record_dir,
                     run_id=run_id, harness_binary=harness_binary, harness_timeout_s=harness_timeout_s,
                     gate_timeout_s=gate_timeout_s, cleanup_wait_s=cleanup_wait_s)
    sandbox = _checked_gate_sandbox(gate_sandbox)
    record_path = Path(record_dir) / f"run-{run_id}.json"
    if os.path.lexists(record_path):
        raise ReplayRefused("a record for this run id already exists, and a record is never overwritten")
    refused_flag = guards.refused_codex_flag(list(harness_flags))
    if refused_flag is not None:
        raise guards.GuardRefused(refused_flag)
    if not isinstance(pin, RunPin):
        raise ReplayRefused("a replay needs a RunPin")
    if pin.prompt_id != task.prompt_id:
        raise PinMismatch("the pin does not match what is on disk: ['prompt_id']")
    if not isinstance(pin_facts, Mapping) or not PIN_FACTS <= set(pin_facts) <= PIN_FACTS | OPTIONAL_PIN_FACTS:
        raise ReplayRefused("pin_facts must hold exactly the verify_pin inputs")
    prompt = task.prompt_text()
    verify_pin(pin, prompt_text=prompt, **dict(pin_facts))
    argv = [harness_binary, *guards.codex_exec_args(list(harness_flags), prompt)]
    problem = guards.exec_args_problem(argv[1:])
    if problem is not None:
        raise guards.GuardRefused(problem)
    env = {key: value for key, value in (os.environ if base_env is None else base_env).items()
           if isinstance(value, str)}
    home = env.get("HOME")
    if not home:
        raise ReplayRefused("the environment has no HOME, so the guards cannot check the worktree")

    source = Path(source_repo)
    work = Path(work_root)
    request = Request(input_tokens=spend.input_token_budget, max_tokens=spend.output_token_budget)
    state = _Run()

    def replay_call(req: Request) -> Response:
        with state.lock:
            if state.cancelled:
                raise _Cancelled("the spend guard stopped the run before it began")
            state.entered = True
        try:
            return _replay(state, task=task, source=source, endpoint=endpoint, argv=argv, env=env, home=home,
                           work_root=work, run_id=run_id, request=req, harness_timeout_s=harness_timeout_s,
                           gate_timeout_s=gate_timeout_s, clock=clock, gate_sandbox=sandbox)
        except ReplayRefused as refused:
            state.refused = str(refused)
            raise
        except _Cancelled as cancelled:
            state.error = str(cancelled)
            raise
        except Exception as error:  # noqa: BLE001 -- recorded by type, then the guard fails the run
            state.error = f"the replay failed with {type(error).__name__}"
            raise
        finally:
            state.done.set()

    try:
        result = spend_guard.run_guarded(
            estimate=spend.estimate,
            token_price=spend.token_price,
            cap_micros=spend.cap_micros,
            max_tokens_per_request=spend.output_token_budget,
            request_timeout_s=spend.request_timeout_s,
            run_time_limit_s=spend.run_time_limit_s,
            requests=[request],
            endpoint=replay_call,
            out=out,
            gpu=None,
            gpu_budget_seconds=None,
            gpu_call_timeout_s=None,
        )
    finally:
        # The guard has returned or raised. A replay still running on the
        # guard's worker thread is stopped and awaited for a bounded time.
        state.cancel()
        with state.lock:
            entered = state.entered
        if entered and not state.done.wait(cleanup_wait_s):
            state.notes.append("the run's cleanup had not finished when the record was written")

    if result.status is RunStatus.REFUSED:
        return ReplayResult(spend=result, record=None, record_path=None)
    outcome, reason = _outcome(result, state)
    record = {
        "schema": SCHEMA,
        "run_id": run_id,
        "task_id": task.task_id,
        "repo": task.repo,
        "frozen_commit": task.commit,
        "head": state.head,
        "pin": pin.record(),
        "pin_digest": pin.digest(),
        "outcome": outcome,
        "reason": reason,
        "gate": state.gate,
        "harness": state.harness,
        "tokens": _tokens_record(state.tokens),
        "wall_time_s": state.wall_time_s,
        "diff": state.diff,
        "egress": state.egress,
        "worktree_removed": state.worktree_removed,
        "spend": result.as_record(),
        "notes": list(state.notes),
    }
    with open(record_path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return ReplayResult(spend=result, record=record, record_path=record_path)
