"""The bench task corpus (issue #5).

Section 6.1 of "Threat model and eval spec: Vines connectors and the
model-fit bench" sets the rules this module enforces:

1. Tasks come only from repositories on a committed allowlist of named
   repositories. "Practice-owned" alone is never enough, because a
   client repository can live inside the practice's organization.
2. Every repository on the client list (the merge gate's list of client
   repositories) is excluded, and the allowlist cannot override that.
   When the two lists overlap, the corpus build refuses to start.
3. A repository joins the allowlist only after the founder clears it for
   egress, that is, for sending its code to the chosen model providers.
   Each allowlist entry names the date and the link of that clearance.
   This is a founder gate: the shipped allowlist names no repository.
4. Each task is frozen at one commit, and its snapshot is an export of
   that commit: a snapshot that holds git metadata is refused.
5. The sanitizer and a secret scan run on each task snapshot, and on the
   task's own files, before it enters the corpus. A finding fails it.
   Both scan file contents and file and folder names. Anything the scan
   cannot read, such as a symlink, an unreadable file, a file too large,
   or a file that is not UTF-8 text, is a finding too.
6. A task names the gates of its own repository, which grade it.

The client list and the sanitizer's deny terms are read from files the
caller names at run time. Neither is copied into this repository. A
missing, empty, unreadable, or malformed list refuses the build, the same
way the merge gate refuses every merge on a bad list.

Deny terms and client names are matched after Unicode normalization
(NFKC, case folding, and removal of invisible format characters), so a
full-width or zero-width variant still matches. A finding and every error
name a rule, and a file and line where the path itself is not a finding,
never the text that matched.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_TASK_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_PROMPT_FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.(?:md|txt)")
_CLEARANCE_URL_RE = re.compile(r"https://github\.com/[A-Za-z0-9._/#?=-]{1,400}")
_GATE_ARG_RE = re.compile(r"[\x20-\x7e]{1,256}")
_TASK_FIELDS = frozenset({"repo", "commit", "prompt", "gates"})
_MAX_GATES = 20
_MAX_GATE_ARGS = 32
_MAX_SCAN_BYTES = 5 * 1024 * 1024
_MAX_NAMED_FINDINGS = 5

_B = r"(?<![A-Za-z0-9])"  # a key starts at a word boundary
_SECRET_PATTERNS = {
    "aws_access_key": re.compile(_B + r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "github_token": re.compile(_B + r"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"),
    "gitlab_token": re.compile(_B + r"glpat-[A-Za-z0-9_-]{20,}"),
    "anthropic_key": re.compile(_B + r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "openai_key": re.compile(_B + r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "stripe_key": re.compile(_B + r"[rs]k_live_[A-Za-z0-9]{20,}"),
    "slack_token": re.compile(_B + r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    "slack_webhook": re.compile(r"hooks\.slack\.com/services/[A-Za-z0-9/]+"),
    "google_api_key": re.compile(_B + r"AIza[0-9A-Za-z_-]{35}"),
    "npm_token": re.compile(_B + r"npm_[A-Za-z0-9]{36}"),
    "sendgrid_key": re.compile(_B + r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}"),
    "jwt": re.compile(_B + r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\."),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "url_with_password": re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s:/@]+:[^\s/@]+@"),
}
_SECRET_FILE_RE = re.compile(
    r"\.env(?:\..*)?|.*\.pem|.*\.p12|.*\.pfx|.*\.jks|.*\.key|id_rsa|id_ed25519|id_ecdsa|\.npmrc|\.netrc|\.pgpass",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


class CorpusRefused(Exception):
    """The corpus cannot be used. The message names rules, files, lines,
    and task ids, never the text that matched."""


@dataclass(frozen=True)
class Clearance:
    repo: str
    cleared_on: date
    reference: str


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str


@dataclass(frozen=True)
class Task:
    task_id: str
    repo: str
    commit: str
    prompt_path: Path
    prompt_id: str
    gates: tuple

    def prompt_text(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")


def _fold(text: str) -> str:
    """NFKC, case folding, and no invisible format characters."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _normal_repo(repo: str) -> str:
    repo = repo.lower()
    return repo[:-4] if repo.endswith(".git") else repo


def _content_lines(path: Path, what: str) -> Iterable[str]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        raise CorpusRefused(f"the {what} cannot be read as UTF-8 text") from None
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def read_allowlist(path: Path) -> dict:
    """The allowlist: one `owner/name cleared_on=YYYY-MM-DD clearance=URL`
    line per repository. An entry without its egress clearance refuses the
    build."""
    entries = {}
    for number, line in enumerate(_content_lines(path, "allowlist"), start=1):
        parts = line.split()
        fields = dict(part.split("=", 1) for part in parts[1:] if "=" in part)
        if (
            len(parts) != 3
            or not _REPO_RE.fullmatch(parts[0])
            or set(fields) != {"cleared_on", "clearance"}
            or not _CLEARANCE_URL_RE.fullmatch(fields["clearance"])
        ):
            raise CorpusRefused(f"allowlist entry {number} needs a repository, cleared_on, and a clearance link")
        try:
            cleared_on = date.fromisoformat(fields["cleared_on"])
        except ValueError:
            raise CorpusRefused(f"allowlist entry {number} has no valid cleared_on date") from None
        repo = _normal_repo(parts[0])
        if repo in entries:
            raise CorpusRefused(f"allowlist entry {number} repeats a repository")
        entries[repo] = Clearance(repo=repo, cleared_on=cleared_on, reference=fields["clearance"])
    return entries


def read_client_list(path: Path) -> frozenset:
    """The client repositories, always excluded. Missing, empty, or
    malformed refuses the build."""
    lines = list(_content_lines(path, "client list"))
    if not lines:
        raise CorpusRefused("the client list has no entries")
    for number, line in enumerate(lines, start=1):
        if not _REPO_RE.fullmatch(line):
            raise CorpusRefused(f"client list entry {number} is not an owner/name repository")
    return frozenset(_normal_repo(line) for line in lines)


def read_deny_terms(path: Path) -> tuple:
    """The sanitizer's deny terms, one per line, matched after Unicode
    folding. Missing or empty refuses the build."""
    terms = tuple(_fold(line) for line in _content_lines(path, "deny terms file"))
    if not terms:
        raise CorpusRefused("the deny terms file has no entries")
    return terms


def _client_patterns(client_repos: frozenset) -> tuple:
    """Each client as owner/name, and its name alone as a whole word when
    the name has 4 or more characters."""
    patterns = []
    for repo in sorted(client_repos):
        patterns.append(re.compile(re.escape(_fold(repo))))
        name = _fold(repo.split("/", 1)[1])
        if len(name) >= 4:
            patterns.append(re.compile(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])"))
    return tuple(patterns)


def _sanitizer_hit(folded: str, deny_terms: tuple, client_patterns: tuple) -> list:
    rules = []
    if any(term in folded for term in deny_terms):
        rules.append("sanitizer:deny_term")
    if any(pattern.search(folded) for pattern in client_patterns):
        rules.append("sanitizer:client_repository")
    return rules


def scan_snapshot(root: Path, deny_terms: tuple, client_repos: frozenset) -> list:
    """Run the secret scan and the sanitizer on every file and folder name
    under root and on every file's text. Anything the scan cannot read is
    a finding, so nothing passes unscanned."""
    root = Path(root)
    patterns = _client_patterns(client_repos)
    findings = []

    def unreadable(error):
        name = getattr(error, "filename", None)
        relative = Path(name).relative_to(root).as_posix() if name and str(name).startswith(str(root)) else "."
        findings.append(Finding(relative, 0, "sanitizer:unreadable"))

    for directory, subdirectories, files in os.walk(root, followlinks=False, onerror=unreadable):
        kept = []
        for name in sorted(subdirectories):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            for rule in _sanitizer_hit(_fold(relative), deny_terms, patterns):
                findings.append(Finding(relative, 0, rule + "_in_path"))
            if path.is_symlink():
                findings.append(Finding(relative, 0, "sanitizer:symlink"))
            elif name == ".git":
                findings.append(Finding(relative, 0, "sanitizer:git_metadata"))
            else:
                kept.append(name)
        subdirectories[:] = kept
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            for rule in _sanitizer_hit(_fold(relative), deny_terms, patterns):
                findings.append(Finding(relative, 0, rule + "_in_path"))
            if path.is_symlink():
                findings.append(Finding(relative, 0, "sanitizer:symlink"))
                continue
            if name == ".git":
                findings.append(Finding(relative, 0, "sanitizer:git_metadata"))
                continue
            if _SECRET_FILE_RE.fullmatch(name):
                findings.append(Finding(relative, 0, "secret:secret_file_name"))
            try:
                if path.stat().st_size > _MAX_SCAN_BYTES:
                    findings.append(Finding(relative, 0, "sanitizer:file_too_large"))
                    continue
                data = path.read_bytes()
            except OSError:
                findings.append(Finding(relative, 0, "sanitizer:unreadable"))
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                findings.append(Finding(relative, 0, "sanitizer:not_utf8_text"))
                continue
            # Split on newlines only, so a form feed or other line separator
            # cannot split a key in two.
            for number, line in enumerate(text.split("\n"), start=1):
                for rule, pattern in _SECRET_PATTERNS.items():
                    if pattern.search(line):
                        findings.append(Finding(relative, number, f"secret:{rule}"))
                if _EMAIL_RE.search(line):
                    findings.append(Finding(relative, number, "sanitizer:email_address"))
                for rule in _sanitizer_hit(_fold(line), deny_terms, patterns):
                    findings.append(Finding(relative, number, rule))
    return findings


def _describe(findings: list) -> str:
    """Which checks failed, and the first findings by rule, file, and
    line. A path that is itself a finding is not shown."""
    kinds = []
    if any(f.rule.startswith("secret:") for f in findings):
        kinds.append("the secret scan")
    if any(f.rule.startswith("sanitizer:") for f in findings):
        kinds.append("the sanitizer")
    hidden_paths = {f.path for f in findings if f.rule.endswith("_in_path")}
    shown = []
    for finding in findings[:_MAX_NAMED_FINDINGS]:
        where = "a path not shown" if finding.path in hidden_paths else f"{finding.path}:{finding.line}"
        shown.append(f"{finding.rule} at {where}")
    return f"failed {' and '.join(kinds)} ({len(findings)} finding(s): {', '.join(shown)})"


def _valid_gates(gates) -> bool:
    if not isinstance(gates, list) or not 1 <= len(gates) <= _MAX_GATES:
        return False
    return all(
        isinstance(gate, list)
        and 1 <= len(gate) <= _MAX_GATE_ARGS
        and all(isinstance(arg, str) and _GATE_ARG_RE.fullmatch(arg) for arg in gate)
        for gate in gates
    )


def _load_task(task_dir: Path, allowed: dict, clients: frozenset, deny_terms: tuple,
               snapshot_root: Path) -> Task:
    task_id = task_dir.name
    try:
        spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise CorpusRefused(f"task {task_id}: task.json is missing or not JSON") from None
    if not isinstance(spec, dict) or set(spec) != _TASK_FIELDS:
        raise CorpusRefused(f"task {task_id}: task.json must hold exactly repo, commit, prompt, and gates")
    repo, commit, prompt, gates = spec["repo"], spec["commit"], spec["prompt"], spec["gates"]
    if not isinstance(repo, str) or not _REPO_RE.fullmatch(repo):
        raise CorpusRefused(f"task {task_id}: repo is not an owner/name repository")
    if _normal_repo(repo) in clients:
        raise CorpusRefused(f"task {task_id}: the repository is on the client list")
    if _normal_repo(repo) not in allowed:
        raise CorpusRefused(f"task {task_id}: the repository is not on the allowlist")
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise CorpusRefused(f"task {task_id}: commit is not a full 40-hex or 64-hex commit")
    if not isinstance(prompt, str) or not _PROMPT_FILE_RE.fullmatch(prompt):
        raise CorpusRefused(f"task {task_id}: prompt must be a .md or .txt file in the task directory")
    prompt_path = task_dir / prompt
    if prompt_path.is_symlink() or not prompt_path.is_file():
        raise CorpusRefused(f"task {task_id}: the prompt file is missing")
    if not _valid_gates(gates):
        raise CorpusRefused(f"task {task_id}: gates must be 1 to {_MAX_GATES} commands, each a list of arguments")

    findings = scan_snapshot(task_dir, deny_terms, clients)
    if findings:
        raise CorpusRefused(f"task {task_id}: its own files {_describe(findings)}")
    root = Path(snapshot_root).resolve()
    snapshot = Path(snapshot_root) / repo / commit
    try:
        inside = snapshot.resolve().relative_to(root) is not None
    except ValueError:
        inside = False
    if not inside or snapshot.is_symlink() or not snapshot.is_dir():
        raise CorpusRefused(f"task {task_id}: no snapshot exists at the frozen commit")
    findings = scan_snapshot(snapshot, deny_terms, clients)
    if findings:
        raise CorpusRefused(f"task {task_id}: the snapshot {_describe(findings)}")

    return Task(
        task_id=task_id,
        repo=repo,
        commit=commit,
        prompt_path=prompt_path,
        prompt_id=f"tasks/{task_id}/{prompt}",
        gates=tuple(tuple(gate) for gate in gates),
    )


def load_corpus(*, corpus_dir: Path, allowlist_path: Path, client_list_path: Path,
                deny_terms_path: Path, snapshot_root: Path) -> list:
    """Read and check every task before a run, or raise CorpusRefused.

    The corpus directory must exist; a corpus with no tasks directory has
    no tasks. Every task must pass; one bad task refuses the whole corpus,
    and the message lists each problem found.
    """
    if not Path(corpus_dir).is_dir():
        raise CorpusRefused("the corpus directory does not exist")
    allowed = read_allowlist(allowlist_path)
    clients = read_client_list(client_list_path)
    if set(allowed) & clients:
        raise CorpusRefused("the allowlist names a repository on the client list")
    deny_terms = read_deny_terms(deny_terms_path)

    tasks_dir = Path(corpus_dir) / "tasks"
    if not tasks_dir.is_dir():
        return []
    tasks, problems = [], []
    for entry in sorted(tasks_dir.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            problems.append("an entry in tasks/ is not a task directory")
            continue
        if not _TASK_ID_RE.fullmatch(entry.name):
            problems.append("a task id is not a lowercase slug")
            continue
        try:
            tasks.append(_load_task(entry, allowed, clients, deny_terms, snapshot_root))
        except CorpusRefused as problem:
            problems.append(str(problem))
    if problems:
        raise CorpusRefused("the corpus is refused: " + "; ".join(problems))
    return tasks
