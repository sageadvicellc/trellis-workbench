"""The bench task corpus (issue #5).

Section 6.1 of "Threat model and eval spec: Vines connectors and the
model-fit bench" sets the rules this module enforces:

1. Tasks come only from repositories on a committed allowlist of named
   repositories. "Practice-owned" alone is never enough, because a
   client repository can live inside the practice's organization.
2. Every repository on the client list (the merge gate's list of client
   repositories) is excluded, and the allowlist cannot override that.
   When the two lists overlap, the corpus build refuses to start. The
   list matches by name only, so a renamed, moved, or forked client
   repository needs its own line.
3. A repository joins the allowlist only after the founder clears it for
   egress, that is, for sending its code to the chosen model providers.
   Each allowlist entry names the date and the link of that clearance.
   This is a founder gate: the shipped allowlist names no repository.
4. Each task is frozen at one commit. Its task.json records that
   commit's tree id, and the loader recomputes the git tree hash of the
   snapshot and refuses a mismatch, so the snapshot is provably that
   commit's content. An empty snapshot, and a snapshot that holds git
   metadata, are refused.
5. The sanitizer and a secret scan run on each task snapshot, on the
   task's own files, and on the task id, before it enters the corpus. A
   finding fails it. Both scan file contents and file and folder names.
   Anything the scan cannot read in full, such as a symlink, a special
   file, an unreadable file, a file too large, a file that is not UTF-8
   text, or a Git LFS pointer, is a finding too.
6. A task names the gates of its own repository, which grade it.
7. The scan records a manifest of every file it read: path, size, and
   SHA-256. A Task carries it, and check_manifest proves the bytes about
   to be sent are the bytes that were scanned. Whatever sends a task
   calls it first.

The client list and the sanitizer's deny terms are read from files the
caller names at run time. Neither is copied into this repository. A
missing, empty, unreadable, or malformed list refuses the build, the same
way the merge gate refuses every merge on a bad list.

Deny terms and client names are matched after folding: invisible format
characters removed, then NFKD, then combining marks removed, then case
folding. White space is collapsed, and each term also matches with every
separator removed on both sides, and across each pair of adjacent lines.
Folding does not map lookalike letters from other scripts, such as a
Cyrillic letter that looks Latin, so those can still evade a term.

A finding and every error name a rule, and a file and line where the path
itself is not a finding, never the text that matched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
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
_TASK_FIELDS = frozenset({"repo", "commit", "tree", "prompt", "gates"})
_MAX_GATES = 20
_MAX_GATE_ARGS = 32
_MAX_SCAN_BYTES = 5 * 1024 * 1024
_MAX_NAMED_FINDINGS = 5
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

_B = r"(?<![A-Za-z0-9])"  # a key starts at a word boundary
_SECRET_PATTERNS = {
    "aws_access_key": re.compile(_B + r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "aws_secret_key": re.compile(r"(?i)aws.{0,20}secret.{0,20}[=:]\s*['\"]?[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
    "github_token": re.compile(_B + r"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"),
    "gitlab_token": re.compile(_B + r"glpat-[A-Za-z0-9_-]{20,}"),
    "huggingface_token": re.compile(_B + r"hf_[A-Za-z0-9]{30,}"),
    "anthropic_key": re.compile(_B + r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "openai_key": re.compile(_B + r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "stripe_key": re.compile(_B + r"[rs]k_live_[A-Za-z0-9]{20,}"),
    "slack_token": re.compile(_B + r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    "slack_webhook": re.compile(r"hooks\.slack\.com/services/[A-Za-z0-9/]+"),
    "google_api_key": re.compile(_B + r"AIza[0-9A-Za-z_-]{35}"),
    "google_service_account": re.compile(r"\"type\"\s*:\s*\"service_account\""),
    "npm_token": re.compile(_B + r"npm_[A-Za-z0-9]{36}"),
    "sendgrid_key": re.compile(_B + r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}"),
    "jwt": re.compile(_B + r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\."),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "url_with_password": re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s:/@]+:[^\s/@]+@"),
    "password_assignment": re.compile(r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"\s]{6,}['\"]"),
}
_SECRET_FILE_RE = re.compile(
    r"\.env(?:\..*)?|.*\.pem|.*\.p12|.*\.pfx|.*\.jks|.*\.key|.*\.keystore|.*\.tfstate(?:\..*)?"
    r"|id_rsa|id_ed25519|id_ecdsa|\.npmrc|\.netrc|\.pgpass|\.git-credentials|credentials|\.htpasswd",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_WHITESPACE_RE = re.compile(r"\s+")
_SEPARATOR_RE = re.compile(r"[\W_]+")


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
class ManifestEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Task:
    task_id: str
    repo: str
    commit: str
    tree: str
    prompt_path: Path
    prompt_id: str
    gates: tuple
    task_manifest: tuple
    snapshot_path: Path
    snapshot_manifest: tuple

    def prompt_text(self) -> str:
        """The prompt, read again and checked against the scanned bytes."""
        check_manifest(self.prompt_path.parent, self.task_manifest)
        return self.prompt_path.read_text(encoding="utf-8")


def _strip_invisible(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _fold(text: str) -> str:
    """Invisible format characters out, NFKD, combining marks out, case
    folded, and white space collapsed to one space."""
    text = unicodedata.normalize("NFKD", _strip_invisible(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn").casefold()
    return _WHITESPACE_RE.sub(" ", text)


def _squash(folded: str) -> str:
    return _SEPARATOR_RE.sub("", folded)


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
    """The sanitizer's deny terms, one per line, matched after folding.
    Missing or empty refuses the build."""
    terms = tuple(_fold(line) for line in _content_lines(path, "deny terms file"))
    if not terms:
        raise CorpusRefused("the deny terms file has no entries")
    return terms


class _Matcher:
    """Deny terms and client names, matched on folded text: each term as
    collapsed text and with separators removed on both sides, and each
    client name as whole words with any separators, or none, between
    them."""

    def __init__(self, deny_terms: tuple, client_repos: frozenset):
        self.terms = tuple((term, _squash(term)) for term in deny_terms if _squash(term))
        patterns = []
        for repo in sorted(client_repos):
            owner, name = repo.split("/", 1)
            for words, minimum in (((owner, name), 0), ((name,), 4)):
                tokens = [t for w in words for t in _SEPARATOR_RE.split(_fold(w)) if t]
                if tokens and len("".join(tokens)) >= minimum:
                    body = r"[\W_]*".join(re.escape(t) for t in tokens)
                    patterns.append(re.compile(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])"))
        self.clients = tuple(patterns)

    def rules(self, text: str) -> list:
        folded = _fold(text)
        squashed = _squash(folded)
        rules = []
        if any(term in folded or squash in squashed for term, squash in self.terms):
            rules.append("sanitizer:deny_term")
        if any(pattern.search(folded) for pattern in self.clients):
            rules.append("sanitizer:client_repository")
        return rules


def _secret_rules(text: str) -> list:
    """Secret and email rules on the raw text and on the text with
    invisible characters removed, so a zero-width character cannot hide a
    key."""
    stripped = unicodedata.normalize("NFKC", _strip_invisible(text))
    rules = [f"secret:{rule}" for rule, pattern in _SECRET_PATTERNS.items()
             if pattern.search(text) or pattern.search(stripped)]
    if _EMAIL_RE.search(text) or _EMAIL_RE.search(stripped):
        rules.append("sanitizer:email_address")
    return rules


def _name_findings(relative: str, name: str, matcher: _Matcher) -> list:
    findings = [Finding(relative, 0, rule + "_in_path") for rule in matcher.rules(relative)]
    findings += [Finding(relative, 0, rule + "_in_path") for rule in _secret_rules(relative)]
    if _SECRET_FILE_RE.fullmatch(name):
        findings.append(Finding(relative, 0, "secret:secret_file_name"))
    return findings


def _read_regular(path: Path, cap: int):
    """Read a regular file without following a link or blocking, at most
    cap plus one bytes. Returns bytes, or a finding rule."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return "sanitizer:unreadable"
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return "sanitizer:not_a_regular_file"
        chunks, total = [], 0
        while total <= cap:
            chunk = os.read(fd, min(1024 * 1024, cap + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > cap:
            return "sanitizer:file_too_large"
        return b"".join(chunks)
    except OSError:
        return "sanitizer:unreadable"
    finally:
        os.close(fd)


def _scan_text(relative: str, text: str, matcher: _Matcher) -> list:
    findings = []
    lines = text.split("\n")  # newlines only, so a form feed cannot split a key
    for number, line in enumerate(lines, start=1):
        for rule in _secret_rules(line) + matcher.rules(line):
            findings.append(Finding(relative, number, rule))
        if number < len(lines):
            joined = line + " " + lines[number]
            for rule in matcher.rules(joined):
                if rule not in matcher.rules(line) and rule not in matcher.rules(lines[number]):
                    findings.append(Finding(relative, number, rule + "_across_lines"))
    return findings


def _scan(root: Path, matcher: _Matcher):
    """Scan every name and every file under root. Returns the findings and
    the manifest of the files read."""
    root = Path(root)
    findings, manifest = [], []

    def unreadable(error):
        name = getattr(error, "filename", None)
        relative = Path(name).relative_to(root).as_posix() if name and str(name).startswith(str(root)) else "."
        findings.append(Finding(relative, 0, "sanitizer:unreadable"))

    for directory, subdirectories, files in os.walk(root, followlinks=False, onerror=unreadable):
        kept = []
        for name in sorted(subdirectories):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            findings.extend(_name_findings(relative, name, matcher))
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
            findings.extend(_name_findings(relative, name, matcher))
            try:
                mode = os.lstat(path).st_mode
            except OSError:
                findings.append(Finding(relative, 0, "sanitizer:unreadable"))
                continue
            if stat.S_ISLNK(mode):
                findings.append(Finding(relative, 0, "sanitizer:symlink"))
                continue
            if name == ".git":
                findings.append(Finding(relative, 0, "sanitizer:git_metadata"))
                continue
            if not stat.S_ISREG(mode):
                findings.append(Finding(relative, 0, "sanitizer:not_a_regular_file"))
                continue
            data = _read_regular(path, _MAX_SCAN_BYTES)
            if isinstance(data, str):
                findings.append(Finding(relative, 0, data))
                continue
            manifest.append(ManifestEntry(relative, len(data), hashlib.sha256(data).hexdigest()))
            if data.startswith(_LFS_POINTER_PREFIX):
                findings.append(Finding(relative, 0, "sanitizer:lfs_pointer"))
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                findings.append(Finding(relative, 0, "sanitizer:not_utf8_text"))
                continue
            findings.extend(_scan_text(relative, text, matcher))
    return findings, tuple(manifest)


def scan_snapshot(root: Path, deny_terms: tuple, client_repos: frozenset) -> list:
    """Run the secret scan and the sanitizer on every file and folder name
    under root and on every file's text. Anything the scan cannot read in
    full is a finding, so nothing passes unscanned."""
    return _scan(root, _Matcher(deny_terms, client_repos))[0]


def check_manifest(root: Path, manifest: tuple) -> None:
    """Prove the files under root are exactly the scanned ones, byte for
    byte, or raise CorpusRefused. Whatever sends a task calls this
    first."""
    findings, current = _scan(root, _Matcher((), frozenset()))
    unreadable = [f for f in findings if not f.rule.startswith(("secret:", "sanitizer:email"))]
    if unreadable or current != tuple(manifest):
        raise CorpusRefused("the files changed after they were scanned")


def git_tree_hash(root: Path, algorithm: str = "sha1") -> str:
    """The git tree id of the regular files under root, computed as git
    computes it: blobs, then trees with entries sorted by name, a folder
    compared as its name plus a slash. A symlink or special file raises,
    because the scan refuses them."""
    def object_id(kind: bytes, body: bytes) -> bytes:
        return hashlib.new(algorithm, kind + b" " + str(len(body)).encode() + b"\0" + body).digest()

    def tree(directory: Path) -> bytes:
        entries = []
        for entry in os.scandir(directory):
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                if entry.name == ".git":
                    raise CorpusRefused("the snapshot holds git metadata")
                entries.append((entry.name + "/", b"40000", entry.name, tree(Path(entry.path))))
            elif stat.S_ISREG(mode):
                data = Path(entry.path).read_bytes()
                file_mode = b"100755" if mode & 0o111 else b"100644"
                entries.append((entry.name, file_mode, entry.name, object_id(b"blob", data)))
            else:
                raise CorpusRefused("the snapshot holds a file git cannot hash")
        body = b"".join(
            file_mode + b" " + name.encode("utf-8") + b"\0" + digest
            for _key, file_mode, name, digest in sorted(entries, key=lambda item: item[0].encode("utf-8"))
        )
        return object_id(b"tree", body)

    return tree(Path(root)).hex()


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


def _load_task(task_dir: Path, allowed: dict, clients: frozenset, matcher: _Matcher,
               snapshot_root: Path) -> Task:
    task_id = task_dir.name
    # The task's own folder is scanned before task.json is parsed.
    findings, task_manifest = _scan(task_dir, matcher)
    if findings:
        raise CorpusRefused(f"task {task_id}: its own files {_describe(findings)}")
    spec_bytes = _read_regular(task_dir / "task.json", _MAX_SCAN_BYTES)
    try:
        if isinstance(spec_bytes, str):
            raise ValueError
        spec = json.loads(spec_bytes.decode("utf-8"))
    except ValueError:
        raise CorpusRefused(f"task {task_id}: task.json is missing or not JSON") from None
    if not isinstance(spec, dict) or set(spec) != _TASK_FIELDS:
        raise CorpusRefused(f"task {task_id}: task.json must hold exactly repo, commit, tree, prompt, and gates")
    repo, commit, tree, prompt, gates = (spec[k] for k in ("repo", "commit", "tree", "prompt", "gates"))
    if not isinstance(repo, str) or not _REPO_RE.fullmatch(repo):
        raise CorpusRefused(f"task {task_id}: repo is not an owner/name repository")
    if _normal_repo(repo) in clients:
        raise CorpusRefused(f"task {task_id}: the repository is on the client list")
    if _normal_repo(repo) not in allowed:
        raise CorpusRefused(f"task {task_id}: the repository is not on the allowlist")
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise CorpusRefused(f"task {task_id}: commit is not a full 40-hex or 64-hex commit")
    if not isinstance(tree, str) or not _COMMIT_RE.fullmatch(tree) or len(tree) != len(commit):
        raise CorpusRefused(f"task {task_id}: tree is not a full tree id in the commit's hash format")
    if not isinstance(prompt, str) or not _PROMPT_FILE_RE.fullmatch(prompt):
        raise CorpusRefused(f"task {task_id}: prompt must be a .md or .txt file in the task directory")
    prompt_path = task_dir / prompt
    if prompt not in {entry.path for entry in task_manifest}:
        raise CorpusRefused(f"task {task_id}: the prompt file is missing")
    if not _valid_gates(gates):
        raise CorpusRefused(f"task {task_id}: gates must be 1 to {_MAX_GATES} commands, each a list of arguments")

    root = Path(snapshot_root).resolve()
    snapshot = Path(snapshot_root) / repo / commit
    try:
        inside = snapshot.resolve().relative_to(root) is not None
    except ValueError:
        inside = False
    if not inside or snapshot.is_symlink() or not snapshot.is_dir():
        raise CorpusRefused(f"task {task_id}: no snapshot exists at the frozen commit")
    findings, snapshot_manifest = _scan(snapshot, matcher)
    if findings:
        raise CorpusRefused(f"task {task_id}: the snapshot {_describe(findings)}")
    if not snapshot_manifest:
        raise CorpusRefused(f"task {task_id}: the snapshot is empty")
    algorithm = "sha1" if len(tree) == 40 else "sha256"
    if git_tree_hash(snapshot, algorithm) != tree:
        raise CorpusRefused(f"task {task_id}: the snapshot is not the recorded tree of that commit")

    return Task(
        task_id=task_id,
        repo=repo,
        commit=commit,
        tree=tree,
        prompt_path=prompt_path,
        prompt_id=f"tasks/{task_id}/{prompt}",
        gates=tuple(tuple(gate) for gate in gates),
        task_manifest=task_manifest,
        snapshot_path=snapshot,
        snapshot_manifest=snapshot_manifest,
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
    matcher = _Matcher(read_deny_terms(deny_terms_path), clients)

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
        if matcher.rules(entry.name) or _secret_rules(entry.name):
            problems.append("a task id failed the sanitizer, so it is not shown")
            continue
        try:
            tasks.append(_load_task(entry, allowed, clients, matcher, snapshot_root))
        except CorpusRefused as problem:
            problems.append(str(problem))
    if problems:
        raise CorpusRefused("the corpus is refused: " + "; ".join(problems))
    return tasks
