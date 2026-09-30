"""The replay harness guards (issue #6).

A Python port of the #24-class guards in sageadvicellc/trellis-crew:
src/adapters/codex-guard.ts and src/adapters/codex-args.ts, at commit
8aae572101f502884b392dcaef071f915dafe9bb (branch work/23-up-harness,
"fix: up --harness review round 5 (#24)"). What each part ports:

- workdir_problem ports workdirProblem: the folder must be the top of a
  git worktree, compared after realpath, and never / or the home folder.
- nested_repo_problem ports nestedRepoProblem: no other repository, a
  .git folder or a gitfile, at depth 1 to 3. A link is never followed,
  and a folder that cannot be read refuses.
- config_problem, parse_config_list, and command_problem port
  configProblem, parseConfigList, and commandProblem: from
  `git config --list --show-origin --includes -z`, no config file or
  include target inside the worktree outside its own .git folder, no
  core.hooksPath inside it, and no command key whose value names a path
  inside it or an interpreter with anything but an absolute outside path
  after it.
- without_git_vars ports withoutGitVars: every GIT_ variable is dropped
  for the check's git calls, so git sees what plain git sees.
- CHILD_ENV and child_env port CODEX_CHILD_ENV and codexChildEnv, with
  three changes: NO_PROXY and OPENAI_API_KEY are not passed, and the
  proxy variables are set to the bench's egress proxy, never passed
  through.
- SANDBOX_FIXED, refused_codex_flag, codex_exec_args, and
  exec_args_problem port the same names in codex-args.ts: workspace-write,
  network access off, an empty writable-roots list (the bench has no
  mailbox), `--` before the prompt, and every launch flag refused, as no
  Codex launch flag is verified.

The judges are pure; git runs only through an injected runner. Every
refusal is a reason string; a check returns None when the folder passes.
This is not a shell parser: a command found on PATH, or a path built at
run time, is not seen.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple, Optional, Sequence

GIT_TIMEOUT_S = 10.0
GIT_TOP = ("rev-parse", "--show-toplevel")
GIT_CONFIG = ("config", "--list", "--show-origin", "--includes", "-z")
WORKDIR_RULE = "A replay harness can write its working folder, so the bench runs one only at the top of a git worktree."
NESTED_DEPTH = 3


class GuardRefused(ValueError):
    """A guard refused a folder or an argument list."""


class GitAnswer(NamedTuple):
    """One git answer: the exit code, or None when git could not run, its
    output, and the first line of standard error or why git could not
    run."""

    code: Optional[int]
    stdout: str
    reason: str


GitRunner = Callable[[Sequence[str], str, Mapping[str, str], float], GitAnswer]


def _printable(text: str) -> str:
    """The text with every character that could hide text on a terminal
    shown as an escape."""
    return "".join(ch if ch.isprintable() else f"\\u{ord(ch):04x}" for ch in str(text))


def without_git_vars(env: Mapping[str, str]) -> dict:
    """The environment with every GIT_ variable removed."""
    return {key: value for key, value in env.items() if value is not None and not key.startswith("GIT_")}


def _first_line(text: str) -> str:
    lines = text.strip().split("\n")
    return lines[0] if lines else ""


def run_git(args: Sequence[str], cwd: str, env: Mapping[str, str], timeout_s: float) -> GitAnswer:
    """Run git with an argument list and no shell. Any failure to run is
    an answer with code None, so the check fails closed."""
    try:
        done = subprocess.run(["git", *args], cwd=cwd, env=dict(env), stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=timeout_s)
    except FileNotFoundError:
        return GitAnswer(None, "", "git is not on PATH")
    except subprocess.TimeoutExpired:
        return GitAnswer(None, "", "git did not answer in time")
    except OSError as error:
        return GitAnswer(None, "", type(error).__name__)
    stdout = done.stdout.decode("utf-8", "surrogateescape")
    return GitAnswer(done.returncode, stdout, _first_line(done.stderr.decode("utf-8", "replace")))


def _real(path: str) -> Optional[str]:
    try:
        if not os.path.exists(path):
            return None
        return os.path.realpath(path)
    except OSError:
        return None


def real_of_existing(path: str) -> str:
    """The realpath of a path whose last parts may not exist yet: the
    realpath of its nearest existing parent, with the rest joined on."""
    rest = []
    current = os.path.abspath(path)
    while True:
        found = _real(current)
        if found is not None:
            return os.path.join(found, *reversed(rest))
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(path)
        rest.append(os.path.basename(current))
        current = parent


def _within(child: str, parent: str) -> bool:
    return child.startswith(parent if parent.endswith(os.sep) else parent + os.sep)


@dataclass(frozen=True)
class ConfigEntry:
    origin: str
    key: str
    # None for a key with no value, which git reads as true.
    value: Optional[str]


def parse_config_list(stdout: str) -> Optional[list]:
    """Read `git config --list --show-origin -z` output: the origin, a
    NUL, the key, then a newline and the value when there is one, and a
    NUL. None when the output does not have that shape."""
    if stdout == "":
        return []
    if not stdout.endswith("\0"):
        return None
    tokens = stdout[:-1].split("\0")
    if len(tokens) % 2 != 0:
        return None
    entries = []
    for i in range(0, len(tokens), 2):
        pair = tokens[i + 1]
        key, newline, value = pair.partition("\n")
        entries.append(ConfigEntry(origin=tokens[i], key=key, value=value if newline else None))
    return entries


def resolve_config_path(value: str, base: str, home: str) -> Optional[str]:
    """Resolve a path from git's config as git does: absolute as it is, a
    leading ~/ against the home folder, any other against base. None for
    a ~user form, which cannot be resolved here."""
    if value == "~" or value.startswith("~/"):
        return real_of_existing(os.path.join(home, value[2:]))
    if value.startswith("~"):
        return None
    return real_of_existing(value if os.path.isabs(value) else os.path.join(base, value))


_GIT_BOOLEAN = re.compile(r"(true|false|yes|no|on|off|1|0)", re.IGNORECASE)


@dataclass(frozen=True)
class KeyPattern:
    """A config key: the section, the subsection ("none", "any", or
    "either"), and the variable ("*" for any). With bang, the value counts
    only when it starts with "!"."""

    section: str
    subsection: str
    variable: str
    bang: bool = False


# The keys whose value is a command git runs, often through a shell,
# with the worktree top as its folder. Ported from COMMAND_KEYS.
COMMAND_KEYS = tuple(KeyPattern(*spec) for spec in (
    ("core", "none", "fsmonitor"), ("core", "none", "sshcommand"), ("core", "none", "editor"),
    ("core", "none", "pager"), ("core", "none", "askpass"), ("core", "none", "gitproxy"),
    ("core", "none", "alternaterefscommand"), ("sequence", "none", "editor"),
    ("alias", "either", "*", True), ("pager", "none", "*"),
    ("filter", "any", "clean"), ("filter", "any", "smudge"), ("filter", "any", "process"),
    ("diff", "none", "external"), ("diff", "any", "textconv"), ("diff", "any", "command"),
    ("merge", "any", "driver"), ("difftool", "any", "cmd"), ("mergetool", "any", "cmd"),
    ("gpg", "none", "program"), ("gpg", "any", "program"), ("gpg", "any", "defaultkeycommand"),
    ("credential", "none", "helper"), ("credential", "any", "helper"),
    ("remote", "any", "uploadpack"), ("remote", "any", "receivepack"),
    ("submodule", "any", "update", True), ("browser", "any", "cmd"), ("man", "any", "cmd"),
    ("sendemail", "either", "smtpserver"), ("sendemail", "either", "sendmailcmd"),
    ("sendemail", "either", "tocmd"), ("sendemail", "either", "cccmd"),
    ("trailer", "any", "command"), ("trailer", "any", "cmd"),
))

_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})
_RUNTIMES = frozenset({"node", "deno", "bun", "perl", "ruby", "php"})
# The characters a shell splits or expands at. The word test splits there.
_SHELL_SPLIT = re.compile(r"""[\s;&|(){}<>`$"'!]+""")


def _is_interpreter(word: str) -> bool:
    name = os.path.basename(word)
    return name in _SHELLS or name in _RUNTIMES or name.startswith("python")


def _split_key(key: str):
    """The section, subsection, and variable. Git lowercases the section
    and the variable; the subsection keeps its case and may hold dots."""
    first, last = key.find("."), key.rfind(".")
    if first == -1:
        return key.lower(), None, ""
    subsection = None if first == last else key[first + 1:last]
    return key[:first].lower(), subsection, key[last + 1:].lower()


def _command_key_of(key: str) -> Optional[KeyPattern]:
    section, subsection, variable = _split_key(key)
    for pattern in COMMAND_KEYS:
        if pattern.section != section:
            continue
        if pattern.subsection != "either" and (pattern.subsection == "none") != (subsection is None):
            continue
        if pattern.variable in ("*", variable):
            return pattern
    return None


def command_problem(entry: ConfigEntry, top: str, home: str) -> Optional[str]:
    """Refuse a command key whose value names a path inside the working
    folder. The value is split into words at white space and each shell
    character. A word with a /, or one that starts with . or ~, is a path
    and resolves against the worktree top. Each interpreter word must be
    followed by an absolute path outside the working folder; an option,
    or nothing, refuses. core.fsmonitor passes as a boolean and refuses
    when empty; alias.* and submodule.*.update count only with a "!"."""
    matched = _command_key_of(entry.key)
    if matched is None or entry.value is None:
        return None
    section, _, variable = _split_key(entry.key)
    value = entry.value
    if section == "core" and variable == "fsmonitor":
        if _GIT_BOOLEAN.fullmatch(value):
            return None
        if value.strip() == "":
            return f"{entry.key} is empty, so it cannot be checked"
    if matched.bang and not value.startswith("!"):
        return None

    def inside(path: str) -> bool:
        return path == top or _within(path, top)

    words = [word for word in _SHELL_SPLIT.split(value) if word]
    for word in words:
        if not ("/" in word or word.startswith(".") or word.startswith("~")):
            continue
        resolved = resolve_config_path(word, top, home)
        if resolved is None:
            return f"{entry.key} is {_printable(value)}, which names another user's home folder, so it cannot be checked"
        if inside(resolved):
            return (f"{entry.key} is {_printable(value)}, and the path {_printable(word)} in it resolves to"
                    f" {_printable(resolved)}, inside the working folder, so a session could write the command git runs")
    for i, word in enumerate(words):
        if not _is_interpreter(word):
            continue
        following = words[i + 1] if i + 1 < len(words) else None
        if following is not None and os.path.isabs(following) and not inside(real_of_existing(following)):
            continue
        after = "nothing" if following is None else _printable(following)
        return (f"{entry.key} is {_printable(value)}, and the interpreter {_printable(word)} in it has {after}"
                " after it, not an absolute path outside the working folder, so it cannot be checked")
    return None


_INCLUDE_KEY = re.compile(r"include(if\..+)?\.path")


def config_problem(top: str, home: str, answer: GitAnswer) -> Optional[str]:
    """Refuse git settings that let a session change what git runs by
    writing inside the working folder: a config file git reads from
    inside it, outside its own top-level .git folder, as an origin or an
    include target (an empty include counts too); a core.hooksPath inside
    it; and a command key that names a path inside it. Every entry is
    checked, and any git failure refuses."""
    def failed(reason: str) -> str:
        return f"git config --list failed, so git's settings cannot be checked ({reason})"

    if answer.code != 0:
        return failed(answer.reason or f"exit code {answer.code}")
    entries = parse_config_list(answer.stdout)
    if entries is None:
        return failed("its answer cannot be read")
    own_git = os.path.join(top, ".git")
    git_folder = real_of_existing(own_git) if os.path.isdir(own_git) and not os.path.islink(own_git) else None

    def inside(path: str) -> bool:
        return path == top or _within(path, top)

    def exposed(path: str) -> bool:
        return inside(path) and not (git_folder is not None and (path == git_folder or _within(path, git_folder)))

    def reads_file(file: str, key: str) -> str:
        return (f"git reads the config file {_printable(file)}, for the key {_printable(key)},"
                " and that file sits inside the working folder")

    for entry in entries:
        file = None
        if entry.origin.startswith("file:"):
            file = real_of_existing(os.path.join(top, entry.origin[len("file:"):]))
        if file is not None and exposed(file):
            return reads_file(file, entry.key)
        if file is not None and _INCLUDE_KEY.fullmatch(entry.key):
            target = resolve_config_path(entry.value or "", os.path.dirname(file), home)
            if target is None:
                return f"{entry.key} is {_printable(entry.value or '')}, which names another user's home folder, so it cannot be checked"
            if exposed(target):
                return reads_file(target, entry.key)
        if entry.key == "core.hookspath":
            value = entry.value or ""
            resolved = resolve_config_path(value, top, home)
            if resolved is None:
                return f"core.hooksPath is {_printable(value)}, which names another user's home folder, so it cannot be checked"
            if inside(resolved):
                return (f"core.hooksPath is {_printable(value)}, which resolves to {_printable(resolved)},"
                        " inside the working folder, so a session could write a git hook")
        command = command_problem(entry, top, home)
        if command is not None:
            return command
    return None


def nested_repo_problem(top: str) -> Optional[str]:
    """Look for another git repository below a worktree top, breadth
    first, at depth 1 to 3. A .git entry of any kind counts. The top's
    own .git is not entered, a link is never followed, and a folder that
    cannot be read refuses."""
    level = [""]
    for depth in range(NESTED_DEPTH + 1):
        if not level:
            break
        following = []
        for rel in level:
            try:
                with os.scandir(os.path.join(top, rel) if rel else top) as found:
                    entries = sorted(found, key=lambda entry: entry.name)
            except OSError as error:
                code = getattr(error, "strerror", None) or type(error).__name__
                return (f"a folder below it cannot be read, so it cannot be checked for other git repositories:"
                        f" {_printable(rel or '.')} ({code})")
            if depth > 0 and any(entry.name == ".git" for entry in entries):
                return f"it holds another git repository at {_printable(rel)}"
            if depth == NESTED_DEPTH:
                continue
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False) or (depth == 0 and entry.name == ".git"):
                    continue
                following.append(os.path.join(rel, entry.name) if rel else entry.name)
        level = following
    return None


def workdir_problem(folder: str, home: str, git: Callable[[Sequence[str]], GitAnswer]) -> Optional[str]:
    """Judge a working folder: the top of a git worktree, compared after
    realpath, not / or the home folder, with no other repository at depth
    1 to 3 and no git setting that runs a file inside it."""
    def why(reason: str) -> str:
        return f"{WORKDIR_RULE} {_printable(folder)}: {reason}."

    real_folder = _real(str(folder))
    if real_folder is None or not os.path.isdir(real_folder):
        return why("the folder cannot be read")
    if real_folder == os.sep:
        return why("it is the root folder")
    if real_folder == (_real(str(home)) or os.path.abspath(home)):
        return why("it is your home folder")
    top = git(GIT_TOP)
    if top.code != 0:
        return why(f"it is not in a git worktree ({top.reason or f'git exited with code {top.code}'})")
    top_path = _first_line(top.stdout)
    if _real(top_path) != real_folder:
        return why(f"it is not the top of a git worktree. The top is {_printable(top_path)}")
    problem = nested_repo_problem(real_folder) or config_problem(real_folder, str(home), git(GIT_CONFIG))
    return None if problem is None else why(problem)


def check_workdir(folder, home, env: Mapping[str, str], *, runner: GitRunner = run_git,
                  timeout_s: float = GIT_TIMEOUT_S) -> Optional[str]:
    """Check a folder with plain git, run in that folder with every GIT_
    variable dropped from env. Returns the reason it is refused, or
    None."""
    git_env = without_git_vars(env)
    real_folder = _real(str(folder))
    answers = {}
    if real_folder is not None and os.path.isdir(real_folder) and real_folder != os.sep:
        for args in (GIT_TOP, GIT_CONFIG):
            answers[args] = runner(list(args), real_folder, git_env, timeout_s)
    return workdir_problem(str(folder), str(home),
                           lambda args: answers.get(tuple(args), GitAnswer(None, "", "git did not run")))


# The environment variables a harness child gets, when set. Ported from
# CODEX_CHILD_ENV without NO_PROXY and OPENAI_API_KEY; the proxy
# variables are set by child_env, never passed through.
CHILD_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "CODEX_HOME",
)
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def child_env(vars: Mapping[str, str], proxy_url: str) -> dict:
    """The allowlisted variables that are set, plus every proxy variable,
    upper and lower case, pointing at the egress proxy. No NO_PROXY."""
    env = {key: vars[key] for key in CHILD_ENV if vars.get(key) is not None}
    for key in PROXY_VARS:
        env[key] = proxy_url
    return env


# The fixed sandbox arguments, read in #24 from `codex exec --help` of
# codex-cli 0.157.0. A later codex build must check these names again.
SANDBOX_FIXED = ("--sandbox", "workspace-write", "-c", "sandbox_workspace_write.network_access=false")
ROOTS_ARG = "sandbox_workspace_write.writable_roots=" + json.dumps([])
_SANDBOX_ARGS = (*SANDBOX_FIXED, "-c", ROOTS_ARG)
_SUBCOMMANDS = frozenset({"resume", "fork", "review", "help"})
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# No Codex launch flag is verified, so every one is refused.
VERIFIED_FLAGS: frozenset = frozenset()


def refused_codex_flag(flag_args: Sequence[str], verified: frozenset = VERIFIED_FLAGS) -> Optional[str]:
    """Why launch arguments are refused, or None. They come in pairs, a
    flag name and its value. A name must be a verified flag, and none is.
    A value must not start with - or hold a control character."""
    flag_args = list(flag_args)
    for i in range(0, len(flag_args), 2):
        name = flag_args[i]
        if not isinstance(name, str) or not name.startswith("-"):
            why = "it is a codex exec subcommand" if name in _SUBCOMMANDS else "it is not a flag"
            return f'the launch argument "{_printable(name)}" is refused on Codex CLI, because {why}.'
        if name not in verified:
            return (f'the launch flag "{_printable(name)}" is refused on Codex CLI, because no Codex launch flag'
                    " is verified. The bench sets the sandbox itself.")
        if i + 1 >= len(flag_args):
            return f'the launch flag "{_printable(name)}" is refused on Codex CLI, because it has no value.'
        value = flag_args[i + 1]
        if not isinstance(value, str) or value.startswith("-"):
            return f'the value "{_printable(value)}" of the launch flag "{name}" is refused, because it starts with -.'
        if _CONTROL.search(value):
            return f'the value "{_printable(value)}" of the launch flag "{name}" is refused, because it holds a control character.'
    return None


def codex_exec_args(flag_args: Sequence[str], prompt: str) -> list:
    """The arguments for one `codex exec` run: exec, the launch flags
    (none pass today), the fixed sandbox, --, and the prompt, so a prompt
    that starts with - is never read as an option. Raises GuardRefused
    when a launch argument is refused."""
    refused = refused_codex_flag(flag_args)
    if refused is not None:
        raise GuardRefused(refused)
    if not isinstance(prompt, str) or not prompt or "\0" in prompt:
        raise GuardRefused("the prompt must be a non-empty string with no NUL")
    return ["exec", *flag_args, *_SANDBOX_ARGS, "--", prompt]


_TAIL = len(_SANDBOX_ARGS) + 2


def exec_args_problem(args: Sequence[str]) -> Optional[str]:
    """Why a full `codex exec` argument list is refused, or None. It must
    be exec, launch flags that pass the flag check, the exact sandbox
    arguments, --, and the prompt. Run before every harness start."""
    args = list(args)
    wrong = "the arguments are not the sandbox arguments the bench sets"
    if len(args) < 1 + _TAIL or args[0] != "exec" or args[-2] != "--":
        return wrong
    if tuple(args[-_TAIL:-2]) != _SANDBOX_ARGS:
        return wrong
    return refused_codex_flag(args[1:-_TAIL])
