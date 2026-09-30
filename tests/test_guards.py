"""Tests for the replay harness guards (issue #6), the Python port of the
trellis-crew #24 guards.

Every test makes its own small git repository in a temporary directory,
with a temporary home, so no user or shared git config is read. Nothing
here runs codex or reaches a network.

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench import guards  # noqa: E402
from bench.guards import (  # noqa: E402
    CHILD_ENV,
    GitAnswer,
    GuardRefused,
    SANDBOX_FIXED,
    check_workdir,
    child_env,
    codex_exec_args,
    exec_args_problem,
    nested_repo_problem,
    parse_config_list,
    refused_codex_flag,
    without_git_vars,
)

SOURCE_COMMIT = "8aae572101f502884b392dcaef071f915dafe9bb"
PROXY = "http://127.0.0.1:40000"


def needs_git(test):
    return unittest.skipIf(shutil.which("git") is None, "git is not installed")(test)


class Repo:
    """A temporary git repository with one commit, and a temporary home."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(os.path.realpath(self.tmp.name))
        self.home = self.root / "home"
        self.home.mkdir()
        self.top = self.root / "repo"
        self.top.mkdir()
        self.env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.home), "LANG": "C"}
        self.git("init", "-q")
        (self.top / "a.txt").write_text("a\n")
        self.git("add", "-A")
        self.git("-c", "user.name=Bench", "-c", "user.email=bench@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-q", "-m", "first")

    def git(self, *args):
        env = dict(self.env, GIT_CONFIG_NOSYSTEM="1")
        return subprocess.run(["git", "-C", str(self.top), *args], check=True, env=env,
                              capture_output=True).stdout.decode()

    def check(self, folder=None):
        return check_workdir(folder or self.top, self.home, self.env)

    def close(self):
        self.tmp.cleanup()


class GuardTestCase(unittest.TestCase):
    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        self.repo = Repo()
        self.addCleanup(self.repo.close)


class SourceTests(unittest.TestCase):
    def test_the_module_cites_the_source_file_and_commit(self):
        for part in ["codex-guard.ts", "codex-args.ts", SOURCE_COMMIT, "trellis-crew"]:
            self.assertIn(part, guards.__doc__)


class WorkdirTests(GuardTestCase):
    def test_a_clean_worktree_top_passes(self):
        self.assertIsNone(self.repo.check())

    def test_a_folder_below_the_top_is_refused(self):
        sub = self.repo.top / "sub"
        sub.mkdir()
        self.assertIn("not the top of a git worktree", self.repo.check(sub))

    def test_a_folder_outside_any_repository_is_refused(self):
        outside = self.repo.root / "plain"
        outside.mkdir()
        self.assertIn("not in a git worktree", self.repo.check(outside))

    def test_the_home_folder_is_refused(self):
        self.assertIn("home folder", check_workdir(self.repo.home, self.repo.home, self.repo.env))

    def test_the_root_folder_is_refused(self):
        self.assertIn("root folder", check_workdir("/", self.repo.home, self.repo.env))

    def test_a_missing_folder_is_refused(self):
        self.assertIn("cannot be read", self.repo.check(self.repo.root / "missing"))

    def test_every_git_variable_is_dropped_for_the_git_calls(self):
        env = dict(self.repo.env, GIT_DIR=str(self.repo.root / "nowhere"), GIT_WORK_TREE="/")
        self.assertIsNone(check_workdir(self.repo.top, self.repo.home, env))
        seen = []

        def runner(args, cwd, run_env, timeout_s):
            seen.append(run_env)
            if args[0] == "rev-parse":
                return GitAnswer(0, str(self.repo.top) + "\n", "")
            return GitAnswer(0, "", "")

        self.assertIsNone(check_workdir(self.repo.top, self.repo.home, env, runner=runner))
        self.assertEqual(len(seen), 2)
        for run_env in seen:
            self.assertFalse([key for key in run_env if key.startswith("GIT_")])
            self.assertEqual(run_env["HOME"], str(self.repo.home))

    def test_without_git_vars_keeps_every_other_variable(self):
        self.assertEqual(without_git_vars({"GIT_DIR": "x", "GITHUB": "y", "PATH": "p"}), {"GITHUB": "y", "PATH": "p"})

    def test_a_git_failure_refuses_so_the_check_fails_closed(self):
        def top_fails(args, cwd, env, timeout_s):
            return GitAnswer(128, "", "fatal: not a git repository")

        problem = check_workdir(self.repo.top, self.repo.home, self.repo.env, runner=top_fails)
        self.assertIn("not in a git worktree", problem)

        def config_fails(args, cwd, env, timeout_s):
            if args[0] == "rev-parse":
                return GitAnswer(0, str(self.repo.top) + "\n", "")
            return GitAnswer(None, "", "git did not answer in time")

        problem = check_workdir(self.repo.top, self.repo.home, self.repo.env, runner=config_fails)
        self.assertIn("git config --list failed", problem)

    def test_config_output_of_the_wrong_shape_refuses(self):
        def bad_config(args, cwd, env, timeout_s):
            if args[0] == "rev-parse":
                return GitAnswer(0, str(self.repo.top) + "\n", "")
            return GitAnswer(0, "file:x\0core.bare\nfalse", "")

        self.assertIn("cannot be read", check_workdir(self.repo.top, self.repo.home, self.repo.env, runner=bad_config))


class NestedRepoTests(GuardTestCase):
    def test_a_nested_repository_at_depth_one_to_three_is_refused(self):
        for rel in ["one", "a/two", "a/b/three"]:
            with self.subTest(rel=rel):
                (self.repo.top / rel / ".git").mkdir(parents=True)
                self.assertIn(f"another git repository at {rel}", self.repo.check())
                shutil.rmtree(self.repo.top / rel.split("/")[0])

    def test_a_gitfile_counts_as_a_nested_repository(self):
        (self.repo.top / "sub").mkdir()
        (self.repo.top / "sub" / ".git").write_text("gitdir: /elsewhere\n")
        self.assertIn("another git repository at sub", self.repo.check())

    def test_a_repository_below_depth_three_is_not_looked_for(self):
        (self.repo.top / "a" / "b" / "c" / "d" / ".git").mkdir(parents=True)
        self.assertIsNone(self.repo.check())

    def test_a_symbolic_link_is_never_followed(self):
        other = self.repo.root / "other"
        (other / ".git").mkdir(parents=True)
        (self.repo.top / "link").symlink_to(other, target_is_directory=True)
        self.assertIsNone(nested_repo_problem(str(self.repo.top)))

    def test_an_unreadable_folder_refuses(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can read every folder")
        locked = self.repo.top / "locked"
        locked.mkdir()
        locked.chmod(0)
        self.addCleanup(locked.chmod, 0o755)
        self.assertIn("cannot be read", self.repo.check())


class ConfigTests(GuardTestCase):
    def test_a_hooks_path_inside_the_worktree_is_refused(self):
        self.repo.git("config", "core.hooksPath", ".githooks")
        self.assertIn("core.hooksPath", self.repo.check())

    def test_a_hooks_path_outside_the_worktree_passes(self):
        for value in ["/dev/null", "~/hooks"]:
            with self.subTest(value=value):
                self.repo.git("config", "core.hooksPath", value)
                self.assertIsNone(self.repo.check())

    def test_an_include_of_a_file_inside_the_worktree_is_refused(self):
        (self.repo.top / "inc.cfg").write_text("[user]\n\tname = x\n")
        self.repo.git("config", "include.path", "../inc.cfg")
        self.assertIn("inc.cfg", self.repo.check())

    def test_an_include_of_a_missing_file_inside_the_worktree_is_refused(self):
        self.repo.git("config", "includeIf.gitdir:/.path", "../later.cfg")
        self.assertIn("later.cfg", self.repo.check())

    def test_a_command_key_naming_a_path_inside_the_worktree_is_refused(self):
        for key, value in [("core.fsmonitor", "./fsmon.sh"), ("filter.x.clean", "./clean.sh"),
                           ("alias.x", "!./run.sh"), ("diff.external", "tools/diff.sh")]:
            with self.subTest(key=key):
                self.repo.git("config", key, value)
                self.assertIn("inside the working folder", self.repo.check())
                self.repo.git("config", "--unset", key)

    def test_an_interpreter_with_a_bare_script_is_refused(self):
        self.repo.git("config", "core.fsmonitor", "sh fsmon.sh")
        self.assertIn("interpreter", self.repo.check())

    def test_safe_command_keys_pass(self):
        for key, value in [("core.fsmonitor", "false"), ("alias.st", "status"), ("credential.helper", "store")]:
            with self.subTest(key=key):
                self.repo.git("config", key, value)
                self.assertIsNone(self.repo.check())
                self.repo.git("config", "--unset", key)

    def test_parse_config_list_reads_the_nul_format(self):
        entries = parse_config_list("file:.git/config\0core.bare\nfalse\0command line:\0core.flag\0")
        self.assertEqual([(e.origin, e.key, e.value) for e in entries],
                         [("file:.git/config", "core.bare", "false"), ("command line:", "core.flag", None)])
        self.assertEqual(parse_config_list(""), [])
        self.assertIsNone(parse_config_list("file:x\0core.bare"))


class ChildEnvTests(unittest.TestCase):
    def test_the_child_environment_is_an_allowlist_with_the_proxy_set(self):
        env = child_env({"PATH": "/bin", "HOME": "/h", "SECRET": "s", "GIT_DIR": "g", "NO_PROXY": "*",
                         "no_proxy": "*", "OPENAI_API_KEY": "k", "HTTP_PROXY": "http://elsewhere"}, PROXY)
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["HOME"], "/h")
        for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
            self.assertEqual(env[name], PROXY)
        for name in ["SECRET", "GIT_DIR", "NO_PROXY", "no_proxy", "OPENAI_API_KEY"]:
            self.assertNotIn(name, env)
        self.assertTrue(set(env) <= set(CHILD_ENV) | {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                                                       "http_proxy", "https_proxy", "all_proxy"})

    def test_the_allowlist_passes_no_key_and_no_proxy_bypass(self):
        for name in ["NO_PROXY", "no_proxy", "OPENAI_API_KEY"]:
            self.assertNotIn(name, CHILD_ENV)


class CodexArgvTests(unittest.TestCase):
    EXPECTED_SANDBOX = ["--sandbox", "workspace-write", "-c", "sandbox_workspace_write.network_access=false",
                        "-c", "sandbox_workspace_write.writable_roots=[]"]

    def test_the_codex_argv_has_the_sandbox_network_off_and_the_prompt_after_a_double_dash(self):
        self.assertEqual(codex_exec_args([], "Fix the parser."),
                         ["exec", *self.EXPECTED_SANDBOX, "--", "Fix the parser."])
        self.assertEqual(list(SANDBOX_FIXED), self.EXPECTED_SANDBOX[:4])

    def test_a_prompt_that_starts_with_a_dash_stays_after_the_double_dash(self):
        args = codex_exec_args([], "--dangerously-bypass-approvals-and-sandbox")
        self.assertEqual(args[-2:], ["--", "--dangerously-bypass-approvals-and-sandbox"])
        self.assertIsNone(exec_args_problem(args))

    def test_every_extra_flag_is_refused(self):
        for flags in (["--dangerously-bypass-approvals-and-sandbox", "x"], ["--sandbox", "danger-full-access"],
                      ["-m", "some-model"], ["-c", "sandbox_workspace_write.network_access=true"],
                      ["resume", "x"], ["loose"]):
            with self.subTest(flags=flags):
                self.assertIsNotNone(refused_codex_flag(flags))
                with self.assertRaises(GuardRefused):
                    codex_exec_args(flags, "prompt")

    def test_a_verified_flag_still_needs_a_clean_value(self):
        verified = frozenset({"--model"})
        self.assertIsNone(refused_codex_flag(["--model", "m"], verified))
        self.assertIn("starts with -", refused_codex_flag(["--model", "-x"], verified))
        self.assertIn("control character", refused_codex_flag(["--model", "m\x1b"], verified))
        self.assertIn("no value", refused_codex_flag(["--model"], verified))

    def test_a_full_argv_is_checked_against_the_fixed_sandbox(self):
        good = codex_exec_args([], "p")
        self.assertIsNone(exec_args_problem(good))
        bad = [
            ["exec", "--sandbox", "danger-full-access", "-c", "sandbox_workspace_write.network_access=false",
             "-c", "sandbox_workspace_write.writable_roots=[]", "--", "p"],
            ["exec", "--sandbox", "workspace-write", "-c", "sandbox_workspace_write.network_access=true",
             "-c", "sandbox_workspace_write.writable_roots=[]", "--", "p"],
            ["exec", "--yolo", "x", *self.EXPECTED_SANDBOX, "--", "p"],
            ["exec", *self.EXPECTED_SANDBOX, "p"],
            ["review", *self.EXPECTED_SANDBOX, "--", "p"],
        ]
        for args in bad:
            with self.subTest(args=args):
                self.assertIsNotNone(exec_args_problem(args))


if __name__ == "__main__":
    unittest.main()
