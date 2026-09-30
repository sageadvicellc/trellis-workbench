"""Tests for the bench task corpus (issue #5).

Every test builds its own corpus, lists, and snapshots in a temporary
directory. Key-shaped values are built from pieces, so no whole
credential-shaped string sits in this file.

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unittest import mock  # noqa: E402

from bench import corpus as corpus_module  # noqa: E402
from bench.corpus import (  # noqa: E402
    CorpusRefused,
    load_corpus,
    read_allowlist,
    read_client_list,
    scan_snapshot,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
REPO = "example-org/example-repo"
CLIENT = "example-org/client-site"
CLEARANCE = "https://github.com/example-org/decisions/issues/1#issuecomment-1"


class Fixture:
    """A temporary corpus with one valid task, and its lists and snapshot."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.corpus = self.root / "corpus"
        self.snapshots = self.root / "snapshots"
        self.allowlist = self.root / "allowlist.txt"
        self.clients = self.root / "clients.txt"
        self.deny = self.root / "deny.txt"
        self.allowlist.write_text(f"# header\n{REPO} cleared_on=2026-09-30 clearance={CLEARANCE}\n")
        self.clients.write_text(f"# client repos\n{CLIENT}\n")
        self.deny.write_text("# deny terms\nzz-private-term\n")
        self.add_task("parser-fix")
        self.snapshot(REPO, COMMIT)

    def add_task(self, task_id, repo=REPO, commit=COMMIT, prompt="prompt.md", gates=None, extra=None,
                 prompt_text="Fix the failing parser test."):
        task_dir = self.corpus / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompt.md").write_text(prompt_text)
        task = {"repo": repo, "commit": commit, "prompt": prompt,
                "gates": gates if gates is not None else [["npm", "run", "test"], ["npm", "run", "lint"]]}
        if extra:
            task.update(extra)
        (task_dir / "task.json").write_text(json.dumps(task))
        return task_dir

    def snapshot(self, repo, commit, files=None):
        path = self.snapshots / repo / commit
        path.mkdir(parents=True, exist_ok=True)
        for name, text in (files or {"src/parser.py": "def parse(text):\n    return text\n"}).items():
            file = path / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text)
        return path

    def load(self):
        return load_corpus(
            corpus_dir=self.corpus,
            allowlist_path=self.allowlist,
            client_list_path=self.clients,
            deny_terms_path=self.deny,
            snapshot_root=self.snapshots,
        )

    def close(self):
        self.tmp.cleanup()


class CorpusTestCase(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(self.fx.close)

    def assertRefused(self, text=None):
        with self.assertRaises(CorpusRefused) as ctx:
            self.fx.load()
        if text is not None:
            self.assertIn(text, str(ctx.exception))
        return str(ctx.exception)


class ValidTaskTests(CorpusTestCase):
    def test_a_valid_task_loads_with_its_repo_commit_prompt_and_gates(self):
        tasks = self.fx.load()
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.task_id, "parser-fix")
        self.assertEqual(task.repo, REPO)
        self.assertEqual(task.commit, COMMIT)
        self.assertEqual(task.prompt_id, "tasks/parser-fix/prompt.md")
        self.assertEqual(task.gates, (("npm", "run", "test"), ("npm", "run", "lint")))
        self.assertEqual(task.prompt_text(), "Fix the failing parser test.")

    def test_a_corpus_with_no_tasks_loads_empty(self):
        for task_dir in (self.fx.corpus / "tasks").iterdir():
            for file in task_dir.iterdir():
                file.unlink()
            task_dir.rmdir()
        self.assertEqual(self.fx.load(), [])


class ForbiddenSourceTests(CorpusTestCase):
    def test_a_task_from_a_repo_off_the_allowlist_is_refused(self):
        self.fx.add_task("other", repo="example-org/not-cleared")
        self.fx.snapshot("example-org/not-cleared", COMMIT)
        message = self.assertRefused("other")
        self.assertIn("not on the allowlist", message)

    def test_a_client_repo_on_the_allowlist_refuses_the_build(self):
        self.fx.clients.write_text(f"{CLIENT}\n{REPO.upper()}\n")
        self.assertRefused("the allowlist names a repository on the client list")

    def test_a_task_naming_a_client_repo_is_refused(self):
        self.fx.add_task("client-task", repo=CLIENT)
        self.fx.snapshot(CLIENT, COMMIT)
        self.assertRefused("task client-task: the repository is on the client list")

    def test_a_task_naming_a_client_repo_with_a_git_suffix_or_capitals_is_refused(self):
        self.fx.add_task("client-git", repo=CLIENT.upper() + ".git")
        self.assertRefused("task client-git: the repository is on the client list")

    def test_a_repo_name_of_dots_is_refused(self):
        for repo in ["example-org/..", "example-org/.", "../example-repo"]:
            with self.subTest(repo=repo):
                self.fx.add_task("dots", repo=repo)
                self.assertRefused("task dots: repo is not an owner/name repository")

    def test_mixed_case_repo_matches_its_allowlist_entry(self):
        (self.fx.corpus / "tasks" / "parser-fix" / "task.json").unlink()
        self.fx.add_task("parser-fix", repo=REPO.upper())
        self.fx.snapshot(REPO.upper(), COMMIT)
        self.assertEqual(len(self.fx.load()), 1)

    def test_a_missing_or_empty_client_list_refuses_the_build(self):
        self.fx.clients.unlink()
        self.assertRefused("client list")
        self.fx.clients.write_text("# no entries\n")
        self.assertRefused("client list")

    def test_a_bad_client_list_line_refuses_the_build(self):
        self.fx.clients.write_text("not a repo line\n")
        self.assertRefused("client list")

    def test_an_allowlist_entry_needs_a_clearance(self):
        for line in [
            f"{REPO}\n",
            f"{REPO} cleared_on=2026-09-30\n",
            f"{REPO} clearance={CLEARANCE}\n",
            f"{REPO} cleared_on=yesterday clearance={CLEARANCE}\n",
            f"{REPO} cleared_on=2026-09-30 clearance=http://example.com/x\n",
            f"{REPO} cleared_on=2026-09-30 clearance={CLEARANCE}\n{REPO} cleared_on=2026-09-30 clearance={CLEARANCE}\n",
        ]:
            with self.subTest(line=line):
                self.fx.allowlist.write_text(line)
                self.assertRefused("allowlist")

    def test_the_shipped_allowlist_names_no_repo(self):
        shipped = REPO_ROOT / "corpus" / "allowlist.txt"
        self.assertEqual(read_allowlist(shipped), {})
        self.assertIn("egress clearance", shipped.read_text())

    def test_a_list_that_is_not_utf8_refuses_the_build(self):
        self.fx.clients.write_bytes(b"example-org/\xffclient\n")
        self.assertRefused("client list cannot be read")

    def test_a_missing_corpus_directory_refuses_the_build(self):
        self.fx.corpus = self.fx.root / "typo"
        self.assertRefused("the corpus directory does not exist")

    def test_a_missing_or_empty_deny_terms_file_refuses_the_build(self):
        self.fx.deny.write_text("# none\n")
        self.assertRefused("deny terms")
        self.fx.deny.unlink()
        self.assertRefused("deny terms")


class TaskShapeTests(CorpusTestCase):
    def test_a_bad_commit_is_refused(self):
        for commit in ["main", "HEAD", "a" * 39, "A" * 40]:
            with self.subTest(commit=commit):
                self.fx.add_task("bad-commit", commit=commit)
                self.assertRefused("task bad-commit: commit is not a full 40-hex or 64-hex commit")

    def test_a_64_hex_commit_loads(self):
        (self.fx.corpus / "tasks" / "parser-fix" / "task.json").unlink()
        self.fx.add_task("parser-fix", commit="e" * 64)
        self.fx.snapshot(REPO, "e" * 64)
        self.assertEqual(self.fx.load()[0].commit, "e" * 64)

    def test_a_prompt_outside_the_task_is_refused(self):
        for prompt in ["../parser-fix/prompt.md", "/etc/passwd", "missing.md", "task.json"]:
            with self.subTest(prompt=prompt):
                self.fx.add_task("bad-prompt", prompt=prompt)
                message = self.assertRefused("task bad-prompt:")
                self.assertIn("prompt", message.split("task bad-prompt:", 1)[1])

    def test_bad_gates_are_refused(self):
        for gates in [[], [[]], ["npm run test"], [["npm", ""]], [["npm", "a\nb"]], [["x"]] * 21]:
            with self.subTest(gates=gates):
                self.fx.add_task("bad-gates", gates=gates)
                self.assertRefused("task bad-gates: gates must be")

    def test_an_extra_task_field_is_refused(self):
        self.fx.add_task("extra", extra={"client": "somebody"})
        self.assertRefused("task extra: task.json must hold exactly")

    def test_a_malformed_task_json_is_refused(self):
        task_dir = self.fx.add_task("broken")
        (task_dir / "task.json").write_text("{not json")
        self.assertRefused("task broken: task.json is missing or not JSON")

    def test_a_stray_file_in_tasks_is_refused(self):
        (self.fx.corpus / "tasks" / "notes.txt").write_text("x")
        self.assertRefused("an entry in tasks/ is not a task directory")

    def test_every_bad_task_is_listed(self):
        self.fx.add_task("bad-one", commit="main")
        self.fx.add_task("bad-two", gates=[])
        message = self.assertRefused()
        self.assertIn("task bad-one:", message)
        self.assertIn("task bad-two:", message)

    def test_a_bad_task_id_is_refused(self):
        self.fx.add_task("Bad Id")
        self.assertRefused("a task id")

    def test_a_missing_snapshot_is_refused(self):
        self.fx.add_task("no-snapshot", commit="b" * 40)
        self.assertRefused("task no-snapshot: no snapshot exists at the frozen commit")


class SnapshotScanTests(CorpusTestCase):
    def planted(self, text, name="config.py"):
        self.fx.add_task("planted", commit="c" * 40)
        self.fx.snapshot(REPO, "c" * 40, {name: text})
        return self.assertRefused("planted")

    def test_a_snapshot_with_a_planted_secret_fails(self):
        mixed = "Zk3x9Qw" + "ErTyUiOp" + "AsDfGhJk2Lm"
        for text in [
            "KEY = '" + "AK" + "IA" + "IOSFODNN7EXAMPL" + "E'\n",
            "TOKEN = '" + "gh" + "p_" + mixed + "abcdefghijklmn" + "'\n",
            "KEY = '" + "sk" + "-ant-" + "api03-" + mixed + "'\n",
            "-----BEGIN " + "RSA PRIVATE KEY" + "-----\nabc\n",
            "SLACK = '" + "xo" + "xb-" + "123456789012-abc'\n",
        ]:
            with self.subTest(text=text[:12]):
                message = self.planted(text)
                self.assertNotIn(mixed, message)
                self.assertIn("secret", message)

    def test_a_snapshot_with_a_deny_term_fails(self):
        message = self.planted("# notes for ZZ-Private-Term only\n")
        self.assertIn("sanitizer", message)
        self.assertNotIn("zz-private-term", message.lower())

    def test_a_snapshot_with_an_email_address_fails(self):
        self.assertIn("sanitizer", self.planted("AUTHOR = 'someone" + "@" + "example.com'\n"))

    def test_a_snapshot_naming_a_client_repo_fails(self):
        self.assertIn("sanitizer", self.planted("see the client-site repo\n"))

    def test_a_secret_file_name_fails(self):
        self.assertIn("secret", self.planted("anything\n", name=".env"))

    def test_the_prompt_is_scanned_too(self):
        self.fx.add_task("secret-prompt", prompt_text="use " + "AK" + "IA" + "IOSFODNN7EXAMPLE")
        self.assertRefused("task secret-prompt: its own files failed the secret scan")

    def test_a_finding_names_the_rule_and_line_never_the_text(self):
        root = self.fx.snapshot(REPO, "d" * 40, {"a.py": "x = 1\ny = '" + "AK" + "IA" + "IOSFODNN7EXAMPLE'\n"})
        findings = scan_snapshot(root, deny_terms=(), client_repos=frozenset())
        self.assertEqual([(f.path, f.line, f.rule) for f in findings], [("a.py", 2, "secret:aws_access_key")])

    def test_a_clean_snapshot_has_no_findings(self):
        root = self.fx.snapshots / REPO / COMMIT
        self.assertEqual(scan_snapshot(root, deny_terms=("zz-private-term",), client_repos=frozenset({CLIENT})), [])

    def scan(self, files, deny=("zz-private-term",), clients=frozenset({CLIENT})):
        root = self.fx.root / "scan"
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(text, bytes):
                path.write_bytes(text)
            else:
                path.write_text(text)
        return scan_snapshot(root, deny_terms=deny, client_repos=clients)

    def test_a_linked_directory_or_file_is_a_finding(self):
        outside = self.fx.root / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text("x = 1\n")
        root = self.fx.root / "scan"
        root.mkdir()
        (root / "linked-dir").symlink_to(outside, target_is_directory=True)
        (root / "linked-file.py").symlink_to(outside / "secret.py")
        rules = sorted(f.rule for f in scan_snapshot(root, deny_terms=(), client_repos=frozenset()))
        self.assertEqual(rules, ["sanitizer:symlink", "sanitizer:symlink"])

    def test_git_metadata_is_a_finding_not_skipped(self):
        rules = {f.rule for f in self.scan({".git/config": "[core]\n", "sub/.git": "gitdir: x\n"})}
        self.assertEqual(rules, {"sanitizer:git_metadata"})

    def test_a_file_too_large_or_not_utf8_is_a_finding(self):
        with mock.patch.object(corpus_module, "_MAX_SCAN_BYTES", 10):
            rules = {f.rule for f in self.scan({"big.txt": "x" * 11, "bin.dat": b"\xff\xfe\x00"})}
        self.assertEqual(rules, {"sanitizer:file_too_large", "sanitizer:not_utf8_text"})

    def test_an_unreadable_file_is_a_finding(self):
        root = self.fx.root / "scan"
        root.mkdir()
        locked = root / "locked.py"
        locked.write_text("x = 1\n")
        locked.chmod(0)
        self.addCleanup(locked.chmod, 0o600)
        if os.access(locked, os.R_OK):
            self.skipTest("running as a user who can read any file")
        rules = {f.rule for f in scan_snapshot(root, deny_terms=(), client_repos=frozenset())}
        self.assertEqual(rules, {"sanitizer:unreadable"})

    def test_names_are_scanned_too(self):
        findings = self.scan({"docs/zz-private-term-notes.md": "clean\n", "client-site/readme.md": "clean\n"})
        rules = {f.rule for f in findings}
        self.assertIn("sanitizer:deny_term_in_path", rules)
        self.assertIn("sanitizer:client_repository_in_path", rules)

    def test_unicode_variants_of_a_deny_term_are_caught(self):
        fullwidth = "ＺＺ-ＰＲＩＶＡＴＥ-ＴＥＲＭ"
        zero_width = "zz-pri\u200bvate-term"
        for text in [fullwidth, zero_width]:
            with self.subTest(text=text):
                rules = {f.rule for f in self.scan({"a.md": text + "\n"})}
                self.assertIn("sanitizer:deny_term", rules)

    def test_a_form_feed_cannot_split_a_key(self):
        rules = {f.rule for f in self.scan({"a.py": "k = '\x0c" + "AK" + "IA" + "IOSFODNN7EXAMPLE'\n"})}
        self.assertIn("secret:aws_access_key", rules)

    def test_ordinary_hyphenated_words_are_not_keys(self):
        self.assertEqual(self.scan({"a.md": "task-runner-configuration and desk-lamp-organizer-settings\n"}), [])

    def test_a_client_name_matches_as_a_whole_word_only(self):
        self.assertEqual(self.scan({"a.md": "my-client-sitemap is fine\n"}), [])
        self.assertTrue(self.scan({"a.md": "see client-site now\n"}))

    def test_more_key_formats_are_caught(self):
        for rule, text in [
            ("gitlab_token", "gl" + "pat-" + "abcdefghijklmnopqrst12"),
            ("stripe_key", "sk" + "_live_" + "abcdefghijklmnopqrstu1"),
            ("npm_token", "np" + "m_" + "a" * 36),
            ("jwt", "ey" + "J" + "a" * 12 + ".ey" + "J" + "b" * 12 + ".sig"),
            ("url_with_password", "postgres://user:" + "hunter2" + "@db.example.com/x"),
        ]:
            with self.subTest(rule=rule):
                rules = {f.rule for f in self.scan({"a.py": text + "\n"})}
                self.assertIn(f"secret:{rule}", rules)

    def test_secret_file_names_match_in_any_case(self):
        for name in [".ENV", "ID_RSA", "KEY.PEM", ".npmrc", ".netrc", "cert.pfx"]:
            with self.subTest(name=name):
                root = self.fx.root / f"scan-{name.lower().replace('.', '_')}"
                root.mkdir()
                (root / name).write_text("x\n")
                rules = {f.rule for f in scan_snapshot(root, deny_terms=(), client_repos=frozenset())}
                self.assertIn("secret:secret_file_name", rules)

    def test_the_message_names_rule_file_and_line_but_hides_a_bad_path(self):
        self.fx.add_task("located", commit="f" * 40)
        self.fx.snapshot(REPO, "f" * 40, {"cfg.py": "x = 1\nk = '" + "AK" + "IA" + "IOSFODNN7EXAMPLE'\n",
                                         "zz-private-term.md": "clean\n"})
        message = self.assertRefused("task located:")
        self.assertIn("secret:aws_access_key at cfg.py:2", message)
        self.assertIn("a path not shown", message)
        self.assertNotIn("zz-private-term", message.lower())

    def test_client_list_entries_are_lowercased(self):
        self.fx.clients.write_text("Example-Org/Client-Site\n")
        self.assertEqual(read_client_list(self.fx.clients), frozenset({"example-org/client-site"}))


if __name__ == "__main__":
    unittest.main()
