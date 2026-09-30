"""Tests for the bench task corpus (issue #5).

Every test builds its own corpus, lists, and snapshots in a temporary
directory. Key-shaped values are built from pieces, so no whole
credential-shaped string sits in this file. No fixture holds a real
client name, deny term, or credential.

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench import corpus as corpus_module  # noqa: E402
from bench.corpus import (  # noqa: E402
    CorpusRefused,
    check_manifest,
    git_tree_hash,
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
AWS_KEY = "AK" + "IA" + "IOSFODNN7EXAMPLE"


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
        self.deny.write_text("# deny terms\nzz private term\n")
        self.snapshot(REPO, COMMIT)
        self.add_task("parser-fix")

    def add_task(self, task_id, repo=REPO, commit=COMMIT, prompt="prompt.md", gates=None, extra=None,
                 prompt_text="Fix the failing parser test.", tree="auto"):
        task_dir = self.corpus / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompt.md").write_text(prompt_text)
        if tree == "auto":
            snapshot = self.snapshots / repo / commit
            tree = git_tree_hash(snapshot) if snapshot.is_dir() else "0" * 40
        task = {"repo": repo, "commit": commit, "tree": tree, "prompt": prompt,
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
            if isinstance(text, bytes):
                file.write_bytes(text)
            else:
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

    def scan(self, files, deny=("zz private term",), clients=frozenset({CLIENT})):
        root = self.fx.root / "scan"
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(text, bytes):
                path.write_bytes(text)
            else:
                path.write_text(text)
        root.mkdir(exist_ok=True)
        return scan_snapshot(root, deny_terms=deny, client_repos=clients)

    def rules(self, files, **kwargs):
        return {f.rule for f in self.scan(files, **kwargs)}


class ValidTaskTests(CorpusTestCase):
    def test_a_valid_task_loads_with_its_repo_commit_tree_prompt_and_gates(self):
        tasks = self.fx.load()
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.task_id, "parser-fix")
        self.assertEqual(task.repo, REPO)
        self.assertEqual(task.commit, COMMIT)
        self.assertEqual(task.prompt_id, "tasks/parser-fix/prompt.md")
        self.assertEqual(task.gates, (("npm", "run", "test"), ("npm", "run", "lint")))
        self.assertEqual(task.prompt_text(), "Fix the failing parser test.")
        self.assertEqual([entry.path for entry in task.snapshot_manifest], ["src/parser.py"])

    def test_a_corpus_with_no_tasks_loads_empty(self):
        shutil.rmtree(self.fx.corpus / "tasks")
        self.assertEqual(self.fx.load(), [])


class ForbiddenSourceTests(CorpusTestCase):
    def test_a_task_from_a_repo_off_the_allowlist_is_refused(self):
        self.fx.snapshot("example-org/not-cleared", COMMIT)
        self.fx.add_task("other", repo="example-org/not-cleared")
        self.assertRefused("task other: the repository is not on the allowlist")

    def test_a_client_repo_on_the_allowlist_refuses_the_build(self):
        self.fx.clients.write_text(f"{CLIENT}\n{REPO.upper()}\n")
        self.assertRefused("the allowlist names a repository on the client list")

    def test_a_task_naming_a_client_repo_is_refused(self):
        self.fx.add_task("client-task", repo=CLIENT)
        # The task's own files are scanned before task.json is parsed, so
        # the sanitizer catches the client name first.
        message = self.assertRefused("task client-task:")
        self.assertIn("client", message.split("task client-task:", 1)[1])
        self.assertNotIn(CLIENT, message)

    def test_a_task_naming_a_client_repo_with_a_git_suffix_or_capitals_is_refused(self):
        self.fx.add_task("client-git", repo=CLIENT.upper() + ".git")
        message = self.assertRefused("task client-git:")
        self.assertIn("client", message.split("task client-git:", 1)[1])

    def test_the_client_check_on_repo_holds_even_when_the_name_scan_misses(self):
        with mock.patch.object(corpus_module._Matcher, "rules", return_value=[]):
            self.fx.add_task("client-task", repo=CLIENT)
            self.assertRefused("task client-task: the repository is on the client list")

    def test_a_repo_name_of_dots_is_refused(self):
        for repo in ["example-org/..", "example-org/.", "../example-repo"]:
            with self.subTest(repo=repo):
                self.fx.add_task("dots", repo=repo)
                self.assertRefused("task dots: repo is not an owner/name repository")

    def test_mixed_case_repo_matches_its_allowlist_entry(self):
        self.fx.snapshot(REPO.upper(), COMMIT)
        self.fx.add_task("parser-fix", repo=REPO.upper())
        self.assertEqual(len(self.fx.load()), 1)

    def test_a_missing_or_empty_client_list_refuses_the_build(self):
        self.fx.clients.unlink()
        self.assertRefused("client list")
        self.fx.clients.write_text("# no entries\n")
        self.assertRefused("client list has no entries")

    def test_a_bad_client_list_line_refuses_the_build(self):
        self.fx.clients.write_text("not a repo line\n")
        self.assertRefused("client list entry 1 is not an owner/name repository")

    def test_a_list_that_is_not_utf8_refuses_the_build(self):
        self.fx.clients.write_bytes(b"example-org/\xffclient\n")
        self.assertRefused("client list cannot be read")

    def test_a_missing_corpus_directory_refuses_the_build(self):
        self.fx.corpus = self.fx.root / "typo"
        self.assertRefused("the corpus directory does not exist")

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
                self.assertRefused("allowlist entry")

    def test_the_shipped_allowlist_names_no_repo(self):
        shipped = REPO_ROOT / "corpus" / "allowlist.txt"
        self.assertEqual(read_allowlist(shipped), {})
        self.assertIn("egress clearance", shipped.read_text())

    def test_codeowners_names_an_owner_for_the_allowlist(self):
        lines = (REPO_ROOT / ".github" / "CODEOWNERS").read_text().splitlines()
        entries = [line.split() for line in lines if line.strip() and not line.startswith("#")]
        self.assertIn("/corpus/allowlist.txt", [entry[0] for entry in entries])
        owners = [entry[1:] for entry in entries if entry[0] == "/corpus/allowlist.txt"][0]
        self.assertTrue(owners and all(owner.startswith("@") for owner in owners))

    def test_a_missing_or_empty_deny_terms_file_refuses_the_build(self):
        self.fx.deny.write_text("# none\n")
        self.assertRefused("deny terms file has no entries")
        self.fx.deny.unlink()
        self.assertRefused("deny terms file cannot be read")


class TaskShapeTests(CorpusTestCase):
    def test_a_bad_commit_is_refused(self):
        for commit in ["main", "HEAD", "a" * 39, "A" * 40]:
            with self.subTest(commit=commit):
                self.fx.add_task("bad-commit", commit=commit)
                self.assertRefused("task bad-commit: commit is not a full 40-hex or 64-hex commit")

    def test_a_64_hex_commit_and_tree_load(self):
        self.fx.snapshot(REPO, "e" * 64)
        self.fx.add_task("parser-fix", commit="e" * 64,
                         tree=git_tree_hash(self.fx.snapshots / REPO / ("e" * 64), "sha256"))
        self.assertEqual(self.fx.load()[0].commit, "e" * 64)

    def test_a_bad_tree_is_refused(self):
        for tree in [None, "main", "b" * 64]:
            with self.subTest(tree=tree):
                self.fx.add_task("bad-tree", tree=tree)
                self.assertRefused("task bad-tree: tree is not a full tree id")

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
        self.fx.add_task("extra", extra={"note": "anything"})
        self.assertRefused("task extra: task.json must hold exactly")

    def test_a_malformed_task_json_is_refused(self):
        task_dir = self.fx.add_task("broken")
        (task_dir / "task.json").write_text("{not json")
        self.assertRefused("task broken: task.json is missing or not JSON")

    def test_a_stray_file_in_tasks_is_refused(self):
        (self.fx.corpus / "tasks" / "notes.txt").write_text("x")
        self.assertRefused("an entry in tasks/ is not a task directory")

    def test_a_bad_task_id_is_refused(self):
        self.fx.add_task("Bad Id")
        self.assertRefused("a task id is not a lowercase slug")

    def test_a_task_id_that_is_a_finding_is_refused_and_not_shown(self):
        self.fx.add_task("client-site-fix")
        message = self.assertRefused("a task id failed the sanitizer, so it is not shown")
        self.assertNotIn("client-site", message)

    def test_every_bad_task_is_listed(self):
        self.fx.add_task("bad-one", commit="main")
        self.fx.add_task("bad-two", gates=[])
        message = self.assertRefused()
        self.assertIn("task bad-one:", message)
        self.assertIn("task bad-two:", message)

    def test_a_missing_snapshot_is_refused(self):
        self.fx.add_task("no-snapshot", commit="b" * 40)
        self.assertRefused("task no-snapshot: no snapshot exists at the frozen commit")


class TreeProofTests(CorpusTestCase):
    def git_write_tree(self, root, object_format="sha1"):
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        work = self.fx.root / f"git-{object_format}"
        shutil.copytree(root, work)
        env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", HOME=str(self.fx.root))
        subprocess.run(["git", "init", "-q", f"--object-format={object_format}", str(work)], check=True, env=env)
        subprocess.run(["git", "-C", str(work), "add", "-A"], check=True, env=env)
        result = subprocess.run(["git", "-C", str(work), "write-tree"], check=True, env=env,
                                capture_output=True, text=True)
        return result.stdout.strip()

    def test_git_tree_hash_matches_git_itself(self):
        root = self.fx.snapshot("example-org/tree-check", "c" * 40, {
            "a.b": "dot file\n", "a/inner.txt": "nested\n", "a-b.txt": "dash\n", "z/y/x.txt": "deep\n",
            "run.sh": "#!/bin/sh\n",
        })
        (root / "run.sh").chmod(0o755)
        self.assertEqual(git_tree_hash(root), self.git_write_tree(root))
        try:
            expected = self.git_write_tree(root, "sha256")
        except subprocess.CalledProcessError:
            self.skipTest("this git has no sha256 object format")
        self.assertEqual(git_tree_hash(root, "sha256"), expected)

    def test_a_snapshot_that_is_not_the_recorded_tree_is_refused(self):
        tree = git_tree_hash(self.fx.snapshots / REPO / COMMIT)
        self.fx.snapshot(REPO, COMMIT, {"src/other.py": "copied in later\n"})
        self.fx.add_task("parser-fix", tree=tree)
        self.assertRefused("task parser-fix: the snapshot is not the recorded tree of that commit")

    def test_an_empty_snapshot_is_refused(self):
        empty = self.fx.snapshots / REPO / ("d" * 40)
        empty.mkdir(parents=True)
        self.fx.add_task("empty", commit="d" * 40, tree="d" * 40)
        self.assertRefused("task empty: the snapshot is empty")


class ManifestTests(CorpusTestCase):
    def test_the_manifest_matches_the_scanned_bytes(self):
        task = self.fx.load()[0]
        check_manifest(task.snapshot_path, task.snapshot_manifest)
        entry = task.snapshot_manifest[0]
        self.assertEqual(entry.size, len("def parse(text):\n    return text\n"))

    def test_a_change_after_the_scan_is_caught(self):
        task = self.fx.load()[0]
        (task.snapshot_path / "src" / "parser.py").write_text("changed after the scan\n")
        with self.assertRaises(CorpusRefused):
            check_manifest(task.snapshot_path, task.snapshot_manifest)
        (task.snapshot_path / "src" / "parser.py").write_text("def parse(text):\n    return text\n")
        (task.snapshot_path / "extra.py").write_text("added\n")
        with self.assertRaises(CorpusRefused):
            check_manifest(task.snapshot_path, task.snapshot_manifest)

    def test_the_prompt_is_checked_before_it_is_read_again(self):
        task = self.fx.load()[0]
        task.prompt_path.write_text("a different prompt")
        with self.assertRaises(CorpusRefused):
            task.prompt_text()


class SnapshotScanTests(CorpusTestCase):
    def planted(self, files):
        self.fx.snapshot(REPO, "c" * 40, files)
        self.fx.add_task("planted", commit="c" * 40)
        return self.assertRefused("task planted:")

    def test_a_snapshot_with_a_planted_secret_fails(self):
        mixed = "Zk3x9Qw" + "ErTyUiOp" + "AsDfGhJk2Lm"
        for text in [
            "KEY = '" + AWS_KEY + "'\n",
            "TOKEN = '" + "gh" + "p_" + mixed + "abcdefghijklmn" + "'\n",
            "KEY = '" + "sk" + "-ant-" + "api03-" + mixed + "'\n",
            "-----BEGIN " + "RSA PRIVATE KEY" + "-----\nabc\n",
            "SLACK = '" + "xo" + "xb-" + "123456789012-abc'\n",
        ]:
            with self.subTest(text=text[:12]):
                self.fx.snapshot(REPO, "c" * 40, {"config.py": text})
                self.fx.add_task("planted", commit="c" * 40)
                message = self.assertRefused("task planted: the snapshot failed the secret scan")
                self.assertNotIn(mixed, message)

    def test_names_get_the_secret_and_email_scans(self):
        rules = self.rules({"keys/" + AWS_KEY + ".txt": "clean\n", "someone" + "@" + "example.com.md": "clean\n"})
        self.assertIn("secret:aws_access_key_in_path", rules)
        self.assertIn("sanitizer:email_address_in_path", rules)

    def test_a_secret_in_a_name_is_never_printed(self):
        message = self.planted({AWS_KEY + ".txt": "k = '" + AWS_KEY + "'\n"})
        self.assertNotIn(AWS_KEY, message)
        self.assertIn("a path not shown", message)

    def test_deny_terms_match_across_spacing_separators_and_lines(self):
        for text in ["notes on zz  private\tterm\n", "zz_private_term\n", "zzprivateterm\n",
                     "zz-Private-TERM\n", "notes on zz private\nterm here\n"]:
            with self.subTest(text=text):
                rules = self.rules({"a.md": text})
                self.assertTrue({"sanitizer:deny_term", "sanitizer:deny_term_across_lines"} & rules)

    def test_client_names_match_across_separators_but_only_as_whole_words(self):
        for text in ["see clientsite now\n", "see client_site now\n", "see Client  Site now\n",
                     "see client\nsite now\n"]:
            with self.subTest(text=text):
                rules = self.rules({"a.md": text})
                self.assertTrue({"sanitizer:client_repository", "sanitizer:client_repository_across_lines"} & rules)
        self.assertEqual(self.scan({"a.md": "my-client-sitemap is fine\n"}), [])

    def test_a_zero_width_character_cannot_hide_a_key(self):
        hidden = "AK" + "IA​" + "IOSFODNN7EXAMPLE"
        self.assertIn("secret:aws_access_key", self.rules({"a.py": "k = '" + hidden + "'\n"}))

    def test_folding_catches_width_and_accent_variants(self):
        for text in ["ＺＺ ＰＲＩＶＡＴＥ ＴＥＲＭ\n", "zz prívate term\n", "zz pri​vate term\n"]:
            with self.subTest(text=text):
                self.assertIn("sanitizer:deny_term", self.rules({"a.md": text}))

    def test_the_fold_order_strips_marks_after_decomposing(self):
        self.assertEqual(corpus_module._fold("É​COLE"), "ecole")

    def test_a_snapshot_with_an_email_address_fails(self):
        self.assertIn("sanitizer:email_address", self.rules({"a.py": "AUTHOR = 'someone" + "@" + "example.com'\n"}))

    def test_a_linked_directory_or_file_is_a_finding(self):
        outside = self.fx.root / "outside"
        outside.mkdir()
        (outside / "file.py").write_text("x = 1\n")
        root = self.fx.root / "scan"
        root.mkdir()
        (root / "linked-dir").symlink_to(outside, target_is_directory=True)
        (root / "linked-file.py").symlink_to(outside / "file.py")
        rules = sorted(f.rule for f in scan_snapshot(root, deny_terms=(), client_repos=frozenset()))
        self.assertEqual(rules, ["sanitizer:symlink", "sanitizer:symlink"])

    @unittest.skipUnless(hasattr(os, "mkfifo"), "no FIFOs here")
    def test_a_fifo_is_a_finding_and_does_not_hang(self):
        root = self.fx.root / "scan"
        root.mkdir()
        os.mkfifo(root / "pipe")
        rules = {f.rule for f in scan_snapshot(root, deny_terms=(), client_repos=frozenset())}
        self.assertEqual(rules, {"sanitizer:not_a_regular_file"})

    def test_git_metadata_is_a_finding_not_skipped(self):
        self.assertEqual(self.rules({".git/config": "[core]\n", "sub/.git": "gitdir: x\n"}),
                         {"sanitizer:git_metadata"})

    def test_a_file_too_large_or_not_utf8_is_a_finding(self):
        with mock.patch.object(corpus_module, "_MAX_SCAN_BYTES", 10):
            rules = self.rules({"big.txt": "x" * 11, "bin.dat": b"\xff\xfe\x00"})
        self.assertEqual(rules, {"sanitizer:file_too_large", "sanitizer:not_utf8_text"})

    def test_a_file_at_the_cap_is_read_in_full(self):
        with mock.patch.object(corpus_module, "_MAX_SCAN_BYTES", 10):
            self.assertEqual(self.scan({"exact.txt": "x" * 10}), [])

    def test_an_lfs_pointer_is_a_finding(self):
        pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 12\n"
        self.assertIn("sanitizer:lfs_pointer", self.rules({"model.bin": pointer}))

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

    def test_names_are_scanned_for_terms_and_clients(self):
        rules = self.rules({"docs/zz-private-term-notes.md": "clean\n", "client-site/readme.md": "clean\n"})
        self.assertIn("sanitizer:deny_term_in_path", rules)
        self.assertIn("sanitizer:client_repository_in_path", rules)

    def test_a_form_feed_cannot_split_a_key(self):
        self.assertIn("secret:aws_access_key", self.rules({"a.py": "k = '\x0c" + AWS_KEY + "'\n"}))

    def test_ordinary_hyphenated_words_are_not_keys(self):
        self.assertEqual(self.scan({"a.md": "task-runner-configuration and desk-lamp-organizer-settings\n"}), [])

    def test_more_key_formats_are_caught(self):
        for rule, text in [
            ("gitlab_token", "gl" + "pat-" + "abcdefghijklmnopqrst12"),
            ("stripe_key", "sk" + "_live_" + "abcdefghijklmnopqrstu1"),
            ("npm_token", "np" + "m_" + "a" * 36),
            ("jwt", "ey" + "J" + "a" * 12 + ".ey" + "J" + "b" * 12 + ".sig"),
            ("url_with_password", "postgres://user:" + "hunter2" + "@db.example.com/x"),
            ("huggingface_token", "h" + "f_" + "a" * 34),
            ("aws_secret_key", "aws_secret_access_key = '" + "a" * 20 + "/" + "B" * 19 + "'"),
            ("password_assignment", "pass" + "word = '" + "hunter22" + "'"),
            ("google_service_account", '{"type": "service' + '_account", "project_id": "x"}'),
        ]:
            with self.subTest(rule=rule):
                self.assertIn(f"secret:{rule}", self.rules({"a.py": text + "\n"}))

    def test_secret_file_names_match_in_any_case(self):
        for name in [".ENV", "ID_RSA", "KEY.PEM", ".npmrc", ".netrc", "cert.pfx", ".git-credentials",
                     "credentials", "main.tfstate", ".htpasswd", "app.keystore"]:
            with self.subTest(name=name):
                root = self.fx.root / f"scan-{name.lower().replace('.', '_')}"
                root.mkdir()
                (root / name).write_text("x\n")
                rules = {f.rule for f in scan_snapshot(root, deny_terms=(), client_repos=frozenset())}
                self.assertIn("secret:secret_file_name", rules)

    def test_the_prompt_is_scanned_too(self):
        self.fx.add_task("secret-prompt", prompt_text="use " + AWS_KEY)
        self.assertRefused("task secret-prompt: its own files failed the secret scan")

    def test_the_message_names_rule_file_and_line_but_hides_a_bad_path(self):
        message = self.planted({"cfg.py": "x = 1\nk = '" + AWS_KEY + "'\n", "zz-private-term.md": "clean\n"})
        self.assertIn("secret:aws_access_key at cfg.py:2", message)
        self.assertIn("a path not shown", message)
        self.assertNotIn("private-term", message.lower())

    def test_a_clean_snapshot_has_no_findings(self):
        root = self.fx.snapshots / REPO / COMMIT
        self.assertEqual(scan_snapshot(root, deny_terms=("zz private term",), client_repos=frozenset({CLIENT})), [])

    def test_client_list_entries_are_lowercased(self):
        self.fx.clients.write_text("Example-Org/Client-Site\n")
        self.assertEqual(read_client_list(self.fx.clients), frozenset({"example-org/client-site"}))


if __name__ == "__main__":
    unittest.main()
