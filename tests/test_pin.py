"""Tests for the reproducibility pin (issue #2).

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench.pin import (  # noqa: E402
    PIN_FIELDS,
    PinInvalid,
    PinMismatch,
    RunPin,
    prompt_sha256,
    provider_snapshot_digest,
    require_pin,
    verify_pin,
)

HARNESS = "a" * 40
CORPUS = "b" * 40
REVISION = "c" * 40
PROVIDER = "example-provider"
SNAPSHOT = "example-model-20260930"
SNAPSHOT_DIGEST = provider_snapshot_digest(PROVIDER, SNAPSHOT)
NUMERIC_FIELDS = {"seed", "temperature", "top_p", "max_output_tokens"}
WEIGHTS = "d" * 64
PROMPT_TEXT = "Fix the failing test in the parser module."
# The digest of the pin fields() builds, frozen so a change to the record's
# key order, separators, or schema shows up as a failure.
GOLDEN_DIGEST = "e8527cf56cc016c53b80d424388bf4d105e57a453a2e451dddbe31d46b38ea2f"


def fields(**overrides):
    """An honest hosted-model pin: a provider_snapshot whose version and
    weights digest are the SHA-256 of "<provider>|<snapshot id>"."""
    base = {
        "model_id": SNAPSHOT,
        "model_version": SNAPSHOT_DIGEST,
        "weights_sha256": SNAPSHOT_DIGEST,
        "weights_kind": "provider_snapshot",
        "quantization": "none",
        "runtime_name": PROVIDER,
        "runtime_version": "0.6.3",
        "client_version": "1.51.0",
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 4096,
        "prompt_id": "tasks/parser-fix/prompt.md",
        "prompt_sha256": prompt_sha256(PROMPT_TEXT),
        "seed": 42,
        "harness_commit": HARNESS,
        "corpus_commit": CORPUS,
    }
    base.update(overrides)
    return base


def file_fields(**overrides):
    """A local-weights pin: a fixed revision and the weights file digest."""
    base = fields(weights_kind="file", model_id="org/local-model", model_version=REVISION,
                  weights_sha256=WEIGHTS, runtime_name="vllm")
    base.update(overrides)
    return base


class CompletePinTests(unittest.TestCase):
    def test_a_complete_pin_records_every_field(self):
        pin = require_pin(fields())
        record = pin.record()
        self.assertEqual(set(record), set(PIN_FIELDS) | {"schema"})
        self.assertEqual(record["model_id"], SNAPSHOT)
        self.assertEqual(record["seed"], 42)
        self.assertEqual(record["harness_commit"], HARNESS)

    def test_two_runs_with_the_same_pin_record_the_same_inputs(self):
        first = require_pin(fields())
        second = require_pin(dict(reversed(list(fields().items()))))
        self.assertEqual(first.record_json(), second.record_json())
        self.assertEqual(first.digest(), second.digest())

    def test_swapped_values_change_the_digest(self):
        base = require_pin(fields()).digest()
        swapped = require_pin(fields(harness_commit=CORPUS, corpus_commit=HARNESS)).digest()
        self.assertNotEqual(base, swapped)

    def test_a_changed_input_changes_the_digest(self):
        base = require_pin(fields()).digest()
        for name, value in [("seed", 43), ("quantization", "q4_k_m"), ("corpus_commit", "c" * 40),
                            ("prompt_sha256", prompt_sha256(PROMPT_TEXT + "!"))]:
            with self.subTest(field=name):
                self.assertNotEqual(require_pin(fields(**{name: value})).digest(), base)

    def test_the_record_holds_the_prompt_hash_never_the_prompt(self):
        record_json = require_pin(fields()).record_json()
        self.assertNotIn(PROMPT_TEXT, record_json)
        self.assertIn(prompt_sha256(PROMPT_TEXT), record_json)

    def test_the_record_and_digest_do_not_drift(self):
        pin = require_pin(fields())
        record_json = pin.record_json()
        self.assertTrue(record_json.startswith('{"client_version":"1.51.0","corpus_commit":"'))
        self.assertTrue(record_json.endswith(
            '"temperature":0.0,"top_p":1.0,"weights_kind":"provider_snapshot","weights_sha256":"'
            + SNAPSHOT_DIGEST + '"}'
        ))
        self.assertEqual(pin.digest(), GOLDEN_DIGEST)

    def test_the_record_is_canonical_json(self):
        record_json = require_pin(fields()).record_json()
        self.assertEqual(record_json, json.dumps(json.loads(record_json), sort_keys=True, separators=(",", ":")))

    def test_real_shaped_values_pass(self):
        for name, value in [
            ("model_id", "meta-llama/llama-3.1-70b-instruct"),
            ("model_id", "qwen/qwen3-coder:free"),
            ("model_id", "llama3.1:8b"),
            ("model_id", "hf.co/org/model:Q4_K_M"),
            ("model_version", "e" * 64),
            ("weights_kind", "manifest"),
            ("runtime_version", "1.0+build5"),
            ("temperature", 0.7),
            ("top_p", 0.95),
            ("quantization", "q4_k_m"),
            ("quantization", "Q4_K_M"),
            ("quantization", "IQ4_XS"),
            ("quantization", "BF16"),
            ("quantization", "fp16"),
            ("seed", 0),
            ("harness_commit", "0123456789abcdef" * 4),
        ]:
            with self.subTest(field=name, value=value):
                require_pin(file_fields(**{name: value}))


class MissingOrBadTests(unittest.TestCase):
    def test_a_run_with_any_field_missing_fails_before_it_starts(self):
        for name in PIN_FIELDS:
            with self.subTest(field=name):
                partial = fields()
                del partial[name]
                with self.assertRaises(PinInvalid) as ctx:
                    require_pin(partial)
                self.assertIn(name, str(ctx.exception))

    def test_an_empty_or_none_field_counts_as_missing(self):
        for name in PIN_FIELDS:
            with self.subTest(field=name):
                with self.assertRaises(PinInvalid):
                    require_pin(fields(**{name: None}))
                if name != "seed":
                    with self.assertRaises(PinInvalid):
                        require_pin(fields(**{name: ""}))

    def test_a_value_of_the_wrong_shape_is_refused_and_never_echoed(self):
        secret = "someone@example.com"
        for name in PIN_FIELDS:
            if name == "seed":
                continue
            with self.subTest(field=name):
                with self.assertRaises(PinInvalid) as ctx:
                    require_pin(fields(**{name: secret}))
                self.assertIn(name, str(ctx.exception))
                self.assertNotIn(secret, str(ctx.exception))

    def test_prompt_text_is_refused_in_place_of_its_hash(self):
        with self.assertRaises(PinInvalid):
            require_pin(fields(prompt_sha256=PROMPT_TEXT))

    def test_bad_commits_are_refused(self):
        for value in ["a" * 39, "A" * 40, "g" * 40, "a" * 41, "HEAD", "main"]:
            with self.subTest(value=value):
                with self.assertRaises(PinInvalid):
                    require_pin(fields(harness_commit=value))

    def test_bad_seeds_are_refused(self):
        for value in [-1, 1.5, "42", True, 2**64]:
            with self.subTest(value=value):
                with self.assertRaises(PinInvalid) as ctx:
                    require_pin(fields(seed=value))
                self.assertIn("seed", str(ctx.exception))

    def test_non_string_values_are_refused(self):
        for name in PIN_FIELDS:
            if name in NUMERIC_FIELDS:
                continue
            for value in [5, ["tasks/x.md"], b"abc"]:
                with self.subTest(field=name, value=value):
                    with self.assertRaises(PinInvalid):
                        require_pin(fields(**{name: value}))

    def test_length_bounds(self):
        require_pin(file_fields(model_id="m" * 128))
        require_pin(fields(prompt_id="p" * 256))
        for name, value in [("model_id", "m" * 129), ("prompt_id", "p" * 257),
                            ("model_version", "v" * 65), ("quantization", "q" * 33)]:
            with self.subTest(field=name):
                with self.assertRaises(PinInvalid):
                    require_pin(fields(**{name: value}))

    def test_a_trailing_newline_is_refused(self):
        for name, value in [("harness_commit", HARNESS + "\n"), ("model_id", "claude-opus-5-5\n"),
                            ("prompt_sha256", prompt_sha256(PROMPT_TEXT) + "\n")]:
            with self.subTest(field=name):
                with self.assertRaises(PinInvalid):
                    require_pin(fields(**{name: value}))

    def test_a_prompt_that_is_not_valid_unicode_is_refused(self):
        with self.assertRaises(PinInvalid):
            prompt_sha256("bad \ud800 surrogate")

    def test_a_prompt_id_is_a_relative_path_inside_the_corpus(self):
        for value in ["/etc/passwd", "../outside.md", "tasks/../../x.md", "tasks/a b.md", "tasks/",
                      "tasks/.../x.md", ".git/config", "tasks//x.md"]:
            with self.subTest(value=value):
                with self.assertRaises(PinInvalid):
                    require_pin(fields(prompt_id=value))

    def test_any_other_field_is_refused_and_its_value_never_echoed(self):
        for name in ["prompt", "api_key", "client", "user.email"]:
            with self.subTest(field=name):
                with self.assertRaises(PinInvalid) as ctx:
                    require_pin(fields(**{name: "private value"}))
                self.assertNotIn("private value", str(ctx.exception))

    def test_an_unknown_field_name_is_never_printed(self):
        for name in ["someone@example.com", "acme_corp_q3", "a" * 64]:
            with self.subTest(name=name):
                with self.assertRaises(PinInvalid) as ctx:
                    require_pin(fields(**{name: 1}))
                self.assertEqual(str(ctx.exception), "1 field(s) that are not pin fields")

    def test_a_credential_shaped_value_is_refused_and_never_echoed(self):
        # Built from pieces, so no whole key-shaped string sits in the source.
        mixed_run = "Zk3x9Qw" + "ErTyUiOp" + "AsDfGhJk2"
        for name, value in [
            ("model_id", "sk" + "-ant-" + "api03-" + mixed_run),
            ("runtime_version", "gh" + "p_" + "1A2b3C4d5E6f7G8h9I0j"),
            ("client_version", "AK" + "IA" + "IOSFODNN7EXAMPLE"),
            ("quantization", "ey" + "j" + "hbgcioijiuzi1nij9"),
            ("prompt_id", "tasks/" + mixed_run + "/prompt.md"),
        ]:
            with self.subTest(field=name, value=value):
                with self.assertRaises(PinInvalid) as ctx:
                    require_pin(fields(**{name: value}))
                self.assertEqual(str(ctx.exception), f"pin fields with a value of the wrong shape: ['{name}']")
                self.assertNotIn(value, str(ctx.exception))

    def test_real_model_names_are_not_taken_for_credentials(self):
        for model_id in ["Meta-Llama-3.1-70B-Instruct", "claude-3-5-sonnet-20241022", "gpt-4.1-mini"]:
            with self.subTest(model_id=model_id):
                require_pin(file_fields(model_id=model_id))

    def test_a_moving_model_version_is_refused(self):
        for value in ["latest", "LATEST", "main", "Master", "HEAD", "2026-09-01", "v1.0"]:
            with self.subTest(value=value):
                with self.assertRaises(PinInvalid) as ctx:
                    require_pin(file_fields(model_version=value))
                self.assertIn("model_version", str(ctx.exception))

    def test_the_weights_digest_is_required(self):
        for value in [None, "", "d" * 63, "D" * 64, "latest"]:
            with self.subTest(value=value):
                with self.assertRaises(PinInvalid):
                    require_pin(fields(weights_sha256=value))

    def test_sampling_settings_are_bounded(self):
        for name, value in [("temperature", -0.1), ("temperature", 2.1), ("temperature", float("nan")),
                            ("temperature", True), ("top_p", 0), ("top_p", 1.1), ("top_p", "1"),
                            ("max_output_tokens", 0), ("max_output_tokens", 1.5), ("max_output_tokens", True)]:
            with self.subTest(field=name, value=value):
                with self.assertRaises(PinInvalid):
                    require_pin(fields(**{name: value}))

    def test_zero_temperature_in_any_form_gives_one_digest(self):
        digests = {require_pin(fields(temperature=value, top_p=1)).digest() for value in (0, 0.0, -0.0)}
        self.assertEqual(len(digests), 1)
        self.assertEqual(require_pin(fields(top_p=1)).digest(), require_pin(fields(top_p=1.0)).digest())

    def test_a_provider_snapshot_needs_a_dated_snapshot_id(self):
        for model_id in ["example-model", "example-model-latest", "example-model-2026", "example-model-20261340"]:
            with self.subTest(model_id=model_id):
                digest = provider_snapshot_digest(PROVIDER, model_id)
                with self.assertRaises(PinInvalid) as ctx:
                    require_pin(fields(model_id=model_id, model_version=digest, weights_sha256=digest))
                self.assertIn("dated snapshot", str(ctx.exception))

    def test_a_provider_snapshot_digest_must_match_the_rule(self):
        for name in ["model_version", "weights_sha256"]:
            with self.subTest(field=name):
                with self.assertRaises(PinInvalid) as ctx:
                    require_pin(fields(**{name: "e" * 64}))
                self.assertIn(name, str(ctx.exception))
        other = provider_snapshot_digest("another-provider", SNAPSHOT)
        with self.assertRaises(PinInvalid):
            require_pin(fields(model_version=other, weights_sha256=other))

    def test_different_weights_kinds_never_compare_equal(self):
        file_pin = require_pin(file_fields())
        manifest_pin = require_pin(file_fields(weights_kind="manifest"))
        self.assertNotEqual(file_pin.digest(), manifest_pin.digest())
        self.assertNotEqual(file_pin, manifest_pin)

    def test_an_unknown_weights_kind_is_refused(self):
        for value in ["File", "snapshot", "", None]:
            with self.subTest(value=value):
                with self.assertRaises(PinInvalid):
                    require_pin(fields(weights_kind=value))

    def test_a_non_mapping_is_refused(self):
        for value in [None, [], ["model_id"], "model_id"]:
            with self.subTest(value=value):
                with self.assertRaises(PinInvalid):
                    require_pin(value)

    def test_seed_zero_is_valid_and_false_is_not(self):
        self.assertEqual(require_pin(fields(seed=0)).seed, 0)
        with self.assertRaises(PinInvalid):
            require_pin(fields(seed=False))

    def test_a_missing_field_message_is_exact(self):
        partial = fields()
        del partial["model_id"]
        with self.assertRaises(PinInvalid) as ctx:
            require_pin(partial)
        self.assertEqual(str(ctx.exception), "the pin is missing: ['model_id']")

    def test_a_pin_cannot_be_built_around_the_check(self):
        with self.assertRaises(PinInvalid):
            RunPin(**fields(seed=-1))
        with self.assertRaises(PinInvalid):
            RunPin(**fields(model_id=""))

    def test_prompt_sha256_is_the_hex_digest_of_the_text(self):
        self.assertEqual(
            prompt_sha256("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        with self.assertRaises(PinInvalid):
            prompt_sha256("")



class VerifyPinTests(unittest.TestCase):
    def setUp(self):
        self.pin = require_pin(fields())

    def disk(self, **overrides):
        values = dict(
            harness_head=HARNESS,
            corpus_head=CORPUS,
            prompt_text=PROMPT_TEXT,
            weights_digest=SNAPSHOT_DIGEST,
            model_version=SNAPSHOT_DIGEST,
            runtime_version="0.6.3",
            client_version="1.51.0",
        )
        values.update(overrides)
        return values

    def test_a_pin_that_matches_the_disk_passes(self):
        verify_pin(self.pin, **self.disk())

    def test_each_mismatch_is_named(self):
        for name, overrides in [
            ("harness_commit", {"harness_head": "e" * 40}),
            ("corpus_commit", {"corpus_head": "e" * 40}),
            ("prompt_sha256", {"prompt_text": PROMPT_TEXT + "!"}),
            ("weights_sha256", {"weights_digest": "e" * 64}),
            ("model_version", {"model_version": "e" * 64}),
            ("runtime_version", {"runtime_version": "0.6.4"}),
            ("client_version", {"client_version": "1.52.0"}),
        ]:
            with self.subTest(field=name):
                with self.assertRaises(PinMismatch) as ctx:
                    verify_pin(self.pin, **self.disk(**overrides))
                self.assertEqual(str(ctx.exception), f"the pin does not match what is on disk: ['{name}']")

    def test_an_empty_prompt_is_a_mismatch_not_a_crash(self):
        with self.assertRaises(PinMismatch):
            verify_pin(self.pin, **self.disk(prompt_text=""))

    def test_no_pin_is_a_mismatch(self):
        with self.assertRaises(PinMismatch):
            verify_pin(None, **self.disk())


if __name__ == "__main__":
    unittest.main()
