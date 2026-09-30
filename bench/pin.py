"""The reproducibility pin for every model-fit run (issue #2).

A later run is comparable with an earlier one only if both ran on the
same inputs. Section 6.5 item 8 of "Threat model and eval spec: Vines
connectors and the model-fit bench" lists them: the model id and
version, the quantization, the prompt, the seed, the harness commit, and
the corpus commit. The pin also records the weights digest, the runtime
and client library versions, and the sampling settings. A run missing
any of them is not counted, so require_pin fails before the run starts.

model_version is a 40-hex revision or a 64-hex digest, never a moving
name such as latest or main, and weights_sha256 is the SHA-256 of the
weights file or the runtime's manifest digest. verify_pin checks a pin
against what is actually on disk; the replay harness (bench item 3,
trellis-workbench#6) calls it before the spend guard.

The record holds the prompt's id in the corpus and the SHA-256 of its
text, never the text. The corpus commit plus the prompt id recovers the
prompt, and the hash proves it is the same one. So no prompt text,
client data, or credential enters the record: every field is checked by
its shape, a value shaped like a credential is refused, any other field
is refused, and an error names pin fields only, never a value or an
unknown field's name.

The prompt hash covers the prompt text byte for byte, so a change in
line endings is a different prompt. Checking that prompt_id exists in
the corpus at corpus_commit is the corpus's job (bench item 2), not the
pin's.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

SCHEMA = 1

PIN_FIELDS = (
    "model_id",
    "model_version",
    "weights_sha256",
    "quantization",
    "runtime_name",
    "runtime_version",
    "client_version",
    "temperature",
    "top_p",
    "max_output_tokens",
    "prompt_id",
    "prompt_sha256",
    "seed",
    "harness_commit",
    "corpus_commit",
)

_MOVING_VERSION_NAMES = frozenset({"latest", "main", "master", "head"})

# Up to three parts split by `/`, such as a host, an organization, and a
# model with a `:tag` (an Ollama tag or a quantization suffix).
_MODEL_ID_RE = re.compile(r"(?=.{1,128}\Z)[A-Za-z0-9][A-Za-z0-9._:-]*(?:/[A-Za-z0-9][A-Za-z0-9._:-]*){0,2}")
# A 40-hex revision or a 64-hex digest, never a moving name.
_MODEL_VERSION_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_RUNTIME_NAME_RE = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_SOFTWARE_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")
_QUANTIZATION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}")
_PROMPT_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SEED_MAX = 2**63 - 1
# Shapes of common credentials: known key prefixes, and a long run of
# letters and digits that mixes upper case, lower case, and digits.
_SECRET_PREFIX_RE = re.compile(r"(?i)(?:^|[^a-z0-9])(?:sk-|sk_|ghp_|gho_|ghs_|github_pat_|xox[abprs]-|akia|eyj)")
_LONG_RUN_RE = re.compile(r"[A-Za-z0-9]{20,}")


class PinInvalid(ValueError):
    """A pin is missing a field, has one of the wrong shape, or has one it
    should not. The message names fields, never a value."""


class PinMismatch(ValueError):
    """A pin does not match what is on disk. The message names fields,
    never a value."""


def _looks_like_a_credential(value: str) -> bool:
    if _SECRET_PREFIX_RE.search(value):
        return True
    return any(
        re.search(r"[A-Z]", run) and re.search(r"[a-z]", run) and re.search(r"[0-9]", run)
        for run in _LONG_RUN_RE.findall(value)
    )


def _matches(pattern):
    return lambda value: (
        isinstance(value, str)
        and pattern.fullmatch(value) is not None
        and not _looks_like_a_credential(value)
    )


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _is_prompt_id(value) -> bool:
    if not isinstance(value, str) or not 0 < len(value) <= 256 or _looks_like_a_credential(value):
        return False
    return all(_PROMPT_SEGMENT_RE.fullmatch(segment) for segment in value.split("/"))


def _is_seed(value) -> bool:
    return type(value) is int and 0 <= value <= _SEED_MAX


def _is_model_version(value) -> bool:
    if isinstance(value, str) and value.lower() in _MOVING_VERSION_NAMES:
        return False
    return isinstance(value, str) and _MODEL_VERSION_RE.fullmatch(value) is not None


def _is_number_in(low, high, low_open=False):
    def check(value) -> bool:
        if type(value) not in (int, float) or not math.isfinite(value):
            return False
        above = value > low if low_open else value >= low
        return above and value <= high

    return check


def _is_output_limit(value) -> bool:
    return type(value) is int and 1 <= value <= 10_000_000


_CHECKS = {
    "model_id": _matches(_MODEL_ID_RE),
    "model_version": _is_model_version,
    "weights_sha256": _matches(_SHA256_RE),
    "runtime_name": _matches(_RUNTIME_NAME_RE),
    "runtime_version": _matches(_SOFTWARE_VERSION_RE),
    "client_version": _matches(_SOFTWARE_VERSION_RE),
    "temperature": _is_number_in(0, 2),
    "top_p": _is_number_in(0, 1, low_open=True),
    "max_output_tokens": _is_output_limit,
    "quantization": _matches(_QUANTIZATION_RE),
    "prompt_id": _is_prompt_id,
    "prompt_sha256": _matches(_SHA256_RE),
    "seed": _is_seed,
    "harness_commit": _matches(_COMMIT_RE),
    "corpus_commit": _matches(_COMMIT_RE),
}


def prompt_sha256(prompt_text: str) -> str:
    """The SHA-256 of a prompt's text, as 64 lowercase hex characters."""
    if not isinstance(prompt_text, str) or not prompt_text:
        raise PinInvalid("prompt text must be a non-empty string")
    try:
        encoded = prompt_text.encode("utf-8")
    except UnicodeEncodeError:
        raise PinInvalid("prompt text is not valid Unicode") from None
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RunPin:
    model_id: str
    model_version: str
    weights_sha256: str
    quantization: str
    runtime_name: str
    runtime_version: str
    client_version: str
    temperature: float
    top_p: float
    max_output_tokens: int
    prompt_id: str
    prompt_sha256: str
    seed: int
    harness_commit: str
    corpus_commit: str

    def __post_init__(self):
        missing = [name for name in PIN_FIELDS if _is_missing(getattr(self, name))]
        if missing:
            raise PinInvalid(f"the pin is missing: {missing}")
        bad = [name for name in PIN_FIELDS if not _CHECKS[name](getattr(self, name))]
        if bad:
            raise PinInvalid(f"pin fields with a value of the wrong shape: {bad}")

    def record(self) -> dict:
        """The run's pin record: the schema version and every pin field."""
        return {"schema": SCHEMA, **asdict(self)}

    def record_json(self) -> str:
        """The record as canonical JSON, so equal pins give equal text."""
        return json.dumps(self.record(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """The SHA-256 of the canonical record: one id for these inputs."""
        return hashlib.sha256(self.record_json().encode("utf-8")).hexdigest()


def require_pin(fields: dict) -> RunPin:
    """Build the pin a run needs before it starts, or raise PinInvalid.

    fields must hold exactly the pin fields. A missing, empty, or None
    field fails, and so does any other field.
    """
    if not isinstance(fields, dict):
        raise PinInvalid("the pin fields must be a mapping of field names to values")
    extra = [name for name in fields if name not in PIN_FIELDS]
    if extra:
        # An unknown field's name is never printed: a name can itself be
        # client data or a credential.
        raise PinInvalid(f"{len(extra)} field(s) that are not pin fields")
    missing = [name for name in PIN_FIELDS if _is_missing(fields.get(name))]
    if missing:
        raise PinInvalid(f"the pin is missing: {missing}")
    return RunPin(**{name: fields[name] for name in PIN_FIELDS})


def verify_pin(pin: RunPin, harness_head: str, corpus_head: str, prompt_text: str, weights_digest: str) -> None:
    """Check a pin against what is actually on disk, or raise PinMismatch
    naming each field that differs.

    Pure: it reads nothing itself. The replay harness (bench item 3,
    trellis-workbench#6) reads the harness and corpus heads, the prompt
    file at the corpus commit, and the weights digest, then calls this
    before it calls the spend guard, so a run never spends on inputs
    that differ from its pin.
    """
    if not isinstance(pin, RunPin):
        raise PinMismatch("there is no pin to verify")
    try:
        prompt_digest = prompt_sha256(prompt_text)
    except PinInvalid:
        prompt_digest = None
    mismatched = [
        name
        for name, actual in (
            ("harness_commit", harness_head),
            ("corpus_commit", corpus_head),
            ("prompt_sha256", prompt_digest),
            ("weights_sha256", weights_digest),
        )
        if getattr(pin, name) != actual
    ]
    if mismatched:
        raise PinMismatch(f"the pin does not match what is on disk: {mismatched}")
