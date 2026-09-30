"""The bench spend guard (issue #1).

Every model-fit run spends money, and spend is the founder's decision.
This guard follows the threat model's spend section, section 6.4 of
docs/specs/2026-09-30-vines-connectors-and-bench-threat-model.md in
sageadvicellc/workbench:

1. Before any run, it prints a cost estimate: the run count times the
   price per run, with the price source named.
2. The run starts only after the founder types an exact "yes" at a
   terminal. No parameter, flag, or environment variable skips it.
3. A hard cap on each run, enforced three ways: a token ceiling on each
   request, a running tally of spend, and a stop before any request that
   could pass the cap. The founder sets the cap value; this module has
   no default for it.
4. Any hosted GPU the run started is torn down at the end, on a stop at
   the cap, and on any error. A run that cannot confirm the teardown is
   reported as failed.

Money is whole micro-dollars (millionths of a US dollar), never a float.
The guard holds no credential and reads none. An endpoint's error text is
never printed or recorded, only its type, so a key in an error message
cannot leak through the guard.
"""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable, Optional, Protocol, TextIO

APPROVAL_WORD = "yes"


def _is_whole(value) -> bool:
    """A plain int only: never a bool, a float, or an int subclass that
    could change its value between checks."""
    return type(value) is int


def format_usd(micros: int) -> str:
    """Micro-dollars as dollars and cents, rounded down to the cent."""
    cents = micros // 10_000
    return f"${cents // 100}.{cents % 100:02d}"


@dataclass(frozen=True)
class PriceSource:
    """Where a price came from, so an estimate always names it."""

    name: str
    reference: str
    read_on: date

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a price source needs a name")
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("a price source needs a reference")
        if not isinstance(self.read_on, date):
            raise ValueError("a price source needs the date it was read")


@dataclass(frozen=True)
class Estimate:
    """The run count times the price per run."""

    run_count: int
    price_per_run_micros: int
    source: PriceSource

    def __post_init__(self):
        if not _is_whole(self.run_count) or self.run_count < 1:
            raise ValueError("run_count must be a whole number of 1 or more")
        if not _is_whole(self.price_per_run_micros) or self.price_per_run_micros < 0:
            raise ValueError("price_per_run_micros must be a whole number of 0 or more")
        if not isinstance(self.source, PriceSource):
            raise ValueError("an estimate needs a price source")

    @property
    def total_micros(self) -> int:
        return self.run_count * self.price_per_run_micros

    def describe(self) -> str:
        runs = "run" if self.run_count == 1 else "runs"
        return (
            f"Estimate: {self.run_count} {runs} x {format_usd(self.price_per_run_micros)} per run"
            f" = {format_usd(self.total_micros)}."
            f" Price source: {self.source.name} ({self.source.reference}),"
            f" read on {self.source.read_on.isoformat()}."
        )


@dataclass(frozen=True)
class Request:
    """One model request: its token ceiling, and the most it can cost."""

    max_tokens: int
    max_cost_micros: int


@dataclass(frozen=True)
class Response:
    input_tokens: int
    output_tokens: int
    cost_micros: int


class TokenCeilingExceeded(Exception):
    """A request asks for more tokens than the per-request ceiling."""


class CapReached(Exception):
    """A request could pass the cap, or spend has passed it."""


class CostUnknown(Exception):
    """A request or response carries no valid cost, so spend is unknown."""


class SpendTally:
    """The running tally for one run, against its hard cap."""

    def __init__(self, cap_micros: int, max_tokens_per_request: int):
        if not _is_whole(cap_micros) or cap_micros < 1:
            raise ValueError("cap_micros must be a whole number of 1 or more")
        if not _is_whole(max_tokens_per_request) or max_tokens_per_request < 1:
            raise ValueError("max_tokens_per_request must be a whole number of 1 or more")
        self.cap_micros = cap_micros
        self.max_tokens_per_request = max_tokens_per_request
        self.spent_micros = 0

    def reserve(self, request: Request) -> None:
        """Refuse a request before it is sent if it breaks the ceiling or
        could pass the cap at its most expensive."""
        if not _is_whole(request.max_tokens) or request.max_tokens < 1:
            raise TokenCeilingExceeded("a request has no valid token ceiling")
        if request.max_tokens > self.max_tokens_per_request:
            raise TokenCeilingExceeded("a request asked for more than the per-request token ceiling")
        if not _is_whole(request.max_cost_micros) or request.max_cost_micros < 1:
            raise CostUnknown("a request has no valid cost bound, so the cap cannot be held")
        if self.spent_micros + request.max_cost_micros > self.cap_micros:
            raise CapReached("the next request could pass the hard cap")

    def settle(self, request: Request, response: Response) -> None:
        """Add a response's cost to the tally, then stop the run if it cost
        more than was reserved or spend has passed the cap.

        The tally counts the larger of the reported cost and the amount
        reserved, so an endpoint that reports too little can never lower
        it. The first response that costs more than its reservation is
        already paid for when it is seen; it is recorded, and the run
        stops there.
        """
        if not _is_whole(response.cost_micros) or response.cost_micros < 0:
            self.spent_micros += request.max_cost_micros
            raise CostUnknown("a response reported no valid cost")
        self.spent_micros += max(response.cost_micros, request.max_cost_micros)
        if not _is_whole(response.output_tokens) or response.output_tokens < 0:
            raise CostUnknown("a response reported no valid token count")
        if response.output_tokens > request.max_tokens:
            raise CapReached("a response used more tokens than its request's ceiling")
        if response.cost_micros > request.max_cost_micros:
            raise CapReached("a response cost more than the amount reserved for it")
        if self.spent_micros > self.cap_micros:
            raise CapReached("spend passed the hard cap")


class GpuLease(Protocol):
    def start(self) -> None: ...

    def teardown(self) -> bool:
        """Release the hosted GPU. Return exactly True only when the
        release is confirmed; any other value counts as unconfirmed."""
        ...


class RunStatus(enum.Enum):
    REFUSED = "refused"
    COMPLETED = "completed"
    STOPPED_AT_CAP = "stopped_at_cap"
    FAILED = "failed"


@dataclass
class RunResult:
    status: RunStatus
    reason: str
    cap_micros: int
    estimate: Estimate
    spent_micros: int = 0
    requests_sent: int = 0
    teardown_confirmed: Optional[bool] = None
    notes: list = field(default_factory=list)

    def as_record(self) -> dict:
        """The fields a run records, for the bench's run log."""
        return {
            "status": self.status.value,
            "reason": self.reason,
            "cap_micros": self.cap_micros,
            "spent_micros": self.spent_micros,
            "requests_sent": self.requests_sent,
            "teardown_confirmed": self.teardown_confirmed,
            "estimate_total_micros": self.estimate.total_micros,
            "price_source": self.estimate.source.name,
        }


def _stdin_is_tty() -> bool:
    return sys.stdin.isatty()


def _read_answer(prompt: str) -> str:
    return input(prompt)


def run_guarded(
    *,
    estimate: Estimate,
    cap_micros: int,
    max_tokens_per_request: int,
    requests: Iterable[Request],
    endpoint: Callable[[Request], Response],
    out: TextIO,
    gpu: Optional[GpuLease],
) -> RunResult:
    """Print the estimate, get the founder's yes, then run the requests
    under the hard cap and tear down any GPU.

    The yes is read from standard input, only when standard input is a
    terminal. No argument approves on the founder's behalf. The requests
    are read in full before the founder is asked, so the run cannot
    change them after the yes.
    """
    tally = SpendTally(cap_micros, max_tokens_per_request)
    planned = list(requests)
    result = RunResult(status=RunStatus.REFUSED, reason="", cap_micros=cap_micros, estimate=estimate)

    out.write(estimate.describe() + "\n")
    out.write(
        f"Hard cap: {format_usd(cap_micros)} per run."
        f" Token ceiling: {max_tokens_per_request} per request.\n"
    )
    if estimate.price_per_run_micros > cap_micros:
        out.write("Note: the price per run is above the hard cap, so the run can stop at the cap.\n")
    out.flush()

    if not _stdin_is_tty():
        result.reason = "the yes must be typed at a terminal, and no terminal is attached"
        out.write(f"Refused: {result.reason}.\n")
        return result
    try:
        answer = _read_answer(f'Type "{APPROVAL_WORD}" to approve this spend: ')
    except EOFError:
        answer = ""
    if answer != APPROVAL_WORD:
        result.reason = f'the founder did not type "{APPROVAL_WORD}"'
        out.write(f"Refused: {result.reason}.\n")
        return result

    result.status = RunStatus.COMPLETED
    result.reason = "every request finished under the cap"
    gpu_started = False
    pending = None
    spent_before = 0
    try:
        if gpu is not None:
            gpu_started = True
            gpu.start()
        for request in planned:
            tally.reserve(request)
            pending, spent_before = request, tally.spent_micros
            response = endpoint(request)
            result.requests_sent += 1
            tally.settle(request, response)
            pending = None
    except (TokenCeilingExceeded, CapReached) as stop:
        result.status = RunStatus.STOPPED_AT_CAP
        result.reason = f"stopped at the cap: {stop}"
    except CostUnknown as unknown:
        result.status = RunStatus.FAILED
        result.reason = f"the run failed: {unknown}"
    except Exception as error:  # noqa: BLE001 -- any endpoint or GPU fault fails the run
        result.status = RunStatus.FAILED
        result.reason = f"the run failed with {type(error).__name__}"
        if pending is not None and tally.spent_micros == spent_before:
            # A request that failed after it was sent may still be billed,
            # so it counts at its full reservation.
            tally.spent_micros += pending.max_cost_micros
            result.notes.append("a failed request was counted at its reservation")
    finally:
        result.spent_micros = tally.spent_micros
        if gpu is not None and gpu_started:
            interrupted = None
            try:
                result.teardown_confirmed = gpu.teardown() is True
            except BaseException as error:  # noqa: BLE001 -- a failed teardown is a failed run
                result.teardown_confirmed = False
                result.notes.append(f"teardown raised {type(error).__name__}")
                if not isinstance(error, Exception):
                    interrupted = error
            if not result.teardown_confirmed:
                result.status = RunStatus.FAILED
                result.reason = f"{result.reason}; the GPU teardown was not confirmed"
            if interrupted is not None:
                raise interrupted

    out.write(
        f"Run {result.status.value}: {result.reason}."
        f" Spent {format_usd(result.spent_micros)} of a {format_usd(cap_micros)} cap.\n"
    )
    return result
