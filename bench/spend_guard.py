"""The bench spend guard (issue #1).

Every model-fit run spends money, and spend is the founder's decision.
This guard follows the spend section, section 6.4, of "Threat model and
eval spec: Vines connectors and the model-fit bench":

1. Before any run, it prints a cost estimate: the run count times the
   price per run, with the price source named. Beside it, it prints the
   total of the planned request reservations and the GPU time budget.
2. The run starts only after the founder types an exact "yes" at a
   terminal. No parameter, flag, or environment variable skips it.
   isatty proves that a terminal is attached, not that a person typed
   the yes.
3. A hard cap on each run, enforced by a token ceiling on each request,
   a running tally, a stop before any request that could pass the cap,
   a timeout on each request, and a wall-clock limit on the run. The GPU
   time budget is held inside the cap before the GPU starts, and the GPU
   seconds used are charged to the tally. The founder sets the cap; this
   module has no default for it.
4. Any hosted GPU the run started is torn down at the end, on a stop at
   the cap, on any error, and on an interrupt, SIGTERM, or SIGHUP. A run
   that cannot confirm the teardown is reported as failed. SIGKILL cannot
   be caught, so every GPU also needs a provider-side time-to-live or
   auto-stop as the backstop.

The GPU lease's start() and teardown() run under a timeout too, so a
hung provider call cannot block the guard while GPU time runs. A
teardown that times out counts as unconfirmed, and the run fails. After
a start times out, the guard waits a bounded time for the start to
finish before it tears down; if the start is still in flight, the GPU
could come up after the teardown, so the teardown counts as unconfirmed.

During cleanup, SIGTERM, SIGHUP, and SIGINT are held, not acted on. The
first statement of cleanup sets a flag that makes the guard's handlers
record a signal instead of raising. Every worker thread starts with the
three signals blocked, so a signal sent to the process lands on the main
thread. After the teardown and the run record, the old handlers come
back and a held signal is raised. A signal that lands between the end of
the run and the flag can still cut the cleanup short; the provider-side
time limit covers that case.

The start worker is known before the start is awaited, so on every exit,
an interrupt included, a start still in flight is awaited for a bounded
time before teardown, and the teardown counts as unconfirmed if the
start is still running.

A request, start, or teardown that times out keeps running on its worker
thread. Repeated run_guarded calls in one process can leave such orphan
threads, so an endpoint or GPU client must tolerate a late call.

The endpoint contract. The guard calls the endpoint once per request.
The endpoint passes request.max_tokens to the provider as its output
limit and reports it back as max_tokens_sent. It makes exactly one
attempt, with the SDK's retries turned off, or it reports the number of
attempts and the summed cost of all of them.

Money is whole micro-dollars (millionths of a US dollar), never a float.
The guard derives each request's cost bound itself, from the request's
input size, its max_tokens, and a named per-token price. It holds no
credential and reads none. An endpoint's error text is never printed or
recorded, only its type, so a key in an error message cannot leak.
"""

from __future__ import annotations

import enum
import json
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable, Optional, Protocol, TextIO

APPROVAL_WORD = "yes"


def _is_whole(value) -> bool:
    """A plain int only: never a bool, a float, or an int subclass that
    could change its value between checks."""
    return type(value) is int


def _is_positive_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


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
class TokenPrice:
    """The per-token price the guard bounds each request with."""

    input_micros_per_token: int
    output_micros_per_token: int
    source: PriceSource

    def __post_init__(self):
        for value in (self.input_micros_per_token, self.output_micros_per_token):
            if not _is_whole(value) or value < 0:
                raise ValueError("a per-token price must be a whole number of 0 or more")
        if not isinstance(self.source, PriceSource):
            raise ValueError("a token price needs a price source")

    def cost(self, input_tokens: int, output_tokens: int) -> int:
        return input_tokens * self.input_micros_per_token + output_tokens * self.output_micros_per_token


@dataclass(frozen=True)
class Request:
    """One model request: its input size in tokens, and its output limit."""

    input_tokens: int
    max_tokens: int


@dataclass(frozen=True)
class Response:
    """What the endpoint reports for one request, under its contract."""

    input_tokens: int
    output_tokens: int
    cost_micros: int
    attempts: int
    max_tokens_sent: int


class TokenCeilingExceeded(Exception):
    """A request asks for more tokens than the per-request ceiling."""


class CapReached(Exception):
    """A request could pass the cap, or spend has passed it."""


class RequestTimedOut(CapReached):
    """A request ran past its timeout."""


class RunTimeLimitReached(CapReached):
    """The run ran past its wall-clock limit."""


class CostUnknown(Exception):
    """A request or response carries no valid cost, so spend is unknown."""


class ContractBroken(Exception):
    """The endpoint broke its contract, so its spend cannot be trusted."""


class Terminated(BaseException):
    """SIGTERM or SIGHUP arrived during a run."""


class CallTimedOut(Exception):
    """A GPU start or teardown ran past its timeout. worker is the thread
    that is still running the call."""

    worker: Optional[threading.Thread] = None


_RUN_SIGNALS = tuple(
    signum for signum in (getattr(signal, name, None) for name in ("SIGTERM", "SIGHUP")) if signum is not None
)
_CLEANUP_SIGNALS = _RUN_SIGNALS + (signal.SIGINT,)


class SpendTally:
    """The running tally for one run, against its hard cap.

    held_micros is spend set aside inside the cap but not yet used: the
    GPU time budget, until the GPU is torn down and its real time is
    charged.
    """

    def __init__(self, cap_micros: int, max_tokens_per_request: int):
        if not _is_whole(cap_micros) or cap_micros < 1:
            raise ValueError("cap_micros must be a whole number of 1 or more")
        if not _is_whole(max_tokens_per_request) or max_tokens_per_request < 1:
            raise ValueError("max_tokens_per_request must be a whole number of 1 or more")
        self.cap_micros = cap_micros
        self.max_tokens_per_request = max_tokens_per_request
        self.spent_micros = 0
        self.held_micros = 0

    def hold(self, amount: int) -> None:
        if self.spent_micros + self.held_micros + amount > self.cap_micros:
            raise CapReached("the GPU time budget does not fit inside the hard cap")
        self.held_micros += amount

    def release(self, held: int, used: int) -> None:
        self.held_micros -= held
        self.spent_micros += used

    def bound(self, request: Request, price: TokenPrice) -> int:
        """Check a request's shape and ceiling, and return its cost bound."""
        if not _is_whole(request.max_tokens) or request.max_tokens < 1:
            raise TokenCeilingExceeded("a request has no valid token ceiling")
        if request.max_tokens > self.max_tokens_per_request:
            raise TokenCeilingExceeded("a request asked for more than the per-request token ceiling")
        if not _is_whole(request.input_tokens) or request.input_tokens < 0:
            raise CostUnknown("a request has no valid input size, so its cost cannot be bounded")
        return price.cost(request.input_tokens, request.max_tokens)

    def reserve(self, bound: int) -> None:
        """Refuse a request before it is sent if it could pass the cap."""
        if self.spent_micros + self.held_micros + bound > self.cap_micros:
            raise CapReached("the next request could pass the hard cap")

    def charge(self, amount: int) -> None:
        self.spent_micros += amount

    def settle(self, request: Request, bound: int, response: Response, price: TokenPrice) -> None:
        """Charge a response, then stop the run if it broke a limit.

        The charge is the largest of the reported cost, the bound times the
        number of attempts, and the cost of the reported tokens at the
        named price, so an endpoint that reports too little cannot lower
        the tally.
        """
        fields = (response.input_tokens, response.output_tokens, response.cost_micros, response.max_tokens_sent)
        if not all(_is_whole(value) and value >= 0 for value in fields) \
                or not _is_whole(response.attempts) or response.attempts < 1:
            self.charge(bound)
            raise CostUnknown("a response did not report a valid cost, token count, or attempt count")
        charged = max(
            response.cost_micros,
            bound * response.attempts,
            price.cost(response.input_tokens, response.output_tokens),
        )
        self.charge(charged)
        if response.max_tokens_sent != request.max_tokens:
            raise ContractBroken("the endpoint did not pass the request's max_tokens to the provider")
        if response.output_tokens > request.max_tokens:
            raise CapReached("a response used more tokens than its request's ceiling")
        if response.input_tokens > request.input_tokens:
            raise CapReached("a response used more input tokens than the request planned")
        if charged > bound:
            raise CapReached("a response cost more than the amount reserved for it")
        if self.spent_micros + self.held_micros > self.cap_micros:
            raise CapReached("spend passed the hard cap")


class GpuLease(Protocol):
    price_per_second_micros: int

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
    gpu_seconds: int = 0
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
            "gpu_seconds": self.gpu_seconds,
            "teardown_confirmed": self.teardown_confirmed,
            "estimate_total_micros": self.estimate.total_micros,
            "price_source": self.estimate.source.name,
            "notes": list(self.notes),
        }


def _stdin_is_tty() -> bool:
    return sys.stdin.isatty()


def _out_is_tty(out: TextIO) -> bool:
    return out.isatty()


def _read_answer(prompt: str) -> str:
    return input(prompt)


def _monotonic() -> float:
    return time.monotonic()


class _Cleanup:
    """Process-wide cleanup state. Once active, the guard's signal
    handlers record a signal instead of raising, so nothing can cut the
    teardown short. held collects the signals recorded."""

    active = False
    held: list = []


def _run_with_timeout(call, timeout_s: float, on_timeout: BaseException, holder: Optional[dict] = None):
    """Run call() on a worker thread and wait at most timeout_s.

    A call that times out keeps running on its thread; the guard raises
    on_timeout and moves on. holder, if given, receives the thread before
    the wait begins, so the caller can reach it on any exit, an interrupt
    included.

    The worker starts with SIGTERM, SIGHUP, and SIGINT blocked, and keeps
    that mask, so a signal sent to the process lands on the main thread.
    """
    box = {}

    def work():
        try:
            box["value"] = call()
        except BaseException as error:  # noqa: BLE001 -- handed back to the caller
            box["error"] = error

    worker = threading.Thread(target=work, name="spend-guard-call", daemon=True)
    if holder is not None:
        holder["worker"] = worker
    can_mask = hasattr(signal, "pthread_sigmask")
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(_CLEANUP_SIGNALS)) if can_mask else None
    try:
        worker.start()
    finally:
        if can_mask:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    worker.join(timeout_s)
    if worker.is_alive():
        if isinstance(on_timeout, CallTimedOut):
            on_timeout.worker = worker
        raise on_timeout
    if "error" in box:
        raise box["error"]
    return box["value"]


def _call_with_timeout(endpoint, request: Request, timeout_s: float, run_limited: bool) -> Response:
    """Call the endpoint under a timeout. A request that times out is
    charged at its reservation, and the GPU is torn down at once."""
    on_timeout = (
        RunTimeLimitReached("the run reached its wall-clock limit")
        if run_limited
        else RequestTimedOut("a request ran past its timeout")
    )
    return _run_with_timeout(lambda: endpoint(request), timeout_s, on_timeout)


def _raise_terminated(signum, _frame):
    if _Cleanup.active:
        _Cleanup.held.append(signum)
        return
    raise Terminated(signal.Signals(signum).name)


def _raise_interrupt(signum, _frame):
    if _Cleanup.active:
        _Cleanup.held.append(signum)
        return
    raise KeyboardInterrupt


def _save_signal_handlers(saved: dict) -> None:
    """Record the handlers in force before a run, before any is replaced,
    so an install cut off halfway never saves the guard's own handler."""
    for signum in _CLEANUP_SIGNALS:
        saved[signum] = signal.getsignal(signum)


def _install_signal_handlers() -> None:
    """While a run is live, SIGTERM and SIGHUP raise Terminated and SIGINT
    raises KeyboardInterrupt, so teardown runs. Once cleanup is active,
    each records the signal instead."""
    for signum in _RUN_SIGNALS:
        signal.signal(signum, _raise_terminated)
    signal.signal(signal.SIGINT, _raise_interrupt)


def _restore_signal_handlers(saved: dict) -> None:
    for signum, handler in saved.items():
        if handler is not None:
            signal.signal(signum, handler)


def _hold_cleanup_signals() -> list:
    """Block the cleanup signals, install handlers that only record them,
    then unblock. A signal that arrived while blocked is delivered to the
    recording handler, so none is dropped. Returns the list of held
    signal numbers."""
    held = _Cleanup.held

    def record(signum, _frame):
        held.append(signum)

    can_mask = hasattr(signal, "pthread_sigmask")
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(_CLEANUP_SIGNALS)) if can_mask else None
    try:
        for signum in _CLEANUP_SIGNALS:
            signal.signal(signum, record)
    finally:
        if can_mask:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    return held


def _held_interrupt(held: list) -> Optional[BaseException]:
    """The interrupt a held signal stands for: Terminated for SIGTERM or
    SIGHUP, which wins, else KeyboardInterrupt for SIGINT."""
    for signum in held:
        if signum in _RUN_SIGNALS:
            return Terminated(signal.Signals(signum).name)
    if held:
        return KeyboardInterrupt()
    return None


def run_guarded(
    *,
    estimate: Estimate,
    token_price: TokenPrice,
    cap_micros: int,
    max_tokens_per_request: int,
    request_timeout_s: float,
    run_time_limit_s: float,
    requests: Iterable[Request],
    endpoint: Callable[[Request], Response],
    out: TextIO,
    gpu: Optional[GpuLease],
    gpu_budget_seconds: Optional[int],
    gpu_call_timeout_s: Optional[float],
) -> RunResult:
    """Print the estimate, get the founder's yes, then run the requests
    under the hard cap and tear down any GPU.

    The estimate and the prompt go to out, which must be the same terminal
    the yes is typed at. No argument approves on the founder's behalf. The
    requests are read in full before the founder is asked, so the run
    cannot change them after the yes.
    """
    tally = SpendTally(cap_micros, max_tokens_per_request)
    if not isinstance(token_price, TokenPrice):
        raise ValueError("a run needs a named per-token price")
    if not _is_positive_number(request_timeout_s) or not _is_positive_number(run_time_limit_s):
        raise ValueError("request_timeout_s and run_time_limit_s must be positive numbers")
    gpu_price = 0
    gpu_hold = 0
    if gpu is not None:
        gpu_price = getattr(gpu, "price_per_second_micros", None)
        if not _is_whole(gpu_price) or gpu_price < 0:
            raise ValueError("a GPU lease needs a whole price_per_second_micros of 0 or more")
        if not _is_whole(gpu_budget_seconds) or gpu_budget_seconds < 1:
            raise ValueError("a run with a GPU needs a whole gpu_budget_seconds of 1 or more")
        if not _is_positive_number(gpu_call_timeout_s):
            raise ValueError("a run with a GPU needs a positive gpu_call_timeout_s")
        gpu_hold = gpu_budget_seconds * gpu_price
    elif gpu_budget_seconds is not None or gpu_call_timeout_s is not None:
        raise ValueError("gpu_budget_seconds and gpu_call_timeout_s are only for a run with a GPU")

    planned = list(requests)
    result = RunResult(status=RunStatus.REFUSED, reason="", cap_micros=cap_micros, estimate=estimate)
    try:
        planned_bound = sum(tally.bound(request, token_price) for request in planned)
    except (TokenCeilingExceeded, CostUnknown) as bad:
        result.reason = f"the plan is invalid: {bad}"
        out.write(f"Refused: {result.reason}.\n")
        return result

    out.write(estimate.describe() + "\n")
    out.write(
        f"Planned: {len(planned)} requests, reserved at up to {format_usd(planned_bound)}"
        f" at the price from {token_price.source.name}.\n"
    )
    if gpu is not None:
        out.write(
            f"GPU budget: {gpu_budget_seconds} s x {format_usd(gpu_price)}/s = {format_usd(gpu_hold)},"
            " held inside the cap.\n"
        )
    out.write(
        f"Hard cap: {format_usd(cap_micros)} per run. Token ceiling: {max_tokens_per_request} per request."
        f" Request timeout: {request_timeout_s} s. Run limit: {run_time_limit_s} s.\n"
    )
    if planned_bound + gpu_hold > cap_micros or estimate.price_per_run_micros > cap_micros:
        out.write("Note: the plan can pass the hard cap, so the run can stop at the cap.\n")
    out.flush()

    if not _stdin_is_tty() or not _out_is_tty(out):
        result.reason = "the yes must be typed at a terminal that shows the estimate, and none is attached"
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
    gpu_started_at = None
    run_limit_s = run_time_limit_s
    if gpu is not None:
        run_limit_s = min(run_time_limit_s, float(gpu_budget_seconds))
    pending = None
    interrupted = None
    start_holder = {}
    saved_handlers = {}
    _save_signal_handlers(saved_handlers)
    _Cleanup.active = False
    _Cleanup.held = []
    try:
        _install_signal_handlers()
        started_at = _monotonic()
        deadline = started_at + run_limit_s
        if gpu is not None:
            tally.hold(gpu_hold)
            gpu_started_at = _monotonic()
            _run_with_timeout(
                gpu.start, gpu_call_timeout_s, CallTimedOut("the GPU start timed out"), holder=start_holder
            )
        for request in planned:
            bound = tally.bound(request, token_price)
            tally.reserve(bound)
            remaining = deadline - _monotonic()
            if remaining <= 0:
                raise RunTimeLimitReached("the run reached its wall-clock limit")
            timeout = min(request_timeout_s, remaining)
            pending = (request, bound, tally.spent_micros)
            response = _call_with_timeout(endpoint, request, timeout, run_limited=timeout < request_timeout_s)
            result.requests_sent += 1
            tally.settle(request, bound, response, token_price)
            pending = None
    except (TokenCeilingExceeded, CapReached) as stop:
        result.status = RunStatus.STOPPED_AT_CAP
        result.reason = f"stopped at the cap: {stop}"
    except (CostUnknown, ContractBroken) as broken:
        result.status = RunStatus.FAILED
        result.reason = f"the run failed: {broken}"
    except CallTimedOut as timed_out:
        result.status = RunStatus.FAILED
        result.reason = f"the run failed: {timed_out}"
    except Exception as error:  # noqa: BLE001 -- any endpoint or GPU fault fails the run
        result.status = RunStatus.FAILED
        result.reason = f"the run failed with {type(error).__name__}"
    except BaseException as error:  # noqa: BLE001 -- an interrupt; recorded, then raised again
        result.status = RunStatus.FAILED
        result.reason = f"interrupted by {type(error).__name__}"
        interrupted = error
    finally:
        _Cleanup.active = True
        held = _hold_cleanup_signals()
        try:
            if pending is not None and tally.spent_micros == pending[2]:
                # A request that was sent but not charged may still be
                # billed, so it counts at its full reservation.
                tally.charge(pending[1])
                result.notes.append("a request in flight was counted at its reservation")
            if gpu is not None and gpu_started_at is not None:
                start_still_running = False
                start_worker = start_holder.get("worker")
                if start_worker is not None and start_worker.is_alive():
                    # A start still running, after a timeout or an interrupt,
                    # may yet bring the GPU up, so wait a bounded time for it
                    # before tearing down.
                    start_worker.join(max(1.0, gpu_call_timeout_s))
                    start_still_running = start_worker.is_alive()
                try:
                    confirmed = _run_with_timeout(
                        gpu.teardown, gpu_call_timeout_s, CallTimedOut("the GPU teardown timed out")
                    )
                    result.teardown_confirmed = confirmed is True and not start_still_running
                    if start_still_running:
                        result.notes.append("the GPU start was still in flight at teardown")
                except CallTimedOut:
                    result.teardown_confirmed = False
                    result.notes.append("the GPU teardown timed out")
                except BaseException as error:  # noqa: BLE001 -- a failed teardown is a failed run
                    result.teardown_confirmed = False
                    result.notes.append(f"teardown raised {type(error).__name__}")
                    if not isinstance(error, Exception) and interrupted is None:
                        interrupted = error
                result.gpu_seconds = max(1, math.ceil(_monotonic() - gpu_started_at))
                tally.release(gpu_hold, result.gpu_seconds * gpu_price)
                if not result.teardown_confirmed:
                    result.status = RunStatus.FAILED
                    result.reason = f"{result.reason}; the GPU teardown was not confirmed"
                elif tally.spent_micros > cap_micros and result.status is RunStatus.COMPLETED:
                    result.status = RunStatus.STOPPED_AT_CAP
                    result.reason = "the GPU time passed the hard cap"
        finally:
            held_interrupt = _held_interrupt(held)
            if held_interrupt is not None:
                result.notes.append(
                    f"{type(held_interrupt).__name__} during cleanup was held until the teardown finished"
                )
                if interrupted is None:
                    interrupted = held_interrupt
            result.spent_micros = tally.spent_micros
            try:
                out.write(
                    f"Run {result.status.value}: {result.reason}."
                    f" Spent {format_usd(result.spent_micros)} of a {format_usd(cap_micros)} cap.\n"
                )
                out.write("Record: " + json.dumps(result.as_record(), sort_keys=True) + "\n")
                out.flush()
            finally:
                _restore_signal_handlers(saved_handlers)
                _Cleanup.active = False
                # A signal held while the record was written still counts.
                if interrupted is None:
                    interrupted = _held_interrupt(held)

    if interrupted is not None:
        try:
            interrupted.run_result = result
        except AttributeError:
            pass
        raise interrupted
    return result
