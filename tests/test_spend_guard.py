"""Tests for the bench spend guard (issue #1).

Every test uses a stub endpoint and a stub GPU lease. Nothing here reaches
a network, reads a credential, or spends money.

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import os
import pathlib
import signal
import sys
import threading
import unittest
from datetime import date
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench import spend_guard  # noqa: E402
from bench.spend_guard import (  # noqa: E402
    CapReached,
    CostUnknown,
    Estimate,
    PriceSource,
    Request,
    Response,
    RunStatus,
    SpendTally,
    Terminated,
    TokenCeilingExceeded,
    TokenPrice,
    format_usd,
    run_guarded,
)

SOURCE = PriceSource(name="stub price list", reference="tests/stub", read_on=date(2026, 9, 30))
# 1 micro-dollar per input token and 10 per output token, so a request of
# 100 input tokens and 90 max tokens is bounded at 1,000.
PRICE = TokenPrice(input_micros_per_token=1, output_micros_per_token=10, source=SOURCE)
BOUND = 1_000


def request(input_tokens=100, max_tokens=90):
    return Request(input_tokens=input_tokens, max_tokens=max_tokens)


def requests(count):
    return [request() for _ in range(count)]


def response(req=None, **overrides):
    req = req or request()
    fields = dict(
        input_tokens=req.input_tokens,
        output_tokens=req.max_tokens,
        cost_micros=0,
        attempts=1,
        max_tokens_sent=req.max_tokens,
    )
    fields.update(overrides)
    return Response(**fields)


class StubEndpoint:
    """Returns a queued or default response per request and records calls."""

    def __init__(self, responses=None, fail_on=None, raise_on=None):
        self.responses = list(responses or [])
        self.fail_on = fail_on
        self.raise_on = raise_on
        self.calls = []

    def __call__(self, req: Request) -> Response:
        self.calls.append(req)
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("stub endpoint failure")
        if self.raise_on is not None:
            raise self.raise_on
        if self.responses:
            return self.responses.pop(0)
        return response(req)


class HangingEndpoint:
    """Blocks until released, like a request that never returns."""

    def __init__(self, before=None):
        self.release = threading.Event()
        self.before = before
        self.calls = 0

    def __call__(self, req):
        self.calls += 1
        if self.before is not None:
            self.before()
        self.release.wait(5)
        return response(req)


class StubGpu:
    """A stub hosted-GPU lease. teardown() returns confirmed, or raises."""

    def __init__(self, confirmed=True, raises=False, start_raises=False, price_per_second_micros=0):
        self.started = 0
        self.torn_down = 0
        self.confirmed = confirmed
        self.raises = raises
        self.start_raises = start_raises
        self.price_per_second_micros = price_per_second_micros

    def start(self):
        self.started += 1
        if self.start_raises:
            raise RuntimeError("stub start failure")

    def teardown(self):
        self.torn_down += 1
        if self.raises:
            raise RuntimeError("stub teardown failure")
        return self.confirmed


def guarded(answer="yes", tty=True, out_tty=True, **overrides):
    """Run the guard with the terminal stubbed. answer is what the founder
    types: a string, a callable taking the prompt, or an exception."""
    args = dict(
        estimate=Estimate(run_count=1, price_per_run_micros=5_000, source=SOURCE),
        token_price=PRICE,
        cap_micros=5_000,
        max_tokens_per_request=100,
        request_timeout_s=5,
        run_time_limit_s=30,
        requests=requests(3),
        endpoint=StubEndpoint(),
        out=io.StringIO(),
        gpu=None,
        gpu_budget_seconds=None,
    )
    args.update(overrides)
    if args["gpu"] is not None and "gpu_budget_seconds" not in overrides:
        args["gpu_budget_seconds"] = 60
    if args["gpu"] is not None and "gpu_call_timeout_s" not in overrides:
        args["gpu_call_timeout_s"] = 5
    args.setdefault("gpu_call_timeout_s", None)
    read = mock.Mock(return_value=answer) if isinstance(answer, str) else mock.Mock(side_effect=answer)
    with mock.patch.object(spend_guard, "_stdin_is_tty", return_value=tty), \
            mock.patch.object(spend_guard, "_out_is_tty", return_value=out_tty), \
            mock.patch.object(spend_guard, "_read_answer", read):
        return args, run_guarded(**args)


class EstimateTests(unittest.TestCase):
    def test_the_estimate_is_run_count_times_price_per_run_with_its_source(self):
        estimate = Estimate(run_count=4, price_per_run_micros=2_500_000, source=SOURCE)
        self.assertEqual(estimate.total_micros, 10_000_000)
        text = estimate.describe()
        for part in ["4 runs", "$2.50 per run", "$10.00", "stub price list", "2026-09-30"]:
            self.assertIn(part, text)

    def test_the_guard_prints_the_plan_budget_and_cap_before_it_asks(self):
        seen = []
        out = io.StringIO()

        def answer(_prompt):
            seen.append(out.getvalue())
            return "no"

        guarded(answer=answer, out=out, gpu=StubGpu(price_per_second_micros=10), gpu_budget_seconds=60)
        for part in ["Estimate", "stub price list", "Planned: 3 requests", format_usd(3 * BOUND),
                     "GPU budget: 60 s", "Hard cap", "Request timeout", "Run limit"]:
            self.assertIn(part, seen[0])

    def test_an_estimate_needs_a_named_price_source(self):
        with self.assertRaises(ValueError):
            PriceSource(name="", reference="x", read_on=date(2026, 9, 30))
        with self.assertRaises(ValueError):
            Estimate(run_count=1, price_per_run_micros=1, source=None)
        with self.assertRaises(ValueError):
            TokenPrice(input_micros_per_token=1, output_micros_per_token=1, source=None)

    def test_an_estimate_or_price_refuses_bad_numbers(self):
        for count, price in [(0, 1), (-1, 1), (1, -1), (True, 1), (1, 1.5)]:
            with self.subTest(count=count, price=price):
                with self.assertRaises(ValueError):
                    Estimate(run_count=count, price_per_run_micros=price, source=SOURCE)
        for price in [-1, 1.5, True]:
            with self.subTest(token_price=price):
                with self.assertRaises(ValueError):
                    TokenPrice(input_micros_per_token=price, output_micros_per_token=1, source=SOURCE)

    def test_usd_formatting(self):
        self.assertEqual(format_usd(0), "$0.00")
        self.assertEqual(format_usd(1_234_567), "$1.23")
        self.assertEqual(format_usd(10_000_000), "$10.00")

    def test_a_plan_that_can_pass_the_cap_is_noted_before_the_ask(self):
        out = io.StringIO()
        guarded(answer="no", cap_micros=2_500, out=out)
        self.assertIn("can pass the hard cap", out.getvalue())


class ApprovalTests(unittest.TestCase):
    def test_a_refused_run_calls_no_endpoint_and_starts_no_gpu(self):
        for answer in ["no", "", "y", "YES", "yes please", " yes"]:
            with self.subTest(answer=answer):
                gpu = StubGpu()
                args, result = guarded(answer=answer, gpu=gpu)
                self.assertEqual(result.status, RunStatus.REFUSED)
                self.assertEqual(args["endpoint"].calls, [])
                self.assertEqual(gpu.started, 0)
                self.assertEqual(result.spent_micros, 0)

    def test_a_yes_without_a_terminal_is_refused(self):
        for tty, out_tty in [(False, True), (True, False)]:
            with self.subTest(stdin=tty, out=out_tty):
                gpu = StubGpu()
                args, result = guarded(tty=tty, out_tty=out_tty, gpu=gpu)
                self.assertEqual(result.status, RunStatus.REFUSED)
                self.assertIn("terminal", result.reason)
                self.assertEqual(args["endpoint"].calls, [])
                self.assertEqual(gpu.started, 0)

    def test_end_of_input_at_the_prompt_is_a_refusal(self):
        args, result = guarded(answer=EOFError)
        self.assertEqual(result.status, RunStatus.REFUSED)
        self.assertEqual(args["endpoint"].calls, [])

    def test_the_requests_are_fixed_before_the_founder_is_asked(self):
        order = []

        def planned():
            for req in requests(2):
                order.append("read")
                yield req

        def answer(_prompt):
            order.append("asked")
            return "yes"

        _args, result = guarded(answer=answer, requests=planned())
        self.assertEqual(order, ["read", "read", "asked"])
        self.assertEqual(result.status, RunStatus.COMPLETED)

    def test_an_invalid_plan_is_refused_before_the_ask(self):
        read = []
        _args, result = guarded(answer=lambda p: read.append(p) or "yes",
                                requests=[Request(input_tokens=-1, max_tokens=10)])
        self.assertEqual(result.status, RunStatus.REFUSED)
        self.assertEqual(read, [])

    def test_no_parameter_skips_the_yes(self):
        self.assertEqual(
            set(inspect.signature(run_guarded).parameters),
            {
                "estimate", "token_price", "cap_micros", "max_tokens_per_request",
                "request_timeout_s", "run_time_limit_s", "requests", "endpoint",
                "out", "gpu", "gpu_budget_seconds", "gpu_call_timeout_s",
            },
        )

    def test_the_guard_reads_no_environment_variable(self):
        tree = ast.parse(pathlib.Path(spend_guard.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn("os", imported)
        self.assertNotIn("environ", names)
        self.assertNotIn("getenv", names)

    def test_an_approved_run_completes_and_records_its_spend(self):
        gpu = StubGpu()
        out = io.StringIO()
        args, result = guarded(gpu=gpu, out=out)
        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(len(args["endpoint"].calls), 3)
        self.assertEqual(result.spent_micros, 3 * BOUND)
        self.assertEqual((gpu.started, gpu.torn_down), (1, 1))
        self.assertTrue(result.teardown_confirmed)
        record = json.loads(out.getvalue().split("Record: ", 1)[1])
        self.assertEqual(record["status"], "completed")


class CapTests(unittest.TestCase):
    def test_a_run_stops_before_a_request_that_could_pass_the_cap(self):
        gpu = StubGpu()
        args, result = guarded(cap_micros=2_500, gpu=gpu)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(len(args["endpoint"].calls), 2)
        self.assertEqual(result.spent_micros, 2_000)
        self.assertIn("cap", result.reason)
        self.assertEqual(gpu.torn_down, 1)

    def test_the_bound_comes_from_the_named_price_not_the_caller(self):
        tally = SpendTally(cap_micros=10_000, max_tokens_per_request=100)
        self.assertEqual(tally.bound(request(input_tokens=100, max_tokens=90), PRICE), BOUND)

    def test_a_response_that_costs_more_than_reserved_stops_the_run(self):
        endpoint = StubEndpoint(responses=[response(cost_micros=4_000)])
        _args, result = guarded(cap_micros=10_000, endpoint=endpoint)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(len(endpoint.calls), 1)
        self.assertEqual(result.spent_micros, 4_000)
        self.assertIn("more than", result.reason)

    def test_a_response_that_passes_the_cap_is_recorded_and_stops_the_run(self):
        endpoint = StubEndpoint(responses=[response(cost_micros=9_000)])
        _args, result = guarded(cap_micros=5_000, endpoint=endpoint)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(result.spent_micros, 9_000)

    def test_an_under_reported_cost_counts_at_the_bound(self):
        endpoint = StubEndpoint(responses=[response(cost_micros=0, output_tokens=0)] * 3)
        _args, result = guarded(cap_micros=2_500, endpoint=endpoint)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(result.spent_micros, 2_000)

    def test_retries_are_charged_for_every_attempt(self):
        endpoint = StubEndpoint(responses=[response(cost_micros=100, attempts=3)])
        _args, result = guarded(cap_micros=10_000, endpoint=endpoint)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(result.spent_micros, 3 * BOUND)
        self.assertEqual(len(endpoint.calls), 1)

    def test_an_endpoint_that_does_not_pass_max_tokens_fails_the_run(self):
        endpoint = StubEndpoint(responses=[response(max_tokens_sent=4_096)])
        _args, result = guarded(endpoint=endpoint)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("max_tokens", result.reason)
        self.assertEqual(result.spent_micros, BOUND)

    def test_a_response_with_invalid_fields_fails_the_run(self):
        for name, value in [("cost_micros", -1), ("cost_micros", 1.5), ("cost_micros", None),
                            ("attempts", 0), ("output_tokens", True), ("input_tokens", "5")]:
            with self.subTest(field=name, value=value):
                endpoint = StubEndpoint(responses=[response(**{name: value})])
                _args, result = guarded(endpoint=endpoint)
                self.assertEqual(result.status, RunStatus.FAILED)
                self.assertEqual(result.spent_micros, BOUND)

    def test_a_response_over_its_token_limits_stops_the_run(self):
        for overrides, text in [({"output_tokens": 91}, "more tokens"), ({"input_tokens": 101}, "input tokens")]:
            with self.subTest(overrides=overrides):
                _args, result = guarded(endpoint=StubEndpoint(responses=[response(**overrides)]))
                self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
                self.assertIn(text, result.reason)

    def test_a_request_over_the_token_ceiling_is_refused_before_the_ask(self):
        endpoint = StubEndpoint()
        _args, result = guarded(requests=[request(max_tokens=101)], endpoint=endpoint)
        self.assertEqual(result.status, RunStatus.REFUSED)
        self.assertIn("token ceiling", result.reason)
        self.assertEqual(endpoint.calls, [])

    def test_the_cap_ceiling_and_limits_must_be_set(self):
        for cap, ceiling in [(0, 1), (-1, 1), (1, 0), (None, 1), (1, None), (True, 1)]:
            with self.subTest(cap=cap, ceiling=ceiling):
                with self.assertRaises(ValueError):
                    SpendTally(cap_micros=cap, max_tokens_per_request=ceiling)
        for name, value in [("request_timeout_s", 0), ("run_time_limit_s", -1),
                            ("run_time_limit_s", float("inf")), ("request_timeout_s", None)]:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    guarded(**{name: value})

    def test_the_tally_itself(self):
        tally = SpendTally(cap_micros=2_500, max_tokens_per_request=100)
        tally.hold(500)
        tally.reserve(BOUND)
        tally.settle(request(), BOUND, response(cost_micros=500), PRICE)
        self.assertEqual(tally.spent_micros, BOUND)
        with self.assertRaises(CapReached):
            tally.reserve(BOUND + 1)
        with self.assertRaises(TokenCeilingExceeded):
            tally.bound(request(max_tokens=101), PRICE)
        with self.assertRaises(CostUnknown):
            tally.bound(request(input_tokens=-1), PRICE)
        tally.release(500, 200)
        self.assertEqual((tally.held_micros, tally.spent_micros), (0, BOUND + 200))

    def test_the_result_records_the_cap_and_why_it_stopped(self):
        _args, result = guarded(cap_micros=2_500)
        record = result.as_record()
        self.assertEqual(record["status"], "stopped_at_cap")
        self.assertEqual(record["cap_micros"], 2_500)
        self.assertEqual(record["spent_micros"], 2_000)
        self.assertTrue(record["reason"])


class TimeAndGpuTests(unittest.TestCase):
    def test_a_hanging_request_stops_at_its_timeout_and_is_charged(self):
        gpu = StubGpu(price_per_second_micros=7)
        endpoint = HangingEndpoint()
        try:
            _args, result = guarded(endpoint=endpoint, gpu=gpu, request_timeout_s=0.05)
        finally:
            endpoint.release.set()
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertIn("timeout", result.reason)
        self.assertEqual(gpu.torn_down, 1)
        self.assertEqual(result.gpu_seconds, 1)
        self.assertEqual(result.spent_micros, BOUND + 7)

    def test_a_run_stops_at_its_wall_clock_limit(self):
        endpoint = HangingEndpoint()
        try:
            _args, result = guarded(endpoint=endpoint, run_time_limit_s=0.05)
        finally:
            endpoint.release.set()
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertIn("wall-clock", result.reason)
        self.assertEqual(result.spent_micros, BOUND)

    def test_gpu_seconds_are_charged_inside_the_cap(self):
        gpu = StubGpu(price_per_second_micros=100)
        _args, result = guarded(gpu=gpu, gpu_budget_seconds=10)
        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.gpu_seconds, 1)
        self.assertEqual(result.spent_micros, 3 * BOUND + 100)

    def test_a_gpu_budget_that_does_not_fit_the_cap_never_starts_the_gpu(self):
        gpu = StubGpu(price_per_second_micros=100)
        args, result = guarded(gpu=gpu, gpu_budget_seconds=60, cap_micros=5_000)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertIn("GPU time budget", result.reason)
        self.assertEqual(gpu.started, 0)
        self.assertEqual(args["endpoint"].calls, [])

    def test_the_gpu_budget_holds_back_request_spend(self):
        gpu = StubGpu(price_per_second_micros=100)
        args, result = guarded(gpu=gpu, gpu_budget_seconds=30, cap_micros=5_000)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(len(args["endpoint"].calls), 2)

    def test_a_gpu_needs_a_price_and_a_budget(self):
        no_price = StubGpu()
        no_price.price_per_second_micros = None
        float_price = StubGpu()
        float_price.price_per_second_micros = 1.5

        for gpu, budget in [(no_price, 60), (float_price, 60), (StubGpu(), None), (StubGpu(), 0)]:
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    guarded(gpu=gpu, gpu_budget_seconds=budget)
        with self.assertRaises(ValueError):
            guarded(gpu_budget_seconds=60)
        with self.assertRaises(ValueError):
            guarded(gpu_call_timeout_s=5)
        with self.assertRaises(ValueError):
            guarded(gpu=StubGpu(), gpu_call_timeout_s=0)

    def test_a_hanging_gpu_start_times_out_and_is_torn_down(self):
        class HangingStartGpu(StubGpu):
            def __init__(self):
                super().__init__()
                self.release = threading.Event()

            def start(self):
                self.started += 1
                self.release.wait(5)

        gpu = HangingStartGpu()
        try:
            args, result = guarded(gpu=gpu, gpu_call_timeout_s=0.05)
        finally:
            gpu.release.set()
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("start timed out", result.reason)
        self.assertEqual(gpu.torn_down, 1)
        self.assertEqual(args["endpoint"].calls, [])

    def test_a_hanging_teardown_counts_as_unconfirmed(self):
        class HangingTeardownGpu(StubGpu):
            def __init__(self):
                super().__init__()
                self.release = threading.Event()

            def teardown(self):
                self.torn_down += 1
                self.release.wait(5)
                return True

        gpu = HangingTeardownGpu()
        try:
            _args, result = guarded(gpu=gpu, gpu_call_timeout_s=0.05)
        finally:
            gpu.release.set()
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertFalse(result.teardown_confirmed)
        self.assertIn("teardown", result.reason)
        self.assertIn("the GPU teardown timed out", result.notes)


class TeardownTests(unittest.TestCase):
    def test_the_gpu_is_torn_down_after_an_endpoint_error(self):
        gpu = StubGpu()
        _args, result = guarded(endpoint=StubEndpoint(fail_on=2), gpu=gpu)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(gpu.torn_down, 1)
        self.assertEqual(result.spent_micros, 2 * BOUND)
        self.assertTrue(result.notes)

    def test_the_gpu_is_torn_down_when_it_fails_to_start(self):
        gpu = StubGpu(start_raises=True)
        args, result = guarded(gpu=gpu)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(gpu.torn_down, 1)
        self.assertEqual(args["endpoint"].calls, [])

    def test_an_unconfirmed_teardown_fails_the_run(self):
        for gpu in [StubGpu(confirmed=False), StubGpu(raises=True)]:
            with self.subTest(gpu=vars(gpu)):
                _args, result = guarded(gpu=gpu)
                self.assertEqual(result.status, RunStatus.FAILED)
                self.assertFalse(result.teardown_confirmed)
                self.assertIn("teardown", result.reason)

    def test_a_truthy_teardown_that_is_not_true_fails_the_run(self):
        class OneGpu(StubGpu):
            def teardown(self):
                super().teardown()
                return 1

        _args, result = guarded(gpu=OneGpu())
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertFalse(result.teardown_confirmed)

    def test_an_unconfirmed_teardown_after_a_cap_stop_still_fails_the_run(self):
        _args, result = guarded(cap_micros=2_500, gpu=StubGpu(confirmed=False))
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("cap", result.reason)
        self.assertIn("teardown", result.reason)

    def test_an_endpoint_failure_and_a_failed_teardown_both_show(self):
        _args, result = guarded(endpoint=StubEndpoint(fail_on=1), gpu=StubGpu(confirmed=False))
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("RuntimeError", result.reason)
        self.assertIn("teardown", result.reason)

    def test_an_endpoint_error_message_is_not_echoed(self):
        leak = RuntimeError("Authorization: Bearer stub-token-not-real")
        out = io.StringIO()
        _args, result = guarded(endpoint=StubEndpoint(raise_on=leak), out=out)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertNotIn("stub-token", result.reason)
        self.assertNotIn("stub-token", out.getvalue())
        self.assertIn("RuntimeError", result.reason)


class InterruptTests(unittest.TestCase):
    def test_an_interrupt_is_recorded_charged_torn_down_and_raised(self):
        gpu = StubGpu()
        out = io.StringIO()
        with self.assertRaises(KeyboardInterrupt) as ctx:
            guarded(endpoint=StubEndpoint(raise_on=KeyboardInterrupt()), gpu=gpu, out=out)
        result = ctx.exception.run_result
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("interrupted", result.reason)
        self.assertEqual(result.spent_micros, BOUND)
        self.assertEqual(gpu.torn_down, 1)
        self.assertIn('"status": "failed"', out.getvalue())

    def test_an_interrupt_during_teardown_is_recorded_and_raised(self):
        class InterruptedGpu(StubGpu):
            def teardown(self):
                super().teardown()
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt) as ctx:
            guarded(gpu=InterruptedGpu())
        self.assertEqual(ctx.exception.run_result.status, RunStatus.FAILED)

    def _signal_test(self, signum):
        before = signal.getsignal(signum)
        gpu = StubGpu()
        endpoint = HangingEndpoint(before=lambda: os.kill(os.getpid(), signum))
        try:
            with self.assertRaises(Terminated) as ctx:
                guarded(endpoint=endpoint, gpu=gpu)
        finally:
            endpoint.release.set()
        result = ctx.exception.run_result
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("interrupted", result.reason)
        self.assertEqual(result.spent_micros, BOUND)
        self.assertEqual(gpu.torn_down, 1)
        self.assertEqual(signal.getsignal(signum), before)

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "no SIGTERM here")
    def test_sigterm_tears_down_and_restores_the_handler(self):
        self._signal_test(signal.SIGTERM)

    @unittest.skipUnless(hasattr(signal, "SIGHUP"), "no SIGHUP here")
    def test_sighup_tears_down_and_restores_the_handler(self):
        self._signal_test(signal.SIGHUP)

    def test_a_second_signal_during_teardown_does_not_stop_it(self):
        before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}

        class SignalledGpu(StubGpu):
            def teardown(self):
                self.torn_down += 1
                os.kill(os.getpid(), signal.SIGTERM)
                os.kill(os.getpid(), signal.SIGINT)
                threading.Event().wait(0.05)
                return True

        gpu = SignalledGpu()
        endpoint = HangingEndpoint(before=lambda: os.kill(os.getpid(), signal.SIGTERM))
        try:
            with self.assertRaises(Terminated) as ctx:
                guarded(endpoint=endpoint, gpu=gpu)
        finally:
            endpoint.release.set()
        result = ctx.exception.run_result
        self.assertTrue(result.teardown_confirmed)
        self.assertEqual(gpu.torn_down, 1)
        self.assertEqual({s: signal.getsignal(s) for s in before}, before)

    def test_a_sigint_during_teardown_is_held_until_it_finishes(self):
        class SignalledGpu(StubGpu):
            def teardown(self):
                self.torn_down += 1
                os.kill(os.getpid(), signal.SIGINT)
                threading.Event().wait(0.05)
                return True

        gpu = SignalledGpu()
        with self.assertRaises(KeyboardInterrupt) as ctx:
            guarded(gpu=gpu)
        result = ctx.exception.run_result
        self.assertTrue(result.teardown_confirmed)
        self.assertIn("a SIGINT during cleanup was held until the teardown finished", result.notes)

    def test_the_handlers_are_restored_after_a_normal_run(self):
        before = signal.getsignal(signal.SIGTERM)
        guarded()
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)


if __name__ == "__main__":
    unittest.main()
