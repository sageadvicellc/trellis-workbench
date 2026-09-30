"""Tests for the bench spend guard (issue #1).

Every test uses a stub endpoint and a stub GPU lease. Nothing here reaches
a network, reads a credential, or spends money.

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import inspect
import io
import pathlib
import sys
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
    TokenCeilingExceeded,
    format_usd,
    run_guarded,
)

SOURCE = PriceSource(name="stub price list", reference="tests/stub", read_on=date(2026, 9, 30))


class StubEndpoint:
    """Returns a fixed response per request and records each call."""

    def __init__(self, responses=None, fail_on=None):
        self.responses = list(responses or [])
        self.fail_on = fail_on
        self.calls = []

    def __call__(self, request: Request) -> Response:
        self.calls.append(request)
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("stub endpoint failure")
        if self.responses:
            return self.responses.pop(0)
        return Response(input_tokens=10, output_tokens=request.max_tokens, cost_micros=request.max_cost_micros)


class StubGpu:
    """A stub hosted-GPU lease. teardown() returns confirmed, or raises."""

    def __init__(self, confirmed=True, raises=False, start_raises=False):
        self.started = 0
        self.torn_down = 0
        self.confirmed = confirmed
        self.raises = raises
        self.start_raises = start_raises

    def start(self):
        self.started += 1
        if self.start_raises:
            raise RuntimeError("stub start failure")

    def teardown(self) -> bool:
        self.torn_down += 1
        if self.raises:
            raise RuntimeError("stub teardown failure")
        return self.confirmed


def requests(count, max_tokens=100, max_cost_micros=1_000):
    return [Request(max_tokens=max_tokens, max_cost_micros=max_cost_micros) for _ in range(count)]


def guarded(answer="yes", tty=True, **overrides):
    """Run the guard with the terminal stubbed: tty is whether standard
    input is a terminal, and answer is what the founder types (a string,
    a callable taking the prompt, or an exception to raise)."""
    args = dict(
        estimate=Estimate(run_count=1, price_per_run_micros=5_000, source=SOURCE),
        cap_micros=5_000,
        max_tokens_per_request=100,
        requests=requests(3),
        endpoint=StubEndpoint(),
        out=io.StringIO(),
        gpu=None,
    )
    args.update(overrides)
    if isinstance(answer, str):
        read = mock.Mock(return_value=answer)
    elif isinstance(answer, type) and issubclass(answer, BaseException):
        read = mock.Mock(side_effect=answer)
    else:
        read = mock.Mock(side_effect=answer)
    with mock.patch.object(spend_guard, "_stdin_is_tty", return_value=tty), \
            mock.patch.object(spend_guard, "_read_answer", read):
        return args, run_guarded(**args)


class EstimateTests(unittest.TestCase):
    def test_the_estimate_is_run_count_times_price_per_run_with_its_source(self):
        estimate = Estimate(run_count=4, price_per_run_micros=2_500_000, source=SOURCE)
        self.assertEqual(estimate.total_micros, 10_000_000)
        text = estimate.describe()
        self.assertIn("4 runs", text)
        self.assertIn("$2.50 per run", text)
        self.assertIn("$10.00", text)
        self.assertIn("stub price list", text)
        self.assertIn("2026-09-30", text)

    def test_the_guard_prints_the_estimate_and_cap_before_it_asks(self):
        asked = []
        out = io.StringIO()

        def confirm(prompt):
            asked.append(out.getvalue())
            return "no"

        guarded(answer=confirm, out=out)
        self.assertIn("Estimate", asked[0])
        self.assertIn("stub price list", asked[0])
        self.assertIn("Hard cap", asked[0])

    def test_an_estimate_needs_a_named_price_source(self):
        with self.assertRaises(ValueError):
            PriceSource(name="", reference="x", read_on=date(2026, 9, 30))
        with self.assertRaises(ValueError):
            Estimate(run_count=1, price_per_run_micros=1, source=None)

    def test_an_estimate_refuses_a_bad_count_or_price(self):
        for count, price in [(0, 1), (-1, 1), (1, -1), (True, 1), (1, 1.5)]:
            with self.subTest(count=count, price=price):
                with self.assertRaises(ValueError):
                    Estimate(run_count=count, price_per_run_micros=price, source=SOURCE)

    def test_usd_formatting(self):
        self.assertEqual(format_usd(0), "$0.00")
        self.assertEqual(format_usd(1_234_567), "$1.23")
        self.assertEqual(format_usd(10_000_000), "$10.00")


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
        gpu = StubGpu()
        args, result = guarded(tty=False, gpu=gpu)
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
            for request in requests(2):
                order.append("read")
                yield request

        def answer(_prompt):
            order.append("asked")
            return "yes"

        _args, result = guarded(answer=answer, requests=planned())
        self.assertEqual(order, ["read", "read", "asked"])
        self.assertEqual(result.status, RunStatus.COMPLETED)

    def test_no_parameter_skips_the_yes(self):
        parameters = set(inspect.signature(run_guarded).parameters)
        self.assertEqual(
            parameters,
            {
                "estimate", "cap_micros", "max_tokens_per_request", "requests",
                "endpoint", "out", "gpu",
            },
        )

    def test_the_guard_reads_no_environment_variable(self):
        tree = ast.parse(pathlib.Path(spend_guard.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
        }
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn("os", imported)
        self.assertNotIn("environ", names)
        self.assertNotIn("getenv", names)

    def test_an_approved_run_completes_and_records_its_spend(self):
        gpu = StubGpu()
        args, result = guarded(gpu=gpu)
        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(len(args["endpoint"].calls), 3)
        self.assertEqual(result.spent_micros, 3_000)
        self.assertEqual(gpu.started, 1)
        self.assertEqual(gpu.torn_down, 1)
        self.assertTrue(result.teardown_confirmed)


class CapTests(unittest.TestCase):
    def test_a_run_stops_before_a_request_that_could_pass_the_cap(self):
        gpu = StubGpu()
        args, result = guarded(cap_micros=2_500, requests=requests(3), gpu=gpu)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(len(args["endpoint"].calls), 2)
        self.assertEqual(result.spent_micros, 2_000)
        self.assertLessEqual(result.spent_micros, 2_500)
        self.assertIn("cap", result.reason)
        self.assertEqual(gpu.torn_down, 1)

    def test_a_response_that_costs_more_than_reserved_stops_the_run(self):
        endpoint = StubEndpoint(responses=[Response(input_tokens=1, output_tokens=1, cost_micros=4_000)])
        _args, result = guarded(cap_micros=10_000, endpoint=endpoint)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(len(endpoint.calls), 1)
        self.assertEqual(result.spent_micros, 4_000)
        self.assertIn("more than", result.reason)

    def test_a_response_that_passes_the_cap_is_recorded_and_stops_the_run(self):
        endpoint = StubEndpoint(responses=[Response(input_tokens=1, output_tokens=1, cost_micros=9_000)])
        _args, result = guarded(cap_micros=5_000, endpoint=endpoint)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(result.spent_micros, 9_000)

    def test_a_request_over_the_token_ceiling_is_never_sent(self):
        endpoint = StubEndpoint()
        _args, result = guarded(
            max_tokens_per_request=100, requests=[Request(max_tokens=101, max_cost_micros=1)], endpoint=endpoint
        )
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(endpoint.calls, [])
        self.assertIn("token ceiling", result.reason)

    def test_the_cap_and_ceiling_must_be_set(self):
        for cap, ceiling in [(0, 1), (-1, 1), (1, 0), (None, 1), (1, None), (True, 1)]:
            with self.subTest(cap=cap, ceiling=ceiling):
                with self.assertRaises(ValueError):
                    SpendTally(cap_micros=cap, max_tokens_per_request=ceiling)

    def test_the_tally_itself(self):
        tally = SpendTally(cap_micros=1_000, max_tokens_per_request=10)
        tally.reserve(Request(max_tokens=10, max_cost_micros=600))
        tally.settle(Request(max_tokens=10, max_cost_micros=600), Response(1, 1, 500))
        # A response that reports less than its reservation counts at the
        # reservation, so an under-reporting endpoint cannot lower the tally.
        self.assertEqual(tally.spent_micros, 600)
        with self.assertRaises(CapReached):
            tally.reserve(Request(max_tokens=10, max_cost_micros=600))
        with self.assertRaises(CostUnknown):
            tally.reserve(Request(max_tokens=10, max_cost_micros=0))
        with self.assertRaises(TokenCeilingExceeded):
            tally.reserve(Request(max_tokens=11, max_cost_micros=1))

    def test_an_endpoint_that_reports_no_cost_is_counted_at_its_reservation(self):
        endpoint = StubEndpoint(responses=[Response(1, 1, 0), Response(1, 1, 0), Response(1, 1, 0)])
        _args, result = guarded(cap_micros=2_500, endpoint=endpoint)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertEqual(result.spent_micros, 2_000)
        self.assertEqual(len(endpoint.calls), 2)

    def test_a_response_over_its_token_ceiling_stops_the_run(self):
        endpoint = StubEndpoint(responses=[Response(input_tokens=1, output_tokens=101, cost_micros=500)])
        _args, result = guarded(endpoint=endpoint)
        self.assertEqual(result.status, RunStatus.STOPPED_AT_CAP)
        self.assertIn("more tokens", result.reason)

    def test_a_response_with_no_valid_cost_fails_the_run(self):
        for cost in [-1, 1.5, "5", None, True]:
            with self.subTest(cost=cost):
                endpoint = StubEndpoint(responses=[Response(input_tokens=1, output_tokens=1, cost_micros=cost)])
                _args, result = guarded(endpoint=endpoint)
                self.assertEqual(result.status, RunStatus.FAILED)
                self.assertEqual(result.spent_micros, 1_000)

    def test_a_request_with_no_cost_bound_is_never_sent(self):
        endpoint = StubEndpoint()
        _args, result = guarded(requests=[Request(max_tokens=10, max_cost_micros=0)], endpoint=endpoint)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(endpoint.calls, [])

    def test_a_price_above_the_cap_is_noted_before_the_ask(self):
        out = io.StringIO()
        guarded(answer="no", cap_micros=4_000, out=out)
        self.assertIn("above the hard cap", out.getvalue())

    def test_the_result_records_the_cap_and_why_it_stopped(self):
        _args, result = guarded(cap_micros=2_500)
        record = result.as_record()
        self.assertEqual(record["status"], "stopped_at_cap")
        self.assertEqual(record["cap_micros"], 2_500)
        self.assertEqual(record["spent_micros"], 2_000)
        self.assertTrue(record["reason"])


class TeardownTests(unittest.TestCase):
    def test_the_gpu_is_torn_down_after_an_endpoint_error(self):
        gpu = StubGpu()
        _args, result = guarded(endpoint=StubEndpoint(fail_on=2), gpu=gpu)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(gpu.torn_down, 1)
        # The failed request may still be billed, so it counts at its
        # reservation.
        self.assertEqual(result.spent_micros, 2_000)
        self.assertTrue(result.notes)

    def test_the_gpu_is_torn_down_when_it_fails_to_start(self):
        gpu = StubGpu(start_raises=True)
        args, result = guarded(gpu=gpu)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(gpu.torn_down, 1)
        self.assertEqual(args["endpoint"].calls, [])

    def test_a_truthy_teardown_that_is_not_true_fails_the_run(self):
        class OneGpu(StubGpu):
            def teardown(self):
                super().teardown()
                return 1

        gpu = OneGpu()
        _args, result = guarded(gpu=gpu)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertFalse(result.teardown_confirmed)

    def test_an_interrupt_during_teardown_is_recorded_and_raised(self):
        class InterruptedGpu(StubGpu):
            def teardown(self):
                super().teardown()
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            guarded(gpu=InterruptedGpu())

    def test_an_endpoint_failure_and_a_failed_teardown_both_show(self):
        gpu = StubGpu(confirmed=False)
        _args, result = guarded(endpoint=StubEndpoint(fail_on=1), gpu=gpu)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("RuntimeError", result.reason)
        self.assertIn("teardown", result.reason)

    def test_an_unconfirmed_teardown_fails_the_run(self):
        gpu = StubGpu(confirmed=False)
        _args, result = guarded(gpu=gpu)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertFalse(result.teardown_confirmed)
        self.assertIn("teardown", result.reason)

    def test_a_teardown_that_raises_fails_the_run(self):
        gpu = StubGpu(raises=True)
        _args, result = guarded(gpu=gpu)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertFalse(result.teardown_confirmed)

    def test_an_unconfirmed_teardown_after_a_cap_stop_still_fails_the_run(self):
        gpu = StubGpu(confirmed=False)
        _args, result = guarded(cap_micros=2_500, gpu=gpu)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("cap", result.reason)
        self.assertIn("teardown", result.reason)

    def test_an_endpoint_error_message_is_not_echoed(self):
        class LeakyEndpoint(StubEndpoint):
            def __call__(self, request):
                raise RuntimeError("Authorization: Bearer stub-token-not-real")

        out = io.StringIO()
        _args, result = guarded(endpoint=LeakyEndpoint(), out=out)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertNotIn("stub-token", result.reason)
        self.assertNotIn("stub-token", out.getvalue())
        self.assertIn("RuntimeError", result.reason)


if __name__ == "__main__":
    unittest.main()
