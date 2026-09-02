"""Closed-environment benchmarks for thinking-bridge.

Everything runs against the in-process mock server (tests/mock_server.py) —
no network, no tokens spent. Measures the library's own overhead, which is
what a developer adds to their request path by adopting it.

Run from the project root:
    PYTHONPATH=.:tests python3 benchmarks/bench.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time

from thinking_bridge import (
    Conversation,
    PrefixGuard,
    ThinkingBridge,
    parse_input_transformations,
    predict_drops,
)

sys.path.insert(0, "tests")
from mock_server import MockClaudeServer  # noqa: E402


def timeit(fn, repeat=7, number=1):
    """Best-of-N wall time per call, in milliseconds."""
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        samples.append((time.perf_counter() - t0) / number * 1000)
    return min(samples), statistics.median(samples)


def build_history(n_messages: int, producer="claude-fable-5-1"):
    messages, producers = [], []
    for i in range(n_messages // 2):
        messages.append({"role": "user", "content": [{"type": "text", "text": f"question {i} " + "x" * 200}]})
        producers.append(None)
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "summary " * 30, "signature": f"sig-{i}"},
                    {"type": "text", "text": f"answer {i} " + "y" * 400},
                ],
            }
        )
        producers.append(producer)
    return messages, producers


def bench_guard(results):
    for n in (100, 1000, 5000):
        messages, _ = build_history(n)
        guard = PrefixGuard()
        guard.check(messages, system="sys " * 100, tools=[{"name": f"t{i}", "input_schema": {}} for i in range(20)])
        extended = messages + [{"role": "user", "content": "next"}]
        best, med = timeit(
            lambda: PrefixGuard.restore(guard.snapshot()).check(
                extended, system="sys " * 100, tools=[{"name": f"t{i}", "input_schema": {}} for i in range(20)]
            )
        )
        results.append((f"PrefixGuard.check, {n}-message history (restore+full check)", best, med))

    # Steady-state: one long-lived guard, incremental appends.
    messages, _ = build_history(1000)
    guard = PrefixGuard()
    guard.check(messages)
    state = {"n": 1000}

    def step():
        messages.append({"role": "user", "content": f"turn {state['n']}"})
        state["n"] += 1
        guard.check(messages)

    best, med = timeit(step, repeat=7, number=20)
    results.append(("PrefixGuard.check, steady-state append (1000+ msgs)", best, med))


def bench_predict(results):
    for n in (100, 1000, 5000):
        messages, producers = build_history(n)
        best, med = timeit(lambda: predict_drops(messages, "claude-opus-5", producers))
        results.append((f"predict_drops, {n}-message history", best, med))


def bench_parse(results):
    payload = {
        "input_transformations": [
            {"type": "thinking_dropped", "path": f"messages.{i}.content.0", "reason": "model_binding_mismatch"}
            for i in range(1000)
        ]
    }
    best, med = timeit(lambda: parse_input_transformations(payload))
    results.append(("parse_input_transformations, 1000 entries", best, med))


def bench_end_to_end(results):
    def run_session(turns=100):
        server = MockClaudeServer()
        bridge = ThinkingBridge(server)
        guard = PrefixGuard()
        conv = Conversation(bridge, "claude-fable-5-1")
        models = ["claude-fable-5-1", "claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"]
        for i in range(turns):
            if i and i % 10 == 0:
                conv.switch_model(models[(i // 10) % len(models)])
            conv.ask(f"turn {i}", guard=guard)
        return conv

    t0 = time.perf_counter()
    conv = run_session(100)
    elapsed = time.perf_counter() - t0
    results.append(("Full session: 100 turns, model switch every 10 (bridge+guard+mock)", elapsed * 1000 / 100, elapsed * 1000 / 100))

    # Library overhead vs calling the mock server directly with a prebuilt body.
    server = MockClaudeServer()
    bridge = ThinkingBridge(server)
    messages, _ = build_history(200)
    # Register signatures by replaying through the server once is not possible
    # for synthetic sigs, so use a fresh growing conversation instead:
    server2 = MockClaudeServer()
    bridge2 = ThinkingBridge(server2)
    conv = Conversation(bridge2, "claude-fable-5-1")
    for i in range(50):
        conv.ask(f"warmup {i}")

    raw_body = dict(
        model="claude-fable-5-1",
        max_tokens=64,
        messages=list(conv.messages) + [{"role": "user", "content": "bench"}],
        thinking={"type": "adaptive", "display": "summarized",
                  "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
        betas=["thinking-binding-controls-2026-08-01"],
    )
    best_raw, med_raw = timeit(lambda: server2.beta.messages.create(**dict(raw_body)), repeat=7, number=5)
    best_bridge, med_bridge = timeit(
        lambda: bridge2.create(
            model="claude-fable-5-1",
            max_tokens=64,
            messages=raw_body["messages"],
        ),
        repeat=7,
        number=5,
    )
    results.append(("mock server alone, 100-msg body (baseline)", best_raw, med_raw))
    results.append(("bridge.create, same body (baseline + library overhead)", best_bridge, med_bridge))
    results.append(("=> library overhead per request", best_bridge - best_raw, med_bridge - med_raw))


def main():
    results = []
    bench_guard(results)
    bench_predict(results)
    bench_parse(results)
    bench_end_to_end(results)

    width = max(len(r[0]) for r in results)
    lines = [
        f"{'benchmark'.ljust(width)}  best(ms)  median(ms)",
        f"{'-' * width}  --------  ----------",
    ]
    for name, best, med in results:
        lines.append(f"{name.ljust(width)}  {best:8.3f}  {med:10.3f}")
    report = "\n".join(lines)
    print(report)
    with open("benchmarks/results.txt", "w") as fh:
        fh.write(report + "\n")
    print("\nsaved to benchmarks/results.txt")


if __name__ == "__main__":
    main()
