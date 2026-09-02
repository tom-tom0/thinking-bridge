"""Property-based fuzz tests (seeded, deterministic).

Property 1 — soundness: a randomly generated LEGAL session (asks, model
switches in both directions, simple-compaction restarts, cache_control
churn, leading-thinking removals) never draws a 400 from the strict mock
server and never produces a prefix_binding_mismatch.

Property 2 — completeness: a randomly TAMPERED history is always caught —
by the client-side PrefixGuard, or by the error-mode server, or (in
drop_block mode) reported as prefix_binding_mismatch drops.

Property 3 — prediction accuracy: predict_switch() names exactly the drops
the server then reports on the next request after a switch.
"""

import copy
import random
import unittest

from thinking_bridge import (
    Conversation,
    PrefixGuard,
    PrefixMismatchError,
    ThinkingBridge,
)

from mock_server import MockAPIError, MockClaudeServer

MODELS = ["claude-fable-5-1", "claude-opus-5", "claude-fable-5", "claude-sonnet-5", "claude-haiku-4-5"]
ITERATIONS = 150


def run_legal_session(seed: int):
    rng = random.Random(seed)
    server = MockClaudeServer(enforced=True)
    bridge = ThinkingBridge(server, prefix_mismatch_behavior="error")
    guard = PrefixGuard()
    conv = Conversation(bridge, rng.choice(MODELS))
    results = []
    for step in range(rng.randint(3, 12)):
        op = rng.random()
        if op < 0.55 or not conv.messages:
            kwargs = {}
            if rng.random() < 0.3 and conv.model == "claude-haiku-4-5":
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1024}
            results.append(conv.ask(f"step {step}", guard=guard, **kwargs))
        elif op < 0.80:
            conv.switch_model(rng.choice(MODELS))
        elif op < 0.90:
            # cache_control churn on a random earlier user message (legal).
            candidates = [m for m in conv.messages if m["role"] == "user" and isinstance(m["content"], list)]
            if candidates:
                target = rng.choice(candidates)
                block = target["content"][0]
                if "cache_control" in block:
                    del block["cache_control"]
                else:
                    block["cache_control"] = {"type": "ephemeral"}
        else:
            # Simple compaction restart: brand-new history, nothing replayed.
            conv = Conversation(bridge, conv.model)
            results.append(conv.ask(f"Summary of earlier work (step {step}). Continue.", guard=guard))
    return results


class TestLegalSessionsNeverBreak(unittest.TestCase):
    def test_fuzz_legal_sessions(self):
        for seed in range(ITERATIONS):
            with self.subTest(seed=seed):
                try:
                    results = run_legal_session(seed)
                except (MockAPIError, PrefixMismatchError) as exc:
                    self.fail(f"seed {seed}: legal session raised {exc!r}")
                for r in results:
                    bad = [t for t in (r.transformations or []) if t.is_history_edit]
                    self.assertEqual(bad, [], f"seed {seed}: legal session got prefix drops {bad}")


def build_session(seed: int, turns: int = 4):
    rng = random.Random(seed)
    server = MockClaudeServer(enforced=True)
    bridge_error = ThinkingBridge(server, prefix_mismatch_behavior="error")
    conv = Conversation(bridge_error, "claude-fable-5-1", system="sys")
    for i in range(turns):
        conv.ask(f"turn {i}")
    return rng, server, conv


TAMPER_KINDS = ("edit_user", "edit_text", "delete_turn", "reorder", "middle_thinking", "system_swap")


def tamper(rng, conv, kind):
    messages = copy.deepcopy(conv.messages)
    system = conv.system
    if kind == "edit_user":
        idx = rng.choice([i for i, m in enumerate(messages[:-1]) if m["role"] == "user"])
        messages[idx]["content"] = "TAMPERED"
    elif kind == "edit_text":
        idx = rng.choice([i for i, m in enumerate(messages[:-1]) if m["role"] == "assistant"])
        for b in messages[idx]["content"]:
            if b["type"] == "text":
                b["text"] = "TAMPERED"
    elif kind == "delete_turn":
        idx = rng.randrange(0, len(messages) - 2)
        del messages[idx]
    elif kind == "reorder":
        messages[0], messages[2] = messages[2], messages[0]
    elif kind == "middle_thinking":
        assistants = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        idx = assistants[len(assistants) // 2]
        messages[idx]["content"] = [b for b in messages[idx]["content"] if b["type"] != "thinking"]
    elif kind == "system_swap":
        system = "sys TAMPERED"
    messages.append({"role": "user", "content": "next"})
    return messages, system


class TestTamperingAlwaysCaught(unittest.TestCase):
    def test_fuzz_tampering_error_mode(self):
        """Error-mode server: every tamper kind must 400 (or be caught by the
        guard first when one is used)."""
        for seed in range(ITERATIONS):
            kind = TAMPER_KINDS[seed % len(TAMPER_KINDS)]
            with self.subTest(seed=seed, kind=kind):
                rng, server, conv = build_session(seed)
                messages, system = tamper(rng, conv, kind)
                bridge = ThinkingBridge(server, prefix_mismatch_behavior="error")
                caught = False
                try:
                    bridge.create(
                        model="claude-fable-5-1", max_tokens=64,
                        messages=messages, system=system,
                    )
                except MockAPIError:
                    caught = True
                self.assertTrue(caught, f"seed {seed}: tamper {kind!r} was not rejected")

    def test_fuzz_tampering_guard_catches_before_server(self):
        """The client-side guard must catch every tamper kind it claims to
        cover (everything except middle-thinking removal, which the guard
        permits structurally but the server's chain check rejects)."""
        guard_covered = ("edit_user", "edit_text", "delete_turn", "reorder", "system_swap")
        for seed in range(ITERATIONS):
            kind = guard_covered[seed % len(guard_covered)]
            with self.subTest(seed=seed, kind=kind):
                rng, server, conv = build_session(seed)
                guard = PrefixGuard()
                guard.check(conv.messages, system=conv.system, tools=None)
                messages, system = tamper(rng, conv, kind)
                with self.assertRaises(PrefixMismatchError):
                    guard.check(messages, system=system, tools=None)

    def test_fuzz_tampering_drop_block_mode_reports(self):
        """drop_block-mode server: every tamper kind must surface as
        prefix_binding_mismatch drops instead of a 400."""
        for seed in range(ITERATIONS):
            kind = TAMPER_KINDS[seed % len(TAMPER_KINDS)]
            with self.subTest(seed=seed, kind=kind):
                rng, server, conv = build_session(seed)
                messages, system = tamper(rng, conv, kind)
                bridge = ThinkingBridge(server, prefix_mismatch_behavior="drop_block")
                result = bridge.create(
                    model="claude-fable-5-1", max_tokens=64,
                    messages=messages, system=system,
                )
                self.assertTrue(
                    result.history_edit_drops,
                    f"seed {seed}: tamper {kind!r} produced no prefix drops",
                )


class TestPredictionMatchesServer(unittest.TestCase):
    def test_fuzz_predicted_drops_equal_reported_drops(self):
        for seed in range(ITERATIONS):
            rng = random.Random(10_000 + seed)
            server = MockClaudeServer()
            bridge = ThinkingBridge(server)
            conv = Conversation(bridge, rng.choice(MODELS))
            for i in range(rng.randint(1, 5)):
                conv.ask(f"turn {i}")
                if rng.random() < 0.4:
                    conv.switch_model(rng.choice(MODELS), carryover=False)
            target = rng.choice(MODELS)
            predicted = {p.path for p in bridge.predict_switch(conv.messages, target, conv.producers)}
            conv.switch_model(target, carryover=False)
            result = conv.ask("final")
            reported = {t.path for t in result.model_switch_drops}
            self.assertEqual(
                predicted, reported,
                f"seed {seed}: predicted {predicted} != reported {reported} (target {target})",
            )


if __name__ == "__main__":
    unittest.main()
