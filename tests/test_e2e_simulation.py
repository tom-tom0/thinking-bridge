"""End-to-end scenarios: the real library against the mock API server that
implements the documented preserved-thinking semantics. Zero network."""

import unittest

from thinking_bridge import (
    Conversation,
    PrefixGuard,
    PrefixMismatchError,
    ThinkingBridge,
)

from mock_server import MockAPIError, MockClaudeServer


def make(server=None, **bridge_kwargs):
    server = server or MockClaudeServer()
    bridge = ThinkingBridge(server, **bridge_kwargs)
    return server, bridge


class TestSameModelSessions(unittest.TestCase):
    def test_ten_turns_no_drops(self):
        server, bridge = make()
        guard = PrefixGuard()
        conv = Conversation(bridge, "claude-fable-5-1", system="You are helpful.")
        for i in range(10):
            result = conv.ask(f"question {i}", guard=guard)
            self.assertEqual(result.transformations, [])
        # All 10 earlier thinking blocks were delivered on the last turn.
        self.assertEqual(len(server.last_delivered_thinking), 9)

    def test_tool_turns_survive(self):
        server, bridge = make()
        tools = [{"name": "search", "input_schema": {"type": "object"}}]
        conv = Conversation(bridge, "claude-fable-5-1", tools=tools)
        guard = PrefixGuard()
        for i in range(3):
            result = conv.ask(f"go {i}", guard=guard)
            self.assertEqual(result.transformations, [])


class TestModelSwitching(unittest.TestCase):
    def test_downgrade_reports_expected_drops_and_carries_over(self):
        server, bridge = make()
        conv = Conversation(bridge, "claude-fable-5-1", handoff_channel="system")
        conv.ask("plan")
        conv.ask("refine")

        predicted = bridge.predict_switch(conv.messages, "claude-opus-5", conv.producers)
        self.assertEqual(len(predicted), 2)

        drops = conv.switch_model("claude-opus-5")
        self.assertEqual([d.reason for d in drops], ["model_binding_mismatch"] * 2)
        result = conv.ask("continue")

        # The API reported exactly the predicted drops.
        reported = {t.path for t in result.model_switch_drops}
        self.assertEqual(reported, {p.path for p in predicted})
        # The Fable 5.1 blocks never reached Opus 5.
        self.assertEqual(server.last_delivered_thinking, [])
        # The handoff note carried the reasoning summaries.
        handoff = next(m for m in conv.messages if m["role"] == "system")
        self.assertIn("Reasoning summary for turn 1", handoff["content"])

    def test_round_trip_restores_readability(self):
        server, bridge = make()
        conv = Conversation(bridge, "claude-fable-5-1")
        conv.ask("plan")
        conv.switch_model("claude-opus-5")
        conv.ask("step 1")
        drops = conv.switch_model("claude-fable-5-1")
        self.assertEqual(drops, [])  # Fable 5.1 reads everything
        result = conv.ask("review")
        self.assertEqual(result.transformations, [])
        # Both the Fable 5.1 block AND the Opus 5 block were delivered.
        self.assertEqual(len(server.last_delivered_thinking), 2)

    def test_downgrade_to_sonnet_uses_user_prefix_and_passes(self):
        # Sonnet 5 rejects mid-conversation system messages; the mock enforces
        # that, so this also proves the carryover picked the right channel.
        server, bridge = make()
        conv = Conversation(bridge, "claude-fable-5-1")
        conv.ask("plan")
        conv.switch_model("claude-sonnet-5")
        result = conv.ask("continue")  # would 400 if a system msg were used
        self.assertEqual(len(result.model_switch_drops), 1)

    def test_silent_drops_without_controls(self):
        server = MockClaudeServer()
        _, bridge = make(server, binding_controls=False)
        conv = Conversation(bridge, "claude-fable-5-1")
        conv.ask("plan")
        conv.switch_model("claude-opus-5", carryover=False)
        result = conv.ask("continue")
        self.assertIsNone(result.transformations)  # absent without the header
        self.assertEqual(server.last_delivered_thinking, [])  # dropped anyway

    def test_haiku_budget_thinking_with_binding_field(self):
        server, bridge = make()
        result = bridge.create(
            model="claude-haiku-4-5",
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 2048},
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertEqual(result.transformations, [])


class TestHistoryEdits(unittest.TestCase):
    def _session(self, **bridge_kwargs):
        server, bridge = make(**bridge_kwargs)
        conv = Conversation(bridge, "claude-fable-5-1", system="sys")
        conv.ask("one")
        conv.ask("two")
        return server, bridge, conv

    def test_edit_with_error_behavior_400s(self):
        server, bridge, conv = self._session(prefix_mismatch_behavior="error")
        conv.messages[0]["content"] = "EDITED"
        conv.messages.append({"role": "user", "content": "three"})
        with self.assertRaises(MockAPIError) as ctx:
            bridge.create(
                model="claude-fable-5-1", max_tokens=1024,
                messages=conv.messages, system="sys",
            )
        self.assertIn("bound to a different conversation", str(ctx.exception))

    def test_edit_with_drop_block_cascades(self):
        server, bridge, conv = self._session()  # drop_block default
        conv.messages[0]["content"] = "EDITED"
        conv.messages.append({"role": "user", "content": "three"})
        result = bridge.create(
            model="claude-fable-5-1", max_tokens=1024,
            messages=conv.messages, system="sys",
        )
        # First mismatched block AND every later thinking block dropped.
        self.assertEqual(len(result.history_edit_drops), 2)
        self.assertEqual(server.last_delivered_thinking, [])

    def test_system_prompt_edit_invalidates(self):
        server, bridge, conv = self._session()
        conv.messages.append({"role": "user", "content": "three"})
        result = bridge.create(
            model="claude-fable-5-1", max_tokens=1024,
            messages=conv.messages, system="DIFFERENT SYSTEM",
        )
        self.assertEqual(len(result.history_edit_drops), 2)

    def test_tools_reorder_is_not_an_edit(self):
        server, bridge = make()
        t1 = {"name": "alpha", "input_schema": {"type": "object"}}
        t2 = {"name": "beta", "input_schema": {"type": "object"}}
        messages = [{"role": "user", "content": "one"}]
        r1 = bridge.create(model="claude-fable-5-1", max_tokens=64, messages=messages, tools=[t1, t2])
        messages.append({"role": "assistant", "content": r1.response.content})
        messages.append({"role": "user", "content": "two"})
        r2 = bridge.create(model="claude-fable-5-1", max_tokens=64, messages=messages, tools=[t2, t1])
        self.assertEqual(r2.transformations, [])

    def test_cache_control_move_is_not_an_edit(self):
        server, bridge = make()
        messages = [{"role": "user", "content": [{"type": "text", "text": "one"}]}]
        r1 = bridge.create(model="claude-fable-5-1", max_tokens=64, messages=messages)
        messages[0]["content"][0]["cache_control"] = {"type": "ephemeral"}
        messages.append({"role": "assistant", "content": r1.response.content})
        messages.append({"role": "user", "content": "two"})
        r2 = bridge.create(model="claude-fable-5-1", max_tokens=64, messages=messages)
        self.assertEqual(r2.transformations, [])

    def test_leading_thinking_removal_is_legal(self):
        server, bridge = make()
        conv = Conversation(bridge, "claude-fable-5-1")
        conv.ask("one")
        conv.ask("two")
        conv.ask("three")
        # Remove the OLDEST thinking block (message 1 is the first assistant turn).
        conv.messages[1]["content"] = [b for b in conv.messages[1]["content"] if b["type"] != "thinking"]
        conv.messages.append({"role": "user", "content": "four"})
        result = bridge.create(
            model="claude-fable-5-1", max_tokens=64, messages=conv.messages
        )
        self.assertEqual(result.transformations, [])

    def test_middle_thinking_removal_breaks_the_chain(self):
        server, bridge = make(prefix_mismatch_behavior="error")
        conv = Conversation(bridge, "claude-fable-5-1")
        conv.ask("one")
        conv.ask("two")
        conv.ask("three")
        # Remove the MIDDLE thinking block (second assistant turn, message 3).
        conv.messages[3]["content"] = [b for b in conv.messages[3]["content"] if b["type"] != "thinking"]
        conv.messages.append({"role": "user", "content": "four"})
        with self.assertRaises(MockAPIError):
            bridge.create(model="claude-fable-5-1", max_tokens=64, messages=conv.messages)

    def test_simple_compaction_restart_passes(self):
        server, bridge = make()
        guard = PrefixGuard()
        conv = Conversation(bridge, "claude-fable-5-1")
        for i in range(4):
            conv.ask(f"q{i}", guard=guard)
        fresh = [{"role": "user", "content": "Summary of the work so far: ... Continue with step 5."}]
        result = bridge.create(
            model="claude-fable-5-1", max_tokens=64, messages=fresh, guard=guard
        )
        self.assertEqual(result.transformations, [])

    def test_tampered_signature_is_a_hard_400(self):
        server, bridge = make()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": "forged-sig"},
                {"type": "text", "text": "x"},
            ]},
            {"role": "user", "content": "next"},
        ]
        with self.assertRaises(MockAPIError) as ctx:
            bridge.create(model="claude-fable-5-1", max_tokens=64, messages=messages)
        self.assertNotIn("bound to a different conversation", str(ctx.exception))


class TestGuardServerAgreement(unittest.TestCase):
    """A request that passes the client-side PrefixGuard must never draw a
    prefix-binding 400 from the (error-mode) server."""

    def test_guard_blocks_before_server_would(self):
        server, bridge = make(prefix_mismatch_behavior="error")
        guard = PrefixGuard()
        conv = Conversation(bridge, "claude-fable-5-1", system="sys")
        conv.ask("one", guard=guard)
        conv.ask("two", guard=guard)
        conv.messages[0]["content"] = "EDITED"
        conv.messages.append({"role": "user", "content": "three"})
        # The guard fires client-side; the server never sees the bad request.
        requests_before = len(server.requests)
        with self.assertRaises(PrefixMismatchError):
            bridge.create(
                model="claude-fable-5-1", max_tokens=64,
                messages=conv.messages, system="sys", guard=guard,
            )
        self.assertEqual(len(server.requests), requests_before)


class TestNoControlsPlatform(unittest.TestCase):
    def test_header_rejected_when_unavailable(self):
        server = MockClaudeServer(controls_available=False)
        _, bridge = make(server)  # controls on -> header sent -> 400
        with self.assertRaises(MockAPIError):
            bridge.create(
                model="claude-fable-5-1", max_tokens=64,
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_strip_and_retry_recovers_on_enforced_account(self):
        server = MockClaudeServer(controls_available=False, enforced=True)
        _, bridge = make(server, binding_controls=False, strip_and_retry=True)
        conv = Conversation(bridge, "claude-fable-5-1", system="sys")
        conv.ask("one")
        conv.ask("two")
        conv.messages[0]["content"] = "EDITED"  # harness bug
        conv.messages.append({"role": "user", "content": "three"})
        result = bridge.create(
            model="claude-fable-5-1", max_tokens=64,
            messages=conv.messages, system="sys",
        )
        self.assertTrue(result.stripped_and_retried)
        self.assertEqual(server.last_delivered_thinking, [])

    def test_unenforced_account_records_only(self):
        server = MockClaudeServer(controls_available=False, enforced=False)
        _, bridge = make(server, binding_controls=False)
        conv = Conversation(bridge, "claude-fable-5-1", system="sys")
        conv.ask("one")
        conv.messages[0]["content"] = "EDITED"
        conv.messages.append({"role": "user", "content": "two"})
        result = bridge.create(
            model="claude-fable-5-1", max_tokens=64,
            messages=conv.messages, system="sys",
        )
        self.assertIsNone(result.transformations)  # silent — the trap the
        # README warns about: no header means no signal at all.


class TestRawFieldMisuse(unittest.TestCase):
    def test_block_binding_without_header_400s(self):
        server = MockClaudeServer()
        with self.assertRaises(MockAPIError) as ctx:
            server.beta.messages.create(
                model="claude-fable-5-1", max_tokens=64,
                thinking={"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
                messages=[{"role": "user", "content": "hi"}],
            )
        self.assertIn("Extra inputs are not permitted", str(ctx.exception))

    def test_bridge_never_triggers_that_400(self):
        # binding_controls=False strips the field even if the caller passes it.
        server, bridge = make(binding_controls=False)
        result = bridge.create(
            model="claude-fable-5-1", max_tokens=64,
            thinking={"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}},
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIsNone(result.transformations)


if __name__ == "__main__":
    unittest.main()
