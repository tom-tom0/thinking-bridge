import unittest
from types import SimpleNamespace

from thinking_bridge import (
    BINDING_BETA,
    Conversation,
    PrefixGuard,
    PrefixMismatchError,
    ThinkingBridge,
    parse_input_transformations,
    strip_thinking_blocks,
)


class FakeResponse(SimpleNamespace):
    pass


def make_response(model="claude-fable-5-1", transformations=None, content=None):
    kwargs = dict(
        model=model,
        content=content
        or [
            {"type": "thinking", "thinking": "I considered X then Y.", "signature": "s"},
            {"type": "text", "text": "Done."},
        ],
        stop_reason="end_turn",
    )
    if transformations is not None:
        kwargs["input_transformations"] = transformations
    return FakeResponse(**kwargs)


class FakeAPIError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self, responses):
        self.beta = SimpleNamespace(messages=FakeMessages(responses))

    @property
    def calls(self):
        return self.beta.messages.calls


class TestRequestShaping(unittest.TestCase):
    def test_beta_header_and_block_binding_injected(self):
        client = FakeClient([make_response(transformations=[])])
        bridge = ThinkingBridge(client)
        bridge.create(
            model="claude-fable-5-1",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
        )
        call = client.calls[0]
        self.assertIn(BINDING_BETA, call["betas"])
        self.assertEqual(call["thinking"]["type"], "adaptive")
        self.assertEqual(
            call["thinking"]["block_binding"]["prefix_mismatch_behavior"],
            "drop_block",
        )
        # capture_reasoning defaults display to summarized
        self.assertEqual(call["thinking"]["display"], "summarized")

    def test_user_betas_merged_not_replaced(self):
        client = FakeClient([make_response(transformations=[])])
        bridge = ThinkingBridge(client)
        bridge.create(
            model="claude-fable-5-1",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            betas=["compact-2026-01-12"],
        )
        self.assertEqual(
            client.calls[0]["betas"], ["compact-2026-01-12", BINDING_BETA]
        )

    def test_enabled_budget_thinking_preserved(self):
        # Older models (e.g. Haiku 4.5) still use enabled + budget_tokens;
        # block_binding is accepted alongside "enabled".
        client = FakeClient([make_response(model="claude-haiku-4-5")])
        bridge = ThinkingBridge(client, capture_reasoning=False)
        bridge.create(
            model="claude-haiku-4-5",
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 2048},
            messages=[{"role": "user", "content": "hi"}],
        )
        thinking = client.calls[0]["thinking"]
        self.assertEqual(thinking["type"], "enabled")
        self.assertEqual(thinking["budget_tokens"], 2048)
        self.assertIn("block_binding", thinking)

    def test_no_controls_mode_sends_neither_header_nor_field(self):
        client = FakeClient([make_response()])
        bridge = ThinkingBridge(client, binding_controls=False)
        bridge.create(
            model="claude-fable-5-1",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
        )
        call = client.calls[0]
        self.assertNotIn("betas", call)
        self.assertNotIn("block_binding", call["thinking"])

    def test_invalid_behavior_rejected(self):
        with self.assertRaises(ValueError):
            ThinkingBridge(FakeClient([]), prefix_mismatch_behavior="mismatch")


class TestDropReporting(unittest.TestCase):
    def test_on_drop_called_and_classified(self):
        seen = []
        client = FakeClient(
            [
                make_response(
                    transformations=[
                        {
                            "type": "thinking_dropped",
                            "path": "messages.1.content.0",
                            "reason": "model_binding_mismatch",
                        },
                        {
                            "type": "future_thing",
                            "path": "messages.3.content.0",
                            "reason": "some_new_reason",
                        },
                    ]
                )
            ]
        )
        bridge = ThinkingBridge(client, on_drop=seen.extend)
        result = bridge.create(
            model="claude-opus-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(result.model_switch_drops), 1)
        self.assertEqual(result.history_edit_drops, [])
        # Unknown types/reasons are kept but not treated as errors.
        self.assertFalse(seen[1].is_recognized)

    def test_absent_field_means_none(self):
        self.assertIsNone(parse_input_transformations({"model": "m"}))
        result = parse_input_transformations({"input_transformations": []})
        self.assertEqual(result, [])


class TestStripAndRetry(unittest.TestCase):
    ERR = (
        "messages.5.content.0: Invalid `signature` in `thinking` block. "
        "The block is bound to a different conversation."
    )

    def _messages(self):
        return [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "s"},
                    {"type": "text", "text": "ok"},
                ],
            },
            {"role": "user", "content": "next"},
        ]

    def test_recovers_once_when_enabled(self):
        client = FakeClient([FakeAPIError(400, self.ERR), make_response()])
        bridge = ThinkingBridge(client, binding_controls=False, strip_and_retry=True)
        result = bridge.create(
            model="claude-fable-5-1", max_tokens=1024, messages=self._messages()
        )
        self.assertTrue(result.stripped_and_retried)
        retry_messages = client.calls[1]["messages"]
        assistant_blocks = retry_messages[1]["content"]
        self.assertEqual([b["type"] for b in assistant_blocks], ["text"])

    def test_tampered_signature_not_retried(self):
        err = FakeAPIError(
            400, "messages.5.content.0: Invalid `signature` in `thinking` block."
        )
        client = FakeClient([err])
        bridge = ThinkingBridge(client, binding_controls=False, strip_and_retry=True)
        with self.assertRaises(FakeAPIError):
            bridge.create(
                model="claude-fable-5-1", max_tokens=1024, messages=self._messages()
            )

    def test_not_retried_when_controls_active(self):
        client = FakeClient([FakeAPIError(400, self.ERR)])
        bridge = ThinkingBridge(client, strip_and_retry=True)  # controls on
        with self.assertRaises(FakeAPIError):
            bridge.create(
                model="claude-fable-5-1", max_tokens=1024, messages=self._messages()
            )

    def test_strip_helper_keeps_text_and_tool_use(self):
        stripped = strip_thinking_blocks(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "", "signature": "s"},
                        {"type": "redacted_thinking", "data": "x"},
                        {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
                    ],
                }
            ]
        )
        self.assertEqual([b["type"] for b in stripped[0]["content"]], ["tool_use"])


class TestConversation(unittest.TestCase):
    def test_switch_down_appends_system_handoff_when_opted_in(self):
        client = FakeClient([make_response(), make_response(model="claude-opus-5")])
        bridge = ThinkingBridge(client)
        conv = Conversation(bridge, "claude-fable-5-1", handoff_channel="system")
        conv.ask("Plan the migration.")

        drops = conv.switch_model("claude-opus-5")
        self.assertEqual(len(drops), 1)
        handoff = conv.messages[-1]
        self.assertEqual(handoff["role"], "system")
        self.assertIn("I considered X then Y.", handoff["content"])

        # The thinking block was NOT stripped — replayed verbatim.
        assistant = conv.messages[1]
        self.assertEqual(assistant["content"][0]["type"], "thinking")

        conv.ask("Continue.")
        sent = client.calls[1]["messages"]
        self.assertEqual(sent[2]["role"], "system")

    def test_default_channel_uses_user_prefix_everywhere(self):
        client = FakeClient([make_response(), make_response(model="claude-sonnet-5")])
        bridge = ThinkingBridge(client)
        conv = Conversation(bridge, "claude-fable-5-1")
        conv.ask("Plan.")
        conv.switch_model("claude-sonnet-5")
        self.assertNotEqual(conv.messages[-1]["role"], "system")

        conv.ask("Continue.")
        user_msg = client.calls[1]["messages"][-1]
        self.assertEqual(user_msg["role"], "user")
        self.assertIn("[Handoff note]", user_msg["content"][0]["text"])

    def test_system_channel_blocks_later_switch_to_sonnet(self):
        from thinking_bridge import IncompatibleSwitchError

        client = FakeClient([make_response(), make_response(model="claude-opus-5")])
        bridge = ThinkingBridge(client)
        conv = Conversation(bridge, "claude-fable-5-1", handoff_channel="system")
        conv.ask("Plan.")
        conv.switch_model("claude-opus-5")  # appends a system handoff
        conv.ask("Continue.")
        with self.assertRaises(IncompatibleSwitchError):
            conv.switch_model("claude-sonnet-5")

    def test_switch_up_no_handoff(self):
        client = FakeClient([make_response(model="claude-opus-5")])
        bridge = ThinkingBridge(client)
        conv = Conversation(bridge, "claude-opus-5")
        conv.ask("Plan.")
        drops = conv.switch_model("claude-fable-5-1")
        self.assertEqual(drops, [])
        self.assertNotEqual(conv.messages[-1]["role"], "system")

    def test_guard_integration(self):
        client = FakeClient([make_response(), make_response()])
        bridge = ThinkingBridge(client)
        guard = PrefixGuard()
        conv = Conversation(bridge, "claude-fable-5-1", system="sys")
        conv.ask("one", guard=guard)
        conv.ask("two", guard=guard)
        # Now simulate a harness bug: edit an earlier turn.
        conv.messages[0]["content"] = "EDITED"
        with self.assertRaises(PrefixMismatchError):
            bridge.create(
                model="claude-fable-5-1",
                max_tokens=1024,
                messages=conv.messages,
                system="sys",
                guard=guard,
            )

    def test_snapshot_restore_roundtrip(self):
        client = FakeClient([make_response()])
        bridge = ThinkingBridge(client)
        conv = Conversation(bridge, "claude-fable-5-1")
        conv.ask("Plan.")
        restored = Conversation.restore(bridge, conv.snapshot())
        self.assertEqual(restored.producers, conv.producers)
        drops = restored.switch_model("claude-opus-5")
        self.assertEqual(len(drops), 1)


if __name__ == "__main__":
    unittest.main()
