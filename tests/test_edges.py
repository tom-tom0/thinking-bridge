"""Coverage for the streaming passthrough and typed-SDK-object parsing paths."""

import unittest
from types import SimpleNamespace

from thinking_bridge import (
    BINDING_BETA,
    PrefixGuard,
    PrefixMismatchError,
    ThinkingBridge,
    parse_input_transformations,
)


class FakeStreamManager:
    def __init__(self, final_message):
        self._final = final_message

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_final_message(self):
        return self._final


class FakeStreamMessages:
    def __init__(self, final_message):
        self.final_message = final_message
        self.stream_calls = []

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return FakeStreamManager(self.final_message)


class TestStreamPassthrough(unittest.TestCase):
    def _client(self, final):
        return SimpleNamespace(beta=SimpleNamespace(messages=FakeStreamMessages(final)))

    def test_stream_injects_header_and_binding(self):
        final = SimpleNamespace(
            model="claude-fable-5-1",
            content=[],
            input_transformations=[
                {"type": "thinking_dropped", "path": "messages.1.content.0", "reason": "model_binding_mismatch"}
            ],
        )
        client = self._client(final)
        bridge = ThinkingBridge(client)
        with bridge.stream(
            model="claude-fable-5-1",
            max_tokens=64000,
            messages=[{"role": "user", "content": "hi"}],
        ) as s:
            message = s.get_final_message()
        call = client.beta.messages.stream_calls[0]
        self.assertIn(BINDING_BETA, call["betas"])
        self.assertEqual(
            call["thinking"]["block_binding"]["prefix_mismatch_behavior"], "drop_block"
        )
        result = bridge.inspect(message)
        self.assertEqual(len(result.model_switch_drops), 1)

    def test_stream_respects_guard(self):
        client = self._client(SimpleNamespace(model="m", content=[]))
        bridge = ThinkingBridge(client)
        guard = PrefixGuard()
        guard.check([{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}])
        with self.assertRaises(PrefixMismatchError):
            bridge.stream(
                model="claude-fable-5-1",
                max_tokens=64,
                messages=[{"role": "user", "content": "EDITED"}, {"role": "assistant", "content": "two"}],
                guard=guard,
            )
        self.assertEqual(client.beta.messages.stream_calls, [])


class _TypedEntry:
    """Mimics a typed SDK object with model_dump()."""

    def __init__(self, d):
        self._d = d

    def model_dump(self):
        return dict(self._d)


class _PydanticishResponse:
    """Mimics a pydantic response where input_transformations is untyped
    extra data (the launch-SDK situation the docs warn about)."""

    def __init__(self, extra):
        self.model_extra = extra

    def __getattr__(self, name):
        extra = object.__getattribute__(self, "__dict__").get("model_extra") or {}
        if name in extra:
            return extra[name]
        raise AttributeError(name)


class TestTypedObjectParsing(unittest.TestCase):
    def test_typed_entries_via_model_dump(self):
        response = SimpleNamespace(
            input_transformations=[
                _TypedEntry({"type": "thinking_dropped", "path": "messages.0.content.0", "reason": "prefix_binding_mismatch"})
            ]
        )
        parsed = parse_input_transformations(response)
        self.assertEqual(len(parsed), 1)
        self.assertTrue(parsed[0].is_history_edit)

    def test_untyped_field_in_model_extra(self):
        response = _PydanticishResponse(
            {"input_transformations": [
                {"type": "thinking_dropped", "path": "messages.2.content.0", "reason": "model_binding_mismatch"}
            ]}
        )
        parsed = parse_input_transformations(response)
        self.assertEqual(len(parsed), 1)
        self.assertTrue(parsed[0].is_model_switch)

    def test_model_extra_present_but_field_absent(self):
        response = _PydanticishResponse({"other": 1})
        self.assertIsNone(parse_input_transformations(response))

    def test_attribute_only_object_without_field(self):
        self.assertIsNone(parse_input_transformations(SimpleNamespace(model="m")))


if __name__ == "__main__":
    unittest.main()
