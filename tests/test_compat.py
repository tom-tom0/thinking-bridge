import unittest

from thinking_bridge import can_read, is_fable_51_tier, predict_drops


class TestCanRead(unittest.TestCase):
    def test_same_model_always_readable(self):
        self.assertTrue(can_read("claude-opus-4-8", "claude-opus-4-8"))
        self.assertTrue(can_read("claude-fable-5-1", "claude-fable-5-1"))

    def test_one_way_binding(self):
        # Fable 5.1 reads older models' blocks...
        self.assertTrue(can_read("claude-fable-5-1", "claude-opus-5"))
        self.assertTrue(can_read("claude-fable-5-1", "claude-fable-5"))
        self.assertTrue(can_read("claude-fable-5-1", "claude-opus-4-8"))
        self.assertTrue(can_read("claude-fable-5-1", "claude-haiku-4-5"))
        # ...but no earlier model reads Fable 5.1's blocks.
        self.assertFalse(can_read("claude-opus-5", "claude-fable-5-1"))
        self.assertFalse(can_read("claude-fable-5", "claude-fable-5-1"))
        self.assertFalse(can_read("claude-opus-4-8", "claude-fable-5-1"))

    def test_51_tier_reads_each_other(self):
        self.assertTrue(can_read("claude-mythos-5-1", "claude-fable-5-1"))
        self.assertTrue(can_read("claude-fable-5-1", "claude-mythos-5-1"))

    def test_mythos_preview_not_readable(self):
        self.assertFalse(can_read("claude-fable-5-1", "claude-mythos-preview"))

    def test_prefix_disambiguation(self):
        # "claude-fable-5" is a string prefix of "claude-fable-5-1" — make
        # sure tier detection doesn't confuse them.
        self.assertTrue(is_fable_51_tier("claude-fable-5-1"))
        self.assertFalse(is_fable_51_tier("claude-fable-5"))

    def test_cross_older_models_not_readable(self):
        self.assertFalse(can_read("claude-sonnet-5", "claude-opus-5"))

    def test_bedrock_narrower(self):
        self.assertFalse(
            can_read("claude-fable-5-1", "claude-opus-4-8", platform="bedrock")
        )
        self.assertTrue(
            can_read("claude-fable-5-1", "claude-mythos-5-1", platform="bedrock")
        )

    def test_bedrock_id_normalization(self):
        self.assertTrue(can_read("anthropic.claude-fable-5-1", "claude-opus-5"))
        self.assertTrue(can_read("us.anthropic.claude-fable-5-1", "claude-fable-5"))


class TestPredictDrops(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig1"},
                    {"type": "text", "text": "hello"},
                ],
            },
        ]

    def test_switch_down_predicts_drop(self):
        drops = predict_drops(
            self.messages, "claude-opus-5", [None, "claude-fable-5-1"]
        )
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0].path, "messages.1.content.0")
        self.assertEqual(drops[0].reason, "model_binding_mismatch")

    def test_switch_up_predicts_nothing(self):
        drops = predict_drops(
            self.messages, "claude-fable-5-1", [None, "claude-opus-5"]
        )
        self.assertEqual(drops, [])

    def test_assumed_producer(self):
        drops = predict_drops(
            self.messages, "claude-opus-5", assumed_producer="claude-fable-5-1"
        )
        self.assertEqual(len(drops), 1)


if __name__ == "__main__":
    unittest.main()
