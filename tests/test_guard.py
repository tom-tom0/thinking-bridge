import unittest

from thinking_bridge import PrefixGuard, PrefixMismatchError


def msgs(*texts):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(texts)]


class TestPrefixGuard(unittest.TestCase):
    def test_append_only_passes(self):
        guard = PrefixGuard()
        guard.check(msgs("a"), system="sys", tools=[{"name": "t1"}])
        report = guard.check(msgs("a", "b", "c"), system="sys", tools=[{"name": "t1"}])
        self.assertTrue(report.ok)
        self.assertEqual(report.appended_messages, 2)

    def test_edited_message_rejected(self):
        guard = PrefixGuard()
        guard.check(msgs("a", "b", "c"), system="sys")
        with self.assertRaises(PrefixMismatchError) as ctx:
            guard.check(msgs("a", "EDITED", "c"), system="sys")
        self.assertEqual(ctx.exception.kind, "message")
        self.assertEqual(ctx.exception.index, 1)

    def test_removed_middle_turn_rejected(self):
        guard = PrefixGuard()
        guard.check(msgs("a", "b", "c"), system="sys")
        with self.assertRaises(PrefixMismatchError):
            guard.check(msgs("a", "c"), system="sys")

    def test_system_change_rejected(self):
        guard = PrefixGuard()
        guard.check(msgs("a"), system="sys v1")
        with self.assertRaises(PrefixMismatchError) as ctx:
            guard.check(msgs("a", "b"), system="sys v2")
        self.assertEqual(ctx.exception.kind, "system")

    def test_tools_change_rejected_but_reorder_allowed(self):
        t1, t2 = {"name": "alpha", "input_schema": {}}, {"name": "beta", "input_schema": {}}
        guard = PrefixGuard()
        guard.check(msgs("a"), tools=[t1, t2])
        # Reorder only: fine (tools are bound as a name-sorted set).
        guard.check(msgs("a", "b"), tools=[t2, t1])
        with self.assertRaises(PrefixMismatchError) as ctx:
            guard.check(msgs("a", "b", "c"), tools=[t1])
        self.assertEqual(ctx.exception.kind, "tools")

    def test_cache_control_changes_allowed(self):
        guard = PrefixGuard()
        base = [{"role": "user", "content": [{"type": "text", "text": "a"}]}]
        guard.check(base)
        cached = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}
                ],
            },
            {"role": "assistant", "content": "b"},
        ]
        report = guard.check(cached)
        self.assertTrue(report.ok)

    def test_full_reset_allowed_when_opted_in(self):
        guard = PrefixGuard(allow_reset=True)
        guard.check(msgs("a", "b", "c"))
        # Simple compaction: brand-new history starting from a summary.
        report = guard.check(msgs("summary of a-c", "next"))
        self.assertTrue(report.reset)

    def test_full_reset_rejected_when_disallowed(self):
        guard = PrefixGuard(allow_reset=False)
        guard.check(msgs("a", "b"))
        with self.assertRaises(PrefixMismatchError):
            guard.check(msgs("different", "history"))

    def test_snapshot_restore(self):
        guard = PrefixGuard()
        guard.check(msgs("a", "b"), system="sys")
        restored = PrefixGuard.restore(guard.snapshot())
        report = restored.check(msgs("a", "b", "c"), system="sys")
        self.assertEqual(report.appended_messages, 1)
        with self.assertRaises(PrefixMismatchError):
            restored.check(msgs("a", "X", "c"), system="sys")


if __name__ == "__main__":
    unittest.main()
