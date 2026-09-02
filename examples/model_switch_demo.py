"""Demo: a conversation that starts on Claude Fable 5.1, drops to Claude
Opus 5 for routine work, and comes back — with drops observed, predicted,
and reasoning carried over.

Needs credentials (ANTHROPIC_API_KEY or an `ant auth login` profile) and
spends real tokens. Fable 5.1 requires 30-day data retention on the org.
"""

import anthropic

from thinking_bridge import Conversation, PrefixGuard, ThinkingBridge


def main() -> None:
    client = anthropic.Anthropic()

    bridge = ThinkingBridge(
        client,
        prefix_mismatch_behavior="drop_block",  # "error" in CI
        on_drop=lambda drops: [
            print(f"  [drop] {t.path}: {t.reason}") for t in drops
        ],
    )
    guard = PrefixGuard()

    conv = Conversation(bridge, "claude-fable-5-1")

    print("== Turn 1 on claude-fable-5-1 ==")
    result = conv.ask(
        "Plan a 3-step approach to deduplicate a 10M-row customer table.",
        guard=guard,
    )
    print(text_of(result.response))

    # Before switching, see what the API would drop.
    predicted = bridge.predict_switch(conv.messages, "claude-opus-5", conv.producers)
    print(f"\nSwitching to claude-opus-5 would drop {len(predicted)} thinking block(s).")

    # Switch. Thinking blocks stay in the history verbatim (the API drops
    # what Opus 5 can't read, unbilled); a handoff note built from the
    # captured reasoning summaries is appended so Opus 5 doesn't re-plan blind.
    conv.switch_model("claude-opus-5")

    print("\n== Turn 2 on claude-opus-5 ==")
    result = conv.ask("Write the SQL for step 1 only.", guard=guard)
    print(text_of(result.response))
    if result.model_switch_drops:
        print(f"(API reported {len(result.model_switch_drops)} expected model-switch drop(s))")

    # Moving back up: Fable 5.1 reads Opus 5's blocks, so nothing is dropped
    # and the earlier Fable 5.1 blocks in the history are readable again.
    conv.switch_model("claude-fable-5-1")

    print("\n== Turn 3 back on claude-fable-5-1 ==")
    result = conv.ask("Review the SQL against your original plan.", guard=guard)
    print(text_of(result.response))


def text_of(response) -> str:
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )


if __name__ == "__main__":
    main()
