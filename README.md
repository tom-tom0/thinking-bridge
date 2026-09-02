# thinking-bridge

Make **Claude Fable 5.1's preserved-thinking rules** observable and survivable
when a conversation moves across Claude models.

## The problem

Every `thinking` block records which model produced it, and the binding is
one-way: Claude Fable 5.1 reads earlier models' thinking blocks, but **no
earlier model reads a Fable 5.1 block**. When a conversation moves from
`claude-fable-5-1` to any other model (a router, a fallback, a cost
downgrade), the API drops the blocks the target can't read — silently, unless
you send the `thinking-binding-controls-2026-08-01` beta header. The target
model then re-plans without that reasoning: higher cost and latency on the
first turn after the switch, and none of the earlier private reasoning
survives.

On top of that, Fable 5.1 binds each thinking block to the **conversation
prefix** that produced it: editing an earlier turn, rebuilding `system` or
`tools` mid-session, or deleting old tool results invalidates every later
thinking block — a 400 on accounts created on or after 2026-08-31 (and
planned for everyone on future models).

## What this library does

| Problem | thinking-bridge answer |
|---|---|
| Drops are silent | Sends the beta header and an **explicit** `prefix_mismatch_behavior` on every request, so every response carries `input_transformations`; parses and classifies each drop (`model_binding_mismatch` = expected switch, `prefix_binding_mismatch` = harness bug) and surfaces them via callback + logging |
| You don't know what a switch will cost until after it happens | `predict_switch()` / `predict_drops()` tell you client-side which blocks the target model can't read, before you send anything |
| Downgrading loses the reasoning entirely | `Conversation.switch_model()` appends a **handoff note** built from captured reasoning summaries (`display: "summarized"`), injected append-only — by default as a labeled block in the next user turn (portable to every model), or opt-in (`handoff_channel="system"`) as a mid-conversation `{"role": "system"}` message with system-prompt authority on the models that accept those |
| History edits 400 in production | `PrefixGuard` validates every request as append-only *in your process*, with an actionable message naming the first edited index — before the API ever sees it |
| Platforms without the binding controls (Foundry; Bedrock/Vertex until they arrive) | `binding_controls=False` mode plus opt-in one-time **strip-and-retry** recovery for the documented signature 400 (and only for the "bound to a different conversation" variant — a tampered signature is never retried) |
| The temptation to strip blocks on a switch | Resisted, deliberately: the bridge replays history verbatim. Dropped blocks are unbilled, stripping can trigger ordering/signature 400s, and blocks left in place become readable again if the conversation moves back to Fable 5.1 |

## Install

```bash
pip install -e .          # from this directory
```

## Quick start

```python
import anthropic
from thinking_bridge import ThinkingBridge, Conversation, PrefixGuard

client = anthropic.Anthropic()
bridge = ThinkingBridge(
    client,
    prefix_mismatch_behavior="drop_block",   # use "error" in CI so edits fail the run
    on_drop=lambda drops: print([t.path for t in drops]),
)
guard = PrefixGuard()

conv = Conversation(bridge, "claude-fable-5-1")
conv.ask("Plan the refactor.", guard=guard)

# See what a downgrade would cost before doing it:
print(bridge.predict_switch(conv.messages, "claude-opus-5", conv.producers))

# Switch down — blocks stay in history, a handoff note carries the reasoning:
conv.switch_model("claude-opus-5")
conv.ask("Apply step 1.", guard=guard)

# Switch back up — Fable 5.1 reads everything again, nothing was lost:
conv.switch_model("claude-fable-5-1")
conv.ask("Review the result.", guard=guard)
```

Using your own message loop instead of `Conversation`:

```python
result = bridge.create(
    model="claude-fable-5-1",
    max_tokens=16000,
    messages=history,          # replayed verbatim, thinking blocks included
    guard=guard,               # optional append-only check
)
for t in result.history_edit_drops:
    alert(f"harness bug: history changed before {t.path}")
```

Streaming:

```python
with bridge.stream(model="claude-fable-5-1", max_tokens=64000, messages=history) as s:
    message = s.get_final_message()
result = bridge.inspect(message)   # input_transformations arrives on message_start
```

## Design rules the library encodes

These come straight from the published preserved-thinking semantics:

1. **Never strip thinking blocks to switch models.** The API drops what the
   target can't read, unbilled — there are no input tokens to save, and
   client-side stripping can trigger ordering/signature 400s. The only strip
   in this library is the documented one-time 400 recovery on platforms
   without the binding controls.
2. **Set `prefix_mismatch_behavior` explicitly.** The defaults differ by
   surface (no header + enforced account → 400; header alone → `drop_block`;
   Batches' unset default drops). The bridge always sends an explicit value
   under the beta header. The canonical field is
   `thinking.block_binding.prefix_mismatch_behavior` — the `mismatch_behavior`
   spelling is an undocumented alias the bridge refuses to emit.
3. **Tolerate unknown transformation types/reasons.** Later checks add
   values; unrecognized entries are reported but never treated as errors.
4. **Keep histories append-only.** Reminders go in as turn-scoped system
   messages or trailing text blocks and *stay* in the transcript; system/tool
   changes go through mid-conversation system messages; shrinking history is
   the server's job (context editing / compaction) or a full simple-compaction
   restart — the `PrefixGuard` distinguishes a legal restart (no shared turns)
   from an edit (old turns kept around a change).
5. **One request body works across models.** `block_binding` is accepted
   alongside both `{"type": "adaptive"}` and `{"type": "enabled",
   "budget_tokens": N}`, and models that don't enforce the conversation check
   just report model-check drops.

## Compatibility matrix (as shipped)

`can_read(consumer, producer)`:

- Fable 5.1 / Mythos 5.1 read each other's blocks, and blocks from Opus 5,
  Fable 5, Mythos 5, and earlier Opus/Sonnet/Haiku models.
- No earlier model reads a Fable 5.1 / Mythos 5.1 block.
- Mythos Preview blocks are not readable by Fable 5.1.
- Between two different non-5.1 models, blocks are silently ignored — only
  same-model replay preserves them.
- `platform="bedrock"` applies Bedrock's narrower own-family-only matrix
  (confirm at launch).
- Unknown model ids are treated conservatively (same-model only).

Bedrock (`anthropic.` / `us.anthropic.` prefixes) and Vertex (`@` snapshot
suffix) id spellings are normalized before matching.

## Caveats

- The handoff note depends on `display: "summarized"` (the bridge's default
  via `capture_reasoning=True`); with the platform default (`"omitted"`)
  there is nothing to carry over — the note still marks the switch. Display
  affects visibility only; thinking is billed the same either way.
- Mid-conversation `{"role": "system"}` messages are supported on Opus 5,
  Opus 4.8, Fable 5/5.1, Mythos 5/5.1 — not Sonnet 5 / Sonnet 4.6 / Haiku.
  That's why the handoff note defaults to the user-turn channel: once a
  system message is in an append-only history, the conversation can never
  legally move to a model that rejects it (removing the message would be a
  history edit). With `handoff_channel="system"`, `switch_model()` raises
  `IncompatibleSwitchError` on such a switch instead of letting the API
  reject the transcript. (This rule was found by the fuzz suite, not by
  design review — see Validation below.)
- The beta controls (`thinking-binding-controls-2026-08-01`,
  `input_transformations`) exist on the Claude API and Claude Platform on AWS
  at launch, arrive per model on Bedrock and Vertex, and are not offered on
  Microsoft Foundry — use `binding_controls=False` (+ optional
  `strip_and_retry=True`) there.
- In the Message Batches API the *unset* default drops failing blocks rather
  than erroring; the bridge's explicit field keeps behavior uniform.
- The `PrefixGuard` mirrors the server's rules but is a client-side
  approximation (e.g. it can't verify that a rotating signed URL serves the
  same bytes). Treat a clean guard plus empty `input_transformations` across
  a session as the real integration test.
- Python only for now; the design (compat matrix, guard, drop parsing,
  carryover) ports directly to TypeScript.

## Validation (all closed-environment — zero network, zero token spend)

```bash
PYTHONPATH=.:tests python3 -m unittest discover -s tests -t .   # 70 tests
PYTHONPATH=.:tests python3 benchmarks/bench.py                  # benchmarks
```

Three layers:

1. **Unit tests** (`test_compat`, `test_guard`, `test_bridge`, `test_edges`)
   — the compatibility matrix against the documented table, guard semantics,
   request shaping, drop parsing (typed objects, pydantic extras, dicts),
   streaming passthrough, strip-and-retry classification.
2. **End-to-end simulation** (`test_e2e_simulation`) against
   `tests/mock_server.py`, a mock Messages API that implements the
   documented preserved-thinking semantics server-side: model-binding drops
   (silent without the beta header, reported with it), conversation-prefix
   signatures with the block chain, the `drop_block` cascade, the
   "different conversation" vs tampered-signature 400 variants,
   `block_binding`-without-header 400s, model-gated mid-conversation system
   messages, enforced vs unenforced accounts, and platforms without the
   controls. Scenarios include multi-turn sessions, down/up/round-trip
   switches, every legal history operation (cache_control churn, tool
   reorder, leading-thinking removal, simple-compaction restart) and every
   illegal one.
3. **Fuzz properties** (`test_fuzz`, 750 seeded scenarios per run):
   randomly generated *legal* sessions never draw a 400 and never produce
   `prefix_binding_mismatch`; randomly *tampered* histories are always
   caught (by the guard client-side, by the error-mode server, or as
   reported drops in `drop_block` mode); and `predict_switch()` names
   exactly the drops the server then reports.

Coverage: 97% of the library. Benchmarks (`benchmarks/results.txt`): the
bridge adds well under 1 ms per request; `PrefixGuard.check` is ~1 ms at 100
messages and ~11 ms at 1,000 (it re-verifies the full history every call by
design — negligible next to a real API round-trip); `predict_drops` on a
5,000-message history is ~7 ms.

One caveat honestly stated: the mock's model-compatibility matrix is built
from the same published rules as the library's (`compat.py`), so layer 2/3
validate the *integration logic*, not the matrix itself — the matrix is
pinned against the documented table in `test_compat.py`, and the final
word on live behavior belongs to a real key via
`examples/model_switch_demo.py`.
