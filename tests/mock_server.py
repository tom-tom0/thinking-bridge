"""A closed-environment mock of the Claude Messages API that implements the
documented *preserved thinking* semantics:

- every thinking block records the model that produced it (model binding);
- Fable 5.1-tier blocks additionally record the conversation prefix and a
  chain to the previous thinking block (conversation binding);
- a block the receiving model can't read is dropped before the model sees it
  (unbilled) — silently without the ``thinking-binding-controls-2026-08-01``
  header, reported in ``input_transformations`` with it;
- a prefix mismatch 400s (behavior ``"error"``, or an enforced account with
  no opt-in), or drops the first mismatched block AND every thinking block
  after it (behavior ``"drop_block"``);
- ``block_binding`` without the header is a 400 ("Extra inputs are not
  permitted");
- removing a *leading* run of thinking blocks (oldest first) is legal;
  removing one from the middle breaks the chain of the next block;
- mid-conversation ``{"role": "system"}`` messages are only accepted on the
  models that support them;
- an unknown/tampered signature is always a 400 (no "different conversation"
  sentence, and ``prefix_mismatch_behavior`` does not apply).

The mock exposes the same ``client.beta.messages.create(**kwargs)`` surface
ThinkingBridge calls, so the whole library can be exercised end-to-end with
zero network access.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from thinking_bridge.compat import can_read, is_fable_51_tier

BINDING_BETA = "thinking-binding-controls-2026-08-01"

_MID_SYSTEM_MODELS = (
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
)


def _supports_mid_system(model: str) -> bool:
    m = model.lower()
    return any(m == p or m.startswith(p + "-") for p in _MID_SYSTEM_MODELS)


class MockAPIError(Exception):
    """Duck-type compatible with the SDK's APIStatusError for the bridge."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class _SignatureRecord:
    signature: str
    producer: str
    prefix_digest: Optional[str]  # only for Fable 5.1-tier producers
    prev_signature: Optional[str]  # chain, only for Fable 5.1-tier producers


def _strip_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, (list, tuple)):
        return [_strip_cache_control(v) for v in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_strip_cache_control(value), sort_keys=True, separators=(",", ":"), default=str)


def _is_thinking(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking")


class MockClaudeServer:
    """One org's mock API endpoint.

    ``enforced=True`` emulates an account created on/after 2026-08-31 (the
    conversation check errors by default); ``enforced=False`` emulates an
    older account (the check only records unless the request opts in).
    ``controls_available=False`` emulates a platform without the beta
    controls (e.g. Microsoft Foundry): the header itself is rejected.
    """

    def __init__(self, *, enforced: bool = True, controls_available: bool = True, platform: str = "anthropic"):
        self.enforced = enforced
        self.controls_available = controls_available
        self.platform = platform
        self._registry: Dict[str, _SignatureRecord] = {}
        self._sig_counter = itertools.count(1)
        self._turn_counter = itertools.count(1)
        # Introspection for tests:
        self.last_delivered_thinking: List[str] = []  # signatures the model "saw"
        self.requests: List[Dict[str, Any]] = []
        # The client-facing surface ThinkingBridge expects:
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    # -- request digestion ----------------------------------------------------

    def _prefix_digest(self, system: Any, tools: Any, messages: List[Any], upto_message: int) -> str:
        """Digest of the conversation prefix: ``system``, the name-sorted
        ``tools`` set, and ``messages[:upto_message]``.

        Thinking blocks are excluded from the digest (they are bound via the
        chain instead, which is what makes leading-run removal legal), and
        cache_control never counts. A block's digest covers the messages
        *before* its own message — an edit inside or after that message is
        caught by the digests of the blocks that follow it, matching the
        "invalidates every later thinking block" rule.
        """
        tools_part = sorted(_canonical(t) for t in (tools or []))
        parts: List[str] = [_canonical(system), _canonical(tools_part)]
        for message in messages[:upto_message]:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                parts.append(_canonical(message))
                continue
            filtered = [b for b in content if not _is_thinking(b)]
            head = {k: v for k, v in message.items() if k != "content"}
            parts.append(_canonical(head) + _canonical(filtered))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    # -- the endpoint -----------------------------------------------------------

    def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        model: str = request["model"]
        messages: List[Any] = request["messages"]
        system = request.get("system")
        tools = request.get("tools")
        betas = list(request.get("betas") or [])
        header = BINDING_BETA in betas
        thinking = dict(request.get("thinking") or {})
        block_binding = thinking.get("block_binding")

        if header and not self.controls_available:
            raise MockAPIError(400, f"Unsupported beta: {BINDING_BETA} is not available on this platform.")
        if block_binding is not None and not header:
            raise MockAPIError(400, "thinking.block_binding: Extra inputs are not permitted")
        behavior = (block_binding or {}).get("prefix_mismatch_behavior")
        if behavior is None and header:
            behavior = "drop_block"  # the header alone opts into the beta default
        if behavior not in (None, "error", "drop_block"):
            raise MockAPIError(400, f"thinking.block_binding.prefix_mismatch_behavior: unexpected value {behavior!r}")

        # Mid-conversation system messages are model-gated.
        for mi, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") == "system" and not _supports_mid_system(model):
                raise MockAPIError(
                    400,
                    f"messages.{mi}: mid-conversation system messages are not supported on this model.",
                )

        transformations: List[Dict[str, str]] = []
        dropped_paths: set = set()

        # Collect replayed thinking blocks in order.
        replayed: List[tuple] = []  # (mi, bi, block, record)
        for mi, message in enumerate(messages):
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for bi, block in enumerate(content):
                if not _is_thinking(block):
                    continue
                sig = block.get("signature") or block.get("data")
                record = self._registry.get(sig)
                if record is None:
                    # Tampered/unknown signature: always a 400, and
                    # prefix_mismatch_behavior does not apply.
                    raise MockAPIError(
                        400,
                        f"messages.{mi}.content.{bi}: Invalid `signature` in `thinking` block.",
                    )
                replayed.append((mi, bi, block, record))

        # 1) Model-binding check (drop before the model sees it, unbilled).
        survivors: List[tuple] = []
        for mi, bi, block, record in replayed:
            if not can_read(model, record.producer, platform=self.platform):
                dropped_paths.add((mi, bi))
                transformations.append(
                    {"type": "thinking_dropped", "path": f"messages.{mi}.content.{bi}", "reason": "model_binding_mismatch"}
                )
            else:
                survivors.append((mi, bi, block, record))

        # 2) Conversation-binding check, for surviving Fable 5.1-tier blocks.
        bound = [(mi, bi, blk, rec) for mi, bi, blk, rec in survivors if is_fable_51_tier(rec.producer)]
        violation_index: Optional[int] = None
        for idx, (mi, bi, block, record) in enumerate(bound):
            # Chain: this block's recorded predecessor must be the previous
            # bound block in this history — except a leading run may have
            # been removed (predecessor absent from the whole history).
            if idx == 0:
                if record.prev_signature is not None and any(
                    r.signature == record.prev_signature for _, _, _, r in bound
                ):
                    violation_index = idx
                    break
            else:
                if record.prev_signature != bound[idx - 1][3].signature:
                    violation_index = idx
                    break
            # Prefix: the conversation before the block must be unchanged.
            digest = self._prefix_digest(system, tools, messages, mi)
            if record.prefix_digest is not None and digest != record.prefix_digest:
                violation_index = idx
                break

        if violation_index is not None:
            mi, bi, _, _ = bound[violation_index]
            check_active = behavior is not None or (self.enforced and behavior is None and not header)
            if check_active and behavior != "drop_block":
                tail = (
                    ' Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to "drop_block". '
                    f"That setting requires the `{BINDING_BETA}` value in the `anthropic-beta` header."
                    if not header
                    else ""
                )
                raise MockAPIError(
                    400,
                    f"messages.{mi}.content.{bi}: Invalid `signature` in `thinking` block. "
                    "The block is bound to a different conversation." + tail,
                )
            if check_active and behavior == "drop_block":
                # Drop the first mismatched block and every thinking block
                # after it (cascade).
                cascade = False
                for cmi, cbi, cblock, crecord in replayed:
                    if (cmi, cbi) == (mi, bi):
                        cascade = True
                    if cascade and (cmi, cbi) not in dropped_paths:
                        dropped_paths.add((cmi, cbi))
                        transformations.append(
                            {
                                "type": "thinking_dropped",
                                "path": f"messages.{cmi}.content.{cbi}",
                                "reason": "prefix_binding_mismatch",
                            }
                        )
            # An unenforced account with no opt-in only records the mismatch.

        self.last_delivered_thinking = [
            rec.signature for mi, bi, _, rec in replayed if (mi, bi) not in dropped_paths
        ]

        # 3) Produce the response.
        turn = next(self._turn_counter)
        signature = f"mock-sig-{next(self._sig_counter):06d}"
        prev = None
        prefix_digest = None
        if is_fable_51_tier(model):
            live_bound = [rec.signature for _, _, _, rec in bound]
            prev = live_bound[-1] if live_bound else None
            prefix_digest = self._prefix_digest(system, tools, messages, len(messages))
        self._registry[signature] = _SignatureRecord(
            signature=signature, producer=model, prefix_digest=prefix_digest, prev_signature=prev
        )

        display = thinking.get("display", "omitted")
        summary = f"Reasoning summary for turn {turn} on {model}." if display == "summarized" else ""
        content = [
            {"type": "thinking", "thinking": summary, "signature": signature},
            {"type": "text", "text": f"Answer for turn {turn} from {model}."},
        ]
        response = SimpleNamespace(
            model=model,
            content=content,
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )
        if header:
            response.input_transformations = transformations
        return response
