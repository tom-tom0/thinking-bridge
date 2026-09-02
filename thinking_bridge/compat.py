"""Model-to-model thinking-block readability rules.

Every ``thinking`` block records which model produced it. The binding is
one-way: Claude Fable 5.1 (and Claude Mythos 5.1) read blocks from each other
and from earlier models, but no earlier model can read a Fable 5.1 block.
When a request carries a block the receiving model can't read, the API drops
it before the model sees it (unbilled). This module predicts those drops
client-side so an application can decide *before* switching models.

The matrix here mirrors the published rules as of 2026-09; unknown model ids
are treated conservatively (only same-model replay is assumed readable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

# Model-id prefixes. Order matters where one id is a prefix of another
# ("claude-fable-5" vs "claude-fable-5-1"), so tier checks always test the
# longer ids first.
_FABLE_51_TIER = ("claude-fable-5-1", "claude-mythos-5-1")
_MYTHOS_PREVIEW = ("claude-mythos-preview",)
# Models whose blocks Fable 5.1 / Mythos 5.1 can read: Opus 5, Fable 5,
# Mythos 5, and earlier models that don't encrypt reasoning in the signature
# (Opus 4.8 and earlier Opus, Sonnet, Haiku 4.5).
_READABLE_BY_51 = (
    "claude-opus-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4",
    "claude-opus-3",
    "claude-sonnet-5",
    "claude-sonnet-4",
    "claude-sonnet-3",
    "claude-haiku-4",
    "claude-haiku-3",
)


def _norm(model: str) -> str:
    model = (model or "").strip().lower()
    # Bedrock ids carry an "anthropic." (or region) prefix.
    for prefix in ("us.anthropic.", "eu.anthropic.", "apac.anthropic.", "anthropic."):
        if model.startswith(prefix):
            model = model[len(prefix):]
            break
    # Vertex dated snapshots use "@" as the version separator.
    return model.split("@", 1)[0]


def _matches(model: str, prefixes: Iterable[str]) -> bool:
    return any(model == p or model.startswith(p + "-") for p in prefixes)


def is_fable_51_tier(model: str) -> bool:
    """True for Claude Fable 5.1 / Claude Mythos 5.1 (any snapshot)."""
    return _matches(_norm(model), _FABLE_51_TIER)


def is_mythos_preview(model: str) -> bool:
    return _matches(_norm(model), _MYTHOS_PREVIEW)


def can_read(consumer: str, producer: str, *, platform: str = "anthropic") -> bool:
    """Can ``consumer`` read a thinking block produced by ``producer``?

    ``platform="bedrock"`` applies Amazon Bedrock's narrower matrix (a
    Fable 5.1-tier consumer reads only its own tier's blocks there).
    """
    c, p = _norm(consumer), _norm(producer)
    if c == p:
        return True
    # Fable 5.1 tier: check before anything else, because "claude-fable-5" is
    # a string prefix of "claude-fable-5-1".
    if _matches(p, _FABLE_51_TIER):
        return _matches(c, _FABLE_51_TIER)
    if _matches(c, _FABLE_51_TIER):
        if _matches(p, _MYTHOS_PREVIEW):
            return False
        if platform == "bedrock":
            return False  # Bedrock reads own family only (confirm at launch).
        return _matches(p, _READABLE_BY_51)
    # Between two non-5.1 models the API silently ignores each other's
    # thinking blocks; only same-model replay preserves them.
    return False


@dataclass(frozen=True)
class PredictedDrop:
    """One thinking block that the API would drop for ``target_model``."""

    message_index: int
    block_index: int
    producer: str
    reason: str  # mirrors the API's input_transformations reasons

    @property
    def path(self) -> str:
        return f"messages.{self.message_index}.content.{self.block_index}"


def predict_drops(
    messages: Sequence[Any],
    target_model: str,
    producers: Optional[Sequence[Optional[str]]] = None,
    *,
    assumed_producer: Optional[str] = None,
    platform: str = "anthropic",
) -> List[PredictedDrop]:
    """Predict which thinking blocks a switch to ``target_model`` would drop.

    ``producers`` maps each message index to the model that produced it
    (``None`` for user messages) — a :class:`thinking_bridge.Conversation`
    ledger provides this. Without a ledger, pass ``assumed_producer`` to
    assume every thinking block came from that model.

    This is advisory only: never strip the predicted blocks yourself. Replay
    the history verbatim and let the API drop them (unbilled) — removing
    blocks client-side can trigger ordering/signature 400s.
    """
    drops: List[PredictedDrop] = []
    for mi, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not isinstance(content, (list, tuple)):
            continue
        producer = None
        if producers is not None and mi < len(producers):
            producer = producers[mi]
        producer = producer or assumed_producer
        if producer is None:
            continue
        for bi, block in enumerate(content):
            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if btype in ("thinking", "redacted_thinking") and not can_read(
                target_model, producer, platform=platform
            ):
                drops.append(
                    PredictedDrop(
                        message_index=mi,
                        block_index=bi,
                        producer=producer,
                        reason="model_binding_mismatch",
                    )
                )
    return drops
