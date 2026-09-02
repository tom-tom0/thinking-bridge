"""ThinkingBridge — a thin wrapper over the Anthropic SDK that makes
Claude Fable 5.1's preserved-thinking rules observable and survivable when a
conversation moves across models.

What it does on every request:

1. Sends the ``thinking-binding-controls-2026-08-01`` beta header and sets
   ``thinking.block_binding.prefix_mismatch_behavior`` **explicitly**
   (default ``"drop_block"``) — the defaults differ by surface, so relying on
   them is a trap. This also adds ``input_transformations`` to responses.
2. Parses ``input_transformations`` and classifies each drop:
   ``model_binding_mismatch`` (the conversation switched models — expected)
   vs ``prefix_binding_mismatch`` (the harness edited history — a bug).
3. Optionally runs a client-side :class:`PrefixGuard` so history edits fail
   fast, in your process, with an actionable message.
4. On platforms without the binding controls, optionally recovers from the
   signature 400 with a one-time strip-and-retry (the documented no-beta
   recovery).

What it deliberately does NOT do: strip thinking blocks when you switch
models. Replay the history verbatim — the API drops what the target model
can't read, unbilled, and stripping client-side can trigger ordering /
signature 400s.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .compat import can_read, predict_drops, PredictedDrop
from .transformations import (
    InputTransformation,
    parse_input_transformations,
)

logger = logging.getLogger("thinking_bridge")

BINDING_BETA = "thinking-binding-controls-2026-08-01"

# Models that accept mid-conversation {"role": "system"} messages (the
# append-only channel used for reasoning carryover). Not Sonnet 5 / 4.6 or
# Haiku — for those the carryover rides in the next user message instead.
_MID_SYSTEM_MODELS = (
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
)

_SIGNATURE_400_MARKER = "Invalid `signature` in `thinking` block"
_CONVERSATION_BOUND_MARKER = "bound to a different conversation"


class IncompatibleSwitchError(ValueError):
    """The target model cannot accept this history as-is (and fixing that
    would require a history edit that invalidates thinking blocks)."""


def _model_supports_mid_system(model: str) -> bool:
    m = (model or "").lower()
    # Longest prefixes are listed first in _MID_SYSTEM_MODELS where one id is
    # a prefix of another.
    return any(m == p or m.startswith(p + "-") for p in _MID_SYSTEM_MODELS)


@dataclass
class BridgeResult:
    """A response plus what the API did to the input on the way in."""

    response: Any
    transformations: Optional[List[InputTransformation]]
    stripped_and_retried: bool = False

    @property
    def model_switch_drops(self) -> List[InputTransformation]:
        return [t for t in (self.transformations or []) if t.is_model_switch]

    @property
    def history_edit_drops(self) -> List[InputTransformation]:
        return [t for t in (self.transformations or []) if t.is_history_edit]


class ThinkingBridge:
    """Wraps an ``anthropic.Anthropic`` client.

    Parameters:
        client: an ``anthropic.Anthropic`` (or compatible) client.
        prefix_mismatch_behavior: ``"drop_block"`` (degrade gracefully,
            default) or ``"error"`` (fail loudly — use in CI so a history
            edit fails the run).
        binding_controls: set False on platforms that reject the beta header
            (Microsoft Foundry; Bedrock/Vertex until the controls arrive per
            model). Without the controls, drops are silent and the recovery
            for a signature 400 is strip-and-retry.
        strip_and_retry: when True and ``binding_controls`` is False, a 400
            naming an invalid thinking-block signature bound to a different
            conversation is retried once with every thinking /
            redacted_thinking block stripped. One-time recovery, not a
            steady-state pattern.
        capture_reasoning: default the ``thinking.display`` to
            ``"summarized"`` so readable reasoning summaries are available
            for carryover when the conversation later moves to a model that
            can't read the blocks. (Display controls visibility only —
            thinking is billed the same either way.)
        on_drop: callback invoked with a list of InputTransformation whenever
            a response reports drops.
        platform: ``"anthropic"`` (default) or ``"bedrock"`` — affects drop
            prediction only.
    """

    def __init__(
        self,
        client: Any,
        *,
        prefix_mismatch_behavior: str = "drop_block",
        binding_controls: bool = True,
        strip_and_retry: bool = False,
        capture_reasoning: bool = True,
        on_drop: Optional[Callable[[List[InputTransformation]], None]] = None,
        platform: str = "anthropic",
    ):
        if prefix_mismatch_behavior not in ("drop_block", "error"):
            raise ValueError(
                "prefix_mismatch_behavior must be 'drop_block' or 'error' "
                "(the canonical field name is prefix_mismatch_behavior; the "
                "'mismatch_behavior' spelling is an undocumented alias)."
            )
        self.client = client
        self.prefix_mismatch_behavior = prefix_mismatch_behavior
        self.binding_controls = binding_controls
        self.strip_and_retry = strip_and_retry
        self.capture_reasoning = capture_reasoning
        self.on_drop = on_drop
        self.platform = platform

    # -- request building ---------------------------------------------------

    def _build_thinking(self, thinking: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        built: Dict[str, Any] = dict(thinking) if thinking else {"type": "adaptive"}
        if self.capture_reasoning:
            built.setdefault("display", "summarized")
        if self.binding_controls:
            # block_binding is accepted alongside both "adaptive" and
            # "enabled", and models that don't enforce the conversation check
            # accept it and report only model-check drops — one body works
            # across models.
            binding = dict(built.get("block_binding") or {})
            binding.setdefault("prefix_mismatch_behavior", self.prefix_mismatch_behavior)
            built["block_binding"] = binding
        else:
            # Sending block_binding without the header is a 400.
            built.pop("block_binding", None)
        return built

    def _build_betas(self, betas: Optional[Sequence[str]]) -> List[str]:
        merged = list(betas or [])
        if self.binding_controls and BINDING_BETA not in merged:
            merged.append(BINDING_BETA)
        return merged

    # -- calls ---------------------------------------------------------------

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Any],
        max_tokens: int,
        thinking: Optional[Dict[str, Any]] = None,
        betas: Optional[Sequence[str]] = None,
        guard: Optional[Any] = None,
        **kwargs: Any,
    ) -> BridgeResult:
        """Non-streaming request via ``client.beta.messages.create``.

        ``guard`` is an optional :class:`thinking_bridge.PrefixGuard`; when
        given, the request is validated as append-only before it is sent.
        """
        if guard is not None:
            guard.check(
                messages,
                system=kwargs.get("system"),
                tools=kwargs.get("tools"),
            )

        request = dict(
            model=model,
            max_tokens=max_tokens,
            messages=list(messages),
            thinking=self._build_thinking(thinking),
            **kwargs,
        )
        merged_betas = self._build_betas(betas)
        if merged_betas:
            request["betas"] = merged_betas

        try:
            response = self.client.beta.messages.create(**request)
        except Exception as exc:  # noqa: BLE001 - classified below
            recovered = self._maybe_strip_and_retry(exc, request)
            if recovered is None:
                raise
            return recovered
        return self._inspect(response)

    def stream(self, **kwargs: Any) -> Any:
        """Passthrough to ``client.beta.messages.stream`` with the same
        header/thinking handling. Call :meth:`inspect` on the stream's
        ``get_final_message()`` result — the accumulated message carries
        ``input_transformations`` (it arrives in ``message_start``, and again
        in the final ``message_delta`` after a mid-stream server-side
        fallback).
        """
        guard = kwargs.pop("guard", None)
        if guard is not None:
            guard.check(
                kwargs.get("messages") or [],
                system=kwargs.get("system"),
                tools=kwargs.get("tools"),
            )
        kwargs["thinking"] = self._build_thinking(kwargs.get("thinking"))
        merged = self._build_betas(kwargs.get("betas"))
        if merged:
            kwargs["betas"] = merged
        return self.client.beta.messages.stream(**kwargs)

    def inspect(self, response: Any) -> BridgeResult:
        """Parse and report drops on a response obtained elsewhere
        (e.g. a streaming final message)."""
        return self._inspect(response)

    # -- internals -----------------------------------------------------------

    def _inspect(self, response: Any) -> BridgeResult:
        transformations = parse_input_transformations(response)
        if transformations:
            for t in transformations:
                if t.is_history_edit:
                    logger.warning(
                        "thinking block dropped at %s: the conversation "
                        "prefix changed since the previous request (harness "
                        "bug — make the history append-only).",
                        t.path,
                    )
                elif t.is_model_switch:
                    logger.info(
                        "thinking block dropped at %s: the conversation "
                        "switched models (expected; the target re-plans "
                        "without that reasoning).",
                        t.path,
                    )
                else:
                    logger.info(
                        "input transformation %s at %s (reason %r) — "
                        "unrecognized, ignoring.",
                        t.type,
                        t.path,
                        t.reason,
                    )
            if self.on_drop:
                self.on_drop(transformations)
        return BridgeResult(response=response, transformations=transformations)

    def _maybe_strip_and_retry(
        self, exc: Exception, request: Dict[str, Any]
    ) -> Optional[BridgeResult]:
        """One-time no-beta recovery for the conversation-binding 400."""
        if not self.strip_and_retry or self.binding_controls:
            return None
        status = getattr(exc, "status_code", None)
        text = str(getattr(exc, "message", None) or exc)
        if status != 400 or _SIGNATURE_400_MARKER not in text:
            return None
        if _CONVERSATION_BOUND_MARKER not in text:
            # A tampered/undecryptable signature is a different failure —
            # always a 400, and stripping won't fix a harness that keeps
            # producing it.
            return None
        logger.warning(
            "Retrying once with thinking blocks stripped (conversation-"
            "binding 400). The model answers this turn without that "
            "reasoning; treat this as a one-time recovery and fix the "
            "history edit."
        )
        retry = dict(request)
        retry["messages"] = strip_thinking_blocks(request["messages"])
        response = self.client.beta.messages.create(**retry)
        result = self._inspect(response)
        result.stripped_and_retried = True
        return result

    def predict_switch(
        self,
        messages: Sequence[Any],
        target_model: str,
        producers: Optional[Sequence[Optional[str]]] = None,
        *,
        assumed_producer: Optional[str] = None,
    ) -> List[PredictedDrop]:
        """What would switching this history to ``target_model`` drop?"""
        return predict_drops(
            messages,
            target_model,
            producers,
            assumed_producer=assumed_producer,
            platform=self.platform,
        )


def strip_thinking_blocks(messages: Sequence[Any]) -> List[Any]:
    """Remove every thinking / redacted_thinking block; text and tool_use
    blocks stay. Only for the documented one-time 400 recovery on platforms
    without the binding controls — never as a steady-state pattern, and never
    as preparation for a model switch (the API drops unreadable blocks itself,
    unbilled)."""
    stripped: List[Any] = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            stripped.append(message)
            continue
        new_message = dict(message)
        new_message["content"] = [
            block
            for block in message["content"]
            if not (
                (isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"))
                or getattr(block, "type", None) in ("thinking", "redacted_thinking")
            )
        ]
        stripped.append(new_message)
    return stripped


# ---------------------------------------------------------------------------
# Conversation ledger
# ---------------------------------------------------------------------------


@dataclass
class _Turn:
    role: str
    producer: Optional[str] = None  # model id for assistant turns


class Conversation:
    """An append-only conversation ledger with model-switch support.

    Tracks which model produced each assistant turn (so drops can be
    predicted before a switch), captures readable thinking summaries, and —
    when the conversation moves to a model that can't read the existing
    blocks — appends a *handoff note* so the target model doesn't re-plan
    from nothing. The note is injected append-only.

    ``handoff_channel`` picks where the note goes:

    - ``"user"`` (default): a labeled text block ahead of the next user
      turn. Portable — every model accepts it, so the conversation can keep
      switching anywhere later.
    - ``"system"``: a mid-conversation ``{"role": "system"}`` message, which
      carries system-prompt authority — but only Opus 5 / Opus 4.8 /
      Fable- and Mythos-tier models accept those. Once one is in the
      history, a later switch to Sonnet or Haiku is impossible (they 400 on
      the message, and removing it would be a history edit that invalidates
      later thinking blocks), so :meth:`switch_model` refuses such a switch
      with a clear error instead of letting the API reject the transcript.
    """

    def __init__(
        self,
        bridge: ThinkingBridge,
        model: str,
        *,
        system: Any = None,
        tools: Any = None,
        handoff_channel: str = "user",
    ):
        if handoff_channel not in ("user", "system"):
            raise ValueError("handoff_channel must be 'user' or 'system'")
        self.bridge = bridge
        self.model = model
        self.system = system
        self.tools = tools
        self.handoff_channel = handoff_channel
        self.messages: List[Dict[str, Any]] = []
        self._turns: List[_Turn] = []
        self._summaries: List[str] = []
        self._pending_user_prefix: Optional[str] = None
        self.last_result: Optional[BridgeResult] = None

    # -- history -------------------------------------------------------------

    @property
    def producers(self) -> List[Optional[str]]:
        return [t.producer for t in self._turns]

    def add_user(self, content: Any) -> None:
        if self._pending_user_prefix is not None:
            blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
            content = [{"type": "text", "text": self._pending_user_prefix}] + list(blocks)
            self._pending_user_prefix = None
        self.messages.append({"role": "user", "content": content})
        self._turns.append(_Turn(role="user"))

    def _append_response(self, response: Any) -> None:
        # Append the FULL content blocks — thinking blocks included, replayed
        # verbatim on later requests.
        content = getattr(response, "content", None)
        blocks: List[Any] = []
        for block in content or []:
            if isinstance(block, dict):
                blocks.append(block)
            else:
                dump = getattr(block, "model_dump", None) or getattr(block, "to_dict", None)
                blocks.append(dump() if callable(dump) else block)
            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if btype == "thinking":
                text = block.get("thinking") if isinstance(block, dict) else getattr(block, "thinking", "")
                if text:
                    self._summaries.append(text)
        self.messages.append({"role": "assistant", "content": blocks})
        producer = getattr(response, "model", None) or self.model
        self._turns.append(_Turn(role="assistant", producer=producer))

    # -- calls ---------------------------------------------------------------

    def ask(self, content: Any, *, max_tokens: int = 16000, guard: Any = None, **kwargs: Any) -> BridgeResult:
        self.add_user(content)
        call_kwargs = dict(kwargs)
        if self.system is not None and "system" not in call_kwargs:
            call_kwargs["system"] = self.system
        if self.tools is not None and "tools" not in call_kwargs:
            call_kwargs["tools"] = self.tools
        result = self.bridge.create(
            model=self.model,
            messages=self.messages,
            max_tokens=max_tokens,
            guard=guard,
            **call_kwargs,
        )
        self._append_response(result.response)
        self.last_result = result
        return result

    # -- switching -----------------------------------------------------------

    def switch_model(self, target_model: str, *, carryover: bool = True) -> List[PredictedDrop]:
        """Move the conversation to ``target_model``.

        Existing thinking blocks stay in the history verbatim (the API drops
        whatever the target can't read, unbilled — and if the conversation
        later moves back to a model that can read them, they're still there).
        When drops are predicted and ``carryover`` is True, a handoff note
        built from captured reasoning summaries is appended so the target
        model doesn't re-plan blind.

        Returns the predicted drops.
        """
        if not _model_supports_mid_system(target_model) and any(
            m.get("role") == "system" for m in self.messages if isinstance(m, dict)
        ):
            raise IncompatibleSwitchError(
                f"The history contains mid-conversation system messages, "
                f"which {target_model} rejects — and removing them would be "
                "a history edit that invalidates later thinking blocks. "
                "Either switch to a model that accepts them (Opus 5, "
                "Opus 4.8, Fable/Mythos tier), or start conversations with "
                "handoff_channel='user' (the default) so the history stays "
                "portable."
            )
        drops = self.bridge.predict_switch(self.messages, target_model, self.producers)
        if drops and carryover:
            note = self._handoff_note(target_model, drops)
            if self.handoff_channel == "system" and _model_supports_mid_system(target_model):
                self.messages.append({"role": "system", "content": note})
                self._turns.append(_Turn(role="system"))
            else:
                self._pending_user_prefix = note
        self.model = target_model
        return drops

    def _handoff_note(self, target_model: str, drops: List[PredictedDrop]) -> str:
        parts = [
            "[Handoff note] This conversation previously ran on "
            f"{drops[-1].producer}; its internal reasoning is not available "
            "to the current model.",
        ]
        if self._summaries:
            recent = self._summaries[-5:]
            joined = "\n".join(f"- {s.strip()}"[:2000] for s in recent if s.strip())
            if joined:
                parts.append("Summaries of that reasoning, most recent last:\n" + joined)
        parts.append(
            "Continue the task from the visible conversation; rely on the "
            "summaries above rather than re-deriving prior decisions."
        )
        return "\n\n".join(parts)

    # -- persistence ---------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "system": self.system,
            "tools": self.tools,
            "handoff_channel": self.handoff_channel,
            "messages": copy.deepcopy(self.messages),
            "producers": self.producers,
            "summaries": list(self._summaries),
            "pending_user_prefix": self._pending_user_prefix,
        }

    @classmethod
    def restore(cls, bridge: ThinkingBridge, snapshot: Dict[str, Any]) -> "Conversation":
        conv = cls(
            bridge,
            snapshot["model"],
            system=snapshot.get("system"),
            tools=snapshot.get("tools"),
            handoff_channel=snapshot.get("handoff_channel", "user"),
        )
        conv.messages = copy.deepcopy(snapshot.get("messages") or [])
        producers = snapshot.get("producers") or []
        for message, producer in zip(conv.messages, list(producers) + [None] * len(conv.messages)):
            conv._turns.append(_Turn(role=message.get("role", "user"), producer=producer))
        conv._summaries = list(snapshot.get("summaries") or [])
        conv._pending_user_prefix = snapshot.get("pending_user_prefix")
        return conv
