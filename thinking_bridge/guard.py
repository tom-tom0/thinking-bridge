"""Client-side append-only history guard.

Claude Fable 5.1 binds each thinking block to the conversation prefix that
produced it (the ``system`` prompt, the ``tools`` set, and every message
before the block). Editing any of those invalidates every later thinking
block — a 400 on enforced accounts. This guard catches the edit in the
client, before the request is sent, with an actionable diagnostic.

What the guard allows (mirroring the API's rules):
- appending new messages (append-only histories always pass);
- adding/moving/removing ``cache_control`` markers (stripped before hashing);
- reordering ``tools`` without changing them (bound as a name-sorted set);
- removing a *leading* run of messages (client-side simple compaction — the
  guard resets when the new history is not an extension of the old one and
  ``allow_reset`` is True).

What it rejects: editing/reordering/removing an earlier turn, injecting text
into an earlier turn, or rebuilding ``system``/``tools`` mid-conversation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


class PrefixMismatchError(ValueError):
    """The request edits an earlier part of the conversation prefix."""

    def __init__(self, message: str, *, kind: str, index: Optional[int] = None):
        super().__init__(message)
        self.kind = kind  # "system" | "tools" | "message"
        self.index = index


def _strip_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _strip_cache_control(v)
            for k, v in value.items()
            if k != "cache_control"
        }
    if isinstance(value, (list, tuple)):
        return [_strip_cache_control(v) for v in value]
    return value


def _to_plain(value: Any) -> Any:
    """Best-effort conversion of SDK objects to JSON-serializable data."""
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    for attr in ("to_dict", "model_dump"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _to_plain(fn())
            except Exception:
                pass
    return value


def _digest(value: Any) -> str:
    canonical = json.dumps(
        _strip_cache_control(_to_plain(value)),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tools_digest(tools: Any) -> str:
    if not tools:
        return _digest(None)
    plain = _to_plain(tools)
    # The API binds tools as a name-sorted set: reordering is fine.
    keyed = sorted(
        (json.dumps(_strip_cache_control(t), sort_keys=True, default=str) for t in plain)
    )
    return hashlib.sha256("\n".join(keyed).encode("utf-8")).hexdigest()


@dataclass
class GuardReport:
    ok: bool
    appended_messages: int
    reset: bool = False


class PrefixGuard:
    """Tracks one conversation's request prefix across requests."""

    def __init__(self, *, allow_reset: bool = True):
        self.allow_reset = allow_reset
        self._system_digest: Optional[str] = None
        self._tools_digest: Optional[str] = None
        self._message_digests: List[str] = []

    def check(
        self,
        messages: Sequence[Any],
        *,
        system: Any = None,
        tools: Any = None,
    ) -> GuardReport:
        """Validate this request against the previously seen prefix.

        Raises :class:`PrefixMismatchError` on an edit; otherwise records the
        new state and returns a report.
        """
        new_system = _digest(system)
        new_tools = _tools_digest(tools)
        new_messages = [_digest(m) for m in messages]

        if self._system_digest is None:
            return self._record(new_system, new_tools, new_messages, reset=False)

        if new_system != self._system_digest:
            raise PrefixMismatchError(
                "The top-level `system` prompt changed mid-conversation. This "
                "invalidates every later Claude Fable 5.1 thinking block. "
                "Freeze `system` at session start and append a "
                '{"role": "system"} message to `messages` instead.',
                kind="system",
            )
        if new_tools != self._tools_digest:
            raise PrefixMismatchError(
                "The `tools` array changed mid-conversation (beyond "
                "reordering). This invalidates every later thinking block. "
                "Declare the full set at session start (defer_loading: true "
                "for hidden tools) and use mid-conversation tool_addition / "
                "tool_removal system messages instead.",
                kind="tools",
            )

        old = self._message_digests
        if len(new_messages) >= len(old) and new_messages[: len(old)] == old:
            appended = len(new_messages) - len(old)
            return self._record(new_system, new_tools, new_messages, reset=False, appended=appended)

        # Not an extension. A brand-new history that shares nothing with the
        # old one is a client-side simple-compaction restart, allowed when
        # opted in. Any overlap means old turns were kept around an edit.
        shares_old_turns = bool(set(old) & set(new_messages))
        first_diff = next(
            (
                i
                for i in range(min(len(old), len(new_messages)))
                if new_messages[i] != old[i]
            ),
            min(len(old), len(new_messages)),
        )
        if not shares_old_turns and self.allow_reset:
            return self._record(new_system, new_tools, new_messages, reset=True)
        raise PrefixMismatchError(
            f"messages[{first_diff}] changed since the previous request "
            "(edited, reordered, or removed while later turns were kept). "
            "Every thinking block after that point is invalidated. Make the "
            "history append-only: leave earlier turns byte-identical, send "
            "per-turn reminders as turn-scoped system messages, and use "
            "server-side context editing or compaction to shrink history."
            + (" (No shared prefix was found.)" if not shares_old_turns else ""),
            kind="message",
            index=first_diff,
        )

    def _record(
        self,
        system_digest: str,
        tools_digest: str,
        message_digests: List[str],
        *,
        reset: bool,
        appended: Optional[int] = None,
    ) -> GuardReport:
        self._system_digest = system_digest
        self._tools_digest = tools_digest
        prev = len(self._message_digests)
        self._message_digests = list(message_digests)
        if appended is None:
            appended = len(message_digests) if reset or prev == 0 else 0
        return GuardReport(ok=True, appended_messages=appended, reset=reset)

    def snapshot(self) -> dict:
        return {
            "system": self._system_digest,
            "tools": self._tools_digest,
            "messages": copy.copy(self._message_digests),
        }

    @classmethod
    def restore(cls, snapshot: dict, *, allow_reset: bool = True) -> "PrefixGuard":
        guard = cls(allow_reset=allow_reset)
        guard._system_digest = snapshot.get("system")
        guard._tools_digest = snapshot.get("tools")
        guard._message_digests = list(snapshot.get("messages") or [])
        return guard
