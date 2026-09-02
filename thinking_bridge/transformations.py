"""Parsing for the ``input_transformations`` response field.

With the ``thinking-binding-controls-2026-08-01`` beta header, every response
from a thinking-capable model carries a top-level ``input_transformations``
array (empty when nothing was dropped, never null); without the header the
field is absent. Each entry names a dropped thinking block:

    {"type": "thinking_dropped", "path": "messages.1.content.0",
     "reason": "model_binding_mismatch"}

``reason`` is ``"prefix_binding_mismatch"`` (your history changed — usually a
bug in the harness) or ``"model_binding_mismatch"`` (the conversation switched
models — expected, not a bug). Later checks add new types/reasons, so unknown
values are kept but never treated as errors.

When streaming, the array arrives on the ``message`` object in
``message_start`` (and again in the final ``message_delta`` after a mid-stream
server-side fallback); the SDK's accumulated final message carries it too, so
parsing the final message covers both paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

REASON_MODEL_MISMATCH = "model_binding_mismatch"
REASON_PREFIX_MISMATCH = "prefix_binding_mismatch"
KNOWN_REASONS = frozenset({REASON_MODEL_MISMATCH, REASON_PREFIX_MISMATCH})
KNOWN_TYPES = frozenset({"thinking_dropped"})


@dataclass(frozen=True)
class InputTransformation:
    type: str
    path: str
    reason: str
    raw: Dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def is_model_switch(self) -> bool:
        """The drop came from a model switch — expected, not a harness bug."""
        return self.reason == REASON_MODEL_MISMATCH

    @property
    def is_history_edit(self) -> bool:
        """The drop came from an edited history — fix the harness."""
        return self.reason == REASON_PREFIX_MISMATCH

    @property
    def is_recognized(self) -> bool:
        return self.type in KNOWN_TYPES and self.reason in KNOWN_REASONS


def _entry_to_dict(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, dict):
        return dict(entry)
    for attr in ("to_dict", "model_dump"):
        fn = getattr(entry, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:
                pass
    return {
        "type": getattr(entry, "type", None),
        "path": getattr(entry, "path", None),
        "reason": getattr(entry, "reason", None),
    }


def parse_input_transformations(response: Any) -> Optional[List[InputTransformation]]:
    """Extract ``input_transformations`` from a Message.

    Returns ``None`` when the field is absent (the beta header was not sent),
    and a list (possibly empty) when it is present. Works on typed SDK
    responses, on responses where the field is untyped extra data, and on
    plain dicts.
    """
    entries = None
    if isinstance(response, dict):
        if "input_transformations" in response:
            entries = response["input_transformations"]
        else:
            return None
    else:
        entries = getattr(response, "input_transformations", None)
        if entries is None:
            # Launch SDKs may not type the field yet — check pydantic extras.
            extra = getattr(response, "model_extra", None)
            if isinstance(extra, dict) and "input_transformations" in extra:
                entries = extra["input_transformations"]
            elif isinstance(extra, dict):
                return None
            elif not hasattr(response, "input_transformations"):
                return None
    if entries is None:
        return None
    parsed: List[InputTransformation] = []
    for entry in entries:
        d = _entry_to_dict(entry)
        parsed.append(
            InputTransformation(
                type=str(d.get("type") or ""),
                path=str(d.get("path") or ""),
                reason=str(d.get("reason") or ""),
                raw=d,
            )
        )
    return parsed
