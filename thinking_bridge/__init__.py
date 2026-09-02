"""thinking-bridge: make Claude Fable 5.1's preserved-thinking rules
observable and survivable when a conversation moves across models.

Quick start::

    import anthropic
    from thinking_bridge import ThinkingBridge, Conversation

    bridge = ThinkingBridge(anthropic.Anthropic())
    conv = Conversation(bridge, "claude-fable-5-1")
    conv.ask("Plan the refactor.")

    drops = conv.switch_model("claude-opus-5")   # predicted drops, carryover note appended
    conv.ask("Now apply step 1.")
"""

from .bridge import (
    BINDING_BETA,
    BridgeResult,
    Conversation,
    IncompatibleSwitchError,
    ThinkingBridge,
    strip_thinking_blocks,
)
from .compat import PredictedDrop, can_read, is_fable_51_tier, predict_drops
from .guard import GuardReport, PrefixGuard, PrefixMismatchError
from .transformations import (
    KNOWN_REASONS,
    REASON_MODEL_MISMATCH,
    REASON_PREFIX_MISMATCH,
    InputTransformation,
    parse_input_transformations,
)

__version__ = "0.1.0"

__all__ = [
    "BINDING_BETA",
    "BridgeResult",
    "Conversation",
    "GuardReport",
    "IncompatibleSwitchError",
    "InputTransformation",
    "KNOWN_REASONS",
    "PredictedDrop",
    "PrefixGuard",
    "PrefixMismatchError",
    "REASON_MODEL_MISMATCH",
    "REASON_PREFIX_MISMATCH",
    "ThinkingBridge",
    "can_read",
    "is_fable_51_tier",
    "parse_input_transformations",
    "predict_drops",
    "strip_thinking_blocks",
]
