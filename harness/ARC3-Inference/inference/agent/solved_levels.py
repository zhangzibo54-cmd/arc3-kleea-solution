"""Solved-level memory pinned to the end of the system prompt (ARC3_SOLVED_LEVELS).

Adapted from lalalia9's ARC-AGI-3 notebook (itself from rellik13/sirikilohit's
M85 write-up, CC-BY 4.0). When a level is cleared, the harness records facts it
already has: the level number, the winning attempt's actions (count and the
last 30) and the tail of the reasoning in the turn that cleared it. The model
is asked, at most three times, for one `Rule for level N:` line, captured from
its reply or reasoning. The records are rendered as one block at the end of the
system prompt.

The block is only rewritten when a context trim drops history: the server has
to re-read the conversation then anyway, so changing the system prompt costs
no extra prefill, and the shared start of the system prompt stays cached.
Before the first trim on a level, the previous level is still in the
conversation itself.

The functions here are pure; `ToolAgent` keeps the records and calls them.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

ACTIONS_KEPT = 30
REASONING_CHARS = 900
RULE_CHARS = 600
FULL_LEVELS = 3
MAX_ASKS = 3

HEADER = "=== SOLVED LEVELS (facts from this game + your own conclusions; kept for the whole game) ==="
FOOTER = "=== END SOLVED LEVELS ==="

# The anchor in LEVEL_START_FEEDBACK_USER_PROMPT the first request goes before:
# after the winning frame has been checked, before the new board is inspected.
ASK_ANCHOR = "Then inspect the current board."

_INSTRUCTION_ECHO = "states the goal, what each control did"


def solved_levels_enabled() -> bool:
    """Whether solved-level memory is on (ARC3_SOLVED_LEVELS, default off)."""
    return os.environ.get("ARC3_SOLVED_LEVELS", "0").strip().lower() in ("1", "true", "yes", "on")


def first_ask_text(level: int) -> str:
    """The request placed in the level-start message, before the new board."""
    return (
        f"Then write a line starting `Rule for level {level}:` that states the goal, "
        "what each control did and the plan that won (at most 5 lines). It is kept "
        "for the rest of the game."
    )


def reminder_text(level: int) -> str:
    """The request appended to a later message while the rule is missing."""
    return (
        f"You have not written `Rule for level {level}:` yet. Write it now (at most "
        "5 lines: the goal, what each control did and the plan that won) before "
        "your next action. It is kept for the rest of the game."
    )


def winning_attempt_actions(entries: Iterable[Any], level: int) -> list[str]:
    """Return the actions of the attempt that cleared `level`, in order.

    Args:
        entries: The game's history entries (`HistoryEntry`-like: `.action`,
            `.frame.level`, `.result`). An entry's frame is the board after its
            action, so the action belongs to the level of the frame before it:
            this holds whether the winning action's own frame is the old
            level's winning board or already the next level's first board.
        level: The 1-based level number.

    Returns:
        The actions after the level's last RESET or death, up to and including
        the winning action; empty when the level has no recorded actions.
    """
    actions: list[str] = []
    previous = None
    for entry in entries:
        action = str(getattr(entry, "action", "") or "").strip()
        if action and previous == level:
            if action.upper().startswith("RESET"):
                actions = []
            elif (getattr(entry, "result", None) or {}).get("game_over"):
                actions = []
            else:
                actions.append(action)
        frame_level = getattr(getattr(entry, "frame", None), "level", None)
        if isinstance(frame_level, int):
            previous = frame_level
    return actions


def find_rule(text: str, level: int, *, from_reasoning: bool = False) -> str | None:
    """Return the last `Rule for level N:` statement in `text`, or None.

    The statement is the rest of that line plus the following non-blank lines,
    at most five lines and RULE_CHARS characters. Reasoning rarely has blank
    lines, so with `from_reasoning` a first line of 40 characters or more is
    the whole statement. An echo of the request itself (a model thinking about
    what it was asked) is not a rule.
    """
    pattern = re.compile(r"Rule for level\s*%d\s*:[*`]*[ \t]*" % int(level), re.I)
    for match in reversed(list(pattern.finditer(text or ""))):
        lines = [line.strip() for line in text[match.end():].split("\n")]
        if from_reasoning and len(lines[0]) >= 40:
            lines = lines[:1]
        kept: list[str] = []
        for line in lines[:5]:
            if line.startswith("```") or (not line and kept):
                break
            if line:
                kept.append(line)
        rule = " ".join(" ".join(kept).split())[:RULE_CHARS]
        if len(rule) >= 20 and _INSTRUCTION_ECHO not in rule and not rule.startswith("`"):
            return rule
    return None


def message_text(message: dict[str, Any], keys: tuple[str, ...]) -> str:
    """The text of one assistant message under the first of `keys` it has."""
    for key in keys:
        value = message.get(key)
        if isinstance(value, list):
            value = "\n".join(
                str(part.get("text", "")) for part in value
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if isinstance(value, str) and value.strip():
            return value
    return ""


_REPLY_KEYS = ("content",)
_REASONING_KEYS = ("reasoning_content", "reasoning")


def _newest_rule(
    assistant: list[dict[str, Any]],
    level: int,
    keys: tuple[str, ...],
) -> str | None:
    for message in reversed(assistant):
        rule = find_rule(
            message_text(message, keys), level, from_reasoning=keys == _REASONING_KEYS
        )
        if rule:
            return rule
    return None


def capture_rules(records: list[dict[str, Any]], messages: Iterable[dict[str, Any]]) -> bool:
    """Fill in rules the model wrote in `messages`; return whether any changed.

    A rule stated in a reply wins over one in reasoning, which may be a draft
    or a recollection; among rules from the same source the newest wins, so a
    revised rule replaces an earlier one.
    """
    assistant = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    changed = False
    for record in records:
        source = "content"
        rule = _newest_rule(assistant, record["level"], _REPLY_KEYS)
        if rule is None and record.get("rule_source") != "content":
            source = "reasoning"
            rule = _newest_rule(assistant, record["level"], _REASONING_KEYS)
        if rule and rule != record.get("rule"):
            record["rule"] = rule
            record["rule_source"] = source
            changed = True
    return changed


def reasoning_tail(messages: Iterable[dict[str, Any]]) -> str:
    """The last REASONING_CHARS of the newest assistant reasoning (or reply)."""
    for message in reversed(list(messages)):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        # reasoning explains the win better than the reply
        text = message_text(message, _REASONING_KEYS + _REPLY_KEYS)
        if text:
            return " ".join(text.split())[-REASONING_CHARS:]
    return ""


def render_block(records: list[dict[str, Any]]) -> str:
    """The SOLVED LEVELS block for the system prompt, or "" without records.

    Every level keeps its rule; the newest FULL_LEVELS also keep their winning
    actions and reasoning tail.
    """
    if not records:
        return ""
    lines = [HEADER]
    full_from = max(0, len(records) - FULL_LEVELS)
    for i, record in enumerate(records):
        lines.append(
            f"Level {record['level']} - cleared after {record['n_actions']} actions "
            "in the winning attempt."
        )
        lines.append(f"  Your rule: {record.get('rule') or 'not stated'}")
        if i >= full_from:
            tail = record["actions"]
            skipped = record["n_actions"] - len(tail)
            lines.append(
                "  Last actions before it cleared"
                + (f" ({skipped} earlier omitted)" if skipped > 0 else "")
                + ": " + (", ".join(tail) if tail else "none recorded")
            )
            if record.get("reasoning"):
                lines.append("  Your reasoning in the turn that cleared it: " + record["reasoning"])
    lines.append(
        "Reuse these where the new level has the same controls; verify with one probe "
        "before relying on them."
    )
    lines.append(FOOTER)
    return "\n".join(lines)
