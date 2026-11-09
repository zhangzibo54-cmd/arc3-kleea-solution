"""Structured runtime state shared with created Python tools."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference.utils.grid_utils import format_grid_ascii


RUNTIME_STATE_FILENAME = "tool_runtime_state.json"


@dataclass(frozen=True)
class Frame:
    grid: tuple[tuple[int, ...], ...]
    step: int
    level: int

    @property
    def shape(self) -> tuple[int, int]:
        rows = len(self.grid)
        cols = max((len(row) for row in self.grid), default=0)
        return rows, cols

    @property
    def ascii(self) -> str:
        return format_grid_ascii(self.grid)

    def __str__(self) -> str:
        rows, cols = self.shape
        return (
            f"Level: {self.level}\n"
            f"Step: {self.step}\n"
            f"Grid shape: {rows} x {cols}\n"
            f"Grid contents:\n{self.ascii}"
        )


@dataclass(frozen=True)
class HistoryEntry:
    """An action and its resulting board, with optional replay metadata.

    Attributes:
        action: The action that produced this board, or empty for a level start.
        frame: The board after the action.
        result: Compact action metadata, including an optional animation hash.
        animations: The intermediate grids feedback shows without being asked
            (a bounded sample), in chronological order.
        animation_timeline: The action's diff timeline over every state it
            passed through, in the form of `last_animation_timeline`; empty
            when it did not animate.
        animation_chain: Every state of the action, the board before it
            first and this entry's board last. Kept in the solver's memory
            only: it is not written to the runtime-state file.
    """

    action: str
    frame: Frame
    result: dict[str, Any] = field(default_factory=dict)
    animations: tuple[tuple[tuple[int, ...], ...], ...] = ()
    animation_timeline: dict[str, Any] = field(default_factory=dict)
    animation_chain: tuple[tuple[tuple[int, ...], ...], ...] = field(
        default=(), compare=False, repr=False,
    )


def normalize_grid(raw: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    rows: list[tuple[int, ...]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)):
            continue
        cells: list[int] = []
        for cell in row:
            try:
                cells.append(int(cell))
            except (TypeError, ValueError):
                cells.append(0)
        rows.append(tuple(cells))
    return tuple(rows)


def frame_from_payload(payload: Any) -> Frame | None:
    if not isinstance(payload, dict):
        return None
    try:
        step = max(0, int(payload.get("step", 0) or 0))
    except (TypeError, ValueError):
        step = 0
    try:
        level = max(1, int(payload.get("level", 1) or 1))
    except (TypeError, ValueError):
        level = 1
    return Frame(
        grid=normalize_grid(payload.get("grid")),
        step=step,
        level=level,
    )


def frame_to_payload(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    return {
        "grid": [list(row) for row in frame.grid],
        "step": frame.step,
        "level": frame.level,
    }


def history_entry_from_payload(payload: Any) -> HistoryEntry | None:
    if not isinstance(payload, dict):
        return None
    frame = frame_from_payload(payload.get("frame"))
    if frame is None:
        return None
    return HistoryEntry(
        action=str(payload.get("action", "")).strip(), frame=frame,
        result=dict(payload["result"]) if isinstance(payload.get("result"), dict) else {},
        animations=tuple(normalize_grid(grid) for grid in payload.get("animations") or ()),
        animation_timeline=(dict(payload["animation_timeline"])
                            if isinstance(payload.get("animation_timeline"), dict) else {}),
    )


def history_entry_to_payload(entry: HistoryEntry) -> dict[str, Any]:
    payload = {
        "action": entry.action,
        "frame": frame_to_payload(entry.frame),
        "result": dict(getattr(entry, "result", {}) or {}),
    }
    animations = getattr(entry, "animations", ())
    if animations:
        payload["animations"] = [[list(row) for row in grid] for grid in animations]
    timeline = getattr(entry, "animation_timeline", None)
    if timeline:
        payload["animation_timeline"] = dict(timeline)
    return payload


def state_delta_enabled() -> bool:
    """ARC3_STATE_DELTA: keep the runtime state in memory, send history deltas (default off).

    Off, every action rewrites the whole runtime-state file, every tool call
    reads it back, and the Python worker is sent the whole history after each
    action: all O(history), and the host part runs under the GIL every game
    shares. On, the solver's history is published in memory (the file is
    written when a game starts and ends), each entry is converted once, and
    the worker is sent only the entries it does not hold yet. What callers
    and the worker see is unchanged.
    """
    return os.environ.get("ARC3_STATE_DELTA", "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class _LiveState:
    """A game's runtime state as published in memory.

    Attributes:
        source: The solver's entries, compared by identity to find new ones.
        loaded: Each source entry as a file round trip returns it (None where
            that drops it), so a load reads what it would read from the file.
        current_frame: The current frame, round-tripped likewise.
    """

    source: list[HistoryEntry] = field(default_factory=list)
    loaded: list[HistoryEntry | None] = field(default_factory=list)
    current_frame: Frame | None = None


_LIVE_STATES: dict[str, _LiveState] = {}


def _round_trip_entry(entry: HistoryEntry) -> HistoryEntry | None:
    return history_entry_from_payload(json.loads(json.dumps(history_entry_to_payload(entry))))


def _publish_live_state(
    path: Path,
    current_frame: Frame | None,
    history: list[HistoryEntry],
) -> None:
    live = _LIVE_STATES.setdefault(str(path), _LiveState())
    # history is append-only in practice; anything else re-converts from the
    # first entry that is not the one converted before
    keep = 0
    for old, new in zip(live.source, history):
        if old is not new:
            break
        keep += 1
    del live.source[keep:], live.loaded[keep:]
    for entry in history[keep:]:
        live.source.append(entry)
        live.loaded.append(_round_trip_entry(entry))
    live.current_frame = frame_from_payload(json.loads(json.dumps(frame_to_payload(current_frame))))


def release_runtime_state(path: Path) -> None:
    """Drop the in-memory state published for `path`; later loads read the file."""
    _LIVE_STATES.pop(str(path), None)


def load_runtime_state(path: Path) -> tuple[Frame | None, list[HistoryEntry]]:
    """Load the current frame and history last written for `path`.

    Returns:
        The current frame (None if there is none) and a new list of history
        entries. Under ARC3_STATE_DELTA the entries are shared between loads
        (they are frozen; their dicts must not be mutated).
    """
    live = _LIVE_STATES.get(str(path))
    if live is not None:
        return live.current_frame, [entry for entry in live.loaded if entry is not None]
    if not path.exists():
        return None, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    current_frame = frame_from_payload(payload.get("current_frame"))
    history_entries = [
        entry
        for raw_entry in payload.get("history", [])
        for entry in [history_entry_from_payload(raw_entry)]
        if entry is not None
    ]
    return current_frame, history_entries


def write_runtime_state(
    path: Path,
    *,
    current_frame: Frame | None,
    history: list[HistoryEntry],
    persist: bool = False,
) -> None:
    """Write the runtime state for `path`.

    Under ARC3_STATE_DELTA the state is published in memory, and the file is
    written only when it does not exist yet or `persist` is set (a game's
    start and end); otherwise the file is always rewritten.
    """
    if state_delta_enabled():
        _publish_live_state(path, current_frame, history)
        if not persist and path.exists():
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_frame": frame_to_payload(current_frame),
        "history": [history_entry_to_payload(entry) for entry in history],
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    # No indent: with one, json falls back to its pure-Python encoder, which made
    # this rewrite of the whole history most of the cost of every action (O(steps)).
    tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp_path.replace(path)
