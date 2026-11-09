"""Identify repeated gameplay animations across translated and scrolled views."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from inference.utils.frame_diff import chained_view_shift, _animation_pair_shift


def animation_fingerprint(
    frames: Sequence[Sequence[Sequence[int]]],
    *,
    expected_shift: tuple[int, int] | None = None,
    border: int = 4,
) -> str | None:
    """Hash changed cell transitions in world coordinates, independent of position.

    Args:
        frames: The complete animation chain, including its start and end boards.
        expected_shift: The action's independently detected camera displacement.
            A mismatch suppresses the hash rather than comparing screen positions.
        border: Screen edge width excluded to avoid timer and HUD changes.

    Returns:
        A short fingerprint, or None when there are no intermediate gameplay
        changes or camera registration cannot be trusted.
    """
    if len(frames) < 3:
        return None
    # Use the same adjacent-frame validation as the solver's long-scroll
    # detector.  Estimating a shift only when a pair changes by 15% silently
    # treated sparse gameplay during a scroll as stationary and put later
    # transitions in screen coordinates.
    total_shift = chained_view_shift(frames)
    if total_shift is None:
        return None
    if expected_shift is not None and total_shift != expected_shift:
        return None
    offsets = [(0, 0)]
    for before, after in zip(frames, frames[1:]):
        step = _animation_pair_shift(
            before, after, step_shift=10, margin=4, min_overlap=600,
            min_agree=0.9, min_gain=0.1,
        )
        if step is None:
            return None
        old_row, old_col = offsets[-1]
        offsets.append((old_row + step[0], old_col + step[1]))

    transitions: list[list[tuple[int, int, int, int]]] = []
    positions: list[tuple[int, int]] = []
    for index, (before, after) in enumerate(zip(frames, frames[1:])):
        before_offset = offsets[index]
        after_offset = offsets[index + 1]
        rows = min(len(before), len(after))
        cols = min(len(before[0]), len(after[0])) if rows else 0
        changed = []
        for row in range(border, rows - border):
            world_row = row - after_offset[0]
            source_row = world_row + before_offset[0]
            if not border <= source_row < rows - border:
                continue
            for col in range(border, cols - border):
                world_col = col - after_offset[1]
                source_col = world_col + before_offset[1]
                if not border <= source_col < cols - border:
                    continue
                old = before[source_row][source_col]
                new = after[row][col]
                if old != new:
                    changed.append((world_row, world_col, old, new))
                    positions.append((world_row, world_col))
        if changed:
            transitions.append(changed)
    if not transitions:
        return None
    origin_row = min(row for row, _ in positions)
    origin_col = min(col for _, col in positions)
    relative = [
        [(row - origin_row, col - origin_col, old, new)
         for row, col, old, new in changed]
        for changed in transitions
    ]
    encoded = json.dumps(relative, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()[:16]
