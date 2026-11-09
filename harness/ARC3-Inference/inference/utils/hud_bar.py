"""Edge bars (timers, remaining-step counters) that changed between two grids.

Used by the [action] echo under ARC3_BAR_ECHO: "bottom bar 42->41". A bar is
a straight segment of two colors along a row or column near the border - a
fill run and a drained run meeting at one boundary - whose boundary moved:

    prev  ffffffffffffff....      (f = fill, . = drained)
    cur   fffffffffff.......

Measured on the 25 local games (2026-10-19): every game has one, on row 0/1
(cn04 lf52 m0r0 sp80 tn36 vc33), the bottom rows (most; ls20 two rows thick,
sb26 and sk48 on row 53), col 0 (lp85 r11l) or the right columns (ar25, sc25
two thick). Steps are 1-5 cells per action; some games drain only on
actions with an effect, sk48 every few actions.
"""

# Lines within this many cells of the border are candidates. Bars on the very
# edge (distance <= _EDGE) need a span of _MIN_SPAN_EDGE cells, inner ones
# (sb26 / sk48 on row 53, distance 10, span 64) at least 3/4 of the width.
_MAX_DIST = 11
_EDGE = 2
_MIN_SPAN_EDGE = 8
_MIN_SPAN_INNER = 48
_MAX_THICKNESS = 3
# How far from a candidate a moving sprite's other edge is looked for.
_SPRITE_REACH = 8


def _lines(grid, orientation: str) -> list[tuple[int, ...]]:
    if orientation == "row":
        return [tuple(row) for row in grid]
    return [tuple(col) for col in zip(*grid)]


def _boundaries(segment) -> int:
    return sum(1 for x, y in zip(segment, segment[1:]) if x != y)


def _line_change(prev, cur, dist: int):
    """(start, end, a, b, d0, d1) when the line's change is a bar boundary move."""
    changed = [i for i, (x, y) in enumerate(zip(prev, cur)) if x != y]
    if not changed:
        return None
    d0, d1 = changed[0], changed[-1]
    if d1 - d0 + 1 != len(changed):
        return None
    olds = {prev[i] for i in changed}
    news = {cur[i] for i in changed}
    if len(olds) != 1 or len(news) != 1:
        return None
    a, b = olds.pop(), news.pop()
    pair = (a, b)
    start, end = d0, d1
    while start > 0 and prev[start - 1] in pair and cur[start - 1] in pair:
        start -= 1
    while end < len(prev) - 1 and prev[end + 1] in pair and cur[end + 1] in pair:
        end += 1
    span = end - start + 1
    if span < (_MIN_SPAN_EDGE if dist <= _EDGE else _MIN_SPAN_INNER):
        return None
    if _boundaries(prev[start:end + 1]) > 1 or _boundaries(cur[start:end + 1]) > 1:
        return None
    return start, end, a, b, d0, d1


def bar_changes(grid_a, grid_b, fill_colors: dict | None = None) -> list[dict]:
    """Edge bars whose length changed from grid_a to grid_b.

    Args:
        grid_a: The grid before the action (rows of color ints).
        grid_b: The grid after the action.
        fill_colors: Optional per-game memory of each bar's fill color, keyed
            by the bar's place. A bar first seen shrinking defines its fill
            color; later changes (a refill too) are then reported as the
            length of that same color. Updated in place.

    Returns:
        One dict per changed bar: `side` ('top', 'bottom', 'left', 'right'),
        `lines` (the row or column indexes, 1-3 thick), `before` and `after`
        (the fill run's length in cells along the bar).
    """
    return [{k: v for k, v in bar.items() if not k.startswith("_")}
            for bar in _bar_records(grid_a, grid_b, fill_colors)]


def bar_cells(grid_a, grid_b) -> set[tuple[int, int]]:
    """The (row, col) cells whose change from grid_a to grid_b is an edge bar's."""
    cells = set()
    for bar in _bar_records(grid_a, grid_b, None):
        cells |= bar["_cells"]
    return cells


def _bar_records(grid_a, grid_b, fill_colors: dict | None) -> list[dict]:
    """bar_changes' bars, each with its changed cells under `_cells`."""
    if not grid_a or not grid_b or len(grid_a) != len(grid_b):
        return []
    if any(len(r1) != len(r2) for r1, r2 in zip(grid_a, grid_b)):
        return []
    memory = fill_colors if fill_colors is not None else {}
    out = []
    for orientation in ("row", "col"):
        prev_lines = _lines(grid_a, orientation)
        cur_lines = _lines(grid_b, orientation)
        n = len(prev_lines)
        found = {}
        for idx in range(n):
            dist = min(idx, n - 1 - idx)
            if dist > _MAX_DIST:
                continue
            change = _line_change(prev_lines[idx], cur_lines[idx], dist)
            if change:
                found[idx] = change
        # parallel lines with the same change form one thick bar
        idxs = sorted(found)
        groups = []
        for idx in idxs:
            if groups and groups[-1][-1] == idx - 1 and found[groups[-1][-1]] == found[idx]:
                groups[-1].append(idx)
            else:
                groups.append([idx])
        for group in groups:
            if len(group) > _MAX_THICKNESS:
                continue
            start, end, a, b, d0, d1 = found[group[0]]
            # a moving sprite changes the neighboring lines too
            clean = True
            for nb in (group[0] - 1, group[-1] + 1):
                if 0 <= nb < n and any(
                    prev_lines[nb][i] != cur_lines[nb][i] for i in range(d0, d1 + 1)
                ):
                    clean = False
            # a sprite moving across the line: its leading edge, a few lines on,
            # changes the same cells the other way (b -> a)
            for nb in range(group[0] - _SPRITE_REACH, group[-1] + _SPRITE_REACH + 1):
                if nb in group or not 0 <= nb < n:
                    continue
                if any(prev_lines[nb][i] == b and cur_lines[nb][i] == a
                       for i in range(d0, d1 + 1)):
                    clean = False
            if not clean:
                continue
            key = (orientation, group[0], group[-1], frozenset((a, b)))
            new = key not in memory
            fill = memory.setdefault(key, a)
            before = prev_lines[group[0]][start:end + 1].count(fill)
            after = cur_lines[group[0]][start:end + 1].count(fill)
            if orientation == "row":
                side = "top" if group[0] < n / 2 else "bottom"
            else:
                side = "left" if group[0] < n / 2 else "right"
            cells = {
                (line, i) if orientation == "row" else (i, line)
                for line in group for i in range(d0, d1 + 1)
            }
            out.append({"side": side, "lines": list(group), "before": before, "after": after,
                        "_cells": cells, "_size": max(before, after), "_new": new})
    # A bar's moving end also changes one cell of each crossing line; when the
    # crossing line has the bar's two colors along it (dc22's left column) it
    # passes as a 1-cell bar of its own. Keep the longer of the two; for the
    # last cell in a corner (both 1 long), the one already seen before.
    return [
        bar for bar in out
        if not any(other is not bar and bar["_cells"] <= other["_cells"]
                   and (other["_size"] > bar["_size"]
                        or (other["_size"] == bar["_size"] and bar["_new"]
                            and not other["_new"]))
                   for other in out)
    ]


def bar_phrase(bars: list[dict]) -> str:
    """'bottom bar 42->41, top bar 64->63' for an action's bar changes."""
    return ", ".join(f"{b['side']} bar {b['before']}->{b['after']}" for b in bars)
