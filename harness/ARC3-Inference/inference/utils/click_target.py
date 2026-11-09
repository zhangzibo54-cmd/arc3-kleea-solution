"""What a click landed on, for the echo of a click that changed nothing.

A no-op click is ambiguous to the model: the mechanic may not exist, or the
click missed (cn04 v34: a bounding-box centre on empty space, block
coordinates passed as cells). Naming the clicked cell's colour and component
tells the two apart.
"""
from __future__ import annotations

from typing import Any

from inference.utils.grid_utils import ARC_COLOR_CHARS

Grid = tuple[tuple[int, ...], ...]


def click_target(grid: Grid, row: int, col: int, background: set[str]) -> dict[str, Any] | None:
    """What a click at (row, col) of `grid` landed on, for a no-op click's echo.

    Returns:
        {"row", "col", "color", "background": True} on a background colour,
        else {"row", "col", "color", "pixels"} with the size of the cell's
        4-connected same-colour component; None off the board.
    """
    if not (0 <= row < len(grid) and 0 <= col < len(grid[row])):
        return None
    value = grid[row][col]
    color = ARC_COLOR_CHARS[value] if 0 <= value < len(ARC_COLOR_CHARS) else "?"
    if color in background:
        return {"row": row, "col": col, "color": color, "background": True}
    seen = {(row, col)}
    stack = [(row, col)]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if ((nr, nc) not in seen and 0 <= nr < len(grid) and 0 <= nc < len(grid[nr])
                    and grid[nr][nc] == value):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return {"row": row, "col": col, "color": color, "pixels": len(seen)}


def click_target_text(target: dict[str, Any] | None) -> str:
    """The clause a no-op click's verdict gains, or "" without a target."""
    if not target:
        return ""
    where = f"({target['row']},{target['col']})"
    if target.get("background"):
        return f"; the clicked cell {where} is background '{target['color']}'"
    return (f"; the clicked cell {where} is '{target['color']}', in a "
            f"{target['pixels']}-cell '{target['color']}' component")
