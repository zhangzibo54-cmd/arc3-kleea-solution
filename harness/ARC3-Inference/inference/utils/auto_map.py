"""ARC3_AUTO_MAP: a block map the harness derives from the board (pure Python, no numpy).

The board is read as a lattice of s x s tiles repeating every p cells (p - s is the gap between
tiles, 0 for plain blocks). The lattice is the one under which the board's 4-connected
one-colour components are best explained as whole unions of tiles; a game gets a map only if
the result is clean enough (`auto_map_decide`). Each tile becomes one character: its colour
letter when uniform, a glyph for a two-colour centre line ('-' '|'), cross ('+' 'x') or centred
square ('o'), otherwise the letter of its most common colour.

Ported from results/step-cost/automap (tile_fit.py, detect.py, gate.py, render_review.py), where
the thresholds were set on the v27 hard12 boards and checked on the level-1 boards of all 25
offline games. This module only uses the standard library: it is also spliced into the Python
sandbox worker, so its top-level names start with `auto_map_` or `_am_`.
"""

_AM_SYMBOLS = "WwgGcBMPRbSYOrNp"  # ARC colour index -> letter (as ARC_COLOR_CHARS)
_AM_SIZE = 64
_AM_HUD = 2  # components only in this edge band are HUD, not evidence
_AM_UI = 4  # edge band left out of the gate's measures (bars, counters)
_AM_IOU = 0.75  # a component is tiled when it matches its tiles this well
_AM_GRIDLINE = 0.9  # share of a component in the gaps that makes it the lattice's lines
_AM_MIN_TILED = 3  # tiled components a lattice needs
_AM_BACKGROUND = 0.3  # components above this share of the board are background
_AM_MIN_SCORE = 0.5  # weakest lattice score accepted
_AM_TIE = 0.03  # near-tied lattices: the larger period wins
# gate: hide the map when the simplified map explains under FLOOR of the foreground, or when
# tiles barely repeat and either the map explains too little or the lattice fits poorly
_AM_REP_MIN = 0.2
_AM_FID_MIN = 0.8
_AM_FID_FLOOR = 0.5
_AM_FIT_MIN = 0.7
_AM_GLYPHS = (("hline_mid", "-"), ("vline_mid", "|"), ("plus", "+"), ("x", "x"),
              ("square", "o"))


def _am_components(grid):
    """4-connected one-colour components as (cells, size, (r1, c1, r2, c2) exclusive)."""
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    found = []
    for r in range(rows):
        for c in range(cols):
            if seen[r][c]:
                continue
            color = grid[r][c]
            seen[r][c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < rows and 0 <= nx < cols and not seen[ny][nx] \
                            and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            ys = [y for y, _ in cells]
            xs = [x for _, x in cells]
            found.append((cells, len(cells), (min(ys), min(xs), max(ys) + 1, max(xs) + 1)))
    limit = _AM_BACKGROUND * rows * cols
    out = []
    for cells, size, (r1, c1, r2, c2) in found:
        if size > limit:
            continue
        if r2 <= _AM_HUD or r1 >= rows - _AM_HUD or c2 <= _AM_HUD or c1 >= cols - _AM_HUD:
            continue
        out.append((cells, size, (r1, c1, r2, c2)))
    return out


def _am_area(r_lo, r_hi, c_lo, c_hi):
    """Cells of a rectangle clipped to the board."""
    r_lo, c_lo = max(0, r_lo), max(0, c_lo)
    r_hi, c_hi = min(_AM_SIZE, r_hi), min(_AM_SIZE, c_hi)
    return max(0, r_hi - r_lo) * max(0, c_hi - c_lo)


def _am_fit(cells, size, box, p, s, r0, c0):
    """IoU of a component with the tiles it covers, or None for sub-tile detail."""
    r1, c1, r2, c2 = box
    if (r1 - r0) // p == (r2 - 1 - r0) // p and (c1 - c0) // p == (c2 - 1 - c0) // p:
        i, j = (r1 - r0) // p, (c1 - c0) // p
        tr, tc = r0 + i * p, c0 + j * p
        inside = sum(1 for y, x in cells if tr <= y < tr + s and tc <= x < tc + s)
        tile = _am_area(tr, tr + s, tc, tc + s)
        iou = inside / (tile + size - inside) if tile + size - inside else 0.0
        return iou if iou >= _AM_IOU else None
    count = {}
    where = []
    for y, x in cells:
        i, oy = divmod(y - r0, p)
        j, ox = divmod(x - c0, p)
        where.append((i, j, oy < s, ox < s))
        if oy < s and ox < s:
            count[(i, j)] = count.get((i, j), 0) + 1
    tiles = set()
    for (i, j), n in count.items():
        core = _am_area(r0 + i * p, r0 + i * p + s, c0 + j * p, c0 + j * p + s)
        if core and n >= 0.5 * core:
            tiles.add((i, j))
    if not tiles:
        return 0.0
    best = 0.0
    for fill in ((False, True) if s < p else (False,)):
        pred = 0
        for i, j in tiles:
            pred += _am_area(r0 + i * p, r0 + i * p + s, c0 + j * p, c0 + j * p + s)
            if fill:
                if (i, j + 1) in tiles:
                    pred += _am_area(r0 + i * p, r0 + i * p + s,
                                     c0 + j * p + s, c0 + (j + 1) * p)
                if (i + 1, j) in tiles:
                    pred += _am_area(r0 + i * p + s, r0 + (i + 1) * p,
                                     c0 + j * p, c0 + j * p + s)
                if (i, j + 1) in tiles and (i + 1, j) in tiles and (i + 1, j + 1) in tiles:
                    pred += _am_area(r0 + i * p + s, r0 + (i + 1) * p,
                                     c0 + j * p + s, c0 + (j + 1) * p)
        inter = 0
        for i, j, in_r, in_c in where:
            if in_r and in_c:
                inter += (i, j) in tiles
            elif fill and in_r:
                inter += (i, j) in tiles and (i, j + 1) in tiles
            elif fill and in_c:
                inter += (i, j) in tiles and (i + 1, j) in tiles
            elif fill:
                inter += ((i, j) in tiles and (i, j + 1) in tiles and (i + 1, j) in tiles
                          and (i + 1, j + 1) in tiles)
        union = pred + size - inter
        best = max(best, inter / union if union else 0.0)
    return best


def _am_evaluate(comps, p, s, r0, c0):
    """(score, tiled components, tiled pixels) of one lattice."""
    tiled = untiled = n_tiled = 0
    for cells, size, box in comps:
        if s < p:
            in_gap = sum(1 for y, x in cells
                         if (y - r0) % p >= s or (x - c0) % p >= s)
            if in_gap >= _AM_GRIDLINE * size:
                tiled += size  # the lines between tiles: the lattice itself
                continue
        iou = _am_fit(cells, size, box, p, s, r0, c0)
        if iou is None:
            continue
        if iou >= _AM_IOU:
            tiled += size
            n_tiled += 1
        else:
            untiled += size
    total = tiled + untiled
    return (tiled / total if total else 0.0), n_tiled, tiled


def _am_best_origin(comps, p, s):
    """(score, r0, c0, tiled pixels) of the best origin for a fixed period and tile."""
    best = None
    for r0 in range(p):
        for c0 in range(p):
            score, n_tiled, tiled = _am_evaluate(comps, p, s, r0, c0)
            key = (score if n_tiled >= _AM_MIN_TILED else 0.0, tiled)
            if best is None or key > best[0]:
                best = (key, r0, c0)
    (score, tiled), r0, c0 = best
    return score, r0, c0, tiled


def auto_map_detect(grid):
    """The board's lattice as {'p', 's', 'r0', 'c0', 'score'}, or None.

    Tries periods 3..10, tiles of p, p - 1 and p - 2 cells and every origin; keeps the best
    origin and tile per period, then the largest period within _AM_TIE of the best score.
    """
    comps = _am_components(grid)
    rows = []
    for p in range(3, 11):
        best = None
        for s in range(max(3, p - 2), p + 1):
            score, r0, c0, tiled = _am_best_origin(comps, p, s)
            key = (score, tiled, s)
            if best is None or key > best[0]:
                best = (key, {"p": p, "s": s, "r0": r0, "c0": c0, "score": score})
        rows.append(best[1])
    top = max(row["score"] for row in rows)
    if top < _AM_MIN_SCORE:
        return None
    return max((row for row in rows if row["score"] >= top - _AM_TIE), key=lambda r: r["p"])


def auto_map_origin(grid, p, s):
    """(r0, c0, score) of the best origin for a known period and tile."""
    score, r0, c0, _ = _am_best_origin(_am_components(grid), p, s)
    return r0, c0, score


_AM_MASKS = {}


def _am_masks(n):
    """Glyph templates for an n x n tile: [(name, set of cells in the second colour)]."""
    if n in _AM_MASKS:
        return _AM_MASKS[n]
    cells = [(r, c) for r in range(n) for c in range(n)]
    out = []
    for t in range(1, max(2, n // 2)):
        for i in range(0, n - t + 1):
            if 0 < i and i + t < n and abs(2 * i + t - n) <= 1:
                out.append((f"hline_mid_t{t}@{i}", {(r, c) for r, c in cells if i <= r < i + t}))
                out.append((f"vline_mid_t{t}@{i}", {(r, c) for r, c in cells if i <= c < i + t}))
    for t in range(1, max(2, n // 2)):
        lo = (n - t) // 2
        out.append((f"plus_t{t}", {(r, c) for r, c in cells
                                   if lo <= r < lo + t or lo <= c < lo + t}))
        out.append((f"square_t{t}", {(r, c) for r, c in cells
                                     if lo <= r < lo + t and lo <= c < lo + t}))
    out.append(("x", {(r, c) for r, c in cells if r == c or r + c == n - 1}))
    lo, hi = (n - 1) // 2, n // 2

    def dist(v):
        return max(lo - v, v - hi, 0)

    for k in range(1, (n - 1) // 2):
        out.append((f"plus_small{k}", {(r, c) for r, c in cells
                                       if (dist(r) == 0 and dist(c) <= k)
                                       or (dist(c) == 0 and dist(r) <= k)}))
        out.append((f"x_small{k}", {(r, c) for r, c in cells
                                    if dist(r) == dist(c) and dist(r) <= k}))
    _AM_MASKS[n] = out
    return out


def _am_tile(grid, r, c, s):
    return [[grid[r + y][c + x] for x in range(s)] for y in range(s)]


def _am_majority(tile):
    counts = {}
    for row in tile:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _am_symbol(tile, s):
    """(symbol, kind) for one tile: kind is 'uniform', 'glyph' or 'majority'."""
    first = tile[0][0]
    if all(v == first for row in tile for v in row):
        return _AM_SYMBOLS[first], "uniform"
    for name, mask in _am_masks(s):
        inside = {tile[r][c] for r, c in mask}
        outside = {tile[r][c] for r in range(s) for c in range(s) if (r, c) not in mask}
        if len(inside) == 1 and len(outside) == 1 and inside != outside:
            for prefix, glyph in _AM_GLYPHS:
                if name.startswith(prefix):
                    return glyph, "glyph"
    return _AM_SYMBOLS[_am_majority(tile)], "majority"


def auto_map_rows(grid, p, s, r0, c0):
    """The map: one string per row of whole tiles, one character per tile."""
    out = []
    for r in range(r0 % p, _AM_SIZE - s + 1, p):
        out.append("".join(_am_symbol(_am_tile(grid, r, c, s), s)[0]
                           for c in range(c0 % p, _AM_SIZE - s + 1, p)))
    return out


def _am_fidelity(grid, p, s, r0, c0):
    """Share of foreground pixels (not background, not UI) the map explains."""
    flat = [v for row in grid for v in row]
    counts = {}
    for v in flat:
        counts[v] = counts.get(v, 0) + 1
    background = max(counts.items(), key=lambda kv: kv[1])[0]
    counted = explained = 0
    for r in range(r0, _AM_SIZE - s + 1, p):
        for c in range(c0, _AM_SIZE - s + 1, p):
            tile = _am_tile(grid, r, c, s)
            keep = [(y, x) for y in range(s) for x in range(s)
                    if tile[y][x] != background
                    and _AM_UI <= r + y < _AM_SIZE - _AM_UI
                    and _AM_UI <= c + x < _AM_SIZE - _AM_UI]
            if not keep:
                continue
            counted += len(keep)
            _, kind = _am_symbol(tile, s)
            if kind == "glyph":
                explained += len(keep)
            else:
                main = _am_majority(tile)
                explained += sum(1 for y, x in keep if tile[y][x] == main)
    return explained / counted if counted else 1.0


def _am_repetition(grid, p, s, r0, c0):
    """Share of non-uniform tiles outside the UI band whose pattern occurs twice."""
    patterns = {}
    for r in range(r0, _AM_SIZE - s + 1, p):
        for c in range(c0, _AM_SIZE - s + 1, p):
            if r < _AM_UI or c < _AM_UI or r + s > _AM_SIZE - _AM_UI or c + s > _AM_SIZE - _AM_UI:
                continue
            tile = tuple(tuple(row) for row in _am_tile(grid, r, c, s))
            if any(v != tile[0][0] for row in tile for v in row):
                patterns[tile] = patterns.get(tile, 0) + 1
    total = sum(patterns.values())
    return sum(n for n in patterns.values() if n > 1) / total if total else 1.0


def auto_map_decide(grid, lattice):
    """Whether a game with this board and lattice gets the map (decided once per game)."""
    p, s, r0, c0 = lattice["p"], lattice["s"], lattice["r0"], lattice["c0"]
    fidelity = _am_fidelity(grid, p, s, r0, c0)
    repetition = _am_repetition(grid, p, s, r0, c0)
    if fidelity < _AM_FID_FLOOR:
        return False
    return not (repetition < _AM_REP_MIN
                and (fidelity < _AM_FID_MIN or lattice["score"] < _AM_FIT_MIN))


class AutoMapState:
    """One game's auto-map decisions: the lattice per level and whether the game gets a map.

    The first level whose initial board shows a lattice decides the game (detection plus the
    gate). Later levels keep that period and tile and only search the origin on their initial
    board, keeping the previous origin when none fits (a level with a limited view). Within a
    level the origin follows the view: `scroll` is the content's total (rows, cols) shift since
    the level's initial board.
    """

    def __init__(self):
        self.decided = False
        self.enabled = False
        self.p = self.s = None
        self.origin = None  # the latest level's origin, carried to the next level
        self.levels = {}  # level -> (r0, c0) on its initial board, or None

    def lattice(self, level, initial_grid):
        """(p, s, r0, c0) on the level's initial board, or None when it gets no map."""
        if len(initial_grid) != _AM_SIZE or any(len(row) != _AM_SIZE for row in initial_grid):
            return None  # the measures and thresholds are for ARC's 64 x 64 boards
        if level in self.levels:
            got = self.levels[level]
            return None if got is None else (self.p, self.s) + got
        if not self.decided:
            found = auto_map_detect(initial_grid)
            if found is None:
                self.levels[level] = None  # try again on the next level
                return None
            self.decided = True
            self.enabled = auto_map_decide(initial_grid, found)
            self.p, self.s = found["p"], found["s"]
            origin = (found["r0"], found["c0"])
        elif not self.enabled:
            self.levels[level] = None
            return None
        else:
            r0, c0, score = auto_map_origin(initial_grid, self.p, self.s)
            origin = (r0, c0) if score >= _AM_MIN_SCORE or self.origin is None else self.origin
        if not self.enabled:
            self.levels[level] = None
            return None
        self.origin = origin
        self.levels[level] = origin
        return (self.p, self.s) + origin

    def rows(self, level, initial_grid, grid, scroll=(0, 0)):
        """The map of `grid` (a board of `level`), or None when the game gets no map."""
        got = self.lattice(level, initial_grid)
        if got is None:
            return None
        p, s, r0, c0 = got
        return auto_map_rows(grid, p, s, (r0 + scroll[0]) % p, (c0 + scroll[1]) % p)

    def describe(self, level, scroll=(0, 0)):
        """(p, s, r0, c0) of the map shown for `level`, after the scroll."""
        got = self.levels.get(level)
        if got is None:
            return None
        return (self.p, self.s, (got[0] + scroll[0]) % self.p, (got[1] + scroll[1]) % self.p)
