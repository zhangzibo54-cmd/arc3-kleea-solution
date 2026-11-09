"""Object-level frame diff core, shared by the sandbox runtime and the host.

This module must stay dependency-free (stdlib only, no imports): its source is
injected verbatim into the sandbox bootstrap template (see
python_tool_sandbox.py), which runs under ``python -I -S`` and cannot import
project modules. The host imports it normally for the auto frame-diff feature.
"""


def _node_topleft(node):
    points = node.get("boundary") or []
    if not points:
        return (0, 0)
    return (min(p[0] for p in points), min(p[1] for p in points))

def _hungarian(cost):
    # Minimum-cost assignment (Jonker-Volgonant style with potentials).
    # cost: n x m matrix with n <= m. Returns list row -> column.
    n = len(cost)
    m = len(cost[0]) if n else 0
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = row assigned to column j (1-based; 0 = none)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment

def _node_bbox(node):
    points = node.get("boundary") or []
    if not points:
        return (0, 0, 0, 0)
    rows = [p[0] for p in points]; cols = [p[1] for p in points]
    return (min(rows), min(cols), max(rows), max(cols))

def _node_center2(node):
    # bbox center in half-cell units (ints): (r0+r1, c0+c1)
    points = node.get("boundary") or []
    if not points:
        return (0, 0)
    rows = [p[0] for p in points]; cols = [p[1] for p in points]
    return (min(rows) + max(rows), min(cols) + max(cols))

def _entry_size(entry):
    # an entry's pixel count; before/after entries (resized, scaled) use the larger side
    if "pixels" in entry:
        return entry["pixels"] or 0
    return max(entry.get("before_pixels") or 0, entry.get("after_pixels") or 0)


def compute_frame_diff(grid_a, grid_b, nodes_a, nodes_b, max_group_match=None):
    # Object-level diff between two same-shaped grids with precomputed
    # segmentation nodes. Returns only what changed; unchanged objects are
    # omitted. When max_group_match is set, identical-hash groups larger than
    # it skip pairwise matching and report plain appeared/disappeared entries
    # (guards against O(n^3) assignment on e.g. hundreds of identical pixels).
    # Every list is ordered largest object first (ties keep detection order):
    # model code often prints only the first few entries, and a 1-pixel corner
    # listed ahead of the 30-pixel block that vanished with it hides the block.
    changed = set()
    for r, (row_a, row_b) in enumerate(zip(grid_a, grid_b)):
        if row_a == row_b:
            continue
        for c, (va, vb) in enumerate(zip(row_a, row_b)):
            if va != vb:
                changed.add((r, c))
    if not changed:
        return {"changed_cell_count": 0, "moved": [], "rotated": [],
                "appeared": [], "disappeared": [],
                "changed_color": [], "resized": [], "scaled": []}

    def touched_nodes(nodes):
        touched = []
        for node in nodes:
            points = node.get("boundary") or []
            if not points:
                continue
            r0 = min(p[0] for p in points); r1 = max(p[0] for p in points)
            c0 = min(p[1] for p in points); c1 = max(p[1] for p in points)
            # dilate by 1 so objects that shrank/grew still register as
            # touched by the cells they vacated/claimed
            if any(r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1 for (r, c) in changed):
                touched.append(node)
        return touched

    before_nodes = touched_nodes(nodes_a)
    after_nodes = touched_nodes(nodes_b)

    moved, appeared, disappeared = [], [], []
    rotated = []

    def _classify_match(node_a, node_b):
        rot_a = node_a.get("rotation")
        rot_b = node_b.get("rotation")
        symmetry = node_a.get("rotational_symmetry") or 1
        rotated_by = None
        if rot_a is not None and rot_b is not None:
            modulus = 180 if symmetry == 2 else 360
            delta = (rot_b - rot_a) % modulus
            if delta:
                rotated_by = delta
        pa, pb = _node_topleft(node_a), _node_topleft(node_b)
        base_entry = {"color": node_a.get("color"),
                      "pixels": node_a.get("pixels"),
                      "hash": node_a.get("hash")}
        if rotated_by is not None:
            ca, cb = _node_center2(node_a), _node_center2(node_b)
            in_place = (abs(ca[0] - cb[0]) <= 2 and abs(ca[1] - cb[1]) <= 2)
            base_entry["rotated_by"] = rotated_by
            if in_place:
                base_entry["at"] = list(pb)
                rotated.append(base_entry)
            else:
                base_entry["from"] = list(pa); base_entry["to"] = list(pb)
                moved.append(base_entry)
        elif pa != pb:
            base_entry["from"] = list(pa); base_entry["to"] = list(pb)
            moved.append(base_entry)
        # same hash, same position, same rotation: unchanged -> omit

    groups_before = {}
    for node in before_nodes:
        groups_before.setdefault(node.get("hash"), []).append(node)
    groups_after = {}
    for node in after_nodes:
        groups_after.setdefault(node.get("hash"), []).append(node)

    for key, group_a in groups_before.items():
        group_b = groups_after.pop(key, [])
        if (
            max_group_match is not None
            and max(len(group_a), len(group_b)) > max_group_match
        ):
            # too many identical objects to match affordably
            for node_a in group_a:
                disappeared.append({"color": node_a.get("color"),
                                    "pixels": node_a.get("pixels"),
                                    "hash": node_a.get("hash"),
                                    "shape_hash": node_a.get("shape_hash"),
                                    "at": list(_node_topleft(node_a)),
                                    "bbox": list(_node_bbox(node_a))})
            for node_b in group_b:
                appeared.append({"color": node_b.get("color"),
                                 "pixels": node_b.get("pixels"),
                                 "hash": node_b.get("hash"),
                                 "shape_hash": node_b.get("shape_hash"),
                                 "at": list(_node_topleft(node_b)),
                                 "bbox": list(_node_bbox(node_b))})
            continue
        if not group_b:
            for node_a in group_a:
                disappeared.append({"color": node_a.get("color"),
                                    "pixels": node_a.get("pixels"),
                                    "hash": node_a.get("hash"),
                                    "shape_hash": node_a.get("shape_hash"),
                                    "at": list(_node_topleft(node_a)),
                                    "bbox": list(_node_bbox(node_a))})
            continue
        # optimal assignment over bbox-center distances; rows = smaller side
        transposed = len(group_a) > len(group_b)
        rows, cols = (group_b, group_a) if transposed else (group_a, group_b)
        cost = []
        for row_node in rows:
            cr = _node_center2(row_node)
            cost.append([
                abs(cr[0] - _node_center2(col_node)[0])
                + abs(cr[1] - _node_center2(col_node)[1])
                for col_node in cols
            ])
        assignment = _hungarian(cost)
        matched_cols = set()
        for row_index, col_index in enumerate(assignment):
            if col_index < 0:
                continue
            matched_cols.add(col_index)
            if transposed:
                _classify_match(cols[col_index], rows[row_index])
            else:
                _classify_match(rows[row_index], cols[col_index])
        if transposed:
            for col_index, node_a in enumerate(cols):
                if col_index not in matched_cols:
                    disappeared.append({"color": node_a.get("color"),
                                        "pixels": node_a.get("pixels"),
                                        "hash": node_a.get("hash"),
                                    "shape_hash": node_a.get("shape_hash"),
                                        "at": list(_node_topleft(node_a)),
                                        "bbox": list(_node_bbox(node_a))})
        else:
            for col_index, node_b in enumerate(cols):
                if col_index not in matched_cols:
                    appeared.append({"color": node_b.get("color"),
                                     "pixels": node_b.get("pixels"),
                                     "hash": node_b.get("hash"),
                             "shape_hash": node_b.get("shape_hash"),
                                     "at": list(_node_topleft(node_b)),
                                     "bbox": list(_node_bbox(node_b))})
            continue
        # transposed: every after-node (a row) got matched; nothing appeared
    for group_b in groups_after.values():
        for node_b in group_b:
            appeared.append({"color": node_b.get("color"),
                             "pixels": node_b.get("pixels"),
                             "hash": node_b.get("hash"),
                             "shape_hash": node_b.get("shape_hash"),
                             "at": list(_node_topleft(node_b)),
                             "bbox": list(_node_bbox(node_b))})
    # Phase A - changed_color: EXACT pairs only. Same position, byte-
    # identical shape (color-independent pose signature), different
    # color. This asserts "these exact cells changed color" and nothing
    # more; fuzzy transformation claims are deliberately not made.
    changed_color = []
    used_gone, used_new = set(), set()
    for gi, gone in enumerate(disappeared):
        for ni, new in enumerate(appeared):
            if ni in used_new:
                continue
            if (gone["at"] == new["at"]
                    and gone.get("shape_hash")
                    and gone.get("shape_hash") == new.get("shape_hash")
                    and gone["color"] != new["color"]):
                changed_color.append({"at": gone["at"],
                                      "pixels": gone["pixels"],
                                      "before_color": gone["color"],
                                      "after_color": new["color"],
                                      "before_hash": gone.get("hash"),
                                      "after_hash": new.get("hash")})
                used_gone.add(gi); used_new.add(ni)
                break
    disappeared = [entry for index, entry in enumerate(disappeared) if index not in used_gone]
    appeared = [entry for index, entry in enumerate(appeared) if index not in used_new]

    # Phase B - resized: same-color pairs whose bboxes substantially
    # overlap (overlap area / smaller bbox area >= 0.5); greedy
    # one-to-one by descending overlap then size similarity.
    resized = []

    def _bbox_area(b):
        return max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)

    def _overlap_ratio(b1, b2):
        inter_r = min(b1[2], b2[2]) - max(b1[0], b2[0]) + 1
        inter_c = min(b1[3], b2[3]) - max(b1[1], b2[1]) + 1
        if inter_r <= 0 or inter_c <= 0:
            return 0.0
        smaller = min(_bbox_area(b1), _bbox_area(b2))
        return (inter_r * inter_c) / smaller if smaller else 0.0

    candidates = []
    for gi, gone in enumerate(disappeared):
        for ni, new in enumerate(appeared):
            if gone["color"] != new["color"]:
                continue
            bg = gone.get("bbox") or (0, 0, -1, -1)
            bn = new.get("bbox") or (0, 0, -1, -1)
            ratio = _overlap_ratio(bg, bn)
            if ratio >= 0.5:
                area_g, area_n = _bbox_area(bg), _bbox_area(bn)
                similarity = (min(area_g, area_n) / max(area_g, area_n)) if max(area_g, area_n) else 0.0
                candidates.append((ratio, similarity, gi, ni))
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    used_gone, used_new = set(), set()
    for ratio, similarity, gi, ni in candidates:
        if gi in used_gone or ni in used_new:
            continue
        used_gone.add(gi); used_new.add(ni)
        gone, new = disappeared[gi], appeared[ni]
        resized.append({"at_before": gone["at"], "at_after": new["at"],
                        "color": gone["color"],
                        "before_pixels": gone["pixels"],
                        "after_pixels": new["pixels"],
                        "before_hash": gone.get("hash"),
                        "after_hash": new.get("hash")})
    disappeared = [entry for index, entry in enumerate(disappeared) if index not in used_gone]
    appeared = [entry for index, entry in enumerate(appeared) if index not in used_new]

    # Phase C - scaled: a disappeared / resized-before object and an appeared /
    # resized-after object of the same color are scaled when one shape, every
    # cell blown up to a k x k block (k = 2..4, optionally rotated), is exactly
    # the other one, and no cell of either keeps the color outside the other. A
    # scaled pair leaves the lists it came from; the other half of a resized
    # pair it broke up goes back to disappeared or appeared.
    befores = [("disappeared", i, e["color"], e["at"], e["pixels"])
               for i, e in enumerate(disappeared)]
    befores += [("resized", i, e["color"], e["at_before"], e["before_pixels"])
                for i, e in enumerate(resized)]
    afters = [("appeared", i, e["color"], e["at"], e["pixels"]) for i, e in enumerate(appeared)]
    afters += [("resized", i, e["color"], e["at_after"], e["after_pixels"])
               for i, e in enumerate(resized)]
    scaled, used = _scaled_pairs(grid_a, grid_b, before_nodes, after_nodes, befores, afters)
    if scaled:
        broken = sorted({i for side, i in used if side.startswith("resized")})
        for i in broken:
            entry = resized[i]
            if ("resized_before", i) not in used:
                disappeared.append(_resized_half(entry, "before", before_nodes))
            if ("resized_after", i) not in used:
                appeared.append(_resized_half(entry, "after", after_nodes))
        disappeared = [e for i, e in enumerate(disappeared) if ("disappeared", i) not in used]
        appeared = [e for i, e in enumerate(appeared) if ("appeared", i) not in used]
        resized = [e for i, e in enumerate(resized) if i not in broken]
    lists = {"moved": moved, "rotated": rotated, "appeared": appeared, "disappeared": disappeared,
             "changed_color": changed_color, "resized": resized, "scaled": scaled}
    return {"changed_cell_count": len(changed),
            **{k: sorted(v, key=_entry_size, reverse=True) for k, v in lists.items()}}


_SCALE_FACTORS = (2, 3, 4)
_SCALE_MIN_PIXELS = 2  # the smaller shape; a single cell grown to a block is too weak a claim
_SCALE_MAX_PIXELS = 1024  # the larger shape; backgrounds are not traced


def _resized_half(entry, half, nodes):
    # one side of a resized pair as an appeared / disappeared entry
    color, at, pixels = entry["color"], entry["at_" + half], entry[half + "_pixels"]
    node = next((n for n in nodes if n.get("color") == color and n.get("pixels") == pixels
                 and list(_node_topleft(n)) == list(at)), None)
    return {"color": color, "pixels": pixels, "hash": entry.get(half + "_hash"),
            "shape_hash": node.get("shape_hash") if node else None, "at": list(at),
            "bbox": list(_node_bbox(node)) if node else None}


def _component_cells(grid, node):
    # the node's cells, flood-filled from its first boundary point (a cell of it)
    points = node.get("boundary") or []
    if not points:
        return None
    r, c = points[0]
    value = grid[r][c]
    rows, cols = len(grid), len(grid[0])
    seen = {(r, c)}
    stack = [(r, c)]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen
                    and grid[nr][nc] == value):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _normalized(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _upscaled(norm, k):
    return frozenset((r * k + i, c * k + j) for r, c in norm
                     for i in range(k) for j in range(k))


def _rotations(norm):
    # (degrees clockwise, the shape rotated that much), 0 first
    current = norm
    for degrees in (0, 90, 180, 270):
        yield degrees, current
        height = max(r for r, _ in current)
        current = _normalized([(c, height - r) for r, c in current])


def _scale_match(before, after):
    # (factor, degrees) when after is before scaled by an integer factor or its
    # inverse (after rotating before by degrees clockwise), else None
    small, large = (before, after) if len(before) < len(after) else (after, before)
    height = max(r for r, _ in before) + 1
    width = max(c for _, c in before) + 1
    # a turned solid rectangle is just another rectangle: an a x b bar becoming a
    # k*b x k*a one says nothing about scaling (ar25's bars), so no turns for it
    turns = 1 if len(before) == height * width else 4
    for k in _SCALE_FACTORS:
        if len(large) != len(small) * k * k:
            continue
        for degrees, turned in list(_rotations(before))[:turns]:
            if len(before) < len(after):
                if _upscaled(turned, k) == after:
                    return k, degrees
            elif turned == _upscaled(after, k):
                return 1.0 / k, degrees
    return None


def _scaled_pairs(grid_a, grid_b, nodes_a, nodes_b, befores, afters):
    # befores / afters: (list, index, color, top-left, pixels) of changed objects
    # on each side. Returns the scaled entries and the {(list, index)} they used,
    # resized entries named "resized_before" / "resized_after" by side.
    def lookup(nodes):
        index = {}
        for node in nodes:
            index.setdefault((node.get("color"), tuple(_node_topleft(node)), node.get("pixels")), node)
        return index

    index_a, index_b = lookup(nodes_a), lookup(nodes_b)
    side_a, side_b = [], []
    for keys, index, side, suffix in ((befores, index_a, side_a, "_before"),
                                      (afters, index_b, side_b, "_after")):
        for source, i, color, at, pixels in keys:
            node = index.get((color, tuple(at), pixels))
            if node is None or not pixels or pixels > _SCALE_MAX_PIXELS:
                continue
            tag = (source + suffix if source == "resized" else source, i)
            side.append((color, node, pixels, tag))
    shapes, components = {}, {}

    def cells_of(grid, node):
        key = id(node)
        if key not in components:
            components[key] = _component_cells(grid, node)
        return components[key]

    def shape(grid, node):
        key = id(node)
        if key not in shapes:
            cells = cells_of(grid, node)
            shapes[key] = _normalized(cells) if cells else None
        return shapes[key]

    def left_behind(node_a, node_b):
        # Cells of either object that keep its color on the other board outside
        # the other object: the old object did not go away, a same-colored piece
        # was cut off or uncovered next to it (lf52's tile with a landing marker
        # drawn on it, dc22's floor a token walked off).
        cells_a, cells_b = cells_of(grid_a, node_a), cells_of(grid_b, node_b)
        r0, c0 = next(iter(cells_a))
        value = grid_a[r0][c0]
        return (any(grid_b[r][c] == value and (r, c) not in cells_b for r, c in cells_a)
                or any(grid_a[r][c] == value and (r, c) not in cells_a for r, c in cells_b))

    candidates = []
    for ai, (color_a, node_a, pixels_a, _) in enumerate(side_a):
        for bi, (color_b, node_b, pixels_b, _) in enumerate(side_b):
            if color_a != color_b or min(pixels_a, pixels_b) < _SCALE_MIN_PIXELS:
                continue
            ratio = max(pixels_a, pixels_b) / min(pixels_a, pixels_b)
            if not any(ratio == k * k for k in _SCALE_FACTORS):
                continue
            box_a, box_b = _node_bbox(node_a), _node_bbox(node_b)
            ca, cb = _node_center2(node_a), _node_center2(node_b)
            span = max(box_a[2] - box_a[0], box_a[3] - box_a[1],
                       box_b[2] - box_b[0], box_b[3] - box_b[1]) + 1
            distance = max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1]))  # half cells
            if distance > 2 * span:
                continue
            before, after = shape(grid_a, node_a), shape(grid_b, node_b)
            match = _scale_match(before, after) if before and after else None
            if match and not left_behind(node_a, node_b):
                candidates.append((distance, ai, bi, match))
    candidates.sort(key=lambda item: item[0])
    used_a, used_b, used, scaled = set(), set(), set(), []
    for distance, ai, bi, (factor, degrees) in candidates:
        if ai in used_a or bi in used_b:
            continue
        used_a.add(ai)
        used_b.add(bi)
        node_a, node_b = side_a[ai][1], side_b[bi][1]
        used.add(side_a[ai][3])
        used.add(side_b[bi][3])
        entry = {"color": side_a[ai][0], "factor": factor,
                 "at_before": list(_node_topleft(node_a)),
                 "at_after": list(_node_topleft(node_b)),
                 "before_pixels": side_a[ai][2], "after_pixels": side_b[bi][2],
                 "before_bbox": list(_node_bbox(node_a)),
                 "after_bbox": list(_node_bbox(node_b))}
        if degrees:
            entry["rotated_by"] = degrees
        scaled.append(entry)
    return scaled, used


# ARC3_FRAME_DIFF_V2: frame_diff without the background and with the parts a
# resized object gained and lost.
_BACKGROUND_SHARE = 0.30  # a colour covering this much of the board is background
_UNCHANGED_SHARE = 0.90  # a component this unchanged was only cut off or rejoined
_TILE_SIZES = range(3, 9)
_TILE_GAPS = range(0, 3)


def background_colors(grid, chars):
    """The colour letters covering >= 30% of `grid`, at most two.

    Summed over every component of a colour, so a background a wall or a flow
    cuts into pieces still counts. One pass over the board: callers compute it
    once (per level or per animation) and pass it on.
    """
    counts = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    need = _BACKGROUND_SHARE * len(grid) * (len(grid[0]) if grid else 0)
    ranked = sorted(counts.items(), key=lambda item: -item[1])[:2]
    return {chars[value] for value, count in ranked if count >= need and 0 <= value < len(chars)}


def _cells_bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return [min(rows), min(cols), max(rows), max(cols)]


def _pieces(cells):
    """The 4-connected pieces of `cells`, largest first, top-left first on ties."""
    left = set(cells)
    pieces = []
    while left:
        start = min(left)
        piece = {start}
        stack = [start]
        left.discard(start)
        while stack:
            r, c = stack.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in left:
                    left.discard(nb)
                    piece.add(nb)
                    stack.append(nb)
        pieces.append(piece)
    return sorted(pieces, key=lambda p: (-len(p), min(p)))


def _tile_fit(cells):
    """(k, period, r0, c0, set of (tile row, tile col)) for equal k x k tiles, or None.

    k is 3..8 and the gap between tiles 0..2. The tiles start at the cells'
    top-left and share one pattern that reaches both far edges of its tile; a
    pattern may be hollow. A single tile must be solid or a one-cell ring,
    since any small shape is "one tile" otherwise. The fewest tiles win.
    """
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    best = None
    for k in _TILE_SIZES:
        ring = {(i, j) for i in range(k) for j in range(k) if i in (0, k - 1) or j in (0, k - 1)}
        for gap in _TILE_GAPS:
            period = k + gap
            tiles = {}
            for r, c in cells:
                tr, ir = divmod(r - r0, period)
                tc, ic = divmod(c - c0, period)
                if ir >= k or ic >= k:
                    tiles = None  # a cell in the gap between tiles
                    break
                tiles.setdefault((tr, tc), set()).add((ir, ic))
            if not tiles:
                continue
            pattern = next(iter(tiles.values()))
            if any(cells_ != pattern for cells_ in tiles.values()):
                continue
            if (max(i for i, _ in pattern) != k - 1
                    or max(j for _, j in pattern) != k - 1):
                continue
            if len(tiles) == 1 and len(pattern) != k * k and pattern != ring:
                continue
            if best is None or len(tiles) < len(best[4]):
                best = (k, period, r0, c0, set(tiles))
            break  # a wider gap at this tile size only adds slack
    return best


def _tile_count(cells):
    """The number of equal square tiles `cells` form (see _tile_fit), or None."""
    fit = _tile_fit(cells)
    return len(fit[4]) if fit else None


def _tile_rectangles(tiles):
    """Cover a set of (row, col) tiles with rectangles, as (r0, c0, r1, c1) tile ranges.

    Each row's tiles are first joined into horizontal runs; a run then extends
    down while the next row has a run with the same column span.
    """
    runs = {}
    for row in sorted({r for r, _ in tiles}):
        cols = sorted(c for r, c in tiles if r == row)
        start = prev = cols[0]
        for col in cols[1:] + [None]:
            if col is not None and col == prev + 1:
                prev = col
                continue
            runs[(row, start, prev)] = True
            if col is not None:
                start = prev = col
    rects = []
    used = set()
    for (row, c0, c1) in sorted(runs):
        if (row, c0, c1) in used:
            continue
        r1 = row
        while (r1 + 1, c0, c1) in runs:
            r1 += 1
            used.add((r1, c0, c1))
        rects.append((row, c0, r1, c1))
    return rects


def _part(cells):
    """{"tiles" or "pixels": n, "bbox": [[r0, c0, r1, c1], ...]}, or None for no cells.

    "tiles" when the cells form n equal square tiles; then `bbox` has one box
    per rectangle of tiles (tiles that join side by side or one above another
    into a larger rectangle share a box). Otherwise "pixels" is their count and
    `bbox` has one box per 4-connected piece.
    """
    if not cells:
        return None
    fit = _tile_fit(cells)
    if fit:
        k, period, r0, c0, tiles = fit
        boxes = [[r0 + a * period, c0 + b * period, r0 + c * period + k - 1, c0 + d * period + k - 1]
                 for a, b, c, d in _tile_rectangles(tiles)]
        return {"tiles": len(tiles), "bbox": boxes}
    return {"pixels": len(cells), "bbox": [_cells_bbox(p) for p in _pieces(cells)]}


def _matching_node(nodes, color, at, pixels):
    for node in nodes:
        if (node.get("color") == color and node.get("pixels") == pixels
                and list(_node_topleft(node)) == list(at)):
            return node
    return None


def frame_diff_v2(grid_a, grid_b, nodes_a, nodes_b, background, hashes=True):
    """compute_frame_diff without the background, for ARC3_FRAME_DIFF_V2.

    Args:
        grid_a, grid_b: The boards before and after, same shape.
        nodes_a, nodes_b: Their segmentation nodes.
        background: Colour letters to leave out (see background_colors).
        hashes: Keep the entries' hash fields (frame_diff() does; the
            animation timeline does not).

    Returns:
        The compute_frame_diff dict with empty lists removed,
        every entry of a background colour dropped, appeared / disappeared
        components dropped when >= 90% of their cells kept their colour (a
        piece cut off from or rejoined to a region that was already there),
        and each resized entry given `added` / `removed`: the cells it gained
        and lost, as {"tiles" or "pixels", "bbox": [one bbox per 4-connected piece]}.
    """
    diff = compute_frame_diff(grid_a, grid_b, nodes_a, nodes_b)
    background = set(background or ())
    out = {"changed_cell_count": diff["changed_cell_count"]}

    def strip(entry):
        return dict(entry) if hashes else {k: v for k, v in entry.items() if "hash" not in k}

    for kind in ("moved", "rotated", "scaled"):
        kept = [strip(e) for e in diff.get(kind, []) if e.get("color") not in background]
        if kept:
            out[kind] = kept
    kept = [strip(e) for e in diff.get("changed_color", [])
            if e.get("before_color") not in background and e.get("after_color") not in background]
    if kept:
        out["changed_color"] = kept
    for kind, grid, nodes in (("appeared", grid_b, nodes_b), ("disappeared", grid_a, nodes_a)):
        kept = []
        for entry in diff.get(kind, []):
            if entry.get("color") in background:
                continue
            node = _matching_node(nodes, entry["color"], entry["at"], entry["pixels"])
            cells = _component_cells(grid, node) if node else None
            if cells and sum(1 for r, c in cells if grid_a[r][c] == grid_b[r][c]) >= _UNCHANGED_SHARE * len(cells):
                continue
            kept.append(strip(entry))
        if kept:
            out[kind] = kept
    kept = []
    for entry in diff.get("resized", []):
        if entry.get("color") in background:
            continue
        row = strip(entry)
        node_a = _matching_node(nodes_a, entry["color"], entry["at_before"], entry["before_pixels"])
        node_b = _matching_node(nodes_b, entry["color"], entry["at_after"], entry["after_pixels"])
        cells_a = _component_cells(grid_a, node_a) if node_a else None
        cells_b = _component_cells(grid_b, node_b) if node_b else None
        if cells_a is not None and cells_b is not None:
            for name, cells in (("added", cells_b - cells_a), ("removed", cells_a - cells_b)):
                part = _part(cells)
                if part:
                    row[name] = part
        kept.append(row)
    if kept:
        out["resized"] = kept
    return out


def detect_view_shift(grid_a, grid_b, margin=4, max_shift=32, min_overlap=600,
                      min_agree=0.9, min_gain=0.1):
    # Whole-view translation (a camera scroll) between two frames, or None.
    # Tries every shift along one axis, rows or columns, up to max_shift.
    # For a shift (dr, dc) it compares grid_b[r][c] with grid_a[r - dr][c - dc]
    # over the cells where both lie inside the interior (margin cells at the
    # border, where HUD bars sit, are ignored); its agreement is the share of
    # equal cells in that overlap. A shift is reported when its overlap has at
    # least min_overlap cells, its agreement is at least min_agree, and it
    # beats the agreement of no shift by at least min_gain. (dr, dc) is how the
    # board content moved on screen: dr > 0 down, dc > 0 right.
    rows = len(grid_b)
    cols = len(grid_b[0]) if rows else 0
    if rows != len(grid_a) or not rows or cols != len(grid_a[0]):
        return None

    def agreement(dr, dc):
        return _shift_agreement(grid_a, grid_b, dr, dc, margin)

    base, _ = agreement(0, 0)
    best = None
    for axis in (0, 1):
        for step in range(-max_shift, max_shift + 1):
            if not step:
                continue
            dr, dc = (step, 0) if axis == 0 else (0, step)
            score, total = agreement(dr, dc)
            if total < min_overlap:
                continue
            if best is None or score > best[0]:
                best = (score, dr, dc)
    if best is None or best[0] < min_agree or best[0] - base < min_gain:
        return None
    return (best[1], best[2])


def scroll_step(grid_a, grid_b, total, margin=4, min_overlap=600, min_agree=0.9, min_gain=0.1):
    """The part of an action's known scroll that one animation step performs.

    Args:
        grid_a, grid_b: Two consecutive animation frames, same shape.
        total: The action's whole view shift (dr, dc), as the solver measured it.
        margin, min_overlap, min_agree, min_gain: As in detect_view_shift.

    Returns:
        (dr, dc, unexplained): the step's shift, tried only along `total`'s
        direction from one cell up to its full length, and the number of
        overlapping interior cells that still differ after it; None when no
        such shift explains the step (it did not scroll, or did more than scroll).
    """
    dr_total, dc_total = total
    sign = lambda v: (v > 0) - (v < 0)
    candidates = [(k * sign(dr_total), 0) for k in range(1, abs(dr_total) + 1)]
    candidates += [(0, k * sign(dc_total)) for k in range(1, abs(dc_total) + 1)]
    if not candidates:
        return None
    base, _ = _shift_agreement(grid_a, grid_b, 0, 0, margin)
    best = None
    for dr, dc in candidates:
        score, compared = _shift_agreement(grid_a, grid_b, dr, dc, margin)
        if compared >= min_overlap and (best is None or score > best[0]):
            best = (score, dr, dc, compared)
    if best is None or best[0] < min_agree or best[0] - base < min_gain:
        return None
    score, dr, dc, compared = best
    return dr, dc, round(compared * (1 - score))


def _shift_agreement(grid_a, grid_b, dr, dc, margin):
    # Share of equal interior cells when grid_a is moved by (dr, dc) onto
    # grid_b, and the number of overlapping cells compared.
    rows = len(grid_b)
    cols = len(grid_b[0]) if rows else 0
    lo_r, hi_r, lo_c, hi_c = margin, rows - margin, margin, cols - margin
    same = total = 0
    for r in range(max(lo_r, lo_r + dr), min(hi_r, hi_r + dr)):
        row_a = grid_a[r - dr]
        row_b = grid_b[r]
        for c in range(max(lo_c, lo_c + dc), min(hi_c, hi_c + dc)):
            total += 1
            if row_a[c - dc] == row_b[c]:
                same += 1
    return (same / total if total else 0.0), total


_Grid = list[list[int]] | tuple[tuple[int, ...], ...]


def chained_view_shift(
    chain: list[_Grid] | tuple[_Grid, ...],
    step_shift: int = 10,
    margin: int = 4,
    min_overlap: int = 600,
    min_agree: float = 0.9,
    min_gain: float = 0.1,
) -> tuple[int, int] | None:
    """Sum camera shifts only when every adjacent frame pair is explained.

    Args:
        chain: Pre-action grid, intermediate animation grids, and final grid.
        step_shift: Maximum row or column displacement per adjacent pair.
        margin: Border width excluded from comparisons to avoid HUD changes.
        min_overlap: Minimum compared cells for each adjacent pair.
        min_agree: Minimum matching fraction for a shift or stationary view.
        min_gain: Required gain over no shift when accepting a displacement.

    Returns:
        Total content displacement (rows, columns), including (0, 0) for a
        verified stationary or round-trip chain. None if there is no
        intermediate frame, dimensions differ, or any pair is unexplained.
        The endpoints need not overlap: every link supplies its own evidence.
    """
    if len(chain) < 3:
        return None
    rows = len(chain[0])
    cols = len(chain[0][0]) if rows else 0
    if not rows or not cols or any(
        len(grid) != rows or any(len(row) != cols for row in grid)
        for grid in chain
    ):
        return None
    dr = dc = 0
    for grid_a, grid_b in zip(chain, chain[1:]):
        step = _animation_pair_shift(
            grid_a, grid_b, step_shift, margin, min_overlap, min_agree, min_gain,
        )
        if step is None:
            return None
        dr += step[0]
        dc += step[1]
    return (dr, dc)


def _animation_pair_shift(
    grid_a: _Grid,
    grid_b: _Grid,
    step_shift: int,
    margin: int,
    min_overlap: int,
    min_agree: float,
    min_gain: float,
) -> tuple[int, int] | None:
    """Validate a displacement or a stationary view with local changes."""
    if grid_a != grid_b:
        shift = detect_view_shift(
            grid_a, grid_b, margin=margin, max_shift=step_shift,
            min_overlap=min_overlap, min_agree=min_agree, min_gain=min_gain,
        )
        if shift is not None:
            return shift
    score, total = _shift_agreement(grid_a, grid_b, 0, 0, margin)
    if total >= min_overlap and score >= min_agree:
        return (0, 0)
    return None


def screen_fixed_cells(grid_a, grid_b, dr, dc, margin=4, min_share=0.9, min_cells=8):
    # HUD cells: whole edge rows or columns (within margin of the border)
    # that stayed put on screen while the board scrolled by (dr, dc). A line
    # counts when, among its cells whose scrolled-in source lies on screen, at
    # least min_share kept their value although that source differs. Judging
    # whole lines keeps walls that merely look alike after the scroll, and the
    # newly revealed strip (no source on screen), on the board.
    rows = len(grid_b)
    cols = len(grid_b[0]) if rows else 0

    def line_fixed(cells):
        candidates = fixed = 0
        for r, c in cells:
            sr, sc = r - dr, c - dc
            if not (0 <= sr < rows and 0 <= sc < cols):
                continue
            candidates += 1
            if grid_a[r][c] == grid_b[r][c] and grid_a[sr][sc] != grid_b[r][c]:
                fixed += 1
        return candidates >= min_cells and fixed >= min_share * candidates

    hud = set()
    edge_rows = [r for r in range(rows) if r < margin or r >= rows - margin]
    edge_cols = [c for c in range(cols) if c < margin or c >= cols - margin]
    for r in edge_rows:
        line = [(r, c) for c in range(cols)]
        if line_fixed(line):
            hud.update(line)
    for c in edge_cols:
        line = [(r, c) for r in range(rows)]
        if line_fixed(line):
            hud.update(line)
    return hud


def animation_revealed_cells(chain, end_shift, margin=4, step_shift=10,
                             min_overlap=600, min_agree=0.9, min_gain=0.1):
    # {(row, col): value} of the cells an action's animation showed outside its
    # end frame's interior (beyond its edges, or in its border band, where a
    # HUD bar may hide the board), in the end frame's screen coordinates. chain is the pre-action grid, the
    # intermediate grids and the end grid; end_shift is the action's view
    # shift ((0, 0) if none). Each frame's position comes from its own link
    # shifts; later frames overwrite earlier ones, so an object the camera
    # followed leaves no trail. Cells within margin of a frame's border (HUD
    # bars) are skipped. Empty when a link is unexplained or the links do not
    # add up to end_shift.
    if len(chain) < 3:
        return {}
    rows = len(chain[0])
    cols = len(chain[0][0]) if rows else 0
    if not rows or any(len(g) != rows or len(g[0]) != cols for g in chain):
        return {}
    offsets = [(0, 0)]
    for grid_a, grid_b in zip(chain, chain[1:]):
        step = _animation_pair_shift(grid_a, grid_b, step_shift, margin,
                                     min_overlap, min_agree, min_gain)
        if step is None:
            return {}
        offsets.append((offsets[-1][0] + step[0], offsets[-1][1] + step[1]))
    total = offsets[-1]
    if total != tuple(end_shift or (0, 0)) or not any(total):
        return {}
    cells = {}
    for grid, (sr, sc) in zip(chain[1:-1], offsets[1:-1]):
        # content at (r, c) here sits at (r + dr, c + dc) on the end screen
        dr, dc = total[0] - sr, total[1] - sc
        for r in range(margin, rows - margin):
            er = r + dr
            for c in range(margin, cols - margin):
                ec = c + dc
                if not (margin <= er < rows - margin and margin <= ec < cols - margin):
                    cells[(er, ec)] = grid[r][c]
    return cells


def level_world_map(entries, chars, cache=None, revealed=None):
    # {(row, col): color char} of every cell seen since the level's last
    # restart (RESET or a new level's start entry), in its starting-view coordinates
    # (row = screen row - view_offset[0], col = screen col - view_offset[1]);
    # later frames overwrite earlier ones. entries is the game history as
    # (action, grid, result) with the solver's view_shift / view_offset /
    # view_offset_unknown / view_shift_long in result. Returns None while the
    # level has not scrolled. Frames from the first unknown offset on are
    # skipped; HUD cells (screen_fixed_cells of any scroll) are left out.
    # Pass the same dict as cache on every call to add only the new entries.
    # revealed(index), if given, returns animation_revealed_cells for a
    # scrolling entry: what its animation showed beyond its end frame's view,
    # written before that frame (ARC3_WORLD_MAP_ANIM).
    cache = {} if cache is None else cache
    start = 0
    for index, (action, _grid, result) in enumerate(entries):
        # a winning entry shows the old level; the next level starts at the
        # level-start entry after it (or at the winning action's entry in
        # histories without one)
        if action == "RESET" or result.get("level_start") or (
            result.get("level_completed") and not result.get("winning_board")
        ):
            start = index
    if cache.get("start") != start or cache.get("done", 0) > len(entries):
        cache.clear()
        cache.update(start=start, done=start, hud=set(), world={}, stopped=False,
                     scrolled=False)
    hud = set(cache["hud"])
    for index in range(max(start + 1, cache["done"]), len(entries)):
        _a, before, _r = entries[index - 1]
        _b, after, result = entries[index]
        shift = result.get("view_shift")
        if shift:
            cache["scrolled"] = True
            if not result.get("view_shift_long"):
                hud |= screen_fixed_cells(before, after, shift[0], shift[1])
    if hud != cache["hud"]:
        # a newly found HUD cell may already sit in the map: start over
        cache.update(hud=hud, done=start, world={}, stopped=False)
    world = cache["world"]
    for index in range(cache["done"], len(entries)):
        _action, grid, result = entries[index]
        if index > start and result.get("view_offset_unknown"):
            cache["stopped"] = True
        if cache["stopped"]:
            continue
        dr, dc = (result.get("view_offset") or (0, 0)) if index > start else (0, 0)
        if revealed is not None and index > start and result.get("view_shift"):
            extra = cache.setdefault("revealed", {})
            if index not in extra:
                try:
                    extra[index] = revealed(index) or {}
                except Exception:
                    extra[index] = {}  # no frames: the end frame alone, as before
            for (r, c), value in extra[index].items():
                world[(r - dr, c - dc)] = chars[value] if value < len(chars) else str(value)
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                if (r, c) not in hud:
                    world[(r - dr, c - dc)] = chars[value] if value < len(chars) else str(value)
    cache["done"] = len(entries)
    return dict(world) if cache["scrolled"] else None


def align_to_view(grid_a, grid_b, dr, dc, margin=4):
    # grid_a moved by (dr, dc) into grid_b's view. Interior cells take grid_a's
    # state where grid_a saw them and grid_b's state where only grid_b does (the
    # newly revealed strip), so only the common part can differ. A margin cell
    # may be a HUD bar (fixed on screen) or scrolled board: it keeps grid_a's
    # unshifted state unless that differs from grid_b and either the shifted
    # state explains grid_b or the cell was scrolled in from beyond the edge, so
    # a HUD change still shows (unless it sits in the revealed edge) and a
    # scrolled edge does not.
    rows = len(grid_b)
    cols = len(grid_b[0]) if rows else 0
    aligned = [list(row) for row in grid_a]
    for r in range(rows):
        sr = r - dr
        for c in range(cols):
            sc = c - dc
            inside = 0 <= sr < rows and 0 <= sc < cols
            if margin <= r < rows - margin and margin <= c < cols - margin:
                if margin <= sr < rows - margin and margin <= sc < cols - margin:
                    aligned[r][c] = grid_a[sr][sc]
                else:
                    aligned[r][c] = grid_b[r][c]
            elif grid_a[r][c] != grid_b[r][c]:
                if not inside:
                    # revealed at the edge: grid_a never saw it
                    aligned[r][c] = grid_b[r][c]
                elif grid_a[sr][sc] == grid_b[r][c]:
                    aligned[r][c] = grid_a[sr][sc]
    return aligned


def view_shift_phrase(dr, dc):
    # "board content moved down 24 rows" style wording for (dr, dc).
    parts = []
    if dr:
        parts.append(f"{'down' if dr > 0 else 'up'} {abs(dr)} rows")
    if dc:
        parts.append(f"{'right' if dc > 0 else 'left'} {abs(dc)} cols")
    return "board content moved " + " and ".join(parts)
