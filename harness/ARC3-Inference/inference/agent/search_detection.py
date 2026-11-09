"""Structural search hints for ARC3_SEARCH_PUSH2, without executing model code.

The index belongs to one game, outside chat history. Runtime function sources
replace its bindings after each call, including aliases and partial execution.
These are advisory heuristics, not a proof of a program's purpose or cost.
"""

from __future__ import annotations

import ast
from functools import lru_cache


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_LOOPS = (ast.For, ast.While, ast.comprehension)
_ENUMERATORS = {"product", "permutations", "combinations",
                "combinations_with_replacement"}
_RANDOM = {"random", "randint", "randrange", "choice", "choices", "sample",
           "shuffle", "uniform"}


def _name(node: ast.AST) -> str:
    """Return a dotted reference, or an empty string for a computed reference."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + "." + node.attr
    return ""


def _calls(node: ast.AST) -> set[str]:
    """Return syntactic calls; strings and comments never enter this set."""
    return {_name(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}


def _names(node: ast.AST) -> set[str]:
    """Return referenced identifiers."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


@lru_cache(maxsize=512)
def _parse(code: str) -> ast.Module:
    """Parse a bounded snippet, returning an empty module on invalid syntax."""
    try:
        if len(code) <= 100_000:
            return ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        pass
    return ast.Module(body=[], type_ignores=[])


def _scope_nodes(root: ast.AST) -> list[ast.AST]:
    """Walk one scope without attributing an inner function's loops to it."""
    result = []
    pending = list(ast.iter_child_nodes(root))
    while pending:
        node = pending.pop()
        if isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
            continue
        result.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return result


class SearchDetector:
    """A game's live function dependencies.

    Call detect before synchronizing the sources returned by the worker.
    Synchronization replaces bindings, so deletion and redefinition cannot
    leave an old search classification attached to a reused function name.
    """

    def __init__(self):
        self._functions: dict[str, ast.AST] = {}
        self._imports: dict[str, str] = {}

    def synchronize(self, sources: dict[str, str]) -> None:
        """Replace live function bindings with the worker's source snapshot.

        Args:
            sources: Global names mapped to the functions actually bound to
                them after execution, including aliases and private names.
        """
        self._functions = {
            name: tree.body[0]
            for name, source in sources.items()
            if (tree := _parse(source)).body
        }

    def detect(self, code: str) -> tuple[str, ...]:
        """Return structural reasons for a search hint in this snippet.

        Args:
            code: Python source about to be, or just, executed. Definitions
                being written are included, even before their first call.

        Returns:
            Sorted reason names, empty for no recognized planning structure.
            Sources are parsed only; no model code is run by the detector.
        """
        tree = _parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self._imports[alias.asname or alias.name] = alias.name
        functions = dict(self._functions)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[node.name] = node
        scopes = [tree]
        scopes.extend(n for n in ast.walk(tree) if isinstance(n, _FUNCTIONS))
        reached = set()
        pending = list(_calls(tree))
        while pending:
            name = pending.pop()
            if name in reached or name not in functions:
                continue
            reached.add(name)
            scope = functions[name]
            scopes.append(scope)
            scopes.extend(n for n in ast.walk(scope)
                          if n is not scope and isinstance(n, _FUNCTIONS))
            pending.extend(_calls(scope))
        reasons = set()
        for scope in scopes:
            reasons.update(self._classify(scope, functions))
        return tuple(sorted(reasons))

    def _classify(
        self,
        scope: ast.AST,
        functions: dict[str, ast.AST],
    ) -> set[str]:
        """Require branching/candidate evaluation in addition to iteration."""
        nodes = _scope_nodes(scope)
        loops = [n for n in nodes if isinstance(n, _LOOPS)]
        if not loops:
            return set()
        calls = [n for n in nodes if isinstance(n, ast.Call)]
        methods = {self._imports.get(_name(n.func), _name(n.func)).split(".")[-1]
                   for n in calls}
        condition = any(isinstance(n, (ast.If, ast.IfExp, ast.Compare))
                        for n in nodes)
        # Recursive branching can enumerate layouts without any action names.
        recursive = (bool(getattr(scope, "name", ""))
                     and scope.name in {_name(n.func) for n in calls})
        if recursive and condition and any(
            getattr(scope, "name", "") in _calls(loop) for loop in loops
        ):
            return {"recursive-branching"}
        # Ordinary pixel alignment also enumerates and scores. It should not
        # receive planning advice unless it has a frontier/recursive planner.
        frontier = bool(methods & {"heappop", "popleft"}) and bool(
            methods & {"heappush", "append"}
        )
        observed = _observation_names(nodes)
        custom = [c for c in calls if (
            _name(c.func).split(".")[-1] in _RANDOM
            or _name(c.func) in functions
            and not any(_names(arg) & observed for arg in c.args)
            and not _observes_or_acts(
                functions[_name(c.func)], functions, set()
            )
        )]
        # Candidate evaluation consumes a loop variable (or a value produced
        # inside that loop), and its result is used in a condition/selection.
        evaluation = False
        for loop in loops:
            loop_nodes = _scope_nodes(loop)
            if isinstance(loop, ast.comprehension):
                # The evaluated expression is a sibling of the generator in
                # Python's AST. Include it and an enclosing rank operation.
                for owner in nodes:
                    if loop in getattr(owner, "generators", []):
                        loop_nodes.extend(_scope_nodes(owner))
                loop_nodes.extend(c for c in calls if loop in ast.walk(c)
                                  and _name(c.func) in {"sorted", "min", "max"})
            produced = set()
            if isinstance(loop, (ast.For, ast.comprehension)):
                produced.update(_names(loop.target))
            for n in loop_nodes:
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    produced.add(n.id)
            candidates = [c for c in custom if c in loop_nodes and (
                _names(c) & produced or _name(c.func).split(".")[-1] in _RANDOM
            )]
            evaluation |= _uses_evaluated_candidate(loop_nodes, candidates)
        selection = condition or bool(methods & {"sorted", "sort", "min", "max"})
        reasons = set()
        path = "reverse" in methods or any(
            isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add)
            and isinstance(n.right, (ast.List, ast.Tuple)) for n in nodes
        )
        if frontier and (evaluation or path) and selection:
            reasons.add("frontier-expansion")
        beam = ("sorted" in methods
                and any(isinstance(n, ast.Slice) for n in nodes))
        if methods & _ENUMERATORS and selection:
            reasons.add("combinatorial-enumeration")
        if evaluation and selection:
            reasons.add("candidate-evaluation")
            if beam:
                reasons.add("beam-selection")
            reachable_methods = set(methods)
            for c in custom:
                if _name(c.func) in functions:
                    reachable_methods.update(
                        name.split(".")[-1]
                        for name in _calls(functions[_name(c.func)])
                    )
            if reachable_methods & _RANDOM:
                reasons.add("random-optimization")
        return reasons


def _compares_pixels(node: ast.AST) -> bool:
    """Recognize paired grid-cell comparisons used by alignment scorers."""
    rows = {
        t.id for n in ast.walk(node) if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Subscript)
        for t in n.targets if isinstance(t, ast.Name)
    }
    for part in ast.walk(node):
        if isinstance(part, ast.Compare):
            values = [part.left, *part.comparators]
            if sum(isinstance(v, ast.Subscript)
                   and (isinstance(v.value, ast.Subscript)
                        or isinstance(v.value, ast.Name) and v.value.id in rows)
                   for v in values) >= 2:
                return True
    return False


def _observes_or_acts(
    node: ast.AST, functions: dict[str, ast.AST], visited: set[str]
) -> bool:
    """Exclude observation extraction and real-action loops from scoring."""
    if _compares_pixels(node) or "action" in _calls(node) or any(
        isinstance(n, ast.Attribute) and n.attr in {"ascii", "segmentation"}
        for n in ast.walk(node)
    ):
        return True
    return any(
        _observes_or_acts(functions[name], functions, visited | {name})
        for name in _calls(node) - visited if name in functions
    )


def _observation_names(nodes: list[ast.AST]) -> set[str]:
    """Track direct frame reads within this scope, not hypothetical states."""
    names = {"current_frame", "previous_frame", "level_history", "history"}
    for _ in range(3):
        for node in nodes:
            if not isinstance(node, ast.Assign):
                continue
            if _names(node.value) & names or any(
                isinstance(n, ast.Attribute)
                and n.attr in {"frame", "ascii", "segmentation", "before_frame",
                               "after_frame"}
                for n in ast.walk(node.value)
            ):
                names.update(n for t in node.targets for n in _names(t))
    return names


def _uses_evaluated_candidate(
    nodes: list[ast.AST], candidates: list[ast.Call]
) -> bool:
    """Trace candidate helper outputs into filtering or ranking, not printing."""
    evaluated: set[str] = set()

    def depends(node: ast.AST) -> bool:
        return bool(_names(node) & evaluated) or any(
            c in ast.walk(node) for c in candidates
        )

    # A few propagation passes cover tuple unpacking, derived scores and
    # score/candidate arrays. This is bounded static analysis, not execution.
    for _ in range(4):
        for node in nodes:
            if isinstance(node, ast.Assign) and depends(node.value):
                evaluated.update(n for t in node.targets for n in _names(t))
            elif isinstance(node, (ast.For, ast.comprehension)) and depends(node.iter):
                evaluated.update(_names(node.target))
        for node in nodes:
            if isinstance(node, (ast.If, ast.IfExp)) and depends(node.test):
                return True
            if isinstance(node, ast.comprehension) and any(
                depends(test) for test in node.ifs
            ):
                return True
            if isinstance(node, ast.Call):
                method = _name(node.func).split(".")[-1]
                if method in {"sorted", "sort", "min", "max"} and depends(node):
                    return True
    return False
