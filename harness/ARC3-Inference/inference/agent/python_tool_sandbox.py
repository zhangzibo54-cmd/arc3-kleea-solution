"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
import shutil
from typing import Any, Callable

from inference.utils import auto_map as _auto_map_module
from inference.utils import frame_diff as _frame_diff_module
from inference.agent.runtime_state import state_delta_enabled
from inference.agent.skills import skill_texts
from inference.utils import segmentation as _segmentation
from inference.utils import retained_functions as _retained_functions
from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.utils.animation import frame_diff_v2_enabled


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import ast
    import builtins
    import copy
    import contextlib
    import ctypes
    import gc
    import io
    import json
    import os
    import re
    import signal
    import sys
    import time
    import threading
    import types
    import traceback
    import weakref

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""

    __SEGMENTATION_SOURCE__

    __FRAME_DIFF_SOURCE__

    __AUTO_MAP_SOURCE__

    __RETAINED_FUNCTIONS_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        # pure-Python sequence comparison, no I/O. Measured across 32
        # transcripts it was imported 9 times and USED none of them - the model
        # reaches for it on "compare two things" and then writes the comparison
        # by hand. Allowing it therefore buys robustness, not capability: a
        # stray import line no longer costs the whole snippet.
        "difflib",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        # Exception types. Only four were present, so anything narrower than
        # `except Exception` failed at the except clause and took the snippet
        # with it. These are inert classes and expose nothing; SystemExit and
        # KeyboardInterrupt stay out because the sandbox uses them for its own
        # timeout and shutdown handling.
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "EOFError",
        "IndexError",
        "KeyError",
        "LookupError",
        "NameError",
        "NotImplementedError",
        "OverflowError",
        "RecursionError",
        "StopIteration",
        "UnboundLocalError",
        "ZeroDivisionError",
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
        "__build_class__",
        "classmethod",
        "property",
        "staticmethod",
    }


    # ARC3_STATE_DELTA: the history items held, one JSON text each. A state
    # with history_from carries only the items from that position on; each
    # message to the host reports how many are held, so the host never sends
    # past a gap (a message this worker dropped).
    _history_texts = []


    def _send(payload):
        payload["history_len"] = len(_history_texts)
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _merge_history(state_payload):
        # idempotent, so a state is merged on arrival (even in a message that
        # is then dropped) and again when it is read
        if not isinstance(state_payload, dict) or "history_from" not in state_payload:
            return
        start = int(state_payload.get("history_from") or 0)
        del _history_texts[start:]
        _history_texts.extend(
            json.dumps(item, ensure_ascii=False) for item in state_payload.get("history") or []
        )
        state_payload["history_from"] = len(_history_texts)
        state_payload["history"] = []


    def _held_history(state_payload):
        if isinstance(state_payload, dict) and "history_from" in state_payload:
            _merge_history(state_payload)
            # parsed afresh each time, as the full history was before
            return json.loads("[" + ",".join(_history_texts) + "]")
        return state_payload.get("history") if isinstance(state_payload, dict) else None


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        message = json.loads(line)
        if isinstance(message, dict):
            _merge_history(message.get("state"))
        return message


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    # ARC3_SHOW_IMAGE: boards the snippet asked to see, sent back with its result
    _SHOW_IMAGE = {"enabled": False, "requests": []}
    _SHOW_IMAGE_MAX = 3

    # ARC3_ANIMATION: reads one history entry's animation frames from the host
    _ENTRY_FRAMES = {"fetch": None}
    # ARC3_FRAME_DIFF_V2: on/off, and each level's background colours (from its first board)
    _FRAME_DIFF_V2 = {"on": False, "background": {}}


    class HistoryEntryView:
        def show_image(self):
            # show this entry's board as an image in the next message (at most 3 per message)
            if not _SHOW_IMAGE["enabled"]:
                return "show_image() is not available in this game."
            frame = self.frame
            if frame is None:
                return "This entry has no board to show."
            if len(_SHOW_IMAGE["requests"]) >= _SHOW_IMAGE_MAX:
                return f"Not shown: at most {_SHOW_IMAGE_MAX} images per message."
            _SHOW_IMAGE["requests"].append(
                {"grid": [list(row) for row in frame._grid], "level": frame.level, "step": frame.step}
            )
            return f"The board of level {frame.level}, step {frame.step} will be shown in your next message."

        def __init__(self, *, action, frame, result=None, animation_timeline=None, index=None):
            self.action = action
            self.frame = frame
            self.result = dict(result) if isinstance(result, dict) else {}
            # the diff timeline over every state the action passed through, in
            # the form of last_animation_timeline; {} when it did not animate
            self._timeline = (
                dict(animation_timeline) if isinstance(animation_timeline, dict) else {}
            )
            self._index = index  # the entry's position in the host's history
            self._frames = None

        @property
        def animation_timeline(self):
            # with ARC3_FRAME_DIFF_V2 the host sends a marker and builds the
            # frame_diff timeline on first use
            if self._timeline.get("lazy"):
                fetch = _ENTRY_FRAMES.get("fetch_timeline")
                timeline = fetch(self._index) if fetch is not None and self._index is not None else {}
                self._timeline = dict(timeline or {})
            return self._timeline

        @animation_timeline.setter
        def animation_timeline(self, value):
            self._timeline = dict(value) if isinstance(value, dict) else {}

        @property
        def animation_frames(self):
            # every state of the action, the board before it first and this
            # entry's board last; read from the host on first use
            if self._frames is None:
                fetch = _ENTRY_FRAMES["fetch"]
                if not self._timeline or fetch is None or self._index is None:
                    self._frames = []
                else:
                    self._frames = fetch(self._index)
            return self._frames

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            frame = _frame_from_payload(entry.get("frame"))
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=frame,
                    result=entry.get("result"),
                    animation_timeline=entry.get("animation_timeline"),
                    index=entry.get("index"),
                )
            )
        return items


    def _transitions_from_history(history):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result=entry.result,
                )
            )
        return transitions


    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)


    def _set_limits(timeout_seconds, memory_limit_bytes=0, cpu_limit_seconds=None):
        if resource is None:
            return
        cpu_limit = max(1, int(cpu_limit_seconds or timeout_seconds)) + 1
        limits = [
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ]
        if memory_limit_bytes:
            limits.append((getattr(resource, "RLIMIT_AS", None), memory_limit_bytes))
        for limit, value in limits:
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    _CLICK_TEXT = re.compile(r"^MOUSE\s*\(\s*row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)\s*\)$", re.IGNORECASE)

    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                # a click as level_history shows it, so recorded actions replay as they are
                click = _CLICK_TEXT.match(action_name)
                if click:
                    normalized.append({"action": "MOUSE", "row": int(click.group(1)), "col": int(click.group(2))})
                    continue
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        view_shift_enabled = bool(initial.get("view_shift"))
        _FRAME_DIFF_V2["on"] = bool(initial.get("frame_diff_v2"))
        world_map_enabled = bool(initial.get("world_map"))
        live_world_map = {}
        replay_enabled = bool(initial.get("replay"))
        _SHOW_IMAGE["enabled"] = bool(initial.get("show_image"))
        world_map_cache = {}
        persistent_mode = bool(initial.get("persistent_process"))
        if persistent_mode:
            SAFE_MODULES.add("time")
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        if persistent_mode:
            # Long-lived workers yield the CPU to the serving and game threads.
            with contextlib.suppress(Exception):
                os.nice(10)
        memory_limit_bytes = int(initial.get("memory_limit_bytes", 0) or 0)
        idle_limit_bytes = int(initial.get("idle_limit_bytes", 0) or 0)
        _set_limits(
            timeout_seconds,
            memory_limit_bytes,
            int(initial.get("cpu_limit_seconds", 0) or 0),
        )
        # Released when model code runs out of memory, so the error reply can
        # still be built; re-acquired afterwards.
        _memory_reserve = [bytearray(8 * 1024 * 1024)] if persistent_mode else []

        action_results = []
        stdout = io.StringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "__name__": "__python_tool__",
            "__package__": None,
            "result": None,
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import
        # These bindings are deliberately available before the first snippet.
        # They are the stable, lightweight views that the harness itself uses;
        # models can inspect or replace them without re-importing implementation
        # modules in every call.
        from collections import Counter, defaultdict, deque

        runtime_globals.update(
            {
                "FrameView": FrameView,
                "HistoryEntryView": HistoryEntryView,
                "TransitionView": TransitionView,
                "Counter": Counter,
                "defaultdict": defaultdict,
                "deque": deque,
            }
        )
        if persistent_mode:
            runtime_globals["time"] = time
        _animation_views = []

        def _level_background(level, fallback):
            # a level's background colours, from its first board, computed once
            cache = _FRAME_DIFF_V2["background"]
            if level not in cache:
                first = fallback
                for entries in runtime_globals.get("level_history") or []:
                    if entries and entries[0].frame is not None and entries[0].frame.level == level:
                        first = entries[0].frame
                        break
                cache[level] = background_colors(first._grid, COLOR_CHARS)
            return cache[level]

        def frame_diff(before=None, after=None):
            # Object-level diff between two frames. Defaults: previous_frame -> current_frame.
            # Returns only what changed; unchanged objects are omitted.
            if before is None:
                before = runtime_globals.get("previous_frame")
            if after is None:
                after = runtime_globals.get("current_frame")
            if before is None or after is None:
                return {"error": "need two frames (no previous_frame available?)"}
            if before.shape != after.shape:
                return {"changed_cell_count": -1, "note": "frame shapes differ; full redraw",
                        "before_shape": before.shape, "after_shape": after.shape}
            shift = None
            if view_shift_enabled and before.level == after.level:
                shift = detect_view_shift(before._grid, after._grid)
            if _FRAME_DIFF_V2["on"]:
                background = _level_background(after.level, after)

                def differ(grid_a, grid_b, nodes_a, nodes_b):
                    return frame_diff_v2(grid_a, grid_b, nodes_a, nodes_b, background)
            else:
                differ = compute_frame_diff
            if shift is None:
                return differ(
                    before._grid,
                    after._grid,
                    before.segmentation.get("nodes", []),
                    after.segmentation.get("nodes", []),
                )
            # a camera scroll: match objects against the previous board moved
            # into the current view, so only real changes are reported
            aligned = align_to_view(before._grid, after._grid, shift[0], shift[1])
            result = differ(
                aligned,
                after._grid,
                segment_layer(aligned, COLOR_CHARS).get("nodes", []),
                after.segmentation.get("nodes", []),
            )
            result["view_shift"] = [shift[0], shift[1]]
            result["view_shift_note"] = (
                view_shift_phrase(shift[0], shift[1])
                + "; the earlier board was shifted to match before comparing"
            )
            return result

        # ARC3_SKILLS: mechanism notes read on demand; absent when skills are off
        _skill_texts = {str(k): str(v) for k, v in (initial.get("skills") or {}).items()}

        def skill(name=None):
            # The text of one skill; without a name, the list of skill names.
            if name in _skill_texts:
                return _skill_texts[name]
            names = ", ".join(sorted(_skill_texts))
            if name is None:
                return f"skills: {names}"
            return f"no skill named {name!r}; skills: {names}"

        if _skill_texts:
            runtime_globals["skill"] = skill
        _replay_fn = []  # ARC3_REPLAY: replay(), defined once action() exists
        schema_enabled = bool(initial.get("schema"))
        _schema_globals = {}  # ARC3_SCHEMA: names restored on every refresh
        _flat_history = []  # the entries in order, for finding an action's own entry
        # ARC3_FUNCTION_SOURCE: name -> source of the latest executed top-level def / class
        _fn_source_on = bool(initial.get("function_source"))
        _fn_sources = {}
        _snippet_defs = []  # (name, first line, source) of the current snippet's top-level defs

        class _FnSource(dict):
            # fn_source['name']: the source of the latest definition of function or class `name`

            def __repr__(self):
                return (f"fn_source: the source of {len(self)} definitions of yours: "
                        f"{', '.join(sorted(self)) or 'none yet'}. print(fn_source['name']) shows one.")

            __str__ = __repr__

        def _refresh_state(state_payload, live=True):
            # live: the host is serving this call's requests, so animation
            # frames can be fetched (not at worker start-up)
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(_held_history(state_payload))
            _flat_history[:] = history
            last_action_call_result = state_payload.get("last_action_call_result")
            action_result = (
                dict(last_action_call_result) if isinstance(last_action_call_result, dict) else {}
            )
            # action_echo is printed by action(), then removed from its return value.
            # Keep the exposed call result identical across subsequent inspections.
            action_result.pop("action_echo", None)
            transitions = _transitions_from_history(history)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            # the flat history stays local: the model reads it only through
            # level_history, history split by the level each board belongs
            # to, in order. A level's list starts with its first board and, once
            # won, ends with the winning board (its own entry, see the solver),
            # so level_history[n - 1][k] is the board after level n's k-th action
            by_level = {}
            for entry in history:
                if entry.frame is not None:
                    by_level.setdefault(entry.frame.level, []).append(entry)
            runtime_globals["level_history"] = [by_level[level] for level in sorted(by_level)]
            # the same entries, only those whose action ended an attempt: a
            # fatal entry sits mid-level once play goes on, so it has no
            # stable place in level_history the way a winning board does
            runtime_globals["level_deaths"] = [
                [entry for entry in by_level[level] if (entry.result or {}).get("game_over")]
                for level in sorted(by_level)
            ]
            if replay_enabled:
                # ARC3_REPLAY: each fatal entry carries its attempt, the actions from
                # the board the level (re)started on up to and including itself
                for level_entries in by_level.values():
                    start = 0
                    for k, entry in enumerate(level_entries):
                        if k and str(entry.action).strip().upper() == "RESET":
                            start = k
                        if (entry.result or {}).get("game_over"):
                            entry.attempt = [e.action for e in level_entries[start + 1:k + 1]]
                            entry._attempt_entries = level_entries[start:k + 1]
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["frame_diff"] = frame_diff
            if _fn_source_on:
                runtime_globals["fn_source"] = _FnSource(_fn_sources)  # restored if the model reassigns it
            if _skill_texts:
                runtime_globals["skill"] = skill  # restored like frame_diff if the model reassigns it
            if replay_enabled and _replay_fn:
                runtime_globals["replay"] = _replay_fn[0]  # restored if the model reassigns it
            runtime_globals.update(_schema_globals)  # restored if the model reassigns them
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["death_ledger"] = state_payload.get("death_ledger") or {"attempts": []}
            runtime_globals["last_action_call_result"] = action_result
            if world_map_enabled:
                # Rebuild from history incrementally, including the initial
                # view before the level has scrolled.
                framed = [entry for entry in history if entry.frame is not None]

                def _revealed(index):
                    # what a scrolling action's animation showed beyond its end
                    # frame's view; a long scroll leaves rows only it saw
                    frames = framed[index].animation_frames
                    return animation_revealed_cells(
                        [frame._grid for frame in frames],
                        framed[index].result.get("view_shift"),
                    )

                level_world_map(
                    [(entry.action, entry.frame._grid, entry.result) for entry in framed],
                    COLOR_CHARS,
                    world_map_cache,
                    revealed=_revealed if live else None,
                )
                # Keep model aliases alive across refreshes and levels.
                # A separate dict prevents model edits from corrupting the
                # incremental cache. RESET/level-start rebuilds it above.
                live_world_map.clear()
                live_world_map.update(world_map_cache.get("world", {}))
                runtime_globals["world_map"] = live_world_map

        # Both refusals derive from BaseException, not Exception, so a
        # defensive `except Exception:` around action(...) - which models write
        # routinely - cannot swallow them. A refusal means the action did not
        # run at all, so the snippet's assumed state is already wrong and every
        # line after it rests on a false premise. One run treated 1500
        # consecutive refusals as proof that every cell on the board was inert.
        # StaleStateActionError below is deliberately catchable: it reports a
        # belief that MAY be stale rather than a verdict on the action.
        class KnownNoOpActionError(BaseException):
            pass  # raised INSTEAD of executing an action already proven inert here

        class KnownDeathActionError(BaseException):
            pass  # raised INSTEAD of executing an action that ended a prior attempt

        # Raised INSTEAD of executing an action this snippet has already run
        # from this exact board. Buggy loop code can spend a whole life bar
        # getting nowhere - a cursor that wraps turns RIGHT into a short cycle,
        # a mistaken `while pos < j: LEFT` walks away from its target forever -
        # and the terminal latch only stops that once the game actually ends,
        # measured at 20 to 30 wasted actions. BaseException like the two above:
        # the shape that produces these loops is `try: action(...) except
        # Exception:` inside a while, which would otherwise swallow the verdict
        # and keep going. No override slot: the snippet is not reading the
        # refusal, its code is running, and writing a new snippet IS the
        # response - the map is rebuilt per execution.
        class RepeatedActionInStateError(BaseException):
            pass

        # Raised when a PREVIOUS action in this snippet changed nothing, and the
        # snippet then tries to act again. Deliberately a plain Exception, not
        # BaseException: unlike the two refusals above this is a warning about
        # the snippet's beliefs rather than a verdict on the action, and a model
        # that has re-read current_frame may legitimately want to proceed.
        class StaleStateActionError(Exception):
            pass

        runtime_globals["KnownNoOpActionError"] = KnownNoOpActionError
        runtime_globals["KnownDeathActionError"] = KnownDeathActionError
        runtime_globals["StaleStateActionError"] = StaleStateActionError
        runtime_globals["RepeatedActionInStateError"] = RepeatedActionInStateError

        # Raised INSTEAD of executing an action once ARC3_SNIPPET_NOOP_LIMIT
        # actions in a row in this snippet changed nothing. BaseException for
        # the same reason as RepeatedActionInStateError: the loops that run up
        # these streaks count their own stalls inside `try ... except Exception`.
        class NoOpRunActionError(BaseException):
            pass

        runtime_globals["NoOpRunActionError"] = NoOpRunActionError
        _noop_run_limit = int(initial.get("snippet_noop_limit") or 0)
        # displays of the executed no-ops in the current streak; reset per snippet
        _noop_run: list[str] = []
        # armed when an action changes nothing; scoped to THIS snippet because
        # the closure is rebuilt per execution, so no reset path can be missed
        _stale_state: list[str] = []

        # Raised when a snippet tries to act after the attempt or level ended.
        # Derives from BaseException, not Exception, so a defensive
        # `try: action(...) except Exception:` around the call - which models
        # write routinely - cannot swallow it. Continuing to execute a path
        # computed against a board that no longer exists is never correct.
        class TerminalStateActionError(BaseException):
            pass

        runtime_globals["TerminalStateActionError"] = TerminalStateActionError
        # latched when an action reports a terminal result; scoped to THIS
        # snippet because these closures are rebuilt for every execution, so
        # no reset path can be missed and the next snippet always starts clear
        _terminal_state = []

        _TERMINAL_DETAIL = {
            "game_over": (
                "the previous action reported game_over. The level has been reset to its "
                "start, so the board you planned against is gone and this action was not "
                "executed. The next turn begins from the new state."
            ),
            "level_completed": (
                "the previous action reported level_completed. A new level has been loaded, "
                "so the board you planned against is gone and this action was not executed. "
                "The next turn begins from the new state."
            ),
            "run_complete": (
                "the previous action reported run_complete. The run is finished and this "
                "action was not executed."
            ),
        }

        def action(actions):
            if _terminal_state:
                raise TerminalStateActionError(_TERMINAL_DETAIL[_terminal_state[0]])
            if _noop_run_limit and len(_noop_run) >= _noop_run_limit:
                raise NoOpRunActionError(
                    f"the last {len(_noop_run)} actions of this snippet "
                    f"({', '.join(_noop_run)}) changed nothing inside the board "
                    "area, and each of them still cost a game action. This action "
                    "was not executed and the snippet can run no more actions. "
                    "Re-read current_frame, work out why nothing moved, and act "
                    "in a new snippet."
                )
            normalized_actions = _normalize_actions(actions)
            _send({
                "type": "action",
                "actions": normalized_actions,
                # the solver decides precedence: a known death or a known no-op
                # is a sharper message than "your state may be stale", and it
                # carries a concrete override
                "stale_after": _stale_state[0] if _stale_state else None,
            })
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            # Cached animations describe the previous action. Invalidate the
            # existing views so aliases also refresh, even if a guard raises.
            for view in _animation_views:
                view.invalidate()
            _refresh_state(reply.get("state") or {})
            _echo = action_result.pop("action_echo", None)
            _progress = action_result.get("terminal_progress")
            if _progress:
                print(_progress["text"])
            if _echo:
                # into the snippet's own stdout, so a loop can see what its
                # call did without waiting for the next turn's opener; printed
                # before the guard refusals below raise, so a refused action
                # is reported as not executed even when the snippet catches
                # the exception and prints only its class name
                print(_echo)
            if action_result.get("stop_reason") in ("reset_not_first", "reset_at_level_start"):
                raise RuntimeError(str(action_result.get("stop_detail")))
            if action_result.get("stop_reason") == "stale_state":
                raise StaleStateActionError(
                    str(action_result.get("stop_detail") or "stale state")
                )
            if action_result.get("stop_reason") == "known_death":
                raise KnownDeathActionError(
                    str(action_result.get("stop_detail") or "known fatal action was not executed.")
                )
            if action_result.get("stop_reason") == "repeated_action_in_state":
                raise RepeatedActionInStateError(
                    str(action_result.get("stop_detail") or "")
                    or "this action was already run from this board in this snippet"
                )
            if action_result.get("stop_reason") == "known_noop":
                # nothing executed and no action was spent; halt the snippet so
                # the model re-plans instead of continuing on a false premise
                raise KnownNoOpActionError(
                    str(action_result.get("stop_detail") or "known no-op action was not executed.")
                )
            for _flag in ("run_complete", "done", "level_completed", "game_over"):
                if action_result.get(_flag):
                    # `done` means the run was won; report it as run_complete
                    _terminal_state.append(
                        "run_complete" if _flag == "done" else _flag
                    )
                    break
            gameplay_changed = action_result.get("gameplay_changed")
            if action_result.get("executed") and gameplay_changed is not None:
                # a batch reports every executed action; a single call only its own
                # (the compact result drops executed_count for single actions)
                _changed = action_result.get("gameplay_changed_per_action") or [gameplay_changed]
                _names = action_result.get("executed_actions") or []
                for _i, _flag in enumerate(_changed):
                    if _flag:
                        del _noop_run[:]
                    else:
                        _noop_run.append(str(
                            _names[_i] if _i < len(_names)
                            else action_result.get("action_display") or "?"
                        ))
            if gameplay_changed is True:
                # a real change means the snippet's beliefs are current again
                del _stale_state[:]
            if gameplay_changed is False and action_result.get("executed"):
                _executed = action_result.get("executed_actions")
                _name = (
                    _executed[-1]
                    if isinstance(_executed, list) and _executed
                    else action_result.get("action_display")
                    or action_result.get("action_name")
                    or "?"
                )
                # Arm rather than raise. The action executed and changed
                # nothing, which is often exactly what the snippet went looking
                # for - it printed the board before, and wants to print it
                # after. Killing it here throws away the second half of the
                # measurement AND the batch stop's own trace, which is computed
                # and then discarded. So let the snippet finish observing, and
                # stop it only if it tries to ACT on the stale belief.
                if not _stale_state:
                    _stale_state.append(str(_name))
            return action_result

        def _animation_fetch(kind, index=None):
            request = {"kind": kind}
            if index is not None:
                request["index"] = index  # kind 'entry_frames': that history entry
            _send({"type": "animation", "request": request})
            reply = _recv()
            # an action request a timeout abandoned in this snippet still gets
            # its reply, ahead of this one: the host answers in order
            while reply.get("type") in ("action_result", "action_error"):
                reply = _recv()
            if reply.get("type") == "animation_error":
                raise RuntimeError(str(reply.get("error", "animation failed")))
            if reply.get("type") != "animation_result":
                raise RuntimeError("Invalid animation response from sandbox host.")
            payload = reply.get("animation")
            if kind not in ("frames", "entry_frames"):
                return payload or {}
            return [
                FrameView(
                    ascii=item.get("ascii", ""),
                    step=item.get("step", 0),
                    level=item.get("level", 0),
                    shape=item.get("shape", (0, 0)),
                    grid=item.get("grid", ()),
                )
                for item in (payload or [])
            ]

        runtime_globals["action"] = action

        def _interior(frame):
            # the board without the 4-cell edge where timer / budget bars sit
            grid = frame._grid if frame is not None else ()
            return tuple(tuple(row[4:-4]) for row in list(grid)[4:-4])

        def replay(steps=-1, death=-1):
            # Replay the actions of a death's attempt on the current level, from a freshly
            # restarted board, up to `steps` actions before the fatal one (-1: the board
            # just before the death, -10: ten actions earlier; a positive n: the first n
            # actions). Every board is compared with the recorded one; the replay stops at
            # the first difference. Uses real actions.
            deaths = (runtime_globals.get("level_deaths") or [[]])[-1]
            if not deaths:
                return {"replayed": 0, "error": "no death on the current level yet"}
            try:
                fatal = deaths[death]
            except (IndexError, TypeError):
                return {"replayed": 0, "error": f"death={death!r}: the current level has {len(deaths)} death(s)"}
            entries, plan = fatal._attempt_entries, list(fatal.attempt)[:steps]
            if _interior(runtime_globals.get("current_frame")) != _interior(entries[0].frame):
                return {"replayed": 0, "error": (
                    "the current board is not the one this attempt started on: replay right after "
                    "the level restarts, before any other action")}
            for i, name in enumerate(plan):
                del _stale_state[:]  # the replay knows each action's board
                del _noop_run[:]  # and re-runs a recorded attempt, no-ops included
                try:
                    result = action([name])
                except (KnownNoOpActionError, RepeatedActionInStateError):
                    result = {}  # refused as inert here; it changed nothing when recorded either
                except KnownDeathActionError:
                    return {"replayed": i, "planned": len(plan), "stopped": f"action {i + 1} ({name}) is known to be fatal here"}
                if _interior(runtime_globals.get("current_frame")) != _interior(entries[i + 1].frame):
                    return {"replayed": i + 1, "planned": len(plan),
                            "stopped": f"the board after action {i + 1} ({name}) differs from the recorded one"}
                for flag in ("level_completed", "game_over", "run_complete", "done"):
                    if result.get(flag):
                        return {"replayed": i + 1, "planned": len(plan), "stopped": f"action {i + 1} reported {flag}"}
            return {"replayed": len(plan), "planned": len(plan), "stopped": None,
                    "next_recorded": list(fatal.attempt)[len(plan):]}

        if replay_enabled:
            _replay_fn.append(replay)
            runtime_globals["replay"] = replay

        # ARC3_SCHEMA: a world model the model writes as step(), checked against
        # every real action that runs with it
        _schema_tolerance = max(1, int(initial.get("schema_tolerance") or 9))
        _schema_report_diff = bool(initial.get("schema_report_diff"))
        # ARC3_SCHEMA_DIFF_AUDIT: every judged-right action's full frame_diff, sent to the
        # host's log only (the model never sees it)
        _schema_diff_audit = bool(initial.get("schema_diff_audit"))
        _diff_audit = []
        _last_model = []  # the WorldModel last passed to action / backtest / check_prediction
        _constructed = []  # WorldModels constructed in THIS snippet; cleared per run
        _backtested = []  # models backtest() ran on in THIS snippet; cleared per run
        _backtest_note = []  # the latest wrong transition of THIS snippet's backtests
        _schema_crop_chars = max(300, int(initial.get("schema_crop_chars") or 1500))
        _schema_hud = 4  # the edge where timer / budget bars sit, never compared
        _letter_value = {ch: i for i, ch in enumerate(COLOR_CHARS)}
        # ARC3_SCHEMA_PARSE: WorldModel(parse, step, is_win_condition); predictions are
        # dicts compared with parse of the real board, never boards
        _schema_parse = bool(initial.get("schema_parse"))
        _schema_parse_map = bool(initial.get("schema_parse_map"))  # ARC3_SCHEMA_PARSE_MAP
        _ctor = "WorldModel(parse, step, ...)" if _schema_parse else "WorldModel(step, ...)"
        _model_fields = (("parse", "step", "is_win_condition") if _schema_parse
                         else ("step", "key_state", "is_win_condition"))

        class WorldModel:
            # The model's mechanism: step predicts the next board; key_state(frame)
            # returns the state of the objects the mechanism depends on (when given,
            # predictions are judged on it instead of the whole board);
            # is_win_condition(frame) is the rule for completing the level.
            # Under ARC3_SCHEMA_PARSE it is WorldModel(parse, step, is_win_condition):
            # parse(frame) returns a dict, step(state, action, row, col) the next dict
            # and is_win_condition(state) the rule, all on parsed states.
            # A function bound to a global name is looked up by that name on every
            # use, so redefining `step` (or key_state ...) updates the model too.
            def __init__(self, *args, **kwargs):
                usage = f"WorldModel({', '.join(_model_fields)})"
                if len(args) > len(_model_fields):
                    raise TypeError(f"{usage} takes at most {len(_model_fields)} functions")
                fns = dict(zip(_model_fields, args))
                for _name, _fn in kwargs.items():
                    if _name not in _model_fields or _name in fns:
                        raise TypeError(f"{usage}: unexpected or repeated argument {_name!r}")
                    fns[_name] = _fn
                required = _model_fields[:2] if _schema_parse else _model_fields[:1]
                for _name in _model_fields:
                    _fn = fns.get(_name)
                    if _name in required and not callable(_fn):
                        raise TypeError(f"{usage}: {_name} must be a function")
                    if _fn is not None and not callable(_fn):
                        raise TypeError(f"WorldModel: {_name} must be a function or None")
                self._fns = {name: fns.get(name)
                             for name in ("parse", "step", "key_state", "is_win_condition")}
                self._names = {
                    field: fn.__name__ for field, fn in self._fns.items()
                    if isinstance(fn, types.FunctionType) and fn.__name__ != "<lambda>"
                    and runtime_globals.get(fn.__name__) is fn}
                _constructed.append(self)

            def _current(self, field):
                name = self._names.get(field)
                value = runtime_globals.get(name) if name else None
                return value if isinstance(value, types.FunctionType) else self._fns[field]

            parse = property(lambda self: self._current("parse"))
            step = property(lambda self: self._current("step"))
            key_state = property(lambda self: self._current("key_state"))
            is_win_condition = property(lambda self: self._current("is_win_condition"))

            def __repr__(self):
                positional = 2 if _schema_parse else 1  # the required functions
                parts = []
                for i, _name in enumerate(_model_fields):
                    _fn = getattr(self, _name)
                    if _fn is None:
                        continue
                    label = getattr(_fn, "__name__", _name)
                    parts.append(label if i < positional else f"{_name}={label}")
                return f"WorldModel({', '.join(parts)})"

        def board(frame):
            # an editable copy of a board: a list of rows, each a list of letters
            if isinstance(frame, HistoryEntryView):
                frame = frame.frame
            return [list(row) for row in str(frame.ascii).split("\n")]

        def _rows_of(value):
            # a predicted board as a tuple of row strings
            if isinstance(value, FrameView):
                return tuple(value.ascii.split("\n"))
            if isinstance(value, str):
                return tuple(value.split("\n"))
            if isinstance(value, (list, tuple)):
                return tuple(row if isinstance(row, str) else "".join(str(ch) for ch in row)
                             for row in value)
            raise TypeError(f"step returned {type(value).__name__}; expected a board "
                            "(a list of row strings or of letter lists) or (board, info)")

        def _frame_of_rows(rows, like):
            grid = []
            for r, row in enumerate(rows):
                try:
                    grid.append([_letter_value[ch] for ch in row])
                except KeyError as exc:
                    raise ValueError(f"predicted row {r} has {exc.args[0]!r}, not a board letter") from None
            return FrameView(ascii="\n".join(rows), step=getattr(like, "step", 0),
                             level=getattr(like, "level", 0),
                             shape=[len(grid), len(grid[0]) if grid else 0], grid=grid)

        def _parse_move(move):
            # (name, row, col) from 'LEFT', 'MOUSE(row=r, col=c)' or an action dict
            item = _normalize_actions([move])[0]
            return item["action"], item.get("row"), item.get("col")

        def _is_undo(name):
            return str(name).strip().upper() in ("UNDO", "ACTION7")

        def _move_text(name, row, col):
            return f"MOUSE(row={row}, col={col})" if row is not None else name

        def _board_diff(pred_rows, real_rows):
            # differing interior cells grouped 8-connected, in board coordinates
            h = _schema_hud
            if len(pred_rows) != len(real_rows) or any(
                    len(a) != len(b) for a, b in zip(pred_rows, real_rows)):
                return {"same": False, "cells": -1, "groups": [],
                        "note": f"predicted shape {len(pred_rows)}x{len(pred_rows[0]) if pred_rows else 0} "
                                f"differs from the real {len(real_rows)}x{len(real_rows[0]) if real_rows else 0}"}
            cells = {(r, c) for r in range(h, len(real_rows) - h)
                     for c in range(h, len(real_rows[r]) - h)
                     if pred_rows[r][c] != real_rows[r][c]}
            groups, seen = [], set()
            for start in sorted(cells):
                if start in seen:
                    continue
                stack, members = [start], []
                seen.add(start)
                while stack:
                    r, c = stack.pop()
                    members.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nb = (r + dr, c + dc)
                            if nb in cells and nb not in seen:
                                seen.add(nb)
                                stack.append(nb)
                members.sort()
                changes = Counter(f"{pred_rows[r][c]}>{real_rows[r][c]}" for r, c in members)
                groups.append({
                    "cells": len(members),
                    "rows": [min(r for r, _ in members), max(r for r, _ in members)],
                    "cols": [min(c for _, c in members), max(c for _, c in members)],
                    "predicted>real": dict(changes.most_common()),
                    "coords": [[r, c] for r, c in members],
                })
            groups.sort(key=lambda g: -g["cells"])
            return {"same": not any(g["cells"] >= _schema_tolerance for g in groups),
                    "cells": len(cells), "groups": groups}

        def _is_node(value):
            return isinstance(value, dict) and {"color", "boundary", "pixels"} <= set(value)

        class _Frozen(tuple):
            # a hashable stand-in for a dict / list / set inside a multiset comparison
            kind = ""

        def _frozen(value):
            # value as a hashable, comparable key; numbers keep 1 == 1.0
            if isinstance(value, dict):
                out = _Frozen(sorted(((repr(k), _frozen(v)) for k, v in value.items()), key=repr))
                out.kind = "dict"
                return out
            if isinstance(value, list):
                # a list inside a list item is unordered too: sorted into one canonical order
                out = _Frozen(sorted((_frozen(v) for v in value), key=repr))
                out.kind = "list"
                return out
            if isinstance(value, tuple):
                out = _Frozen(_frozen(v) for v in value)
                out.kind = "tuple"
                return out
            if isinstance(value, (set, frozenset)):
                out = _Frozen(sorted((_frozen(v) for v in value), key=repr))
                out.kind = "set"
                return out
            try:
                hash(value)
                return value
            except TypeError:
                return repr(value)

        def _thawed(value):
            # a _frozen key back to plain values, for the report
            if isinstance(value, _Frozen):
                if value.kind == "dict":
                    return {k.strip("'\""): _thawed(v) for k, v in value}
                items = [_thawed(v) for v in value]
                return items if value.kind == "list" else tuple(items)
            return value

        def _state_diff(pred, real, path=""):
            # differences between two key_state values; nodes by color and coordinates
            if _is_node(pred) and _is_node(real):
                keys = ("color", "boundary", "pixels")
                pred = {k: pred[k] for k in keys}
                real = {k: real[k] for k in keys}
                return {} if pred == real else {path or "value": {"predicted": pred, "real": real}}
            if isinstance(pred, dict) and isinstance(real, dict):
                out = {}
                # in the prediction's order, so the first difference named is stable
                for key in [*pred, *(k for k in real if k not in pred)]:
                    sub = f"{path}[{key!r}]" if path else repr(key)
                    if key not in pred or key not in real:
                        out[sub] = {"predicted": pred.get(key, "<missing>"),
                                    "real": real.get(key, "<missing>")}
                    else:
                        out.update(_state_diff(pred[key], real[key], sub))
                return out
            if (_schema_parse and isinstance(pred, (list, tuple)) and isinstance(real, (list, tuple))
                    and len(pred) <= 4 and len(real) <= 4
                    and all(isinstance(v, (int, float)) for v in (*pred, *real))):
                # a center or bbox is named whole: 'center' predicted (9, 6), real (9, 9)
                return {} if list(pred) == list(real) else {
                    path or "value": {"predicted": tuple(pred), "real": tuple(real)}}
            if (_schema_parse and isinstance(pred, (list, tuple)) and isinstance(real, (list, tuple))
                    and (isinstance(pred, list) or isinstance(real, list))):
                # ARC3_SCHEMA_PARSE: a list is a multiset (order ignored, repeats counted);
                # a tuple keeps its order. Reported as the items only one side has.
                a = Counter(_frozen(v) for v in pred)
                b = Counter(_frozen(v) for v in real)
                if a == b:
                    return {}
                # the whole lists, and the items only one side has
                return {f"{path or 'value'} (unordered)": {
                    "predicted": list(pred), "real": list(real),
                    "only_predicted": [_thawed(v) for v in (a - b).elements()],
                    "only_real": [_thawed(v) for v in (b - a).elements()]}}
            if isinstance(pred, (list, tuple)) and isinstance(real, (list, tuple)):
                if len(pred) != len(real):
                    return {path or "value": {"predicted": f"{len(pred)} items",
                                              "real": f"{len(real)} items"}}
                out = {}
                for i, (a, b) in enumerate(zip(pred, real)):
                    out.update(_state_diff(a, b, f"{path}[{i}]"))
                return out
            if (hasattr(pred, "__dict__") and hasattr(real, "__dict__")
                    and not callable(pred) and not callable(real)):
                # instances of the model's own classes: same class, then their attributes
                if type(pred).__name__ != type(real).__name__:
                    return {path or "value": {"predicted": type(pred).__name__,
                                              "real": type(real).__name__}}
                a, b = vars(pred), vars(real)
                out = {}
                for key in set(a) | set(b):
                    sub = f"{path}.{key}" if path else key
                    if key not in a or key not in b:
                        out[sub] = {"predicted": a.get(key, "<missing>"),
                                    "real": b.get(key, "<missing>")}
                    else:
                        out.update(_state_diff(a[key], b[key], sub))
                return out
            if pred != real:
                return {path or "value": {"predicted": pred, "real": real}}
            return {}

        def _real_flags(after):
            if not isinstance(after, HistoryEntryView):
                return None  # a bare frame carries no outcome
            result = after.result or {}
            # win: the action completed the level (or the whole game)
            return {"dead": bool(result.get("game_over")),
                    "win": bool(result.get("level_completed") or result.get("done")
                                or result.get("run_complete"))}

        def _predict(model, before, move):
            # (rows, flags) from model.step: info {'dead', 'win'}; a board alone means
            # the action is neither fatal nor completes the level
            name, row, col = _parse_move(move)
            if _is_undo(name):
                return None, {"dead": False, "win": False}  # not judged: step is not asked
            if _schema_parse:
                # step moves a copy of parse(before), so editing its argument is fine
                state = _parsed(model, before)
                out = model.step(copy.deepcopy(state), name, row, col)
                info = {}
                if (isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict)
                        and isinstance(out[1], dict)):
                    out, info = out
                if not isinstance(out, dict):
                    raise TypeError(f"step returned {type(out).__name__}; expected the next state "
                                    "as a dict (like parse returns), or (state, {'dead': bool, "
                                    "'win': bool})")
                return _StatePrediction(out, state), {"dead": bool(info.get("dead", False)),
                                                      "win": bool(info.get("win", False))}
            out = model.step(before, name, row, col)
            info = {}
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                out, info = out
            return _rows_of(out), {"dead": bool(info.get("dead", False)),
                                   "win": bool(info.get("win", False))}

        # ARC3_SCHEMA_PARSE: the most numbers / strings / booleans a parsed state may hold.
        # An object costs about 10 (center 2, bbox 4, cell_num, main_color, symbol, a
        # property), so 30 objects fit; a 64x64 board is 4096 and one pixel-template piece
        # (cn04 v56 rerun) a few hundred.
        _STATE_MAX_SCALARS = 300

        def _scalar_count(value, limit=100000):
            # the number of leaf values in a parsed state (stops counting past `limit`)
            stack, n = [value], 0
            while stack and n <= limit:
                v = stack.pop()
                if isinstance(v, dict):
                    stack.extend(v.values())
                elif isinstance(v, (list, tuple, set, frozenset)):
                    stack.extend(v)
                else:
                    n += 1
            return n

        class _StatePrediction:
            # ARC3_SCHEMA_PARSE: step's predicted state and the parsed state it started from
            def __init__(self, state, before):
                self.state, self.before = state, before

        def _parsed(model, frame):
            # parse(frame), which must be a dict
            if isinstance(frame, HistoryEntryView):
                frame = frame.frame
            state = model.parse(frame)
            if not isinstance(state, dict):
                raise TypeError(f"parse returned {type(state).__name__}; expected a dict")
            _check_objects(state)
            scalars = _scalar_count(state)
            if scalars > _STATE_MAX_SCALARS:
                # a guard, not a switch: a state this large lists cells, not objects
                raise ValueError(
                    f"parse: the state holds {scalars} numbers / strings (at most "
                    f"{_STATE_MAX_SCALARS}); it lists cells instead of objects. Describe each "
                    "object by its center, bbox, cell_num, main_color, symbol and properties "
                    "(an irregular shape by one point per n*n block in 'centers'), never by "
                    "all its cells")
            if _schema_parse_map and not _level_scrolled(frame.level):
                block_map = state.get("map")
                if not (isinstance(block_map, (list, tuple)) and block_map
                        and all(isinstance(row, str) and row for row in block_map)):
                    raise ValueError(
                        "parse: the state has no 'map' (got " + _short(block_map) + "); "
                        "return the block map under 'map': a list of strings, one character "
                        "per block, e.g. printed with cellstat")
            return state

        def _error_line(exc):
            # the exception, and where in the model's own functions it was raised:
            # the innermost frame of a function defined in a snippet, with its line
            text = f"{exc.__class__.__name__}: {exc}"
            where = None
            for frame, lineno in traceback.walk_tb(exc.__traceback__):
                code = frame.f_code
                source = _def_sources.get(code)
                if source is None:
                    continue
                lines = source.splitlines()
                offset = lineno - code.co_firstlineno
                line = lines[offset].strip() if 0 <= offset < len(lines) else ""
                where = f"in {code.co_name}, line {offset + 1}" + (f": `{line[:80]}`" if line else "")
            return f"{text} ({where})" if where else text

        def _judge(model, before, after, move, predicted=None):
            # one comparison record; `predicted` is (rows, flags), or an exception
            # `pred` and `real` are frames, so frame_diff(pred, real) lists what differs;
            # `largest_difference` is where on the board the prediction is most wrong
            name, row, col = _parse_move(move)
            real_frame = after.frame if isinstance(after, HistoryEntryView) else after
            record = {"action": _move_text(name, row, col), "consistent": False, "kind": None,
                      "pred": None, "real": real_frame, "differing_cells": None,
                      "largest_difference": None, "key_state_diff": None, "flags": None,
                      "win_condition": None, "error": None}
            if _schema_parse:
                record.update({"pred_state": None, "real_state": None, "state_diff": None})
            if _is_undo(name):
                # UNDO is never judged: neither the board nor key_state is compared,
                # so step need not model it
                record["consistent"] = True
                return record
            try:
                if predicted is None:
                    predicted = _predict(model, before, move)
                if isinstance(predicted, BaseException):
                    raise predicted
                if _schema_parse:
                    return _judge_state(model, before, after, record, predicted)
                rows, pred_flags = predicted
                pred_frame = _frame_of_rows(rows, real_frame)
                record["pred"] = pred_frame
                real_rows = tuple(real_frame.ascii.split("\n"))
                real_flags = _real_flags(after)
                # a winning entry's board is the moment of the win on the old level, so
                # the prediction is compared with it; a death's board, or a win without
                # that board (the next level's first board), is not compared
                won_board = bool(isinstance(after, HistoryEntryView)
                                 and (after.result or {}).get("winning_board"))
                ended = bool(real_flags and (real_flags["dead"]
                                             or (real_flags["win"] and not won_board)))
                board_diff = _board_diff(rows, real_rows)
                record["differing_cells"] = board_diff["cells"]
                if board_diff["groups"]:
                    g = board_diff["groups"][0]
                    record["largest_difference"] = {"cells": g["cells"], "rows": g["rows"],
                                                    "cols": g["cols"]}
                kinds = []
                if real_flags is not None:
                    wrong = {k: {"predicted": pred_flags[k], "real": real_flags[k]}
                             for k in pred_flags if pred_flags[k] != real_flags[k]}
                    record["flags"] = {"predicted": pred_flags, "real": real_flags, "wrong": wrong}
                    if wrong:
                        kinds.append("flag")
                if model.is_win_condition is not None and not (real_flags and real_flags["dead"]):
                    said = bool(model.is_win_condition(real_frame))
                    won = bool(real_flags and real_flags["win"])
                    # a bare frame does not say whether the level was completed
                    record["win_condition"] = {"says_won": said, "level_completed": won if real_flags else None}
                    if real_flags is not None and said != won:
                        kinds.append("win_condition")
                if model.key_state is not None:
                    diff = _state_diff(model.key_state(pred_frame), model.key_state(real_frame))
                    record["key_state_diff"] = diff
                    if diff and not ended:
                        kinds.append("key_state")
                elif not board_diff["same"] and not ended:
                    if board_diff.get("note"):
                        record["error"] = board_diff["note"]
                    kinds.append("board")
                record["kind"] = kinds[0] if kinds else None
                record["consistent"] = not kinds
                record["ended"] = ended
                record["view_shift"] = _view_shift_of(after)
                # ARC3_SCHEMA_REPORT_DIFF: what a judged-right action changed beyond the
                # prediction is reported, never stopped
                if (record["consistent"] and not ended and _schema_report_diff
                        and board_diff["cells"]):
                    hits = _filter_diff(_raw_diff(pred_frame, real_frame, _view_shift_of(after)),
                                        real_frame)
                    if hits:
                        record["unpredicted"] = hits
            except Exception as exc:
                record["kind"], record["error"] = "error", _error_line(exc)
            return record

        def _judge_state(model, before, after, record, predicted):
            # ARC3_SCHEMA_PARSE: step(parse(before)) against parse of the real board. The
            # flags and is_win_condition are judged as for boards; a death's board, or a win
            # without the winning board, is not parsed for the comparison
            prediction, pred_flags = predicted
            real_frame = record["real"]
            record["pred_state"] = prediction.state
            real_flags = _real_flags(after)
            won_board = bool(isinstance(after, HistoryEntryView)
                             and (after.result or {}).get("winning_board"))
            ended = bool(real_flags and (real_flags["dead"]
                                         or (real_flags["win"] and not won_board)))
            kinds, real_state = [], None
            if real_flags is not None:
                wrong = {k: {"predicted": pred_flags[k], "real": real_flags[k]}
                         for k in pred_flags if pred_flags[k] != real_flags[k]}
                record["flags"] = {"predicted": pred_flags, "real": real_flags, "wrong": wrong}
                if wrong:
                    kinds.append("flag")
            if not ended:
                real_state = _parsed(model, real_frame)
                record["real_state"] = real_state
            if model.is_win_condition is not None and not (real_flags and real_flags["dead"]):
                state = real_state if real_state is not None else _parsed(model, real_frame)
                said = bool(model.is_win_condition(state))
                won = bool(real_flags and real_flags["win"])
                record["win_condition"] = {"says_won": said,
                                           "level_completed": won if real_flags else None}
                if real_flags is not None and said != won:
                    kinds.append("win_condition")
            if real_state is not None:
                diff = _state_diff(prediction.state, real_state)
                record["state_diff"] = diff
                if diff:
                    kinds.append("state")
            record["kind"] = kinds[0] if kinds else None
            record["consistent"] = not kinds
            record["ended"] = ended
            record["view_shift"] = _view_shift_of(after)
            if real_state is not None:
                record["uncovered"] = _coverage(before, real_frame, prediction.before,
                                                real_state, record["view_shift"])
            return record

        # ARC3_SCHEMA_PARSE: frame_diff objects of the real transition that no parsed
        # object's bbox covers, the same objects ARC3_SCHEMA_REPORT_DIFF names
        _COVER_PIXELS = 4  # smaller objects are noise (as _REPORT_MOVE_PIXELS)
        _COVER_SHARE = 0.5  # share of a changed object's bbox inside the parsed bboxes

        def _objects_of(state, path="", out=None):
            # [(path, bbox, value)] of every object in `state`: a dict with a 'bbox'
            out = [] if out is None else out
            if isinstance(state, dict):
                if "bbox" in state:
                    out.append((path or "state", tuple(int(v) for v in state["bbox"]), state))
                for key, value in state.items():
                    if key not in ("center", "bbox"):
                        _objects_of(value, f"{path}[{key!r}]", out)
            elif isinstance(state, (list, tuple)):
                for i, value in enumerate(state):
                    _objects_of(value, f"{path}[{i}]", out)
            return out

        def _check_objects(state, path=""):
            # every object (a dict with a 'center' or a 'bbox') has both a 'center'
            # (row, col) and a 'bbox' (r0, c0, r1, c1): coverage is judged by the bbox
            if isinstance(state, dict):
                if {"center", "bbox"} & set(state):
                    where = path or "the state"
                    center, box = state.get("center"), state.get("bbox")
                    if not (isinstance(center, (list, tuple)) and len(center) == 2
                            and all(isinstance(v, (int, float)) for v in center)):
                        raise ValueError(f"parse: {where} is an object but its 'center' is "
                                         f"{center!r}; every object needs 'center': (row, col) "
                                         "and 'bbox': (r0, c0, r1, c1)")
                    if not (isinstance(box, (list, tuple)) and len(box) == 4
                            and all(isinstance(v, (int, float)) for v in box)):
                        raise ValueError(f"parse: {where} is an object but its 'bbox' is "
                                         f"{box!r}; every object needs 'center': (row, col) "
                                         "and 'bbox': (r0, c0, r1, c1)")
                for key, value in state.items():
                    if key not in ("center", "bbox"):
                        _check_objects(value, f"{path}[{key!r}]")
            elif isinstance(state, (list, tuple)):
                for i, value in enumerate(state):
                    _check_objects(value, f"{path}[{i}]")

        def _boxes_at(nodes):
            # {top-left: [(hash, bbox)]} of a board's segmentation objects
            out = {}
            for n in nodes:
                out.setdefault(tuple(_node_topleft(n)), []).append((n.get("hash"), _node_bbox(n)))
            return out

        def _box_of(index, at, hash_=None):
            found = index.get(tuple(at)) if at is not None else None
            if not found:
                return None
            return next((box for h, box in found if h == hash_), found[0][1])

        def _changed_objects(before, real_frame):
            # [(kind, entry, [bbox, ...])]: frame_diff(before, real) objects of 4+ pixels,
            # not wholly in the border band, with where they are before and after
            nodes_a = before.segmentation.get("nodes", [])
            nodes_b = real_frame.segmentation.get("nodes", [])
            d = compute_frame_diff(before._grid, real_frame._grid, nodes_a, nodes_b)
            at_a, at_b = _boxes_at(nodes_a), _boxes_at(nodes_b)
            shape = (len(real_frame._grid), len(real_frame._grid[0]))
            out = []
            for kind in ("moved", "rotated", "appeared", "disappeared", "changed_color", "resized"):
                for e in d.get(kind, []):
                    if kind == "moved":
                        boxes = [_box_of(at_a, e["from"], e.get("hash")),
                                 _box_of(at_b, e["to"], e.get("hash"))]
                        big = e.get("pixels", 0) >= _COVER_PIXELS
                    elif kind == "rotated":
                        boxes = [_box_of(at_b, e["at"], e.get("hash"))]
                        big = e.get("pixels", 0) >= _COVER_PIXELS
                    elif kind == "resized":
                        boxes = [_box_of(at_a, e["at_before"], e.get("before_hash")),
                                 _box_of(at_b, e["at_after"], e.get("after_hash"))]
                        big = abs(e["after_pixels"] - e["before_pixels"]) >= _COVER_PIXELS
                    elif kind == "changed_color":
                        boxes = [_box_of(at_b, e["at"], e.get("after_hash"))]
                        big = e.get("pixels", 0) >= _COVER_PIXELS
                    else:
                        boxes = [tuple(e["bbox"])]
                        big = e.get("pixels", 0) >= _COVER_PIXELS
                    boxes = [b for b in boxes if b]
                    if big and boxes and not all(_in_border(shape, b[:2], b) for b in boxes):
                        out.append((kind, e, boxes))
            return out

        def _cells_of(box):
            r0, c0, r1, c1 = box
            return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

        def _covered_by(box, area):
            cells = _cells_of(box)
            return len(cells & area) >= _COVER_SHARE * len(cells)

        def _entry_text(kind, e, boxes):
            # one changed object, as the report names it
            out = {"kind": kind, "bbox": list(boxes[-1])}
            for key in ("color", "after_color", "pixels", "from", "to", "at"):
                if key in e:
                    out[key] = e[key]
            return out

        def _level_scrolled(level):
            # whether `level` belongs to a scrolling game: once the view has scrolled in
            # any level, every level counts (bp35's level 2 opened before its first scroll)
            return any(any((entry.result or {}).get("view_shift") or ())
                       for entry in _flat_history)

        def _coverage(before, real_frame, before_state, real_state, shift=None):
            # The frame_diff objects of the real transition that parse missed: ones
            # outside every parsed object's bbox (of the state before or after), and
            # ones inside an object whose parsed value did not change. None if nothing,
            # and in a level whose view has scrolled, where parse's screen positions
            # cannot be shared between views.
            if isinstance(before, HistoryEntryView):
                before = before.frame
            if before is None or real_frame is None or (shift and any(shift)):
                return None
            if _level_scrolled(real_frame.level):
                return None  # parse there holds only what scrolling does not change
            if before._grid == real_frame._grid:
                return None
            changed = _changed_objects(before, real_frame)
            if not changed:
                return None
            objects_a, objects_b = _objects_of(before_state), _objects_of(real_state)
            area = set()
            for _, (r0, c0, r1, c1), _ in objects_a + objects_b:
                area |= _cells_of((r0 - 1, c0 - 1, r1 + 1, c1 + 1))
            values_a = {path: value for path, _, value in objects_a}
            outside, same = [], []
            for kind, e, boxes in changed:
                if not all(_covered_by(b, area) for b in boxes):
                    outside.append(_entry_text(kind, e, boxes))
                    continue
                # covered: by an object whose parsed value stayed the same?
                for path, (r0, c0, r1, c1), value in objects_b:
                    own = _cells_of((r0 - 1, c0 - 1, r1 + 1, c1 + 1))
                    if (any(_covered_by(b, own) for b in boxes) and path in values_a
                            and not _state_diff(values_a[path], value)):
                        same.append({"object": path, **_entry_text(kind, e, boxes)})
                        break
            out = {}
            if outside:
                out["outside_objects"] = outside
                if not objects_a and not objects_b:
                    # nothing to match against: objects kept as tuples or lists are not seen
                    out["no_objects"] = True
            if same:
                out["unchanged_objects"] = same
            return out or None

        def _uncovered_parts(found, seen=None):
            # the report's phrases; `seen` keeps each region or object once per level
            parts = []
            for g in found.get("outside_objects", []):
                r0, c0 = g["bbox"][0], g["bbox"][1]
                key = ("out", r0 // 8, c0 // 8)
                if seen is not None:
                    if key in seen:
                        continue
                    seen.add(key)
                parts.append(f"{g['kind']} {_short({k: v for k, v in g.items() if k != 'kind'}, 120)}"
                             " lies outside every object's bbox")
            for o in found.get("unchanged_objects", []):
                key = ("same", o["object"])
                if seen is not None:
                    if key in seen:
                        continue
                    seen.add(key)
                parts.append(f"{o['object']} {o['kind']} on the board but its parsed value did "
                             "not change")
            return parts

        _cover_seen = {"level": None, "keys": set()}  # what action() already reported

        def _coverage_note(record, i, n):
            # ARC3_SCHEMA_PARSE: one line on what parse missed, each region or object once
            # per level and at most 3 per action
            found = record.get("uncovered")
            real = record.get("real")
            if not found or real is None:
                return ""
            if _cover_seen["level"] != real.level:
                _cover_seen.update(level=real.level, keys=set())
            parts = _uncovered_parts(found, _cover_seen["keys"])[:3]
            if not parts:
                return ""
            if found.get("no_objects"):
                # one reason for all of them, said plainly
                return (f"[model] action {i} of {n} ({record['action']}): " + "; ".join(parts)
                        + ". Your parsed state has no object with a 'bbox', so no change can be "
                        "matched to one: store each object as a dict with 'center' and 'bbox' "
                        "(objects kept as tuples or lists are not seen; lists are compared "
                        "ignoring order, so they need no sorting).")
            return (f"[model] action {i} of {n} ({record['action']}): " + "; ".join(parts)
                    + ". Unless it is noise, parse is missing an object or property, and step "
                    "cannot model it.")

        _REPORT_PIXELS = 8  # smaller appeared / disappeared / recolor / resize changes are noise
        # smaller moved / rotated objects are noise: a scrolled view against an unscrolled
        # prediction moves every 1-pixel wall dot (bp35 v50), while ls20's 4-pixel logo
        # piece still counts
        _REPORT_MOVE_PIXELS = 4

        def _in_border(shape, at, box=None):
            # whether an object lies wholly in the border band (HUD bars and counters)
            h, (height, width) = _schema_hud, shape
            r0, c0, r1, c1 = box if box else (at[0], at[1], at[0], at[1])
            return r1 < h or c1 < h or r0 >= height - h or c0 >= width - h

        def _raw_diff(pred_frame, real_frame, hint=None):
            # frame_diff(pred, real), hashes included. When the real view scrolled away from
            # the prediction (a scroll level's constant `board(f)` step), the prediction is
            # first moved into the real view, as frame_diff() does for a scroll, so only real
            # changes are left and the scroll itself does not move every object. A scroll too
            # long to detect between the two boards falls back to `hint`, the action's own
            # recorded view_shift, when the boards agree under it (or no longer overlap)
            pred_grid, real_grid = pred_frame._grid, real_frame._grid
            shift = detect_view_shift(pred_grid, real_grid) if pred_grid != real_grid else None
            if shift is None and hint and any(hint) and pred_grid != real_grid:
                agree, total = _shift_agreement(pred_grid, real_grid, hint[0], hint[1], _schema_hud)
                if not total or agree >= 0.9:
                    shift = (hint[0], hint[1])
            if shift is None:
                return compute_frame_diff(pred_grid, real_grid,
                                          pred_frame.segmentation.get("nodes", []),
                                          real_frame.segmentation.get("nodes", []))
            aligned = align_to_view(pred_grid, real_grid, shift[0], shift[1])
            return compute_frame_diff(aligned, real_grid,
                                      segment_layer(aligned, COLOR_CHARS).get("nodes", []),
                                      real_frame.segmentation.get("nodes", []))

        def _view_shift_of(after):
            # the action's recorded camera scroll, when `after` is its history entry
            result = after.result if isinstance(after, HistoryEntryView) else None
            shift = (result or {}).get("view_shift")
            return tuple(shift) if shift else None

        def _hashless(d):
            return {kind: [{k: v for k, v in e.items() if "hash" not in k} for e in entries]
                    if isinstance(entries, list) else entries for kind, entries in d.items()}

        def _filter_diff(d, real_frame):
            # the frame_diff entries step did not predict, without noise and hashes
            shape = (len(real_frame._grid), len(real_frame._grid[0]))
            # moved / rotated entries carry no bbox: take it from the real board's object
            real_boxes = {(n.get("hash"), _node_topleft(n)): _node_bbox(n)
                          for n in real_frame.segmentation.get("nodes", [])}

            def keep(kind, e):
                # an object of 4+ pixels moving or turning is reported, unless it lies
                # wholly in the border band (HUD pieces)
                if kind in ("moved", "rotated"):
                    at = e.get("to") or e.get("at")
                    return (e.get("pixels", 0) >= _REPORT_MOVE_PIXELS
                            and not _in_border(shape, at, real_boxes.get((e.get("hash"), tuple(at)))))
                if kind == "resized":
                    return (abs(e["after_pixels"] - e["before_pixels"]) >= _REPORT_PIXELS
                            and not _in_border(shape, e["at_after"]))
                return (e["pixels"] >= _REPORT_PIXELS
                        and not _in_border(shape, e["at"], e.get("bbox")))

            out = {}
            for kind in ("moved", "rotated", "appeared", "disappeared", "changed_color", "resized"):
                entries = [{k: v for k, v in e.items() if "hash" not in k}
                           for e in d.get(kind, []) if keep(kind, e)]
                if entries:
                    out[kind] = entries
            return out

        def _audit(record):
            # ARC3_SCHEMA_DIFF_AUDIT: log a judged-right, unended action whose board differs
            if not (_schema_diff_audit and record.get("consistent") and not record.get("ended")
                    and record.get("differing_cells") and record.get("pred") is not None):
                return
            pred, real = record["pred"], record["real"]
            try:
                d = _raw_diff(pred, real, record.get("view_shift"))
                _diff_audit.append({
                    "level": real.level, "step": real.step, "action": record["action"],
                    "key_state": record.get("key_state_diff") is not None,
                    "differing_cells": record["differing_cells"],
                    "largest_difference": record["largest_difference"],
                    "reported": record.get("unpredicted"),
                    "kept": _filter_diff(d, real), "frame_diff": _hashless(d),
                    "pred": pred.ascii, "real": real.ascii})
            except Exception as exc:
                _diff_audit.append({"action": record["action"], "error": _error_line(exc)})

        def _wrong_hint(what):
            # where the model reads what a wrong prediction got wrong
            if _schema_parse:
                return f"last_prediction['state_diff'] lists what {what} got wrong"
            return (f"frame_diff(last_prediction['pred'], last_prediction['real']) lists what "
                    f"{what} got wrong")

        def _unpredicted_note(record, i, n):
            # ARC3_SCHEMA_REPORT_DIFF: what a judged-right action changed beyond step's prediction
            found = record.get("unpredicted")
            if not found:
                return ""
            return (f"[model] action {i} of {n} ({record['action']}) ran as predicted, but "
                    "frame_diff(last_prediction['pred'], last_prediction['real']) also shows "
                    f"changes step did not predict: {json.dumps(found, separators=(',', ':'))}. "
                    "Unless they are noise, a rule is missing from step or key_state.")

        def _where(record):
            # the reason a record is wrong, in one phrase
            kind = record.get("kind")
            if kind == "error":
                return f"error: {record['error']}"
            box = record.get("largest_difference")
            board = (f"board: {record['differing_cells']} cells differ, the largest group "
                     f"{box['cells']} cells in rows {box['rows'][0]}-{box['rows'][1]}, "
                     f"cols {box['cols'][0]}-{box['cols'][1]}") if box else ""
            if kind == "board":
                return board or f"board: {record['error']}"
            if kind == "flag":
                text = "flag " + ", ".join(f"{k}: predicted {v['predicted']}, actual {v['real']}"
                                           for k, v in record["flags"]["wrong"].items())
            elif kind == "win_condition":
                wc = record["win_condition"]
                text = (f"is_win_condition said {wc['says_won']} but the level "
                        f"{'was' if wc['level_completed'] else 'was not'} completed")
            elif kind == "key_state":
                keys = list(record["key_state_diff"])
                text = "key_state: " + ", ".join(keys[:4]) + (" ..." if len(keys) > 4 else "") + " differ"
            elif kind == "state":
                items = list(record["state_diff"].items())
                def one(path, v):
                    if "only_predicted" in v:  # an unordered list: what differs, then both lists
                        return (f"{path} only predicted {_short(v['only_predicted'])}, only real "
                                f"{_short(v['only_real'])} (predicted {_short(v['predicted'], 120)}, "
                                f"real {_short(v['real'], 120)})")
                    return f"{path} predicted {_short(v['predicted'])}, real {_short(v['real'])}"
                text = "state: " + "; ".join(one(path, v) for path, v in items[:3]) + (
                    f" (and {len(items) - 3} more)" if len(items) > 3 else "")
            else:
                text = "consistent"  # board differences key_state allowed are still located
            return f"{text}; {board}" if board else text

        def _short(value, limit=60):
            text = repr(value)
            return text if len(text) <= limit else text[:limit - 3] + "..."

        def _public(record, index=None):
            # what the model reads of one comparison: where it differs and both boards
            # (ARC3_SCHEMA_PARSE: both states, and the real board as real_frame)
            out = {} if index is None else {"index": index}
            if _schema_parse:
                out.update({"action": record["action"], "where": _where(record),
                            "pred": record.get("pred_state"), "real": record.get("real_state"),
                            "state_diff": record.get("state_diff"), "real_frame": record["real"]})
                if record.get("uncovered"):
                    out["uncovered"] = record["uncovered"]
                return out
            out.update({"action": record["action"], "where": _where(record),
                        "pred": record["pred"], "real": record["real"],
                        "key_state_diff": record["key_state_diff"]})
            if record.get("unpredicted"):
                out["unpredicted"] = record["unpredicted"]
            return out

        def check_prediction(model, before, after, action=None):
            # Predict `after` from `before` with model.step and compare (the rule action() uses).
            # before: a frame or a history entry; after: a history entry (its .action is used
            # and its dead / win are compared) or a frame (then pass `action`).
            if not isinstance(model, WorldModel):
                raise TypeError("check_prediction(model, before, after): model must be a WorldModel")
            _last_model[:] = [model]
            if isinstance(before, HistoryEntryView):
                before = before.frame
            if action is None:
                if not isinstance(after, HistoryEntryView):
                    raise TypeError("check_prediction: pass action=... when `after` is a frame, "
                                    "or pass the history entry, which records its action")
                action = after.action
            record = _judge(model, before, after, action)
            return {"consistent": record["consistent"], **_public(record)}

        def backtest(model, level=-1):
            # Replay the model over a level's recorded transitions (level_history[level],
            # default the current level), each from the real board before it. Returns a
            # summary with the latest wrong one; every wrong one is in last_backtest.
            if not isinstance(model, WorldModel):
                raise TypeError("backtest(model): model must be a WorldModel")
            _last_model[:] = [model]
            _backtested.append(model)
            return _backtest(model, level)

        def _crop(frame, box, pad=1):
            # the rows of `frame` around a difference box, as strings
            if frame is None or not box:
                return None
            rows = frame.ascii.split("\n")
            r0, r1 = max(0, box["rows"][0] - pad), min(len(rows) - 1, box["rows"][1] + pad)
            c0, c1 = max(0, box["cols"][0] - pad), box["cols"][1] + pad + 1
            return {"rows": [r0, r1], "cols": [c0, c1 - 1],
                    "lines": [row[c0:c1] for row in rows[r0:r1 + 1]]}

        def _backtest(model, level=-1, deadline=None):
            history = runtime_globals.get("level_history") or [[]]
            entries = history[level]
            L = level if level >= 0 else len(history) + level  # index = (L, k): level_history[L][k]
            checked, mismatches, stopped = 0, [], False
            uncovered = []  # ARC3_SCHEMA_PARSE: transitions with changes parse missed
            for k in range(1, len(entries)):
                if deadline is not None and time.monotonic() > deadline:
                    stopped = True
                    break
                entry = entries[k]
                name = str(entry.action or "").strip()
                if not name or name.upper() == "RESET" or _is_undo(name) or entries[k - 1].frame is None:
                    continue  # a level start, a restart or an UNDO is not judged
                if (entries[k - 1].result or {}).get("game_over"):
                    continue  # the board after a death is the restarted one
                checked += 1
                record = _judge(model, entries[k - 1].frame, entry, name)
                if record.get("uncovered"):
                    uncovered.append(((L, k), name, record["uncovered"]))
                if not record["consistent"]:
                    mismatches.append((record, _public(record, (L, k))))
            wrong = [item for _, item in mismatches]
            # the return value: the counts and the latest 5 wrong transitions with the
            # predicted and real rows around their largest difference; last_backtest
            # keeps every wrong one in full (frames included)
            recent = []
            for record, item in mismatches[-5:]:
                if _schema_parse:
                    recent.append({"index": item["index"], "action": item["action"],
                                   "where": item["where"], "state_diff": item["state_diff"]})
                    continue
                box = record.get("largest_difference")
                recent.append({"index": item["index"], "action": item["action"],
                               "where": item["where"], "key_state_diff": item["key_state_diff"],
                               "predicted_crop": _crop(item["pred"], box),
                               "real_crop": _crop(item["real"], box)})
            out = {"checked": checked, "correct": checked - len(wrong),
                   "wrong_indices": [item["index"] for item in wrong], "recent": recent}
            if _schema_parse:
                # check_prediction(model, level_history[L][k-1], level_history[L][k]) shows
                # one transition's 'uncovered' changes
                out["uncovered_indices"] = [index for index, _, _ in uncovered][-10:]
            if stopped:
                out["stopped_early"] = True
            if mismatches:
                record, item = mismatches[-1]
                _backtest_note[:] = [_crop_note(item, record)]
            runtime_globals["last_backtest"] = {**out, "latest": wrong[-1] if wrong else None,
                                                "wrong": wrong}
            if _schema_parse:
                runtime_globals["last_backtest"]["uncovered"] = [
                    {"index": index, "action": name, "uncovered": found}
                    for index, name, found in uncovered]
            return out

        def _crop_note(item, record):
            # the latest wrong transition for the next user message: where, then the
            # predicted and real rows around the difference, side by side
            L, k = item["index"]
            head = (f"Your latest backtest's latest wrong transition, level_history[{L}][{k}] "
                    f"({item['action']}): {item['where']}.")
            pred_crop = _crop(item["pred"], record.get("largest_difference"))
            real_crop = _crop(item["real"], record.get("largest_difference"))
            if not pred_crop or not real_crop:
                return head
            lines = [f"{p} | {q}" for p, q in zip(pred_crop["lines"], real_crop["lines"])]
            budget = _schema_crop_chars - len(head) - 120
            kept = []
            for line in lines:
                if sum(len(x) + 1 for x in kept) + len(line) + 1 > budget:
                    break
                kept.append(line)
            r0, c0, c1 = pred_crop["rows"][0], pred_crop["cols"][0], pred_crop["cols"][1]
            cut = "" if len(kept) == len(lines) else f" (first {len(kept)} of {len(lines)} rows)"
            return (f"{head}\nPredicted | real, rows {r0}-{r0 + len(kept) - 1}, cols {c0}-{c1}{cut}:\n"
                    + "\n".join(kept))

        def _action_entry(start):
            # this action's own history entry: the first new one that has an action
            for entry in _flat_history[start:]:
                if str(entry.action or "").strip():
                    return entry
            return None

        # BaseException like the guards above, so `except Exception:` around action()
        # cannot turn the refusal into a loop that keeps going
        class ActionStoppedError(BaseException):
            pass

        # the misses of THIS snippet: actions that ran without a prediction vouching for
        # them (no model, mispredicted, or step failed). Once there are
        # ARC3_SCHEMA_MISS_BUDGET of them the snippet may not act again; cleared per run.
        _schema_stop = []
        _miss_budget = max(1, int(initial.get("schema_miss_budget") or 1))
        # ARC3_SCHEMA_FREE_ACTIONS above 1: actions this snippet may run without a model,
        # unchecked, before a further call without one is refused; cleared per run
        _free_actions = max(1, int(initial.get("schema_free_actions") or 1))
        _free_used = []
        # ARC3_CONST_PARSE_WARN=N: after N action() calls in a row whose model's parse never
        # changed while the board did (a constant parse), the host warns in the next message.
        # The streak spans Python calls; _const_parse_due is cleared per run.
        _const_parse_limit = max(0, int(initial.get("const_parse_warn") or 0))
        _const_parse_streak = [0]
        _const_parse_due = []
        _call_obs = []

        def _observe_constant_parse(before, record, predicted):
            # one board-changing action of this call: True when parse missed the change
            try:
                parsed_before = predicted[0].before
                changed = bool(_changed_objects(before, record["real"]))
            except Exception:
                return
            if changed:
                _call_obs.append(parsed_before == record["real_state"])

        def _tracked_action(actions, model=None):
            # schema_action, counting calls whose parse stayed constant over board changes
            del _call_obs[:]
            try:
                return schema_action(actions, model=model)
            finally:
                if _const_parse_limit and model is not None and _call_obs:
                    if all(_call_obs):
                        _const_parse_streak[0] += 1
                        if _const_parse_streak[0] >= _const_parse_limit:
                            _const_parse_due[:] = [True]
                            _const_parse_streak[0] = 0
                    else:
                        _const_parse_streak[0] = 0
        # ARC3_SCHEMA_STOP=0: nothing is refused or dropped; wrong predictions are reported
        _report_only = not initial.get("schema_stop", True)
        # ARC3_SCHEMA_STOP=call: a wrong prediction drops only the rest of its own call;
        # only a second call without a model is refused (the v27 rules)
        _call_scope = initial.get("schema_stop_mode") == "call"

        def _schema_action_report(items, model):
            # ARC3_SCHEMA_STOP=0: without a model the actions run as one batch (with a
            # reminder); with one, every action runs and each wrong prediction is printed
            if model is None:
                result = action(items)
                if len(items) > 1:
                    print(f"[model] these {len(items)} actions ran without a world model: build "
                          f"one with {_ctor} and pass model=... so that each "
                          "action is checked against its prediction.")
                return result
            if not isinstance(model, WorldModel):
                raise TypeError("action(actions, model=...): model must be a WorldModel")
            _last_model[:] = [model]
            result, wrong, stop, record = {}, [], None, None
            for i, item in enumerate(items):
                before = runtime_globals.get("current_frame")
                try:
                    predicted = _predict(model, before, item)
                except Exception as exc:
                    predicted = exc
                start = len(_flat_history)
                if i:
                    del _stale_state[:]  # the model is checking these boards, change or not
                result = action([item])
                entry = _action_entry(start)
                if entry is None:
                    stop = "the action was not executed"
                    break
                record = _judge(model, before, entry, item, predicted)
                runtime_globals["last_prediction"] = {"consistent": record["consistent"],
                                                      **_public(record)}
                _audit(record)
                if record.get("unpredicted"):
                    print(_unpredicted_note(record, i + 1, len(items)))
                note = _coverage_note(record, i + 1, len(items))
                if note:
                    print(note)
                if not record["consistent"]:
                    wrong.append(f"{i + 1} ({record['action']})")
                    print(f"[model] action {i + 1} of {len(items)} ({record['action']}) was "
                          f"MISPREDICTED ({_where(record)}).")
                for flag in ("level_completed", "game_over", "run_complete", "done"):
                    if result.get(flag):
                        stop = flag
                        break
                if stop:
                    break
            done = len(items) if stop is None else i + 1 - (stop == "the action was not executed")
            if wrong:
                print(f"[model] {len(wrong)} of {done} actions were mispredicted: "
                      + ", ".join(wrong) + ". " + _wrong_hint("the latest one") + "; fix step.")
            elif done > 1:
                print(f"[model] all {done} actions ran as predicted.")
            result = dict(result)
            result["plan"] = {"executed": done, "planned": len(items), "stopped": stop,
                              "mispredicted": len(wrong)}
            return result

        def schema_action(actions, model=None):
            # action() under ARC3_SCHEMA: with a model, each action is predicted first
            # and checked after it runs, and the first wrong prediction drops the rest;
            # without one, only the first action runs.
            items = _normalize_actions(actions)
            if _report_only:
                return _schema_action_report(items, model)
            frame = runtime_globals.get("current_frame")
            if _schema_parse and frame is not None and _level_scrolled(frame.level):
                # ARC3_SCHEMA_PARSE: in a level whose view has scrolled, screen positions
                # cannot be parsed into a state step could share, so a plan runs unchecked,
                # with or without a model (as if none were passed), and nothing is counted
                # against the call
                return action(items)
            # call scope: only unchecked calls are counted, and only they are refused
            if len(_schema_stop) >= _miss_budget and (model is None or not _call_scope):
                count = (f" (this Python call's limit of {_miss_budget} such actions is used up)"
                         if _miss_budget > 1 else "")
                raise ActionStoppedError(
                    f"{_schema_stop[-1]}{count}, so this Python call cannot run more actions; "
                    "this action was not executed.")
            if model is None and _free_actions > 1:
                # ARC3_SCHEMA_FREE_ACTIONS: up to that many unchecked actions per Python
                # call, across its calls without a model, to explore without a world model
                left = _free_actions - len(_free_used)
                if left <= 0:
                    raise ActionStoppedError(
                        f"this Python call already ran {_free_actions} actions without a model "
                        f"(its limit; pass model={_ctor} to run more), so this action was not "
                        "executed.")
                run = items[:left]
                _free_used.extend(run)
                result = action(run)
                if len(items) > len(run):
                    print(f"[model] only {len(run)} of {len(items)} actions ran without a model "
                          f"(at most {_free_actions} per Python call); pass model={_ctor} to "
                          "run more (each is checked against it).")
                    result["plan"] = {"executed": len(run), "planned": len(items),
                                      "stopped": "no model"}
                return result
            if model is None:
                # a loop of single unchecked calls would run a plan nothing checks
                _schema_stop.append("an action already ran without a model in this Python call "
                                    f"(pass model={_ctor} to run several)")
                result = action(items[:1])
                if len(items) > 1:
                    print(f"[model] only the first action ran ({len(items) - 1} dropped): pass "
                          f"model={_ctor} to run several actions per call "
                          "(each is checked against it).")
                    result["plan"] = {"executed": 1, "planned": len(items),
                                      "stopped": "no model"}
                return result
            if not isinstance(model, WorldModel):
                raise TypeError("action(actions, model=...): model must be a WorldModel")
            _last_model[:] = [model]
            result = {}
            for i, item in enumerate(items):
                before = runtime_globals.get("current_frame")
                try:
                    predicted = _predict(model, before, item)
                except Exception as exc:
                    predicted = exc
                if isinstance(predicted, BaseException):
                    record = _judge(model, before, before, item, predicted)
                    record["real"] = None  # the action was not run
                    runtime_globals["last_prediction"] = {"consistent": False, **_public(record)}
                    print(f"[model] action {i + 1} of {len(items)} ({record['action']}) was not run: "
                          + (f"the model's parse or step raised {record['error']}; fix it before "
                             "planning again." if _schema_parse else
                             f"model.step raised {record['error']}; fix step before planning again."))
                    if not _call_scope:
                        _schema_stop.append("model.step raised an error earlier in this Python call")
                    result = dict(result)
                    result["plan"] = {"executed": i, "planned": len(items), "stopped": "step error"}
                    return result
                start = len(_flat_history)
                if i:
                    del _stale_state[:]  # the model predicted this board, change or not
                result = action([item])
                entry = _action_entry(start)
                if entry is None:
                    result["plan"] = {"executed": i, "planned": len(items),
                                      "stopped": "the action was not executed"}
                    return result
                record = _judge(model, before, entry, item, predicted)
                runtime_globals["last_prediction"] = {"consistent": record["consistent"],
                                                      **_public(record)}
                if _const_parse_limit and record.get("real_state") is not None:
                    _observe_constant_parse(before, record, predicted)
                stop = None
                note = _coverage_note(record, i + 1, len(items))
                if note:
                    print(note)
                if not record["consistent"]:
                    stop = "mispredicted"
                    if not _call_scope:
                        _schema_stop.append(f"the model mispredicted {record['action']} earlier "
                                            "in this Python call (fix step first)")
                    dropped = len(items) - i - 1
                    print(f"[model] action {i + 1} of {len(items)} ({record['action']}) was MISPREDICTED "
                          f"({_where(record)}); "
                          + (f"the remaining {dropped} action(s) were dropped. " if dropped else "")
                          + _wrong_hint("your prediction")
                          + "; fix step before planning again.")
                else:
                    _audit(record)
                    if record.get("unpredicted"):
                        print(_unpredicted_note(record, i + 1, len(items)))
                    for flag in ("level_completed", "game_over", "run_complete", "done"):
                        if result.get(flag):
                            stop = flag
                            break
                if stop or i == len(items) - 1:
                    if stop is None and len(items) > 1:
                        print(f"[model] all {len(items)} actions ran as predicted.")
                    result["plan"] = {"executed": i + 1, "planned": len(items), "stopped": stop}
                    result["prediction"] = {"consistent": record["consistent"],
                                            "where": _where(record)}
                    return result
            return result

        if schema_enabled:
            _schema_globals.update({
                "WorldModel": WorldModel,
                "board": board,
                "check_prediction": check_prediction,
                "backtest": backtest,
                "action": _tracked_action,
                "ActionStoppedError": ActionStoppedError,
            })
            runtime_globals.update(_schema_globals)
            runtime_globals["last_prediction"] = None
            runtime_globals["last_backtest"] = None

        # ARC3_GRID_MAP: a stub the model replaces with its block map; the host shows
        # grid_map(current_frame) after every Python call
        _grid_map_view = bool(initial.get("grid_map"))
        # the name the host shows: grid_map, or harness_grid_map (ARC3_GRID_MAP_HOOK), which
        # the model sets to whichever of its map functions it wants shown
        _grid_map_name = initial.get("grid_map_name") or "grid_map"
        _GRID_MAP_MAX_CHARS = 3000

        def _grid_map_stub(frame):
            return None

        # ARC3_AUTO_MAP: grid_map starts as the harness's own block map instead of the stub,
        # for games whose board reads as a lattice of tiles (AutoMapState decides per game);
        # the model replaces it by defining its own grid_map
        _auto_map_on = _grid_map_view and bool(initial.get("auto_map"))
        _auto_map_state = AutoMapState() if _auto_map_on else None

        def _auto_map_context(frame):
            # (level, its initial grid, view scroll since then) for a board of the current
            # level, or None; the scroll sums the recorded view shifts of the level's actions
            history = runtime_globals.get("level_history") or []
            if not history or not history[-1] or frame is None:
                return None
            first = history[-1][0].frame
            if first is None or getattr(frame, "level", None) != first.level:
                return None
            dr = dc = 0
            for entry in history[-1][1:]:
                shift = (entry.result or {}).get("view_shift")
                if shift:
                    dr, dc = dr + shift[0], dc + shift[1]
            return first.level, first._grid, (dr, dc)

        def auto_grid_map(frame):
            # The harness's block map of a board of the current level, or None: one string
            # per row of tiles, one character per tile (its colour letter when uniform, '-'
            # '|' for a centre line, '+' 'x' for a cross, 'o' for a centred square, otherwise
            # its most common colour). None when the board does not read as a lattice.
            context = _auto_map_context(frame)
            if context is None:
                return None
            level, initial_grid, scroll = context
            return _auto_map_state.rows(level, initial_grid, frame._grid, scroll)

        if _grid_map_view:
            runtime_globals[_grid_map_name] = auto_grid_map if _auto_map_on else _grid_map_stub
            if _auto_map_on:
                runtime_globals["auto_grid_map"] = auto_grid_map

        def _collect_grid_map_sources():
            # ARC3_GRID_MAP: the source of the model's grid_map and of the functions of its
            # own it calls (cellstat, ...), for the host to restate after a context trim
            fn = runtime_globals.get(_grid_map_name)
            if (not _grid_map_view or fn is _grid_map_stub or fn is auto_grid_map
                    or not isinstance(fn, types.FunctionType)):
                return {}
            sources = _function_chain_sources([(_grid_map_name, fn)])
            if sources and fn.__name__ not in (_grid_map_name, "<lambda>"):
                # the hook names another function: restate the assignment with it
                sources[_grid_map_name] = f"{_grid_map_name} = {fn.__name__}"
            return {"grid_map_sources": sources} if sources else {}

        def _collect_grid_map():
            # {"grid_map": rows text, None while unset, False when the model set
            # grid_map = False (no block map for this game), or {"error": ...}}
            if not _grid_map_view:
                return {}
            fn = runtime_globals.get(_grid_map_name)
            frame = runtime_globals.get("current_frame")
            if fn is False:
                return {"grid_map": False}
            if fn is _grid_map_stub or not callable(fn) or frame is None:
                return {"grid_map": None}
            try:
                rows = fn(frame)
                if rows is None:
                    return {"grid_map": None}
                if isinstance(rows, str):
                    rows = rows.split("\n")
                rows = [row if isinstance(row, str) else "".join(str(ch) for ch in row)
                        for row in rows]
            except Exception as exc:
                return {"grid_map": {"error": _error_line(exc)}}
            text = "\n".join(rows)
            if len(text) > _GRID_MAP_MAX_CHARS:
                text = text[:_GRID_MAP_MAX_CHARS] + "\n... (cut)"
            if fn is auto_grid_map:
                # the host labels the harness's map with its lattice
                context = _auto_map_context(frame)
                lattice = _auto_map_state.describe(context[0], context[2]) if context else None
                if lattice:
                    return {"grid_map": text, "grid_map_auto": list(lattice)}
            return {"grid_map": text}
        if initial.get("animation_enabled"):
            _ENTRY_FRAMES["fetch"] = lambda index: _animation_fetch("entry_frames", index)
            _ENTRY_FRAMES["fetch_timeline"] = lambda index: _animation_fetch("entry_timeline", index)
            # Globals rather than attributes on last_transition: after a death
            # the harness appends a RESET entry, so last_transition is the
            # reset and not the action that animated. The record is simply
            # "the last executed action that animated", and naming it that way
            # removes the mismatch. Empty when the last executed action
            # returned a single frame.
            class _LazyAnimation:
                def __init__(self, kind, empty):
                    self._kind = kind
                    self._empty = empty
                    self._value = None

                def _get(self):
                    if self._value is None:
                        self._value = _animation_fetch(self._kind) or self._empty
                    return self._value

                def invalidate(self):
                    self._value = None

                # isinstance(last_animation_timeline, dict) must hold: model code
                # guards on it, and a False there skipped the whole read without
                # a word (bp35, rerun). Reports the wrapped type without fetching.
                @property
                def __class__(self):
                    return type(self._empty)

                def __getattr__(self, name):
                    # items(), values(), index() ... of the wrapped dict / list
                    if name.startswith("_"):
                        raise AttributeError(name)
                    return getattr(self._get(), name)

                def __len__(self):
                    return len(self._get())

                def __iter__(self):
                    return iter(self._get())

                def __getitem__(self, key):
                    return self._get()[key]

                def __bool__(self):
                    return bool(self._get())

                def __contains__(self, key):
                    return key in self._get()

                def keys(self):
                    return self._get().keys()

                def get(self, key, default=None):
                    return self._get().get(key, default)

                def __repr__(self):
                    return repr(self._get())

            runtime_globals["last_animation_frames"] = _LazyAnimation("frames", [])
            runtime_globals["last_animation_timeline"] = _LazyAnimation("timeline", {})
            _animation_views.extend((runtime_globals["last_animation_frames"],
                                     runtime_globals["last_animation_timeline"]))

        _refresh_state(initial.get("state") or {}, live=False)

        _persistence_enabled = bool(initial.get("persistence_enabled"))
        _retention_options = {
            "retain_imports": bool(initial.get("retain_imports")),
            "repair_hints": bool(initial.get("repair_hints")),
        }
        _kept_pool = {}
        _retention_rejected = []
        _available_names = set(runtime_globals) | set(runtime_globals["__builtins__"])
        if _persistence_enabled:
            _before_restore = dict(runtime_globals)
            try:
                _kept_pool, _retention_rejected = restore_functions(
                    initial.get("kept_functions") or [], runtime_globals, _available_names,
                    **_retention_options)
            except Exception:
                # Optional cache maintenance must never prevent this snippet running.
                runtime_globals.clear()
                runtime_globals.update(_before_restore)
                _kept_pool = {}
                _retention_rejected = [{"name": "all", "reason": "restoration failed; redefine needed functions"}]

        def _collect_show_requests():
            # boards requested with entry.show_image() during this snippet
            requests = list(_SHOW_IMAGE["requests"])
            _SHOW_IMAGE["requests"].clear()
            return {"show_images": requests} if requests else {}

        def _collect_retention_payload():
            if not _persistence_enabled:
                return {}
            try:
                kept, rejected = collect_functions(
                    _code_text, _kept_pool, runtime_globals, _available_names,
                    **_retention_options)
                return {
                    "keepable_functions": [{"name": n, "source": src} for n, src in kept.items()],
                    "retention_rejected": _retention_rejected + rejected,
                }
            except Exception:
                # Bookkeeping failures may drop helpers, but must never replace
                # the snippet's original error/output or replay its actions.
                return {"keepable_functions": [], "retention_rejected": [
                    {"name": "all", "reason": "retention failed; redefine needed functions"}]}

        previous_inventory = {}
        _code_text = ""
        _search_sources = weakref.WeakKeyDictionary()
        _def_sources = weakref.WeakKeyDictionary()  # ARC3_SCHEMA: code object -> source
        _search_tracking = bool(initial.get("search_tracking"))

        def _compile_snippet(tree, mode, source):
            # Code objects identify the definition that actually executed,
            # even after aliases, a failed redefinition, timeout or rollback.
            compiled = compile(tree, "<python_tool>", mode)
            if _fn_source_on and mode == "exec":
                lines = source.splitlines()
                _snippet_defs[:] = []
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
                        _snippet_defs.append(
                            (node.name, first, "\n".join(lines[first - 1:node.end_lineno])))
            if _search_tracking or schema_enabled or _grid_map_view:
                definitions = {
                    (n.name, min([n.lineno] + [d.lineno for d in n.decorator_list])):
                        ast.get_source_segment(source, n)
                    for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if schema_enabled:
                    # a key_state or step is often a lambda inside WorldModel(...)
                    for n in ast.walk(tree):
                        if isinstance(n, ast.Lambda):
                            definitions.setdefault(("<lambda>", n.lineno),
                                                   ast.get_source_segment(source, n))
                pending = [compiled]
                while pending:
                    obj = pending.pop()
                    text = definitions.get((obj.co_name, obj.co_firstlineno))
                    if text:
                        if _search_tracking and obj.co_name != "<lambda>":
                            _search_sources[obj] = text
                        if schema_enabled or _grid_map_view:
                            # the world model's chain, and grid_map's (ARC3_GRID_MAP)
                            _def_sources[obj] = text
                    pending.extend(c for c in obj.co_consts
                                   if isinstance(c, types.CodeType))
            return compiled

        def _collect_world_model_sources():
            # ARC3_SCHEMA: the model last used with action() / backtest() /
            # check_prediction(), with the source of its functions and of every
            # function of the model's own they call, directly or not
            if not schema_enabled or not _last_model:
                return {}
            model = _last_model[0]
            bound = sorted(name for name, value in runtime_globals.items()
                           if value is model and not name.startswith("_"))
            return {"world_model_sources": {"model": repr(model), "bound_to": bound,
                                            "sources": _chain_sources(model)}}

        def _chain_sources(model):
            # {name: source} of the model's functions and of the functions of the
            # model's own they call, directly or not, breadth first
            return _function_chain_sources(
                [(field, getattr(model, field))
                 for field in ("parse", "step", "key_state", "is_win_condition")
                 if getattr(model, field) is not None])

        def _function_chain_sources(roots):
            # {name: source} of the (label, function) roots and of the functions of the
            # model's own they call, directly or not, breadth first
            names = {}
            for name, value in runtime_globals.items():
                if isinstance(value, types.FunctionType) and not name.startswith("_"):
                    names.setdefault(value, name)
            queue = list(roots)
            sources, seen = {}, set()
            while queue:
                label, fn = queue.pop(0)
                if not isinstance(fn, types.FunctionType) or fn in seen:
                    continue
                seen.add(fn)
                text = _def_sources.get(fn.__code__)
                if text is None:
                    continue  # not defined in a snippet (a harness name, say)
                name = names.get(fn) or label
                if fn.__name__ == "<lambda>":
                    text = f"{label} = {text}"
                elif name != fn.__name__:
                    name = f"{name} (= {fn.__name__})"
                sources[name] = text
                codes, called = [fn.__code__], []
                while codes:
                    code = codes.pop()
                    called.extend(code.co_names)
                    codes.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
                for ref in called:
                    value = runtime_globals.get(ref)
                    if isinstance(value, types.FunctionType):
                        queue.append((ref, value))
            return sources

        def _auto_backtest(tree):
            # ARC3_SCHEMA: after a snippet that built a WorldModel, or redefined a
            # function of the last used model's chain, judge that model on this
            # level's recorded transitions (at most 5 s) and print one line
            if not schema_enabled:
                return
            defined = {n.name for n in tree.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            model, why = None, ""
            if _constructed:
                model, why = _constructed[-1], "this call built it"
            elif _last_model and defined:
                hit = sorted(defined & {k.split(" ")[0] for k in _chain_sources(_last_model[0])})
                if hit:
                    model, why = _last_model[0], "this call redefined " + ", ".join(hit)
            frame = runtime_globals.get("current_frame")
            if _schema_parse and frame is not None and _level_scrolled(frame.level):
                return  # ARC3_SCHEMA_PARSE: a scrolled level's plans are not judged either
            if model is None or any(m is model for m in _backtested):
                return
            _last_model[:] = [model]  # the model a trim restates
            try:
                result = _backtest(model, -1, time.monotonic() + 5)
            except Exception as exc:
                print(f"[model] automatic backtest of {model!r} failed: {_error_line(exc)}")
                return
            if not result["checked"]:
                return
            text = (f"[model] automatic backtest of {model!r} ({why}): {result['correct']}/"
                    f"{result['checked']} recorded transitions of this level predicted")
            if result.get("stopped_early"):
                text += " (stopped after 5 s)"
            if result["wrong_indices"]:
                latest = result["recent"][-1]
                ks = [k for _, k in result["wrong_indices"]]
                text += (f"; wrong at level_history[{latest['index'][0]}][k] for k in {ks[:12]}"
                         + (" ..." if len(ks) > 12 else "")
                         + f"; the latest, k={latest['index'][1]} {latest['action']}: {latest['where']}."
                         " last_backtest['wrong'] holds every wrong one")
            if result.get("uncovered_indices"):
                # ARC3_SCHEMA_PARSE: board changes parse missed, reported, never stopped
                missed = runtime_globals["last_backtest"]["uncovered"]
                last = missed[-1]
                ks = [k for _, k in result["uncovered_indices"]]
                text += (f"; board changes parse missed at level_history[{last['index'][0]}][k] "
                         f"for k in {ks}; the latest, k={last['index'][1]} {last['action']}: "
                         + "; ".join(_uncovered_parts(last["uncovered"])[:3])
                         + ". last_backtest['uncovered'] holds every one")
            print(text + ".")

        def _collect_search_sources():
            if not _search_tracking:
                return {}
            return {"search_function_sources": {
                name: _search_sources[value.__code__]
                for name, value in runtime_globals.items()
                if isinstance(value, types.FunctionType)
                and value.__code__ in _search_sources
            }}

        def _commit_fn_sources():
            # ARC3_FUNCTION_SOURCE: keep the source of each def / class of the snippet
            # that actually executed (an error before the def leaves the old one), and
            # drop names that no longer hold a function or class
            def _function(value):
                # the function under decorators such as functools.lru_cache
                for _ in range(8):
                    if isinstance(value, types.FunctionType):
                        return value
                    value = getattr(value, "__wrapped__", None)
                return None

            for name, first, text in _snippet_defs:
                value = runtime_globals.get(name)
                function = _function(value)
                if function is not None:
                    if (function.__code__.co_filename == "<python_tool>"
                            and function.__code__.co_firstlineno == first):
                        _fn_sources[name] = text
                elif isinstance(value, type) and getattr(value, "__module__", None) == "__python_tool__":
                    _fn_sources[name] = text
            _snippet_defs[:] = []
            for name in list(_fn_sources):
                value = runtime_globals.get(name)
                if _function(value) is None and not isinstance(value, type):
                    del _fn_sources[name]
            runtime_globals["fn_source"] = _FnSource(_fn_sources)

        def _namespace_inventory():
            # Return the kind of every model-created name. Complete, never
            # truncated: deltas are computed from it, and a truncated list
            # reports names that merely fell off the end as removed. The
            # harness-provided names are described in the prompt instead.
            if _fn_source_on:
                _commit_fn_sources()
            inventory = {}
            for name, value in runtime_globals.items():
                if (name.startswith("_") or name == "__builtins__"
                        or name in _protected_names):
                    continue
                if isinstance(value, types.FunctionType):
                    kind = "function"
                elif isinstance(value, type):
                    kind = "class"
                elif isinstance(value, types.ModuleType):
                    kind = "module"
                else:
                    kind = "value"
                inventory[str(name)] = kind
            return dict(sorted(inventory.items()))

        def _namespace_sizes(inventory):
            # Approximate bytes per data variable, for ordering the display.
            return {
                name: _deep_size(runtime_globals.get(name), node_budget=20_000)
                for name, kind in inventory.items()
                if kind == "value"
            }

        def _namespace_delta(current):
            added = [name for name in current if name not in previous_inventory]
            removed = [name for name in previous_inventory if name not in current]
            changed = [
                name for name in current
                if name in previous_inventory and current[name] != previous_inventory[name]
            ]
            return {
                "added": [f"{name} ({current[name]})" for name in sorted(added)],
                "removed": sorted(removed),
                "changed": [f"{name} ({current[name]})" for name in sorted(changed)],
            }

        _protected_names = set()  # harness-provided names; filled before the first snippet
        deque_type = __import__("collections").deque

        def _deep_size(value, node_budget=2_000_000):
            # Approximate bytes reachable from `value` (containers and object
            # attributes), stopping after `node_budget` objects; functions,
            # classes and modules count as their own header only.
            seen_ids, stack, total, nodes = set(), [value], 0, 0
            while stack and nodes < node_budget:
                obj = stack.pop()
                if id(obj) in seen_ids:
                    continue
                seen_ids.add(id(obj))
                nodes += 1
                # the lazy last_animation_* wrappers claim to be a dict / list;
                # walking one would fetch from the host, which can be mid-reply
                # to an action a timeout abandoned (all25 v2: worker died)
                if type(obj).__name__ == "_LazyAnimation":
                    continue
                try:
                    total += sys.getsizeof(obj)
                except Exception:
                    continue
                if isinstance(obj, (str, bytes, bytearray, int, float, bool, type(None),
                                    types.FunctionType, types.ModuleType, type)):
                    continue
                if isinstance(obj, dict):
                    stack.extend(obj.keys())
                    stack.extend(obj.values())
                elif isinstance(obj, (list, tuple, set, frozenset, deque_type)):
                    stack.extend(obj)
                else:
                    attrs = getattr(obj, "__dict__", None)
                    if isinstance(attrs, dict):
                        stack.append(attrs)
            return total

        def _rss_bytes():
            try:
                with open("/proc/self/statm") as handle:
                    return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except Exception:
                return 0

        def _trim_heap():
            gc.collect()
            with contextlib.suppress(Exception):
                ctypes.CDLL("libc.so.6").malloc_trim(0)

        def _enforce_idle_memory():
            # Called after every snippet: the process may not idle above the
            # budget, because a paused game keeps its worker for a long time.
            # Freed memory is first returned to the OS; if the namespace itself
            # is too big, the largest model-created variables are deleted.
            if not idle_limit_bytes:
                return None
            rss = _rss_bytes()
            if not rss or rss <= idle_limit_bytes:
                return None
            _trim_heap()
            rss = _rss_bytes()
            if rss <= idle_limit_bytes:
                return None
            sizes = sorted(
                (
                    # enough to tell variables over 1 MB apart; a full walk of a
                    # search table takes long enough to miss the host's deadline
                    (_deep_size(value, node_budget=200_000), name)
                    for name, value in runtime_globals.items()
                    if not name.startswith("_") and name not in _protected_names
                    and not isinstance(value, (types.FunctionType, types.ModuleType, type))
                ),
                reverse=True,
            )
            evicted = []
            for size, name in sizes:
                if rss <= idle_limit_bytes or size < 1024 * 1024:
                    break
                runtime_globals.pop(name, None)
                evicted.append(f"{name} (~{size // (1024 * 1024)} MB)")
                _trim_heap()
                rss = _rss_bytes()
            return {
                "rss_mb": rss // (1024 * 1024),
                "idle_limit_mb": idle_limit_bytes // (1024 * 1024),
                "evicted": evicted,
            }

        def _archive_namespace():
            # Keep a best-effort checkpoint of the small variables before
            # executing model code; large ones (search tables) are skipped so
            # the checkpoint cannot double the process's memory.
            archive = {}
            for name, value in runtime_globals.items():
                if name.startswith("_") or name == "__builtins__" or name in _protected_names:
                    continue
                if isinstance(value, (types.FunctionType, types.ModuleType, type)):
                    archive[name] = value
                    continue
                if _deep_size(value, node_budget=200_000) > 16 * 1024 * 1024:
                    continue
                try:
                    archive[name] = copy.deepcopy(value)
                except Exception:
                    archive[name] = value
            return archive

        def _run_code(code_text):
            # Execute one request while leaving persistent globals untouched.
            nonlocal stdout, previous_inventory, _code_text
            _code_text = str(code_text)
            del action_results[:]
            del _stale_state[:]
            del _noop_run[:]
            del _terminal_state[:]
            del _schema_stop[:]
            del _free_used[:]
            del _const_parse_due[:]
            del _constructed[:]
            del _backtested[:]
            del _backtest_note[:]
            del _diff_audit[:]
            stdout = io.StringIO()
            # `result` reports this snippet's value; a stale one from an
            # earlier snippet would be re-sent and block the last-expression
            # fallback below.
            runtime_globals["result"] = None
            runtime_globals["_python_process_archive"] = _archive_namespace()
            timeout_handler = None
            timer_started = False
            portable_timer = None
            if persistent_mode:
                signal.signal(signal.SIGINT, signal.default_int_handler)
            if persistent_mode and hasattr(signal, "setitimer"):
                def _raise_timeout(_signum, _frame):
                    raise TimeoutError(
                        f"Python tool timed out after {timeout_seconds}s; definitions and "
                        "intermediate variables created before the timeout remain available."
                    )
                timeout_handler = signal.signal(signal.SIGALRM, _raise_timeout)
                signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
                timer_started = True
            elif persistent_mode:
                # Windows has no SIGALRM. Injecting into the main interpreter
                # thread keeps the worker alive and preserves definitions that
                # executed before the timeout (including while `time.sleep`
                # releases the GIL).
                main_thread_id = threading.get_ident()

                def _inject_timeout():
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(main_thread_id), ctypes.py_object(TimeoutError)
                    )

                portable_timer = threading.Timer(timeout_seconds, _inject_timeout)
                portable_timer.daemon = True
                portable_timer.start()

            def _stop_timer():
                # The timeout and the host interrupt are for the model's code
                # only: one landing in the bookkeeping after it (sizing a large
                # namespace, say) would escape the handlers and kill the worker.
                if persistent_mode:
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                if timer_started:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                if portable_timer is not None:
                    portable_timer.cancel()

            try:
                _tree = ast.parse(str(code_text), "<python_tool>")
                _last_expr = None
                if _tree.body and isinstance(_tree.body[-1], ast.Expr):
                    _last_expr = ast.Expression(_tree.body[-1].value)
                    ast.copy_location(_last_expr, _tree.body[-1])
                    _tree.body = _tree.body[:-1]
                with contextlib.redirect_stdout(stdout):
                    if _tree.body:
                        exec(
                            _compile_snippet(_tree, "exec", str(code_text)),
                            runtime_globals,
                            runtime_globals,
                        )
                    if _last_expr is not None:
                        _value = eval(
                            _compile_snippet(_last_expr, "eval", str(code_text)),
                            runtime_globals,
                            runtime_globals,
                        )
                        if _value is not None:
                            print(repr(_value))
                            if runtime_globals.get("result") is None:
                                runtime_globals["result"] = _value
                    _auto_backtest(_tree)
                _stop_timer()
                result_payload = _json_safe(runtime_globals.get("result"))
                memory_note = _enforce_idle_memory()
                current = _namespace_inventory()
                _send({
                    "type": "final",
                    **_collect_retention_payload(),
                    **_collect_show_requests(),
                    **_collect_search_sources(),
                    **_collect_world_model_sources(),
                    **_collect_grid_map(),
                    **_collect_grid_map_sources(),
                    **({"const_parse_warn": True} if _const_parse_due else {}),
                    **({"schema_note": _backtest_note[0]} if _backtest_note else {}),
                    **({"diff_audit": list(_diff_audit)} if _diff_audit else {}),
                    "stdout": stdout.getvalue(),
                    "result": result_payload,
                    "action_results": _json_safe(action_results),
                    "namespace_inventory": current,
                    "namespace_sizes": _namespace_sizes(current),
                    "namespace_delta": _namespace_delta(current),
                    "archive_created": True,
                    "memory": memory_note,
                })
                previous_inventory = current
            except (
                KnownNoOpActionError,
                RepeatedActionInStateError,
                NoOpRunActionError,
                KnownDeathActionError,
                TerminalStateActionError,
                ActionStoppedError,
                TimeoutError,
                KeyboardInterrupt,
                Exception,
            ) as exc:
                _stop_timer()
                # Under the address-space cap an allocation can also fail where
                # CPython loses the MemoryError and reports this SystemError
                # instead; which one surfaces depends on the memory layout.
                lost_memory_error = bool(
                    memory_limit_bytes
                    and isinstance(exc, SystemError)
                    and "without setting an exception" in str(exc)
                )
                out_of_memory = isinstance(exc, MemoryError) or lost_memory_error
                if out_of_memory:
                    # The traceback still references the frames of the code
                    # that ran out of memory (a search's visited set, say).
                    # Free those locals and the reserve before building the
                    # reply, which otherwise cannot be allocated either.
                    del _memory_reserve[:]
                    traceback.clear_frames(exc.__traceback__)
                    gc.collect()
                try:
                    error_text = _sanitize_exception(exc)
                    if lost_memory_error:
                        error_text = f"MemoryError (Python reported: {exc})"
                    if out_of_memory:
                        limit = (
                            f" ({memory_limit_bytes // (1024 * 1024)} MB)"
                            if memory_limit_bytes else ""
                        )
                        error_text += (
                            "\nThis snippet exceeded the Python process memory limit"
                            f"{limit} and was stopped. Variables and functions defined "
                            "before it are kept. Bound the search (depth or number of "
                            "states), store compact state keys, and `del` large "
                            "variables you no longer need."
                        )
                    memory_note = _enforce_idle_memory()
                    current = _namespace_inventory()
                    _send({
                        "type": "error",
                        **_collect_retention_payload(),
                        **_collect_show_requests(),
                        **_collect_search_sources(),
                        **_collect_world_model_sources(),
                        **_collect_grid_map(),
                        **_collect_grid_map_sources(),
                        **({"const_parse_warn": True} if _const_parse_due else {}),
                        **({"schema_note": _backtest_note[0]} if _backtest_note else {}),
                        **({"diff_audit": list(_diff_audit)} if _diff_audit else {}),
                        "memory": memory_note,
                        "error": error_text,
                        "stdout": stdout.getvalue(),
                        "action_results": _json_safe(action_results),
                        "namespace_inventory": current,
                        "namespace_sizes": _namespace_sizes(current),
                        "namespace_delta": _namespace_delta(current),
                        "archive_created": True,
                        "timed_out": isinstance(exc, TimeoutError),
                        "interrupted": isinstance(exc, KeyboardInterrupt),
                        "out_of_memory": out_of_memory,
                    })
                    previous_inventory = current
                except MemoryError:
                    # Still short of memory (a large global holds it): send
                    # the smallest reply that keeps the protocol in step.
                    gc.collect()
                    _send({
                        "type": "error",
                        "error": "MemoryError: the Python process is out of memory; "
                                 "`del` large variables. Definitions are kept.",
                        "action_results": [],
                        "out_of_memory": True,
                    })
                if not _memory_reserve:
                    with contextlib.suppress(MemoryError):
                        _memory_reserve.append(bytearray(8 * 1024 * 1024))
            finally:
                if persistent_mode:
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                if timer_started:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, timeout_handler)
                if portable_timer is not None:
                    portable_timer.cancel()

        _protected_names.update(runtime_globals)
        if world_map_enabled:
            _protected_names.add("world_map")
        if persistent_mode:
            # A host interrupt that arrives after the snippet has already
            # finished must not kill the idle worker (it would otherwise raise
            # KeyboardInterrupt inside _recv and take the namespace with it).
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        _run_code(initial.get("code", ""))
        if persistent_mode:
            while True:
                request = _recv()
                request_type = str(request.get("type", "")).strip()
                if request_type == "shutdown":
                    break
                if request_type != "execute":
                    # A reply to an action or animation request that an interrupt
                    # abandoned arrives here; answering it would hand the host a
                    # response for a request it is no longer waiting on.
                    continue
                timeout_seconds = max(1, int(request.get("timeout_seconds", timeout_seconds)))
                _refresh_state(request.get("state") or {})
                _run_code(request.get("code", ""))


    if __name__ == "__main__":
        main()
    """
).replace("__RETAINED_FUNCTIONS_SOURCE__\n", textwrap.dedent(inspect.getsource(_retained_functions))).replace(
    "__SEGMENTATION_SOURCE__\n", textwrap.dedent(inspect.getsource(_segmentation))
).replace(
    "__FRAME_DIFF_SOURCE__\n", textwrap.dedent(inspect.getsource(_frame_diff_module))
).replace(
    "__AUTO_MAP_SOURCE__\n", textwrap.dedent(inspect.getsource(_auto_map_module))
)



log = logging.getLogger(__name__)


def replay_enabled() -> bool:
    """ARC3_REPLAY: `replay()` and `.attempt` on fatal entries (default off)."""
    return os.environ.get("ARC3_REPLAY", "0").strip().lower() in ("1", "true", "yes", "on")


def schema_enabled() -> bool:
    """ARC3_SCHEMA: `WorldModel`, checked `action(actions, model=...)` (default off)."""
    return os.environ.get("ARC3_SCHEMA", "0").strip().lower() in ("1", "true", "yes", "on")


def schema_parse_enabled() -> bool:
    """ARC3_SCHEMA_PARSE: the world model is WorldModel(parse, step, is_win_condition).

    Default off. parse(frame) turns a board into a dict of objects and values, step moves
    that dict, and a prediction is right when it equals parse of the real next board; no
    board is predicted or compared cell by cell. Changes the dict's objects do not cover
    (by their 'bbox' / 'cells') are reported, never stopped.
    """
    return schema_enabled() and os.environ.get("ARC3_SCHEMA_PARSE", "0").strip().lower() in (
        "1", "true", "yes", "on")


def schema_parse_map_enabled() -> bool:
    """ARC3_SCHEMA_PARSE_MAP: parse must return a block map under 'map' (default off).

    Needs ARC3_SCHEMA_PARSE. A parsed state without a non-empty 'map' (a list of strings,
    one character per block) fails the prediction, so the rest of the plan is dropped;
    levels whose view has scrolled are exempt."""
    return schema_parse_enabled() and os.environ.get(
        "ARC3_SCHEMA_PARSE_MAP", "0").strip().lower() in ("1", "true", "yes", "on")


def grid_map_name() -> str:
    """The function name the host shows each message: grid_map, or harness_grid_map.

    ARC3_GRID_MAP_HOOK (default off): the model writes its map function under any name and
    sets `harness_grid_map = that_function`; the host shows harness_grid_map(current_frame)."""
    hook = os.environ.get("ARC3_GRID_MAP_HOOK", "0").strip().lower() in ("1", "true", "yes", "on")
    return "harness_grid_map" if hook else "grid_map"


def function_source_enabled() -> bool:
    """ARC3_FUNCTION_SOURCE: `fn_source['name']` gives the source of the model's own definitions (default off).

    A context trim drops the snippets that defined the model's helpers while the helpers stay
    in the persistent namespace; `inspect.getsource` cannot read them (they come from exec'd
    text), so the model rewrote them from memory (su15, s5i5, m0r0, tu93 and re86 in the 4 px
    all25: "Can't use inspect", "I don't remember them exactly"). With this switch the worker
    keeps the text of the latest executed top-level `def` / `class` of every name, and the
    namespace listing shown after a trim says how to read it."""
    return os.environ.get("ARC3_FUNCTION_SOURCE", "0").strip().lower() in ("1", "true", "yes", "on")


def grid_map_view_enabled() -> bool:
    """ARC3_GRID_MAP: show the model's own grid_map(current_frame) in every message (default off).

    The namespace starts with a stub `grid_map(frame)` returning None; the model replaces it
    with a function that returns its block map (a list of strings). Until it does, each
    message carries a reminder of how to build one. Nothing is refused or judged."""
    return os.environ.get("ARC3_GRID_MAP", "0").strip().lower() in ("1", "true", "yes", "on")


def auto_map_enabled() -> bool:
    """ARC3_AUTO_MAP: grid_map starts as the harness's block map (default off).

    Needs ARC3_GRID_MAP. For games whose board reads as a lattice of tiles (detected once per
    game, see inference/utils/auto_map.py), the namespace's grid_map is `auto_grid_map`
    instead of the stub, so every message shows the harness's map until the model defines its
    own grid_map; other games keep the stub and its reminder."""
    return grid_map_view_enabled() and os.environ.get(
        "ARC3_AUTO_MAP", "0").strip().lower() in ("1", "true", "yes", "on")


def _schema_report_diff() -> bool:
    """ARC3_SCHEMA_REPORT_DIFF: name what an action judged right changed beyond step.

    Default off. It never stops a plan. frame_diff(pred, real) objects are named: every
    moved or rotated object (e.g. ls20's corner logo, which the 9-cell stop tolerance
    misses), and appeared, disappeared, recolored or resized objects of 8+ pixels.
    """
    return os.environ.get("ARC3_SCHEMA_REPORT_DIFF", "0").strip().lower() in (
        "1", "true", "yes", "on")


def schema_diff_audit_enabled() -> bool:
    """ARC3_SCHEMA_DIFF_AUDIT: log every judged-right action's frame_diff (default off).

    The log is for offline study of what ARC3_SCHEMA_REPORT_DIFF filters out; the model
    never sees it. Each executed Python call's records come back as `diff_audit`.
    """
    return os.environ.get("ARC3_SCHEMA_DIFF_AUDIT", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _schema_tolerance() -> int:
    """ARC3_SCHEMA_TOLERANCE: the smallest connected board difference that is wrong."""
    try:
        return max(1, int(os.environ.get("ARC3_SCHEMA_TOLERANCE", "9")))
    except ValueError:
        return 9


def schema_stop_mode() -> str:
    """ARC3_SCHEMA_STOP: "snippet" (1, the default), "call" or "report" (0).

    snippet: after an unchecked, mispredicted or failed action the Python call may not act
    again; call: a wrong prediction drops only the rest of its own call, and only a second
    call without a model is refused; report: nothing is refused or dropped."""
    value = os.environ.get("ARC3_SCHEMA_STOP", "1").strip().lower()
    if value == "call":
        return "call"
    return "snippet" if value in ("1", "true", "yes", "on") else "report"


def schema_stop_enabled() -> bool:
    """Whether actions are ever refused or dropped (ARC3_SCHEMA_STOP other than 0)."""
    return schema_stop_mode() != "report"


def const_parse_warn() -> int:
    """ARC3_CONST_PARSE_WARN: calls in a row with a constant parse before a warning (0 = off)."""
    try:
        return max(0, int(os.environ.get("ARC3_CONST_PARSE_WARN", "0")))
    except ValueError:
        return 0


def schema_free_actions() -> int:
    """ARC3_SCHEMA_FREE_ACTIONS: actions a Python call may run without a model (default 1)."""
    try:
        return max(1, int(os.environ.get("ARC3_SCHEMA_FREE_ACTIONS", "1")))
    except ValueError:
        return 1


def snippet_noop_limit() -> int:
    """ARC3_SNIPPET_NOOP_LIMIT: no-ops in a row after which a snippet may not act (0 = off).

    The batch no-op stop (ARC3_BATCH_NOOP_BLOCK) only cuts a single
    action([...]) call. A loop that calls action() once per step is never cut,
    and every no-op in it still costs a game action: wa30 level 4 ran 55 of
    them in search loops that counted their own stalls (frames15 run, 10-28).
    With a limit N, once N executed actions in a row in one snippet changed
    nothing inside the board area, every later action() call of that snippet
    raises NoOpRunActionError instead of executing. A change resets the count;
    the next snippet starts clear."""
    try:
        return max(0, int(os.environ.get("ARC3_SNIPPET_NOOP_LIMIT", "0")))
    except ValueError:
        return 0


def _schema_miss_budget() -> int:
    """ARC3_SCHEMA_MISS_BUDGET: actions a Python call may run unverified or mispredicted."""
    try:
        return max(1, int(os.environ.get("ARC3_SCHEMA_MISS_BUDGET", "1")))
    except ValueError:
        return 1


def _schema_crop_chars() -> int:
    """ARC3_SCHEMA_CROP_MAX_CHARS: the most text a backtest's latest-error note takes."""
    try:
        return max(300, int(os.environ.get("ARC3_SCHEMA_CROP_MAX_CHARS", "1500")))
    except ValueError:
        return 1500


def _show_image_enabled() -> bool:
    """ARC3_SHOW_IMAGE: entries get `.show_image()`, shown in the next message (default off)."""
    return os.environ.get("ARC3_SHOW_IMAGE", "0").strip().lower() in ("1", "true", "yes", "on")


def _world_map_enabled() -> bool:
    """Whether `world_map` is kept in the namespace (ARC3_WORLD_MAP, default on; 0 disables).

    It maps every cell seen since the level's last restart to its color, in the
    level's starting-view coordinates, updating the same dict in place."""
    return _view_shift_enabled() and os.environ.get("ARC3_WORLD_MAP", "1").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _view_shift_enabled() -> bool:
    """Whether camera scrolls are detected (ARC3_VIEW_SHIFT, default on; 0 disables).

    With it, frame_diff() aligns the earlier board to the current view when the
    two differ by a whole-view translation, and reports the shift."""
    return os.environ.get("ARC3_VIEW_SHIFT", "1").strip().lower() in ("1", "true", "yes", "on")


def _sanitize_host_error_text(text: str) -> str:
    """What the MODEL is told when the sandbox process dies.

    Deliberately fixed: the child's traceback carries host paths and internals
    that have no business in the model's context. But the text was previously
    read and thrown away entirely, so a crash left no trace anywhere - the one
    thing that knows what went wrong said nothing. It now goes to the harness
    log instead, where it is useful and nobody is prompted by it.
    """
    detail = str(text or "").strip()
    if detail:
        log.warning(
            "sandbox process died; stderr follows:\n%s",
            detail[-4000:],
        )
    return "Sandbox process exited unexpectedly."


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
        # Keep numerical libraries imported by model code from consuming all
        # host cores.  The sandbox itself is intentionally single-worker.
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except OSError:
        return

    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _write_bootstrap(directory: str) -> str:
    """Write the generated child program so Windows avoids command-line limits."""
    path = os.path.join(directory, "python_tool_worker.py")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_SANDBOX_BOOTSTRAP)
    return path


class PersistentPythonProcess:
    """A per-game Python worker that keeps the model namespace alive."""

    def __init__(
        self,
        *,
        initial_state: dict[str, Any],
        action_handler: Callable[..., dict[str, Any]],
        animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None,
        timeout_seconds: int,
        memory_limit_bytes: int = 2048 * 1024 * 1024,
        cpu_limit_seconds: int = 300,
        idle_limit_bytes: int = 0,
    ) -> None:
        self._action_handler = action_handler
        self._animation_handler = animation_handler
        self._timeout_seconds = max(1, int(timeout_seconds))
        self._initial_state = initial_state
        self._memory_limit_bytes = max(0, int(memory_limit_bytes))
        self._idle_limit_bytes = max(0, int(idle_limit_bytes))
        self._cpu_limit_seconds = max(1, int(cpu_limit_seconds))
        self._process: subprocess.Popen[str] | None = None
        self._sandbox_dir: str | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._started = False
        self._closed = False
        # ARC3_STATE_DELTA: the history items last sent, and how many of them
        # the worker reported holding in its latest message
        self._sent_history: list[Any] = []
        self._worker_history_len = 0
        self._start_process()

    def _history_delta(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return `state` with only the history items the worker lacks."""
        history = state.get("history") if isinstance(state, dict) else None
        if not state_delta_enabled() or not isinstance(history, list):
            return state
        keep = 0
        held = self._sent_history[:self._worker_history_len]
        for old, new in zip(held, history):
            # items are cached per entry, so an unchanged entry is the same object
            if old is not new:
                break
            keep += 1
        self._sent_history = list(history)
        return {**state, "history": history[keep:], "history_from": keep}

    def _restart(self) -> None:
        """Replace a dead or wedged worker with a fresh, empty one."""
        process = self._process
        if process is not None and process.poll() is None:
            _kill_process_group(process)
            _wait_for_process_exit(process, timeout=1.0)
        if self._sandbox_dir:
            shutil.rmtree(self._sandbox_dir, ignore_errors=True)
            self._sandbox_dir = None
        self._stdout_queue = queue.Queue()
        self._started = False
        self._sent_history = []
        self._worker_history_len = 0
        self._start_process()

    def _start_process(self) -> None:
        self._sandbox_dir = tempfile.mkdtemp(prefix="rgb_python_process_")
        try:
            bootstrap_path = _write_bootstrap(self._sandbox_dir)
            creation_flags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if os.name == "nt" else 0
            )
            self._process = subprocess.Popen(
                [sys.executable, "-I", "-S", bootstrap_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self._sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise RuntimeError("Sandbox process could not start.") from exc
        assert self._process.stdout is not None

        process, lines = self._process, self._stdout_queue

        def _stdout_reader() -> None:
            # Bound to this worker's pipe and queue so a restart never mixes
            # output from the replaced worker into the new one.
            for raw_line in process.stdout:
                lines.put(raw_line)
            lines.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

    def _send(self, payload: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Sandbox process is unavailable.")
        _send_json_line(self._process.stdin, payload)

    def _interrupt(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name != "nt":
                process.send_signal(signal.SIGINT)
        except (OSError, ValueError):
            pass

    def _read_response(
        self,
        *,
        timeout_seconds: int,
        should_stop: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        process = self._process
        if process is None:
            return {"error": "Sandbox process is unavailable.", "action_results": []}
        host_action_results: list[dict[str, Any]] = []
        # The worker's own timer fires first; the host interrupt backs it up for
        # code that swallows it, and a restart backs up the interrupt.
        deadline = time.monotonic() + max(1, int(timeout_seconds)) + 1.0
        interrupted = False
        expired = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if interrupted:
                    self._restart()
                    return {
                        "error": (
                            f"Tool timed out after {timeout_seconds}s and did not stop "
                            "when interrupted, so the Python process was restarted. Its "
                            "namespace is now empty; redefine what you need."
                        ),
                        "stdout": "",
                        "action_results": host_action_results,
                        "timed_out": True,
                        "restarted": True,
                    }
                self._interrupt()
                interrupted = True
                expired = True
                deadline = time.monotonic() + 2.0
                remaining = 2.0
            if should_stop is not None and not interrupted:
                try:
                    if should_stop():
                        self._interrupt()
                        interrupted = True
                        deadline = time.monotonic() + 2.0
                except Exception:  # noqa: BLE001
                    pass
            try:
                line = self._stdout_queue.get(timeout=min(0.2, max(0.01, remaining)))
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read() if process.stderr is not None else ""
                self._restart()
                return {
                    "error": (
                        _sanitize_host_error_text(stderr)
                        + " The Python process was restarted with an empty namespace "
                        "(it may have hit its memory or CPU limit); redefine what you need."
                    ),
                    "stdout": "",
                    "action_results": host_action_results,
                    "restarted": True,
                }
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._interrupt()
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": host_action_results,
                }
            if isinstance(message, dict):
                self._worker_history_len = int(message.get("history_len") or 0)
            msg_type = str(message.get("type", "")).strip()
            if msg_type == "animation":
                if self._animation_handler is None:
                    self._send({"type": "animation_error", "error": "Animation unavailable."})
                    continue
                try:
                    result = self._animation_handler(dict(message.get("request") or {}))
                except Exception:  # noqa: BLE001
                    result = {"error": "animation failed in sandbox host."}
                self._send({"type": "animation_result", "animation": result})
                continue
            if msg_type == "action":
                try:
                    result = self._action_handler(
                        list(message.get("actions") or []),
                        stale_after=message.get("stale_after"),
                    )
                except Exception:  # noqa: BLE001
                    self._send({"type": "action_error", "error": "action failed in sandbox host."})
                    continue
                raw = result.get("action_result") or {}
                if isinstance(raw, dict):
                    host_action_results.append(dict(raw))
                self._send({
                    "type": "action_result",
                    "action_result": raw,
                    "state": self._history_delta(result.get("state") or {}),
                })
                continue
            if msg_type in {"final", "error"}:
                timed_out = bool(message.get("timed_out")) or expired
                error = str(message.get("error", "") or "")
                if expired and not error:
                    error = (
                        f"Tool timed out after {timeout_seconds}s; execution was interrupted. "
                        "Definitions and intermediate values created before the interruption "
                        "remain available when the worker resumes."
                    )
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": error,
                    "action_results": list(message.get("action_results") or host_action_results),
                    "namespace_inventory": dict(message.get("namespace_inventory") or {}),
                    "namespace_sizes": dict(message.get("namespace_sizes") or {}),
                    "namespace_delta": dict(message.get("namespace_delta") or {}),
                    "archive_created": bool(message.get("archive_created")),
                    "timed_out": timed_out,
                    "interrupted": bool(message.get("interrupted")),
                    "out_of_memory": bool(message.get("out_of_memory")),
                    "memory": message.get("memory"),
                    "show_images": list(message.get("show_images") or []),
                    **({"search_function_sources": message["search_function_sources"]}
                       if "search_function_sources" in message else {}),
                    **({"world_model_sources": message["world_model_sources"]}
                       if "world_model_sources" in message else {}),
                    **({"grid_map": message["grid_map"]} if "grid_map" in message else {}),
                    **({"grid_map_auto": message["grid_map_auto"]}
                       if "grid_map_auto" in message else {}),
                    **({"grid_map_sources": message["grid_map_sources"]}
                       if "grid_map_sources" in message else {}),
                    **({"const_parse_warn": True} if message.get("const_parse_warn") else {}),
                    **({"schema_note": message["schema_note"]}
                       if "schema_note" in message else {}),
                    **({"diff_audit": message["diff_audit"]}
                       if "diff_audit" in message else {}),
                }
            return {
                "error": "Sandbox process returned an unknown message type.",
                "action_results": host_action_results,
            }

    def execute(
        self,
        *,
        code: str,
        state: dict[str, Any],
        timeout_seconds: int | None = None,
        should_stop: Callable[[], bool] | None = None,
        action_handler: Callable[..., dict[str, Any]] | None = None,
        animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Execute code in the live worker and return its compact response.

        The namespace persists across calls, but the action and animation
        handlers belong to the call: the host builds them per snippet with
        per-snippet guard state (e.g. "a level completed earlier in this
        snippet"), so passing them here keeps that state from leaking into
        later snippets. Without them the handlers given at construction are
        used.
        """
        if action_handler is not None:
            self._action_handler = action_handler
        if animation_handler is not None:
            self._animation_handler = animation_handler
        if self._closed:
            return {"error": "Sandbox process is closed.", "action_results": []}
        timeout = max(1, int(timeout_seconds or self._timeout_seconds))
        if self._process is None or self._process.poll() is not None:
            self._restart()
        try:
            self._send_request(code, state, timeout)
        except OSError:
            # The worker died between calls (memory or CPU limit).
            self._restart()
            self._send_request(code, state, timeout)
        return self._read_response(timeout_seconds=timeout, should_stop=should_stop)

    def _send_request(self, code: str, state: dict[str, Any], timeout: int) -> None:
        if not self._started:
            self._send({
                "code": code,
                "persistent_process": True,
                "search_tracking": os.environ.get("ARC3_SEARCH_PUSH2", "0").strip().lower()
                in ("1", "true", "yes", "on"),
                "persistence_enabled": False,
                "animation_enabled": self._animation_handler is not None,
                "timeout_seconds": timeout,
                "cpu_limit_seconds": self._cpu_limit_seconds,
                "memory_limit_bytes": self._memory_limit_bytes,
                "idle_limit_bytes": self._idle_limit_bytes,
                "sandbox_cwd": self._sandbox_dir or ".",
                "state": self._history_delta(state),
                "color_chars": ARC_COLOR_CHARS,
                "view_shift": _view_shift_enabled(),
                "frame_diff_v2": frame_diff_v2_enabled(),
                "world_map": _world_map_enabled(),
                "show_image": _show_image_enabled(),
                "skills": skill_texts(),
                "replay": replay_enabled(),
                "schema": schema_enabled(),
                "schema_tolerance": _schema_tolerance(),
                "schema_report_diff": _schema_report_diff(),
                "schema_diff_audit": schema_diff_audit_enabled(),
                "schema_crop_chars": _schema_crop_chars(),
                "schema_miss_budget": _schema_miss_budget(),
                "schema_free_actions": schema_free_actions(),
                "const_parse_warn": const_parse_warn(),
                "schema_stop": schema_stop_enabled(),
                "schema_stop_mode": schema_stop_mode(),
                "snippet_noop_limit": snippet_noop_limit(),
                "schema_parse": schema_parse_enabled(),
                "schema_parse_map": schema_parse_map_enabled(),
                "grid_map": grid_map_view_enabled(),
                "grid_map_name": grid_map_name(),
                "auto_map": auto_map_enabled(),
                "function_source": function_source_enabled(),
            })
            self._started = True
        else:
            self._send({
                "type": "execute",
                "code": code,
                "timeout_seconds": timeout,
                "state": self._history_delta(state),
            })

    def close(self) -> None:
        """Stop the worker process and release its private resources."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None or process.poll() is not None:
            if self._sandbox_dir:
                shutil.rmtree(self._sandbox_dir, ignore_errors=True)
                self._sandbox_dir = None
            return
        try:
            self._send({"type": "shutdown"})
        except (OSError, RuntimeError):
            pass
        _wait_for_process_exit(process, timeout=1.0)
        if self._sandbox_dir:
            shutil.rmtree(self._sandbox_dir, ignore_errors=True)
            self._sandbox_dir = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[..., dict[str, Any]],
    animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    kept_functions: list[str] | None = None,
    retain_imports: bool = False,
    repair_hints: bool = False,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        try:
            bootstrap_path = _write_bootstrap(sandbox_dir)
            creation_flags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if os.name == "nt" else 0
            )
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", bootstrap_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
                creationflags=creation_flags,
            )
        except OSError:
            return {
                "error": "Sandbox process could not start.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _stdout_reader() -> None:
            for raw_line in process.stdout:
                stdout_queue.put(raw_line)
            stdout_queue.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

        _send_json_line(
            process.stdin,
            {
                "code": code,
                "kept_functions": list(kept_functions or []),
                "persistence_enabled": kept_functions is not None,
                "retain_imports": retain_imports,
                "repair_hints": repair_hints,
                "animation_enabled": animation_handler is not None,
                "timeout_seconds": timeout_seconds,
                "sandbox_cwd": sandbox_dir,
                "state": initial_state,
                "color_chars": ARC_COLOR_CHARS,
                "view_shift": _view_shift_enabled(),
                "frame_diff_v2": frame_diff_v2_enabled(),
                "world_map": _world_map_enabled(),
                "show_image": _show_image_enabled(),
                "skills": skill_texts(),
                "replay": replay_enabled(),
                "schema": schema_enabled(),
                "schema_tolerance": _schema_tolerance(),
                "schema_report_diff": _schema_report_diff(),
                "schema_diff_audit": schema_diff_audit_enabled(),
                "schema_crop_chars": _schema_crop_chars(),
                "schema_miss_budget": _schema_miss_budget(),
                "schema_free_actions": schema_free_actions(),
                "const_parse_warn": const_parse_warn(),
                "schema_stop": schema_stop_enabled(),
                "schema_stop_mode": schema_stop_mode(),
                "snippet_noop_limit": snippet_noop_limit(),
                "schema_parse": schema_parse_enabled(),
                "schema_parse_map": schema_parse_map_enabled(),
                "grid_map": grid_map_view_enabled(),
                "grid_map_name": grid_map_name(),
                "auto_map": auto_map_enabled(),
            },
        )

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": f"Tool timed out after {timeout_seconds}s",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "animation":
                if animation_handler is None:
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error",
                         "error": "Animation frames are not available in this session."},
                    )
                    continue
                try:
                    animation_payload = animation_handler(dict(message.get("request") or {}))
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error",
                         "error": "animation failed in sandbox host."},
                    )
                    continue
                _send_json_line(
                    process.stdin,
                    {"type": "animation_result", "animation": animation_payload},
                )
                continue
            if msg_type == "action":
                try:
                    # the message carries more than the action list: stale_after
                    # tells the solver a previous action in this snippet changed
                    # nothing, which it weighs after its own guards
                    action_result_payload = action_handler(
                        list(message.get("actions") or []),
                        stale_after=message.get("stale_after"),
                    )
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
                    },
                )
                continue

            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "action_results": list(message.get("action_results") or host_action_results),
                    "keepable_functions": list(message.get("keepable_functions") or []),
                    **({"retention_rejected": message.get("retention_rejected", [])}
                       if kept_functions is not None else {}),
                }

            _wait_for_process_exit(process)
            return {
                "error": "Sandbox process returned an unknown message type.",
                "stdout": "",
                "action_results": list(host_action_results),
            }
