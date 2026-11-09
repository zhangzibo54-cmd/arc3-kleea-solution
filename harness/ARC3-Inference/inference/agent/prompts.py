"""Prompt templates for the analyzer agent."""

import os

from inference.utils.grid_utils import ARC_COLOR_LEGEND

LEVEL_TRANSFER_SYSTEM_GUIDANCE = (
    "- Levels usually build on mechanics learned in earlier levels, especially the most "
    "recent one. Carry forward supported knowledge as a starting hypothesis, while "
    "re-checking anything contradicted by new evidence. New levels often introduce "
    "additional mechanics, sometimes through unfamiliar board elements. These additions "
    "are often important for solving the level. The goal may remain the same but require "
    "new mechanics to reach it, or the goal itself may change.\n"
)

# ARC3_NEW_ELEMENT_PROBE: replaces LEVEL_TRANSFER_SYSTEM_GUIDANCE. bp35 stalled on
# level 3 for not trying a known action (a click) on a newly appeared element.
NEW_ELEMENT_LEVEL_GUIDANCE = (
    "- Levels usually build on mechanics learned in earlier levels, especially the most "
    "recent one. Carry forward supported knowledge as a starting hypothesis, while "
    "re-checking anything contradicted by new evidence. When a level shows an element "
    "you have not seen before, you must try every applicable action on it to learn its "
    "mechanic - including actions you already know, such as clicking it, or moving onto, "
    "next to or into it - and note what changes. This matters: such levels are usually "
    "designed so that the original goal can only be reached through the new mechanic. "
    "The goal itself rarely changes; more often your description of it was incomplete or "
    "wrong, so re-check it against the winning boards of earlier levels (for example "
    "`level_history[0][-1].frame`, the board at the moment level 1 was won).\n"
)

GOAL_REFORMULATION_GUIDANCE = (
    "The same winning state can fit different descriptions of the goal. "
    "When inferring the goal from completed levels, consider alternative "
    "formulations, including weaker or stronger conditions. Use small, "
    "targeted actions to distinguish these formulations and refine your "
    "description of the goal from the results."
)

# ARC3_LEVEL_FEEDBACK: keep goal review beside the existing transfer guidance.
NEW_ELEMENT_FEEDBACK_GUIDANCE = NEW_ELEMENT_LEVEL_GUIDANCE.replace(
    "wrong, so re-check it against the winning boards of earlier levels "
    "(for example `level_history[0][-1].frame`, the board at the moment "
    "level 1 was won).\n",
    "wrong, so re-check it against the winning boards of earlier levels "
    "using their labelled, fixed non-negative history references. "
    + GOAL_REFORMULATION_GUIDANCE + "\n",
)
# ARC3_NEW_ELEMENT_RETRY: inserted after this anchor of NEW_ELEMENT_LEVEL_GUIDANCE.
# In sp80 the model found clicks useless in level 1 and never clicked again.
NEW_ELEMENT_RETRY_ANCHOR = "and note what changes. "
NEW_ELEMENT_RETRY_SENTENCE = (
    "An applicable action that did nothing in an earlier level is likely to work in "
    "this one, so do not rule it out from earlier levels: try it here again. "
)

# ARC3_REPLAY: inserted after the level_deaths line. bp35 re-entered the same
# 20-action route by hand after every death, over several turns each time.
REPLAY_DEATHS_ANCHOR = "(the death replay).\n"
REPLAY_ADDENDUM = (
    "- Each fatal entry also has `.attempt`: the actions of that attempt, from the board the level "
    "(re)started on up to and including the fatal action, as strings `action(...)` accepts. "
    "`replay(steps=-1, death=-1)` re-executes the attempt of `level_deaths[-1][death]` on the restarted "
    "level, up to `steps` actions before the fatal one: `replay(-1)` brings you back to the board right "
    "before the death, `replay(-10)` to the board ten actions earlier. It compares every board with the "
    "recorded one, stops at the first difference and returns how many actions it replayed. Call it right "
    "after the level restarts, before any other action. It redoes all of those actions for real, so it "
    "spends as many actions from the budget as redoing them by hand: it saves turns, not actions.\n"
)
# completes "...the fatal action itself caused the death (hazard or forbidden move) — " in the
# death message, in place of "do NOT retry the same path; a DIFFERENT approach must be tried",
# which pushed toward a fresh route instead of back to the board before the death
REPLAY_DEATH_LINE = (
    "do not repeat the fatal action from the same board. If the actions before it were sound, "
    "`replay(-k)` brings you back to the board k actions before the death (`replay(-1)`: right "
    "before the fatal action), from where you can try a different action. Call it now, before any "
    "other action; it re-executes this attempt's recorded actions (`{fatal_ref}.attempt`) and "
    "spends them from the budget again, stopping early if a board differs from the recorded one."
)

LEVEL_START_USER_PROMPT = (
    "You have completed the previous level. `current_frame` now contains the starting "
    "board of the next level; any accompanying current-grid image shows this new board. "
    "Build a new plan for this layout rather than continuing the previous level's action "
    "sequence.\n\n"
    "Start from the mechanics you established on the previous level; do not rediscover "
    "them without reason. Inspect the new board for unfamiliar elements, changed "
    "arrangements, or interactions your previous understanding does not explain. New "
    "elements often introduce mechanics needed to solve this level, so prioritize small, "
    "informative tests when their behavior is unclear.\n\n"
    "Reassess the goal: does the previous objective still apply, now requiring the new "
    "mechanics, or does the evidence suggest a different objective? Combine retained "
    "knowledge with new findings to plan for this board."
)

# ARC3_LEVEL_FEEDBACK: verify the observed win before transferring its rules.
LEVEL_START_FEEDBACK_USER_PROMPT = LEVEL_START_USER_PROMPT.replace(
    "sequence.\n\n",
    "sequence.\n\n"
    "First load the previous level's actual winning frame using the fixed, "
    "non-negative `level_history` reference provided below. Inspect its "
    "`.segmentation['nodes']`, selecting the elements you believe caused the "
    "win, and carefully check their final states and spatial relationships "
    "against the frame. Use this observed winning state to verify or revise "
    "your explanation of why the previous level was completed.\n\n"
    + GOAL_REFORMULATION_GUIDANCE + "\n\nThen inspect "
    "the current board.\n\n",
).replace(
    "informative tests when their behavior is unclear.\n\n",
    "informative tests when their behavior is unclear. You must try every "
    "applicable action on each unfamiliar element to learn its mechanics "
    "- including actions you already know, such as clicking it, or moving "
    "onto, next to, or into it - and record what changes.\n\n",
)


def apply_level_transfer_system_guidance(prompt: str) -> str:
    """Opt-in system wording; the caller gates ARC3_LEVEL_TRANSFER_GUIDANCE.

    Preserve the original constants for the default arm. The first user prompt
    after completion gives the new-board instructions; the system retains flag
    definitions and stop instructions for failure and run completion.
    """
    return (
        prompt.replace(
            "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n",
            LEVEL_TRANSFER_SYSTEM_GUIDANCE,
        )
        .replace(
            "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n",
            "",
        )
        .replace(
            "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n",
            "",
        )
        .replace(
            "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n",
            "- `WIN` means the whole game is solved.\n",
        )
        .replace(
            "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n",
            "- If an action result reports `game_over`, `run_complete`, or `done`, stop acting immediately and re-ground on the next turn.\n",
        )
    )


TOOL_CALL_FORMAT_GUIDANCE = (
    "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. "
    "Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax. "
    "Do not quote or place tool-call markup inside explanatory text; when you decide to call the tool, emit the tool call itself."
)

GAME_OVERVIEW_ADDENDUM = (
    "\n\nGame overview:\n"
    "- You are solving a multi-level grid puzzle game. \n"
    "- You are called repeatedly over the course of a run. Treat each turn as one observe-plan-act cycle: re-understand the current state from the newest frame, update your working world model in Python, choose the next best action or short sequence against the goal as currently understood, execute it, and expect to re-evaluate on the next turn from the updated state.\n"
    "- Your job is to solve the entire game by clearing every level, not just the current screen.\n"
    "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n"
    "- Optimize for as few in-game actions as possible while still being reliable.\n"
    "- In this environment, boards are presented as 64 x 64 color grids rendered with ARC color symbols.\n"
    f"- Color legend: {ARC_COLOR_LEGEND}.\n"
)

# ARC3_OVERVIEW_EXPLORE_NOTES: two bullets after the Game overview (user wording, 2026-10-23). In the
# a18 all25 run the slow levels (s5i5 L2, tn36 L2, vc33 L4, lf52 L2) wrote simulators on rules taken
# from too little probing and never re-tested elements judged inert in one state or alone.
GAME_OVERVIEW_EXPLORE_NOTES = (
    "- The mechanics of every level are simple: they match physical intuition or are easy for an "
    "ordinary person to understand, because these games are designed so that an ordinary person can "
    "clear them after some exploration and trial. Do not overcomplicate the mechanics.\n"
    "- Mechanics established in levels you have already cleared are not removed, but later levels "
    "often add new objects or new mechanics. So a mechanic that worked before can usually be assumed "
    "to still work, but do not assume that a mechanic or action that had no effect before must still "
    "have no effect in a new level. Try different actions on different objects in different states to "
    "confirm new mechanics, especially whether a mechanic you previously judged ineffective now works "
    "in the new level.\n"
)
# ARC3_OVERVIEW_EXPLORE_NOTES=v2 (user wording, 2026-10-25): the simple-mechanics bullet moves up,
# right after the actions bullet, which now exempts exploration; the second bullet is shorter and
# says the model's description of a mechanic may only fit the earlier levels (sk48 L2 in six LOW8
# runs: "retracting pulls every covered bead" fit level 1 only; contradicting observations were
# patched with special cases instead of re-tested)
OVERVIEW_ACTIONS_LINE = "- Optimize for as few in-game actions as possible while still being reliable.\n"
OVERVIEW_ACTIONS_LINE_V2 = (
    "- Optimize for as few in-game actions as possible while still being reliable, but do not save "
    "actions when they serve to explore the game's mechanics.\n"
)
OVERVIEW_SIMPLE_MECHANICS = GAME_OVERVIEW_EXPLORE_NOTES.split("\n", 1)[0].replace(
    "they match physical intuition", 'they totally follow "physical intuition"') + "\n"
OVERVIEW_NEW_ELEMENTS_V2 = (
    "- Mechanics established in levels you have already cleared are not removed, but your description of "
    "a mechanic may match the real mechanic only in the earlier levels, and later levels often add new "
    "elements or new mechanics. When you find a new element, do not save actions: try different actions on "
    "the new element in different states, to explore the new mechanics and to check whether your earlier "
    "description of a mechanic was only partial.\n"
)

# ARC3_BAR_ECHO=note: the [action] line's bar report, announced after this
# sentence of the edge-bar guidance below
BAR_NOTE_ANCHOR = "It often shrinks or changes each step."
BAR_ECHO_NOTE = (
    " (If such a bar exists, its change is reported after each action in the [action] "
    "line, e.g. `top bar 42->40`.)"
)

VISUAL_GAME_ADDENDUM = (
    "\n\nVisual-game guidance:\n"
    "- Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.\n"
    "- Game entities are usually rendered as connected multi-tile shapes such as 2×2, 2×3, 3×3, or longer patterned structures. Sometimes they might also be 1x1 tokens.\n"
    "- Some games are logic or layout puzzles with no explicit player avatar or controllable sprite on the board. Do not assume a player exists; the relevant state may be an object, region, cursor, selector, or whole-board configuration.\n"
    "- Background colors are often white or gray/black-ish large regions, but not always. Verify background hypotheses by area, stability, and object boundaries rather than assuming them.\n"
    "- In many games, a long horizontal or vertical line near an edge is a timer or remaining-steps bar. It often shrinks or changes each step. If you identify such a bar, do not get distracted by it or treat it as core gameplay state unless there is concrete evidence that it interacts with the puzzle mechanics.\n"
    "A common failure mode is to mistake a segmented edge bar for clickable puzzle pieces. If a repeated strip of small blocks sits flush against the top, bottom, left, or right border and actions only change that strip while the interior board stays the same, classify it as HUD/timer state, not as an object to click through segment by segment. DON'T DO THIS!\n"
    "- Use coordinates only to target actions or describe local evidence. Do not frame the objective as reaching a specific absolute row or column.\n"
    "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n"
    "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n"
    "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n"
    "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n"
)

ACTION_INFO_ADDENDUM = (
    "\n\nAction meanings (use only actions currently available):\n"
    "- `UP`, `DOWN`, `LEFT`, and `RIGHT` are directional controls; what they affect "
    "depends on the game.\n"
    "- When available, `SPACE` performs a game-specific action, such as interacting, "
    "selecting, rotating, attaching/detaching, or executing. Test its effect rather "
    "than assuming what it does.\n"
    "- When available, `MOUSE` clicks a board location. Pass integer `row` and `col` "
    "fields from 0 to 63. Coordinates are zero-based from the top-left: `row` "
    "increases downward and `col` increases rightward.\n"
)

# After UNDO in every user message's list of valid actions.
UNDO_ACTION_NOTE = (
    "UNDO (usually reverts your last action, and repeated UNDOs step further back; it "
    "spends action budget like any other action. You can use it to explore what "
    "different action plans do from the same state, and so understand the game's "
    "mechanics. When an earlier mistake has made the goal impossible to reach, UNDO back "
    "to a state before it and try again)"
)

# ARC3_USER_WORKFLOW_RECAP: closes the text of every user message, before the images (user wording,
# 2026-10-25): points at the system prompt's Workflow instead of restating it
USER_WORKFLOW_RECAP = (
    "Follow the Workflow in the system prompt step by step. The mechanics are simple and intuitive; "
    "if your model is getting complicated, stop and test the simplest explanation first."
)

UNDO_INFO_ADDENDUM = (
    "- When available, `UNDO` reverses a previous action, usually the last turn. "
    "Check what it restores. It cannot recover a failed attempt after game over.\n"
)

# Appended after the action meanings when ARC3_PROMPT_PROBE_ACTIONS is on.
PROBE_ACTIONS_ADDENDUM = (
    "- If `SPACE`, `MOUSE`, `UNDO` or a directional control (`UP`, `DOWN`, `LEFT`, "
    "`RIGHT`) is in `valid_actions` and you have not yet "
    "seen what it does, try it early with a cheap probe (for `MOUSE`, click one "
    "object of each distinct kind) and note what changed. Do the same whenever "
    "progress stalls. These actions are not offered in every game; when they are, "
    "they usually control an important mechanic, even if the previous level could "
    "be solved without them.\n"
)

# Appended to the persistent-process section when ARC3_FUNCTION_DOCS is on.
FUNCTION_DOCS_ADDENDUM = (
    "- When you define a function you may call again, describe it in a docstring or "
    "comments: what it does, its inputs and what it returns, including any convention it "
    "relies on. After a context trim removes earlier turns, the namespace listing shows "
    "each function's signature and that description, so you can call it instead of "
    "rewriting it.\n"
)

# Appended to the namespace listing (shown after context trims) when ARC3_FUNCTION_SOURCE is on.
FUNCTION_SOURCE_HINT = (
    " The snippets that defined these functions and classes may have been removed with "
    "earlier turns; `print(fn_source['name'])` shows the code of your latest definition "
    "of `name`, and `print(fn_source)` lists the names it has."
)

# Appended to the user prompt when ARC3_LEVEL_TOKEN_NUDGE fires; {k} is the
# tokens spent on the current level, in thousands.
LEVEL_TOKEN_NUDGE = (
    "Note: you have spent about {k}K tokens on this level without clearing it. If "
    "the open question is how the game works, a few cheap actions may settle it faster "
    "than more analysis - for example, an action you already know, used in a "
    "situation where you have not tried it yet (with different objects attached, "
    "adjacent or selected). Check what changes after each one."
)
# ARC3_LEVEL_NUDGE_CAUSES: replaces LEVEL_TOKEN_NUDGE with the usual reasons a level stays
# unsolved; {k} as there, {check} one of the two lookups below. The levels that fell behind in
# the 3-switch and fnsrc-aidx all25 runs stalled on rules made ever more complicated (ka59, tn36,
# r11l), on notebook conclusions carried across trims (tu93, wa30) and on wrong goals.
LEVEL_TOKEN_NUDGE_CAUSES = (
    "Note: you have spent about {k}K tokens on this level without clearing it. A level usually "
    "stays unsolved for one of these reasons:\n"
    "1. You are making the mechanics more complicated than they are. Game mechanics are usually "
    "simple and consistent: they are designed to match physical intuition, or to be easily "
    "understood by an ordinary person.\n"
    "2. There is a mechanic you have not found yet. A level can bring new elements, and the same "
    "action can do different things in a different state (with different objects attached, "
    "adjacent or selected); an action that did nothing before may act on a new element on this "
    "level.\n"
    "3. You trust the lab notebook too much. Its entries are condensed summaries of earlier "
    "experiments, written with thinking off, and they lose detail; when one matters, check it "
    "against its boards: {check}.\n"
    "4. Your goal for this level is wrong. The goal also matches ordinary intuition: usually it "
    "is the same as on earlier levels (though the way you stated it may be too narrow), or it is "
    "to bring the board into its most orderly, most harmonious state - the one that takes the "
    "least effort for a person to understand.\n"
    "Work out from the current state which of these is keeping you on this level, and change "
    "your approach to clear it."
)
# {check} of LEVEL_TOKEN_NUDGE_CAUSES: with ARC3_ACTION_INDEX the entries carry [k<a>-<b>].
LEVEL_NUDGE_CHECK_INDEXED = "`level_history[n - 1][a:b + 1]` for an entry `L<n> [k<a>-<b>]`"
LEVEL_NUDGE_CHECK_STEPS = (
    "`[e for e in level_history[n - 1] if a <= e.frame.step <= b]` for an entry `L<n> [s<a>-<b>]`"
)
# ARC3_IDLE_ACT_NUDGE_TOKENS: a request-boundary note each time the model has generated
# that many more tokens without a game action; {k} is the tokens since the last one, in
# thousands. bp35 level 2 stalled for 45K-78K tokens of offline planning (v39, v40, v41)
# where the 4-level run never went past 8K.
IDLE_ACT_NUDGE = (
    "You have generated about {k}K tokens since your last game action. The time for "
    "this game is limited: thinking that does not reach a useful conclusion only wastes "
    "it, so you must act now. Run the next actions of your plan, or a short probe if "
    "you have none, and look at what happens. A death only restarts the level, and its "
    "animation shows what went wrong."
)
# Inserted after the first sentence of IDLE_ACT_NUDGE in a level whose view scrolled.
IDLE_ACT_NUDGE_SCROLL = (
    " This level scrolls: the board you plan from shows only part of it, and moving "
    "reveals the rest."
)

# Appended to LEVEL_TOKEN_NUDGE when UNDO is a valid action.
# Appended to LEVEL_TOKEN_NUDGE: the valid actions (RESET excluded) not yet taken on the current
# level (sp80 rerun 2026-10-23: level 2 for 100 min, MOUSE never clicked, the note fired 7 times).
LEVEL_TOKEN_NUDGE_UNTRIED = " Available actions you have not tried on this level yet: {actions}."

LEVEL_TOKEN_NUDGE_UNDO = (
    " UNDO is available: use it to test a mechanic or rule cheaply and repeatedly - "
    "take the action, check what changed, then UNDO it."
)

# ARC3_LEVEL_FEEDBACK or legacy ARC3_LEVEL_WINS: history description.
LEVEL_WINS_ADDENDUM = (
    "- The last entry of a level you have won is the winning action, and its `.frame` "
    "is the board at the moment the level was won, before the next level loaded "
    "(`.result['winning_board']` is True). The harness labels its stable, "
    "non-negative reference as `level_history[n - 1][k].frame`; use that exact "
    "`k` instead of a moving `[-1]` index. "
    "The next level's list starts with its first board (`.action == ''`). "
    "`transitions` ends each level with the winning action, from the board before it to "
    "the winning board.\n"
)
# ARC3_ACTION_ANIMATIONS: beside the level_history description.
ACTION_ANIMATIONS_ADDENDUM = (
    "- Each entry also has `.animation_frames`: every state its action passed "
    "through, the board before it first and `.frame` last (empty when it did not "
    "animate). The images shown after an action are a sample of them. Each "
    "exposes `.ascii`, `.segmentation` and `.shape` like `.frame`.\n"
)
# ARC3_ANIMATION_FINGERPRINTS: indexed replay without sending every frame.
ANIMATION_FINGERPRINTS_ADDENDUM = (
    "- Repeated animation notices name a short hash and stable, non-negative "
    "`level_history[level_index][entry_index]` references. An entry's "
    "`.animation_frames` holds every state of its action in time order, the "
    "board before it first and `.frame` last; inspect their `.ascii` or "
    "`.segmentation` to infer what the animation means. "
    "`entry.result['animation_hash']` is a fingerprint of changing interior "
    "cells across the full animation, aligned to world coordinates when the "
    "camera scrolls. The same hash is evidence of a repeated visual transition, "
    "not proof of identical game state or cause. Actions without intermediate "
    "frames have no animation hash.\n"
)
# ARC3_ANIMATION_GUIDE: the explanation of `.animation_frames` from the addendum above,
# without hashes or repeated-animation notices. In hard10 v23 the notices (up to 56 a game)
# were ignored, but tn36 read the frames (then `.animations`) from its 8th call on and
# cleared 6 levels (v22: 1).
ANIMATION_GUIDE_ADDENDUM = (
    "- An entry's `.animation_frames` holds every state of its action in time order, "
    "the board before it first and `.frame` last (empty when the action did not "
    "animate); inspect their `.ascii` or `.segmentation` to infer what the animation "
    "means.\n"
)
# ARC3_SHOW_IMAGE: beside the level_history description.
SHOW_IMAGE_ADDENDUM = (
    "- Call `.show_image()` on any `level_history` entry to see its board as an image in "
    "your next message (at most 3 per message), for example "
    "the labelled non-negative `level_history[0][k].show_image()` entry for the board "
    "at the moment level 1 was won. "
    "Use it when you need a board that is no longer in your context.\n"
)
# First prompt of a new level when the winning board is attached.
LEVEL_WIN_USER_LINE = (
    "Level {level} was won by {action}. The winning board image below shows the moment "
    "it was won (`level_history[{level_index}][{history_index}].frame`); the current "
    "grid image shows the new level."
)

# ARC3_SEARCH_DECOMPOSE: subgoal guidance, shown as a system-prompt bullet...
SEARCH_DECOMPOSE_TEXT = (
    "When the goal is clear but a direct search for it would be too large (several "
    "pieces, positions and orientations at once), split it into a sequence of "
    "plausible intermediate goals. Reach each one with a small search or by direct "
    "play, and check the result before moving on. If an intermediate goal turns out "
    "to be unreachable, pick a different one; if the overall goal looks unreachable, "
    "re-examine how you defined it against how the earlier levels were won - it "
    "may only describe level 1's winning state. A search that finds nothing is "
    "evidence against your model or your search code, not proof that the goal is "
    "impossible."
)
SEARCH_DECOMPOSE_ADDENDUM = "- " + SEARCH_DECOMPOSE_TEXT + "\n"
# ARC3_SEARCH_PUSH: all search guidance in one place - it replaces the default-loop bullet
# (and the bullet above), the other mentions of searches lose that word, and the same text
# is put into the next user message whenever a snippet searches over game moves or runs out
# of time or memory. sk48 (hard10 v18) wrote eleven searches over the whole level and never
# split it.
SEARCH_PUSH_TEXT = (
    "Work subgoal by subgoal: first split the goal into intermediate states that build on each "
    "other (for example, one object moved or reshaped into place, then the next, then the goal). "
    "For each subgoal: summarize the board, infer the change you need, search over candidate "
    "action sequences or plan directly, execute with `action(...)`, then inspect until you "
    "understand exactly what changed and whether the subgoal was reached; only then move on.\n"
    "  Example: a maze whose exit opens only after a series of switches is triggered in order - "
    "search for the path to the first switch, take it and check that it triggered, then the next "
    "switch, and finally the exit.\n"
    "  Notes: check your simulated rules against the recorded transitions on what decides the "
    "outcome (which actions are accepted or refused, where the objects the goal depends on end "
    "up), and fix a rule when they disagree. A search that finds nothing is evidence against "
    "your rules or your search code, not proof that the goal is unreachable. Keep a search's "
    "tables local to a function and its progress in a dict, list or tuple at top level, so a "
    "search stopped by the time limit resumes in the next call."
)
SEARCH_PUSH_REPLACEMENTS = (
    ("- A strong default loop is: summarize the board, infer the desired environment change, "
     "write a small scorer or search over candidate sequences, execute the best probe or plan "
     "with `action(...)`, then inspect again until you understand exactly what changed.\n",
     "- " + SEARCH_PUSH_TEXT + "\n"),
    ("Save long searches in a dict, list, or tuple so you can resume from a later call.",
     "Save long-running progress in a dict, list, or tuple so you can resume from a later call."),
    ("Keep search tables (visited sets, queues) local to a function;",
     "Keep large tables local to a function;"),
)

# ARC3_SEARCH_PUSH2 shares one policy between the system and a short reminder.
# The number is an estimated work budget, not a measured sandbox enforcement.
SEARCH_PUSH2_TEXT = (
    "Before each search, state the next subgoal. Search or plan for that subgoal, "
    "then use `action(...)` to make progress before planning the next one. "
    "A whole-level search is allowed only when you can estimate at most "
    "{max_states:,} state expansions or candidate evaluations and it fits within "
    "one Python call's time and memory budget. Otherwise split the goal; later "
    "levels usually need smaller subgoals. Do not blindly increase the depth, "
    "width, restarts or search space after failure or an exhausted budget. "
    "Change the subgoal, or act to obtain new evidence and revise possibly "
    "incorrect rules. If predicted success fails in the real game, fix the "
    "simulator or goal interpretation before searching again. "
    "Before broad enumeration, use observed structure to narrow candidates "
    "(where relevant: connector counts, relative offsets, rotation invariants "
    "and possible pairing graphs). A convenient anchor or placement order "
    "must not silently exclude valid layouts. On failure, briefly record the "
    "goal and applicable assumptions: anchor, coordinate bounds, rotations, "
    "placement order, pairing/connectivity, overlap/coverage rules, pruning, "
    "and budget actually used; distinguish exhausted searches from cutoffs. "
    "Conclude only 'no solution found under these assumptions', not that the "
    "game goal is impossible. Check candidate generation, coordinates and "
    "pruning before blaming the rules. When relaxing a constraint, name the "
    "change and retain the original assumptions and result for comparison. "
    "One level or one successful layout does not establish the complete win "
    "condition for other levels; keep inferred rules as hypotheses until "
    "tested against relevant observations."
)
SEARCH_PUSH2_REMINDER = (
    "Search reminder:\n"
    "1. Scope: search the whole level only when you estimate <= {max_states:,} "
    "expansions or candidate evaluations within one call's budget. Otherwise "
    "split the level into subgoals (the next object to reach, piece to place "
    "or state to set), state the next one, and search for it, using observed "
    "structure to narrow the candidates.\n"
    "2. When a search finds nothing, or a plan predicted to succeed fails in "
    "play, go through these in order; do not just enlarge the search:\n"
    "   a. State the search's assumptions (coordinates, bounds, rotations, "
    "placement order, pruning, goal test) and the budget used, including any "
    "cutoff; then check the code against your own understanding of the "
    "mechanics by running it on a case whose answer you know (a state you "
    "already reached must be found). If you relax a constraint, name it and "
    "keep the original result. 'No solution found under these assumptions' "
    "is not proof of impossibility.\n"
    "   b. If the code is right, use `action(...)` to explore mechanics you "
    "have not tested yet, and update your world model with what you learn.\n"
    "   c. If the mechanics are explored, look again at the earlier winning "
    "boards (`level_history[L][-1].frame` for a completed level L) and "
    "reinterpret the goal; one level or layout does not establish a "
    "universal win condition."
)
OTHER_AGENT_SYSTEM_ADDENDUM = (
    "\n- When an observed object moves or acts on its own, find out whether it "
    "is a teammate or competitor. Watch a few actions to learn what it seeks, "
    "carries or changes, where it delivers things, and how it reacts to you. "
    "Taking an object you need may be helping to deliver it. With a teammate, "
    "plan a division of work: let it handle its useful tasks while you do the "
    "rest, and position yourself so it can work. Account for the remaining "
    "action budget; avoid long waits based on untested predictions. Trust "
    "observed behavior over these possibilities.\n"
)
# ...and a reminder on a Python call that timed out or ran out of memory, the
# moment a direct search has proved too large.
SEARCH_DECOMPOSE_NOTE = (
    "If this was a search, its space is too large for one call. When the goal is "
    "clear, split it into a sequence of plausible intermediate goals: reach each one "
    "with a small search or by direct play, and check the result before moving on."
)

RESET_INFO_ADDENDUM = (
    "- When available, `RESET` usually restores the current level to its starting "
    "state, including the remaining-action/time bar, while keeping completed levels. "
    "Use it to recover from an unrecoverable position or start a different approach. "
    "RESET itself counts as one action, and actions already spent still count toward "
    "your score. RESET must be the first game action in a Python snippet; inspection "
    "or computation may precede it.\n"
)

GAMEPLAY_CHANGED_ADDENDUM = (
    # One region, named once and referred back to twice. "outside that margin"
    # and "inside that border" both read as possibly including the border
    # cells themselves; "ignores those edge cells" states the rule before any
    # geometry, and "further in" cannot be read as the border.
    "- `board_changed` is true when ANY cell differs, including cells within {border} of the "
    "edge where a timer or remaining-steps bar usually sits; `gameplay_changed` ignores those "
    "edge cells and is true only when a cell further in differs. An action that only shrinks "
    "a timer bar in those edge cells sets `board_changed` but not `gameplay_changed`.\n"
)

STRUCTURED_RUNTIME_STATE_ADDENDUM = (
    "\n\nRuntime variables inside every `python` tool call:\n"
    "- `current_frame` is a lightweight frame view for the latest environment state.\n"
    "- `current_frame` exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- `current_frame.ascii` is a single newline-delimited string containing the latest board rendered with the letter-coded ARC color symbols.\n"
    "- `current_frame.segmentation` parses the board into objects. It returns `{'nodes': [...], 'adjacency_list': [...]}`.\n"
    "- Each node in `segmentation['nodes']` is one 4-connected same-color object with: `id` (index, ordered top-most-left-most), `color` (ARC color character), `hash` (a signature of the object's color and shape that ignores position AND rotation -- equal hashes indicate matching color and shape up to rotation; they do not establish object identity), `rotation` (degrees clockwise from the object's canonical orientation: 0/90/180/270, reduced to 0/90 for 2-fold symmetric shapes, `None` for fully symmetric ones), `rotational_symmetry` (1, 2, or 4), `pose_hash` (the rotation-SENSITIVE variant, if you need to match exact orientation), `shape_hash` (color-independent pose shape signature), `pixels` (cell count), `boundary` (clockwise outer-perimeter corner points as `[row, col]` cell coordinates), and `children` (ids of objects fully enclosed by this one).\n"
    "- `segmentation['adjacency_list']` is a list of `[i, j]` node-id pairs whose objects share an edge.\n"
    
    "- `current_frame.step` is the current environment step count.\n"
    "- `current_frame.level` is the current level number.\n"
    "- `current_frame.shape` is a `(rows, cols)` tuple.\n"
    "- The raw numeric grid is intentionally not exposed. Use `current_frame.segmentation` as your primary view of the board -- objects, colors, shapes, containment, adjacency, and cross-frame object hashes. Use `current_frame.ascii` only to read a small, specific region; do not scan the whole board with it.\n"

    "- `level_history` is the game so far, split by level: a Python list with one list per level, in order. `level_history[n - 1]` is level n and `level_history[-1]` is the current level.\n"
    "- Each level's list holds entries exposing `.action`, `.frame`, and `.result`; entries are objects, not subscriptable like `entry['action']`. Its first entry shows the level's first board, and entry k shows the board after the level's k-th action (`entry.frame` is the frame after `entry.action`). Each frame exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- Important semantics: `level_history[-1][-1].frame` is the same latest/post-action board as `current_frame`. It is not the previous board. To inspect the state before the latest action, use `previous_frame` or `level_history[-1][-2].frame` when available.\n"
    "- `level_deaths` lists the deaths so far, split by level like `level_history`: `level_deaths[n - 1]` holds level n's fatal actions, oldest first, and `level_deaths[-1][-1]` is the latest death on the current level. They are the same entries as in `level_history`: `.action` is the fatal action, `.frame` the death frame, `.animation_timeline` the diff timeline of its animation and `.animation_frames` every state of its animation in order (the death replay).\n"
    "- `previous_frame` is the frame before the most recent real environment action, or `None` if no previous frame is available.\n"
    "- `last_action` is the most recent real environment action name/display, or `None` before any real action.\n"
    "- `last_action_frame` is the post-action frame for `last_action`; it matches `current_frame` after a real action.\n"
    "- `transitions` is a chronological list of actual action transitions, excluding the initial seeded frame. Each transition exposes `.action`, `.before_frame`, `.after_frame`, `.frame` (alias of `.after_frame`), and `.result`. Each `.result` describes that individual action and is retained with its `level_history` entry.\n"
    "- `last_transition` is `transitions[-1]` or `None`. A refused action creates no transition and does not overwrite existing transition results. For before/after diffs, compare `last_transition.before_frame` to `last_transition.after_frame`; do not compare `current_frame` to `level_history[-1][-1].frame`.\n"
    "- `last_action_call_result` contains the result of the latest model-issued `action(...)` call, including batch totals, executed/skipped actions, and stop reasons. `action(...)` returns this result. It remains available across later Python inspection calls and analyzer turns, and is `{}` before any action call result exists.\n"
    "- On the next analyzer turn after automatic RESET, `current_frame` shows the restarted level. The fatal action and RESET are separate transitions. Each transition result has `automatic=True` for a harness-initiated action and `automatic=False` for a model-issued action. Automatic RESET does not overwrite `last_action_call_result`: it still describes the call that caused death.\n"
    "- `valid_actions` is the current list of valid action names.\n"
    "- Call `action(actions)` to execute one or more real environment actions from Python.\n"
    "- Pass `action(actions)` a list like `['LEFT']` or `[{'action': 'MOUSE', 'row': 4, 'col': 7}]`.\n"
    "- One action usually returns one frame, but a single action can result in a short multi-frame animation; `current_frame` is its final frame.\n"
    "- After `action(actions)` returns, `current_frame`, `previous_frame`, `level_history`, `level_deaths`, `transitions`, `valid_actions`, and `last_action_call_result` are refreshed.\n"
)

ANIMATION_ADDENDUM_TIMELINE = (
    "- `last_animation_timeline`, and each `level_history` entry's "
    "`.animation_timeline` for its own action, is a diff timeline of those "
    "frames: it shows which cells changed at each step. It is a dict with "
    "`action`, `frames` (the number of states after the board before the "
    "action) and `steps`, one entry per step that changed anything. Step k is "
    "the change from frame k - 1 to frame k of the matching "
    "`last_animation_frames` / `.animation_frames`. Each entry has `step`, "
    "`changed` (the number of changed cells), `bbox`, and either `changes` "
    "(the cells, as `old>new @ (row,col) ...`) or `transitions` (a count per "
    "colour change, when there were too many cells to list). Empty when the "
    "action did not animate.\n"
)

# ARC3_FRAME_DIFF_V2: the timeline is frame_diff per step
ANIMATION_ADDENDUM_TIMELINE_V2 = (
    "- `last_animation_timeline`, and each `level_history` entry's "
    "`.animation_timeline` for its own action, is `frame_diff` applied to each "
    "step of those frames. It is a dict with `action`, `frames` (the number of "
    "states after the board before the action) and `steps`, one entry per step "
    "that changed anything. Step k is the change from frame k - 1 to frame k of "
    "the matching `last_animation_frames` / `.animation_frames`. Each entry has "
    "`step`, `changed_cell_count`, `bbox` (of the changed cells, as text) and the "
    "lists `frame_diff` returns for that step (`moved`, `rotated`, `appeared`, "
    "`disappeared`, `changed_color`, `resized`, `scaled`; empty ones are left "
    "out). The background colours of the board before the action are left out. "
    "Empty when the action did not animate.\n"
)

ANIMATION_ADDENDUM = (
    "- When the final executed action of the latest model-issued call animated, `last_action_call_result['animation']` "
    "reports `frames`, `transient_pixels` and `transient_bbox`. Transient cells "
    "changed and then changed BACK during the animation, so they appear in no "
    "board before or after an action - not in `current_frame`, not in "
    "`previous_frame`, not in any `level_history` entry's `.frame` - only in "
    "the animation's frames.\n"
    "- `last_animation_frames` is every state of the last executed action, as "
    "views with "
    "`.ascii`, `.shape` and `.segmentation`, the same as `current_frame`. Index 0 is "
    "the board before the action and the last one the board it settled on. "
    "After a death these are the frames of the FATAL action, not of the automatic "
    "reset that followed it. Empty when that action did not animate: the next "
    "action replaces them, and an earlier action's animation stays in its "
    "`level_history` entry's `.animation_frames`. Crop "
    "their `.ascii` yourself; printing a whole 64x64 frame will not fit the tool "
    "budget. Reading them costs no in-game action.\n"
    "- Each `level_history` entry has the same for its own action: "
    "`.animation_frames` is every state of it, the board before the action "
    "first and the entry's `.frame` last (for a winning action, the winning "
    "board), read on first use; empty when the action did not animate.\n"
)

MULTIMODAL_CONTEXT_ADDENDUM = (
    "\n\nMultimodal context:\n"
    "- User turns include an attached image of the current ARC grid.\n"
    "- The image and `current_frame.ascii` are two representations of the same current frame.\n"
    "- You can use images and other tools to understand the game state and guide your strategy, each may be useful depending on the current uncertainty.\n"
)

# ARC3_IMAGE_SCALE_NOTE: after the "two representations" line of MULTIMODAL_CONTEXT_ADDENDUM;
# {size} = 64 * {scale} pixels, {last} = {scale} - 1.
IMAGE_SCALE_NOTE = (
    "- Each board image is {size} x {size} pixels: one cell is a {scale} x {scale}-pixel square, "
    "so cell (row, col) covers pixel rows {scale}*row to {scale}*row+{last} and pixel columns "
    "{scale}*col to {scale}*col+{last}. Images of other frames (per-action, diff, animation) use "
    "the same scale.\n"
)

PYTHON_ADDENDUM_HEAD = (
    "\n\nPython tool guidance:\n"
    "- Use `current_frame.segmentation` as your primary view of the board -- objects, colors, containment, adjacency, and cross-frame object hashes.\n"
    "- Use `current_frame.ascii` only to read a small, specific region of the board when `segmentation` is not enough; never use it to scan or summarize the whole board.\n"
    "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\n"
    "- The only importable standard-library modules are: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string.\n"
    "- The only tool is `python`; call it with one ephemeral `code` string.\n"
    "- Always inspect `current_frame`, `level_history`, and `valid_actions` from Python instead of reasoning from the raw board by eye.\n"
    "- For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`. `level_history[-1][-1].frame` is the current frame, so comparing it to `current_frame` only compares the board to itself.\n"
)

# This block is appended only when the persistent-process experiment is enabled.
PERSISTENT_PROCESS_ADDENDUM = (
    "\n\nPersistent Python process mode is enabled for this game.\n"
    "- All code runs in one private Python process for this game. Variables, functions, "
    "classes, and imports survive between `python` calls and can be inspected or "
    "redefined directly.\n"
    "- The harness preloads frame views (`FrameView`, `HistoryEntryView`, "
    "`TransitionView`), `action`, `frame_diff`, `Counter`, `defaultdict`, `deque`, "
    "and `time`. `import time` is allowed.\n"
    "- Besides running your code, the process keeps these names in step with the game: "
    "at the start of every `python` call and after every `action(...)` it overwrites "
    "`current_frame`, `latest_frame`, `previous_frame`, `last_action_frame`, `level_history`, "
    "`transitions`, `last_transition`, `last_action`, `valid_actions`, "
    "`last_action_call_result`, `level_deaths`, `last_animation_frames` and "
    "`last_animation_timeline` with the live state. Read the game state from them. "
    "Anything you store under your own names (for example `f = current_frame`, or a "
    "list of grids) is a snapshot and does not update; an assignment to one of the "
    "names above lasts only until the next refresh, so keep your own data under other "
    "names.\n"
    "- Each turn lists the live namespace, and each tool result lists the names it "
    "added, removed, or changed. Use those names directly; do not rebuild helpers "
    "just because a context trim removed the earlier conversation.\n"
    "- A checkpoint is created before each snippet. If execution times out or is "
    "interrupted, definitions and intermediate values created before the interruption "
    "remain available. Save long searches in a dict, list, or tuple so you can resume "
    "from a later call.\n"
    "- The process is resource-limited: a snippet may use about "
    "{python_peak_mb} MB, but between calls the process keeps at most about "
    "{python_idle_mb} MB, and the largest variables are deleted to get there. Keep "
    "search tables (visited sets, queues) local to a function; keep only compact "
    "results and resumable progress at top level. Keep output compact and never "
    "print a full frame or a large collection.\n"
)

# split so the world-model lines can be swapped in at their original
# position - immediately before the IMPORTANT search guidance - rather than
# appended after the whole addendum
PYTHON_ADDENDUM_TAIL = (
    "- IMPORTANT: Especially when the game is about making an agent navigate to a target, it is usually safer to write an explicit search algorithm such as BFS. More generally, when the objective is understood but the best action order is unclear, pathfinding, flood fill, BFS, DFS, beam search, shortest-path search, limited action-sequence search, or custom heuristics are all valid.\n"
    "- Once the important state variables and action effects are sufficiently understood, stop probing and search in the inferred state space.\n"
    "- Inspect current and earlier frames (`level_history`) from Python instead of describing frames freehand.\n"
    "- Never print or echo full board frames. Return only compact derived summaries such as object lists, diffs, coordinates, counts, or tiny local crops.\n"
    "- Keep tool-output context size minimal and decision-oriented so you can quickly compare before/after state. It's fine to write a lot of python code, just make the output short and interpretable\n"
    "- A strong default loop is: summarize the board, infer the desired environment change, write a small scorer or search over candidate sequences, execute the best probe or plan with `action(...)`, then inspect again until you understand exactly what changed.\n"
    "- For object tracking, match objects by color, overlap, bounding box proximity, area change, and edge contact rather than by exact coordinates alone.\n"
    "- For frame diffs, summarize changed cells, color transitions, appearing/disappearing components, movement candidates, and small local row slices around the changed region.\n"
    "- After every action, verify whether gameplay objects changed or whether only a timer, progress bar, or remaining-step bar moved. Do not treat HUD-only changes as evidence that the move worked.\n"
    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\n"
    "- Call `action(...)` inside Python rather than returning action text in the chat.\n"
    "- `action(...)` accepts an ordered list of one or more actions. Once your code has selected a reliable sequence, it is often useful to batch it.\n"
    "- You can also call `action(...)` multiple times in one Python snippet, including inside loops. Each call updates the preloaded variables before execution continues.\n"
    "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n"
    "- Flag semantics: `game_over` = this attempt FAILED (death or a limit ran out) — the level auto-resets to its initial state and completed levels are kept; it never means the run is won. `level_completed` = advanced one level. `run_complete`/`done` = the whole game is won. Deaths come either from the action itself (e.g., a hazard cell) or from a step/time budget depleting — watch for a bar at the grid border that shrinks each action, and count remaining budget into your plans.\n"
)

# ARC3_SCALE_REPORT (default off): the frame_diff text lists and explains `scaled`, and an
# action that scales an object names it in its [action] line (solver). demo10 v4-v6 (51.4-51.5)
# ran without these; v7 on, with them, scored 43-46 in all but one run.
FRAME_DIFF_SCALED_KEYS_ON = "`changed_color`, `resized`, and `scaled`. Unchanged"
FRAME_DIFF_SCALED_KEYS_OFF = "`changed_color`, and `resized`. Unchanged"
FRAME_DIFF_SCALED_LINE = (
    "- `scaled` describes a component shrunk or enlarged by an integer factor: its shape with every cell blown up to a k x k block is exactly the other one (`factor` k when it grew, 1/k when it shrank; `rotated_by` when it also turned). Entries contain `color`, `factor`, `at_before`, `at_after`, `before_pixels`, `after_pixels`, `before_bbox`, and `after_bbox`; a scaled component is not listed again as appeared, disappeared or resized. An action that scales something also says so in its `[action]` line.\n"
)
FRAME_DIFF_HINT_ADDENDUM = (
    "- `frame_diff(before=None, after=None)` compares two frames. Omitted arguments default to `previous_frame` and `current_frame`, respectively.\n"
    "- It returns a dictionary containing `changed_cell_count` (an integer) and lists named `moved`, `rotated`, `appeared`, `disappeared`, `changed_color`, `resized`, and `scaled`. Unchanged objects are omitted.\n"
    "- Object-level changes are inferred by matching same-color, 4-connected components between frames: cells connect through shared edges, not diagonally. A game object may therefore be split into several components, or touching same-color objects may form one component.\n"
    "- `moved` describes components matched at different positions. Matching does not establish persistent object identity: with multiple similar components, the reported correspondence may be ambiguous or incorrect. Movement combined with rotation includes `rotated_by` in degrees clockwise; `rotated` describes rotation in place.\n"
    "- `changed_color` requires identical occupied cells with a different color. Entries contain `at`, `pixels`, `before_color`, `after_color`, `before_hash`, and `after_hash`.\n"
    "- `resized` describes same-color components with overlapping footprints whose size changed. This can include backgrounds and budget bars. `at_before` and `at_after` can differ.\n"
    "- `scaled` describes a component shrunk or enlarged by an integer factor: its shape with every cell blown up to a k x k block is exactly the other one (`factor` k when it grew, 1/k when it shrank; `rotated_by` when it also turned). Entries contain `color`, `factor`, `at_before`, `at_after`, `before_pixels`, `after_pixels`, `before_bbox`, and `after_bbox`; a scaled component is not listed again as appeared, disappeared or resized. An action that scales something also says so in its `[action]` line.\n"
    "- `appeared` and `disappeared` entries include `bbox` in `[r0, c0, r1, c1]` order. Object hashes use the same rotation-invariant shape-and-color representation as segmentation; they are not unique object identities, and multiple objects can share a hash.\n"
    "- Examples: `frame_diff()` compares the latest action’s before/after frames; `frame_diff(level_history[-1][-3].frame, level_history[-1][-2].frame)` compares two explicitly selected earlier frames of this level, when available.\n"
)

# ARC3_FRAME_DIFF_V2: replacements in FRAME_DIFF_HINT_ADDENDUM
FRAME_DIFF_RESIZED_LINE = (
    "- `resized` describes same-color components with overlapping footprints whose size changed. This can include backgrounds and budget bars. `at_before` and `at_after` can differ.\n"
)
FRAME_DIFF_RESIZED_LINE_V2 = (
    "- `resized` describes same-color components with overlapping footprints whose size changed. This can include budget bars. `at_before` and `at_after` can differ. `added` and `removed` give the cells the component gained and lost: `{\"tiles\": n, \"bbox\": [[r0, c0, r1, c1], ...]}` when they form n equal square tiles (3 to 8 cells a side, solid or hollow), with one box per rectangle the tiles join into; otherwise `{\"pixels\": n, \"bbox\": [[r0, c0, r1, c1], ...]}`, with one box per separate (4-connected) piece.\n"
    "- Background colours are left out of every list: a colour covering at least 30% of the level's first board (at most two colours). So are pieces of a region that were only cut off from it or rejoined to it, with their cells keeping their colour.\n"
)

# ARC3_RULE_NOTES: a single parallel Python-tool bullet, kept out of the
# numbered workflow so the workflow remains compact.
RULE_NOTES_FRAME_DIFF_LINE = (
    "- For important rules, maintain a persistent Python variable `rule_notes`. "
    "Each record should include `claim`, `status` (`hypothesis`, `confirmed`, "
    "or `refuted`), and an evidence reference; include scope or conditions "
    "when useful. Mark a rule `confirmed` only after an action, frame "
    "comparison, or history entry supports it; preserve the original note and "
    "downgrade or mark it `refuted` when later evidence conflicts. Do not treat "
    "an unverified hypothesis as a fact.\n"
)

SEGMENTATION_SCREEN_COORDINATES_NOTE = (
    "Coordinate reminder: segmentation boundaries and positions, including "
    "those in history and animation frames, are in each frame's own screen "
    "coordinates; camera scroll has NOT been subtracted. They are not "
    "world_map coordinates. Compare movement only after aligning the shared "
    "visible area using the scroll between those frames. Do not apply the "
    "current cumulative offset to earlier animation frames; if their offsets "
    "are unknown, screen displacement alone does not establish object motion."
)

WORLD_MAP_LIVE_ADDENDUM = (
    "\n- `world_map` is a live dict of {(row, col): color char} in this "
    "level's starting-view coordinates, available even before scrolling. "
    "The harness updates the same dict after actions and before Python calls. "
    "RESET and new levels clear and refill this same dict with the new starting "
    "view, discarding the old level's cells. Read `world_map` itself in each "
    "call; a copy such as `dict(world_map)` never updates and goes stale after "
    "a scroll or a new level. "
    "Off-screen cells retain their last observed color; an unknown scroll "
    "offset stops map updates until RESET or a new level.\n"
)

LEVEL_FEEDBACK_ADDENDUM = (
    "\n- Death/win feedback shows up to 10 intermediate images of the action, "
    "evenly sampled in time order, plus the terminal board. These are samples "
    "and can omit brief events; the entry's `.animation_frames` and "
    "`.animation_timeline` cover every state of the action.\n"
    "\n- Completion/death feedback counts actually executed game actions. "
    "A submitted action list has a known size; future action calls inside "
    "Python loops or branches do not have a known total. If a sequence ends "
    "on a win or death, infer its cause from the last executed action, its "
    "animation and the terminal board, not the intended final pose of the whole "
    "plan. A death is "
    "the fatal entry in `level_deaths[n - 1][d]`; its death frame is also the "
    "corresponding `level_history[n - 1][k].frame`, and its animation is in "
    "`.animation_frames` and `.animation_timeline`. The harness labels the exact non-negative `n - 1`, `d`, "
    "and `k` for each death image. The current frame after a death is already "
    "the automatic RESET state, so inspect the labelled death image or those "
    "entries rather than the current board. Do not use moving references such "
    "as `level_deaths[-1][-1]` or `level_history[-1][-3]` for terminal evidence. "
    "After context compression, labelled images restore the recorded winning "
    "boards of completed levels and give their stable `level_history[n - 1][k]` "
    "references; use them to check your rule hypotheses.\n"
)

STEP_VERIFICATION_ADDENDUM = (
    "- During a multi-action plan, verify important effects by comparing the actual game "
    "state with your predictions—for example, changed cells, object properties, positions, "
    "or counters. Make these comparisons in code so the batch can stop when a mismatch "
    "undermines the remaining plan. Test uncertain interactions before committing to long "
    "batches.\n"
    "- If your search finds no solution under your current model of the game, remember "
    "that the game is solvable. Reconsider your mechanics, goal, search implementation, "
    "or search limits, including interactions with new elements. Take a targeted action "
    "to test an uncertain rule or overlooked interaction, then update your model from "
    "the result.\n"
)

WORLD_MODEL_ADDENDUM = (
    "- Maintain a compact working world model: what entities or regions exist, what actions "
    "seem to do, what the goal likely is, what remains uncertain, and what plan best fits the "
    "evidence so far.\n"
    "- Optimize for the shortest reliable sequence that advances the current goal as described "
    "by your world model. If confidence is low, program a discriminating probe and revise the "
    "world model from the result.\n"
)

WORLD_MODEL_FREE_ADDENDUM = (
    "- Optimize for the shortest reliable sequence that advances the current goal. If "
    "confidence is low, program a discriminating probe and revise your approach from the "
    "result.\n"
)

DEATH_GUARD_ADDENDUM = (
    "- An action that ended a previous attempt from this EXACT board state is refused before "
    "execution: it raises `KnownDeathActionError`, nothing runs, and no attempt is lost. "
    "`stop_reason` is `known_death`. The guard checks every position of a batch, not just the "
    "first, so a fatal move is caught whichever route reached it. To repeat such an action "
    "deliberately, issue it again as your very next action.\n"
)

NOOP_GUARD_ADDENDUM = (
    "- An action already proven to change nothing in this EXACT board state is refused before "
    "execution: it raises `KnownNoOpActionError`, which is NOT catchable and ends the snippet - "
    "nothing runs, "
    "and no action budget is spent. `stop_reason` is `known_noop`. If you have a specific reason "
    "to believe the action will behave differently now, issue that same action again as your very "
    "next action and it will execute.\n"
)

PREFER_TOOL_CALLS_LINE = (
    # The closing clause used to warn that a long chain "wastes the whole turn".
    # That overstated it - a truncated reply keeps its reasoning in history, and
    # what is lost is only that nothing executed - and it aimed at the tail,
    # since truncation touched about 4% of replies. The speed claim is the
    # stronger one and it is true: decode runs at tens of tokens per second per
    # stream, while a tool call is a short generation plus a mostly-cached
    # prefill, so 2,000 tokens of further reasoning costs far more wall-clock
    # than running the code that would settle the question.
    "- Reading and computing cost nothing; only `action(...)` spends the level budget. When you "
    "are unsure, prefer another tool call over more reasoning: the code answers what the "
    "reasoning would only guess at, and running it is faster than thinking your way to the "
    "same answer.\n"
)

EPHEMERAL_FUNCTIONS_LINE = (
    "- The `python` tool code is not saved between calls, so rewrite any custom utility logic you still need.\n"
)

PERSISTENT_FUNCTIONS_LINE = (
    "- Python variables reset between tool calls. Eligible functions you define are "
    "retained automatically {lifetime}. Call your retained functions directly without "
    "repeating their definitions. Redefine a function when you need to change it.\n"
    "- For a function to be retained, its dependencies must also be available next call. "
    "Pass snippet-specific data as arguments. You may use Python builtins, provided "
    "globals such as `current_frame`, and other retained functions. Provided globals "
    "are refreshed for each call. {import_guidance} Use plain "
    "top-level definitions without decorators or annotations, and literal defaults. "
    "Tool results report which functions were retained and explain any rejection.\n"
)


# ARC3_ACT_BALANCE: the two tool-call lines below become one that also weighs analysis against
# a cheap action. bp35 spent dozens of turns analysing death frames pixel by pixel and s5i5
# forty turns on a simulator without acting (all25 v3, hard10 v18).
ACT_BALANCE_OLD = (
    "- You can call the `python` tool as many times as you want per step. Investigate until your code has a clear probe or plan.\n"
    "- Do not ration tool calls when the state is unclear. Spend extra tool calls to confirm what changed between frames and whether the last action affected gameplay state or only HUD elements such as countdown bars.\n"
)
ACT_BALANCE_NEW = (
    "- You can call the `python` tool as many times as you want per step; there is no need to ration calls. "
    "Balance analysis with action, though: when a question can be settled by one cheap action in the game, "
    "taking that action is often simpler and faster than analysing further.\n"
)


COMPACT_TOOL_SESSION_ADDENDUM = (
    "\n\nTool session rules:\n"
    "- You have exactly one tool: `python`.\n"
    f"- {TOOL_CALL_FORMAT_GUIDANCE}\n"
    "{persistent_functions}"
    "- You can call the `python` tool as many times as you want per step. Investigate until your code has a clear probe or plan.\n"
    "- Do not ration tool calls when the state is unclear. Spend extra tool calls to confirm what changed between frames and whether the last action affected gameplay state or only HUD elements such as countdown bars.\n"
    "{prefer_tool_calls}"
    "- After `action(...)` returns, the structured runtime state is refreshed before the next Python statement and before the next tool call. Inspection-only Python calls do not clear `last_action_call_result`.\n"
    "- Each `python` tool call has a hard time limit of {python_timeout_seconds} seconds.\n"
    "- Tool responses are capped to about {tool_output_tokens} tokens. If a response is cut off, the tool result will tell you that.\n"
    "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n"
)


# Asked once at each context trim, before the older turns are removed; the
# reply is attached to the newest user message of the trimmed history.
COMPACT_NOTE_PROMPT = (
    "Older turns are about to be removed from your context to make room. Write a short note "
    "(keep it brief, under 500 tokens) that will be shown to you after the removal. Include only:\n"
    "1. Goal: what the current level most likely requires; say so if it is uncertain.\n"
    "2. Actions: what each action or key does, as observed (for example SPACE, MOUSE on a "
    "given object, UNDO).\n"
    "3. Rules: rules you have established, marked [verified], and hypotheses still to test, "
    "marked [to verify].\n"
    "Do not include coordinates, the current board state, or a plan. Do not call any tool; "
    "reply with the note only."
)
COMPACT_NOTE_HEADER = (
    "Note you wrote just before older turns were removed from your context "
    "(goal, action effects, rules):"
)
# ARC3_COMPACT_NOTE_STYLE=experiments: a lab notebook instead of the rule note.
# The rule note turns a narrow negative finding into a rule with no scope (sp80:
# four level-1 clicks became "MOUSE = no-op", carried through three notes, and
# the red bars in level 2 were never clicked); an entry keeps what was tried,
# where and why next to what was concluded from it.
# {max_tokens} is ARC3_COMPACT_NOTE_MAX_TOKENS and caps each reply. The model
# writes only new entries; the harness keeps the earlier ones verbatim and
# shows the goal line first, then the entries with the current level first
# (_render_lab_notebook). Copying the notebook forward let the model reword
# and strengthen old entries (cn04 in v11: a partial test became "[confirmed
# negative]"), and a growing copy was cut at the end (sp80 lost its goal).
# Entries hold observations only: the readings and death causes were where the
# errors were (v14: sp80 "MOUSE inert in L2 as in L1" after one click on the
# selected bar; cn04 read a selection switch as a weld; v11 cn04 blamed a step
# budget on real time).
COMPACT_NOTE_EXPERIMENTS_PROMPT = (
    "Older turns are about to be removed from your context. Add to your lab notebook: write "
    "only the new entries, for what you have tried since the notebook shown above (or since "
    "the game started, if there is none). The earlier entries are kept for you as they are. "
    "Keep this reply under {max_tokens} tokens. Start with one line: `Goal (current level): "
    "your best guess, and how sure you are.` Then one line per new entry, oldest first:\n"
    "`L<level> | purpose: what you wanted to find out or achieve | actions: what you did, "
    "naming the objects involved | observed: what changed in which objects, or that nothing "
    "changed`\n"
    "For each death (game over) write an entry in this form instead:\n"
    "`L<level> | death | actions: what you did just before it | observed: what the death "
    "frames show, including any budget bar`\n"
    "In `observed`, describe only what happened to the objects: no interpretation, cause, rule "
    "or judgement. Merge consecutive actions taken toward the same purpose into one entry, and "
    "deaths after the same actions into one entry with their count. Keep an action that did "
    "nothing tied to the objects and level where you tried it. In `actions`, say exactly which "
    "state you acted on (which objects, how many, where), so it can be told apart from "
    "variants you have not tried. Do not include coordinates, the board state or a "
    "plan. Reply with the goal line and the new entries only."
)
COMPACT_NOTE_EXPERIMENTS_HEADER = (
    "Lab notebook from earlier in this game (what you tried and what you observed). It was "
    "written with thinking off, just before older turns were removed; earlier entries are kept "
    "as they were. After its level each entry carries `[s<a>-<b>]`, the steps of the turns it "
    "was written for: `[e for e in level_history[L - 1] if a <= e.frame.step <= b]` returns "
    "those boards on level L, each with `.action` and `.frame` (deaths are also in "
    "`level_deaths`). The "
    "entries are condensed and may be wrong in detail; when one matters, check it against those "
    "boards:"
)

# ARC3_ACTION_INDEX: each entry names the level_history indices of its own actions, which the
# message after every executed batch states; replaces the step range the harness stamps on all
# entries of one note alike (su15: every level-4 entry of a note carried [s0-81]).
COMPACT_NOTE_INDEX_FORMAT = (
    "After the level, every entry (deaths too) gives the `level_history` indices of the first "
    "and last action it covers as `[k<a>-<b>]` (one action: `[k<a>]`), copied from the line "
    "\"In `level_history` these actions are ...\" of the messages after those actions: "
    "`L<level> [k<a>-<b>] | purpose: ...`.\n"
)
COMPACT_NOTE_EXPERIMENTS_HEADER_INDEXED = (
    "Lab notebook from earlier in this game (what you tried and what you observed). It was "
    "written with thinking off, just before older turns were removed; earlier entries are kept "
    "as they were. After its level each entry carries `[k<a>-<b>]`: `level_history[L - 1][a:b + 1]` "
    "are the boards after the actions it describes, each with `.action` and `.frame` (deaths are "
    "also in `level_deaths`); an entry without one carries `[s<a>-<b>]`, the steps of the turns "
    "it was written for: `[e for e in level_history[L - 1] if a <= e.frame.step <= b]`. The "
    "entries are condensed and may be wrong in detail; when one matters, check it against those "
    "boards:"
)


SUMMARY_REQUEST_PROMPT = (
    "Summarize everything above: what you have established about this game, what you have "
    "ruled out, what you were in the middle of, and anything else a later turn would need. "
    "Keep uncertain ideas tentative, and preserve unresolved alternatives. "
    "Do not act and do not call any tool - write the summary only.\n"
    # No "the turns after this one stay visible": the summary is appended at
    # the end of history, so nothing follows it when it is written. That
    # sentence belonged to an earlier design where the summary was inserted
    # ahead of a retained tail, and here it points at messages that do not
    # exist yet.
    "Write it to stand completely on its own. Everything above this point will eventually be "
    "dropped and replaced by this summary, and earlier summaries go with it, so anything you "
    "leave out is gone.\n"
    "Length is not a problem here; leaving something out is."
)


# ARC3_SCHEMA_STOP=1 (the default): how action() stops a plan; SCHEMA_REPORT_TEXT under 0
SCHEMA_STOP_TEXT = (
    "; at the first wrong prediction (by the rule above, a wrong "
    "dead or win, or an `is_win_condition` that disagrees with whether the level was completed) "
    "the rest of the actions are dropped. `action(actions)` without a model runs only its "
    "first action. Calls with a model may follow one another while every prediction holds; "
    "once an action in a Python call ran without a model, or was mispredicted, any "
    "further `action()` in that call is refused with `ActionStoppedError`. Each wrong "
    "prediction prints its own line saying where it differs.\n"
)
SCHEMA_STOP_TEXT_CALL = (
    "; at the first wrong prediction (by the rule above, a wrong "
    "dead or win, or an `is_win_condition` that disagrees with whether the level was completed) "
    "the rest of the actions of that call are dropped, and you may call `action()` again. "
    "`action(actions)` without a model runs only its first action, and only one such call "
    "per Python call (a second one is refused with `ActionStoppedError`); calls with a model "
    "are not limited. Each wrong prediction prints its own line saying where it differs.\n"
)
# ARC3_SCHEMA_FREE_ACTIONS above 1 (with ARC3_SCHEMA_STOP=call): the limit on calls without
# a model, in SCHEMA_STOP_TEXT_CALL and in the batching line, becomes this
SCHEMA_FREE_OLD = (
    "`action(actions)` without a model runs only its first action, and only one such call "
    "per Python call (a second one is refused with `ActionStoppedError`)"
)
SCHEMA_FREE_NEW = (
    "`action(actions)` without a model runs up to {n} actions per Python call, unchecked, "
    "which is meant for exploring (once {n} have run, a further call without a model is "
    "refused with `ActionStoppedError`)"
)
SCHEMA_FREE_BATCH_OLD = (
    "- `action(actions)` without a model runs only its first action; to run a sequence,"
)
SCHEMA_FREE_BATCH_NEW = (
    "- `action(actions)` without a model runs up to {n} actions per Python call, unchecked, "
    "to explore; to run a longer sequence,"
)
SCHEMA_REPORT_TEXT = (
    ". Every wrong prediction (by the rule above, a wrong dead or win, or an `is_win_condition` "
    "that disagrees with whether the level was completed) prints its own line saying where "
    "it differs, and the plan goes on; at the end the call lists the actions it "
    "mispredicted. `action(actions)` without a model runs all the actions unchecked.\n"
)

# ARC3_SCHEMA: the world model section, before "Tool session rules"
SCHEMA_ADDENDUM = (
    "\n\nWorld model (step):\n"
    "- Write the game's mechanism as a Python function:\n"
    "      def step(frame, action, row=None, col=None):\n"
    "          # frame: a board like current_frame (.ascii, .segmentation); action: 'LEFT', 'SPACE', "
    "'MOUSE', ... (row, col: the click of a MOUSE)\n"
    "          # return the predicted next board as 64 strings in the letters of .ascii, or "
    "(board, {'dead': bool, 'win': bool}): dead when the action ends the attempt, win when "
    "it completes the level (then the board is the one at the moment of the win); a board "
    "alone means neither\n"
    "  Build it from recorded transitions: `level_history[-1][0].frame` is this level's first "
    "board, and each entry's `.action` and `.frame` is a real transition from the entry before "
    "it. Identify objects from `.segmentation['nodes']` and write how actions move or change "
    "them; read positions from the board instead of hard-coding them. `board(frame)` returns "
    "an editable 64x64 list of letters to start a prediction from. UNDO is never judged: "
    "step is not called for it and its prediction counts as right, so step need not "
    "model it.\n"
    "- Bundle it into one world model: `model = WorldModel(step, key_state=None, "
    "is_win_condition=None)`. `key_state(frame)` returns the state of the objects the game "
    "mechanism actually depends on, as a dict, e.g. `{'player': <a segmentation node>, "
    "'keys': 2, 'door_open': False}`. When given, a prediction is right if key_state of the "
    "predicted board equals key_state of the real board (same keys; equal numbers and "
    "strings; nodes with the same color and coordinates), and other cells may differ; "
    "without it, the whole board is compared and a connected difference of 9 or more cells "
    "inside the board is wrong. `is_win_condition(frame)` is the model's rule for completing "
    "the level. Use the model in your own searches for a plan.\n"
    "- `action(actions, model=model)` runs the actions one at a time: before each it predicts "
    "the next board with `model.step`, and after it judges the prediction with "
    "`check_prediction`" + SCHEMA_STOP_TEXT +
    "- `check_prediction(model, before, after)` is that same judgement, for you to call: it "
    "runs `model.step` on `before` with the action of the history entry `after` and compares "
    "the result with it (for a bare frame `after`, pass `action=`; whether it was fatal or "
    "completed the level is then unknown). It returns {'consistent', 'action', 'where', "
    "'pred', 'real', 'key_state_diff'}, and `last_prediction` holds the same for the latest "
    "action of `action()`. `where` says what is wrong and where on the board the predicted "
    "and real boards differ most (rows and cols, given even when key_state judged the "
    "prediction right); `pred` and `real` are the predicted and real boards as frames. To "
    "list the objects your prediction got wrong, call `frame_diff(pred, real)`: here "
    "frame_diff compares your prediction with the real board, not the board before an "
    "action with the board after it.\n"
    "- `backtest(model, level=-1)` judges the model on every recorded transition of "
    "`level_history[level]` and returns {'checked', 'correct', 'wrong_indices', 'recent'}: "
    "`recent` holds the latest 5 wrong transitions, each with its 'index', 'action', 'where', "
    "'key_state_diff' and the predicted and real rows around where they differ "
    "('predicted_crop', 'real_crop'); index (L, k) means the prediction was of "
    "`level_history[L][k].frame`. `last_backtest['wrong']` holds every wrong transition in "
    "full, each with `pred` and `real` frames for frame_diff. backtest also runs "
    "by itself after a call that builds a WorldModel or redefines a function the model you "
    "last used calls. A model looks its functions up by name, so redefining step or a "
    "function it calls updates the model.\n"
    "- Your functions and the model stay in the Python process across turns and context "
    "trims; after a trim, the next message restates the model you last used and the source "
    "of its functions and of the functions of yours they call.\n"
)

# ARC3_SCHEMA_PARSE: the world model section instead of SCHEMA_ADDENDUM. The state is a dict
# of objects (center / bbox / cells, color, symbol, properties) and plain values; the block
# map example is bp35's cellstat from run 38 with other colours and symbols.
# The block-map guidance: in the World model section with ARC3_SCHEMA_PARSE, and in its
# place (BLOCK_MAP_SECTION) without ARC3_SCHEMA when ARC3_GRID_MAP asks for a block map.
BLOCK_MAP_TEXT = (
    " Objects are often "
    "drawn as blocks of n*n cells: write `cellstat(frame, row, col, h, w)`, which returns one "
    "symbol for what the h*w rectangle centred at (row, col) holds, decided from its color "
    "counts (or a shape), and print the board by sampling centres spaced by the block size. "
    "For 5*5 blocks whose first centre is (2, 2):\n"
    "      def cellstat(frame, row, col, h=5, w=5):\n"
    "          rows = frame.ascii.split('\\n')\n"
    "          count = {}\n"
    "          for r in range(row - h // 2, row + h // 2 + 1):\n"
    "              for c in range(col - w // 2, col + w // 2 + 1):\n"
    "                  ch = rows[r][c] if 0 <= r < len(rows) and 0 <= c < len(rows[r]) else '?'\n"
    "                  count[ch] = count.get(ch, 0) + 1\n"
    "          if count.get('Y', 0) >= 8: return 'K'   # key\n"
    "          if count.get('M', 0) >= 8: return 'D'   # door\n"
    "          if count.get('g', 0) >= 15: return '.'  # floor\n"
    "          if count.get('c', 0) >= 15: return '#'  # wall\n"
    "          return '?'\n"
    "      for r in range(2, 64, 5):\n"
    "          print(f'{r:>2} ' + ''.join(cellstat(current_frame, r, c) for c in range(2, 64, 5)))\n"
)
BLOCK_MAP_SECTION = (
    "\n\nBlock map:\n"
    "- A block map helps you read the board." + BLOCK_MAP_TEXT
    + "  Add a symbol when you meet a new kind of object.\n"
)

SCHEMA_PARSE_ADDENDUM = (
    "\n\nWorld model (parse, step):\n"
    "- Write the game's mechanism as Python functions on a parsed state:\n"
    "      def parse(frame):\n"
    "          # frame: a board like current_frame (.ascii, .segmentation); return a dict of what "
    "the mechanism depends on\n"
    "      def step(state, action, row=None, col=None):\n"
    "          # state: a copy of parse(frame); action: 'LEFT', 'SPACE', 'MOUSE', ... (row, col: "
    "the click of a MOUSE)\n"
    "          # return the predicted next state, or (state, {'dead': bool, 'win': bool}): dead "
    "when the action ends the attempt, win when it completes the level; a state alone means "
    "neither\n"
    "      def is_win_condition(state):\n"
    "          # the rule for completing the level\n"
    "  In the state, each object is a dict with its 'center' (row, col) and its 'bbox' (r0, "
    "c0, r1, c1, inclusive), both required, and its 'cell_num' (how many cells it has), "
    "'main_color', a one-character 'symbol' and any property that matters ('rotation', "
    "'shape', ...); do not list an object's cells one by one: an irregular shape made of "
    "square blocks is given by the centres of its blocks ('centers'), one point per n*n "
    "square block. Objects of one kind may share a list, each item an object dict (a list is "
    "compared ignoring its order, so it needs no sorting or numbering). Everything else is a "
    "plain value (a counter, a switch, a mode). For example:\n"
    "      {'player': {'center': (34, 21), 'bbox': (32, 19, 36, 23), 'cell_num': 25, "
    "'main_color': 'b', 'symbol': 'P'},\n"
    "       'boxes': [{'center': (20, 40), 'bbox': (18, 38, 22, 42), 'cell_num': 25, "
    "'main_color': 'O', 'symbol': 'B'},\n"
    "                 {'center': (20, 52), 'bbox': (18, 50, 22, 54), 'cell_num': 25, "
    "'main_color': 'O', 'symbol': 'B'}],\n"
    "       'T_piece': {'center': (46, 13), 'bbox': (42, 9, 50, 17), 'cell_num': 45, "
    "'main_color': 'Y', 'symbol': 'T', 'rotation': 0, 'centers': [(43, 10), (43, 13), "
    "(43, 16), (46, 13), (49, 13)]},\n"
    "       'door_open': False, 'moves_left': 12}\n"
    "  Never put the coordinates of all of an object's cells in the state: a state holding "
    "more than 300 numbers and strings in all is refused, and the plan stops.\n"
    "  Build parse from `.segmentation['nodes']` (each object's color, boundary and pixels) "
    "and the board, reading positions from the board instead of hard-coding them. Build step "
    "from recorded transitions: `level_history[-1][0].frame` is this level's first board, and "
    "each entry's `.action` and `.frame` is a real transition from the entry before it. UNDO "
    "is never judged: step is not called for it and its prediction counts as right.\n"
    "- Bundle them into one world model: `model = WorldModel(parse, step, "
    "is_win_condition=None)`. A prediction is right when step(parse(the board before)) "
    "equals parse(the board after): the same keys and equal numbers, strings, tuples and lists; "
    "a list is compared ignoring its order (as a multiset), so use a tuple where order "
    "matters. Boards are not compared cell by cell, so "
    "parse only what the mechanism depends on, but parse it all: after each action, every "
    "object frame_diff finds changed on the board (moved, rotated, appeared, disappeared, "
    "recolored or resized by 4 or more pixels) should lie inside the 'bbox' of one of "
    "parse's objects, before or after the action. One outside every bbox, or inside an "
    "object whose parsed value did not change, is reported and the plan goes on: parse is "
    "then missing an object or a property, and step cannot model it. Use the model in your "
    "own searches for a plan.\n"
    "- `action(actions, model=model)` runs the actions one at a time: before each it predicts "
    "the next state with `model.step`, and after it judges the prediction with "
    "`check_prediction`" + SCHEMA_STOP_TEXT +
    "- `check_prediction(model, before, after)` is that same judgement, for you to call: it "
    "runs `model.step` on parse(`before`) with the action of the history entry `after` and "
    "compares the result with parse(`after`) (for a bare frame `after`, pass `action=`; "
    "whether it was fatal or completed the level is then unknown). It returns {'consistent', "
    "'action', 'where', 'pred', 'real', 'state_diff', 'real_frame'}, and `last_prediction` "
    "holds the same for the latest action of `action()`: `pred` is step's state, `real` is "
    "parse of the real board, `state_diff` lists each key that differs with its predicted and "
    "real value, `real_frame` is the real board, and 'uncovered' holds what the report above "
    "found.\n"
    "- `backtest(model, level=-1)` judges the model on every recorded transition of "
    "`level_history[level]` and returns {'checked', 'correct', 'wrong_indices', 'recent', "
    "'uncovered_indices'}: `recent` holds the latest 5 wrong transitions, each with its "
    "'index', 'action', 'where' and 'state_diff'; index (L, k) means the prediction was of "
    "`level_history[L][k].frame`; `uncovered_indices` are the transitions with board changes "
    "parse missed, and `last_backtest['uncovered']` lists each with what it missed. backtest "
    "also runs by itself after a call that builds a WorldModel or redefines a function the "
    "model you last used calls, and reports both its wrong predictions and the changes parse "
    "missed. A model looks its functions up by "
    "name, so redefining parse, step or a function they call updates the model.\n"
    "- Your functions and the model stay in the Python process across turns and context "
    "trims; after a trim, the next message restates the model you last used and the source "
    "of its functions and of the functions of yours they call.\n"
    + "- A block map helps you read the board (it is not part of the state)." + BLOCK_MAP_TEXT
    + "  Add a symbol when you meet a new kind of object, and give parse's objects the same "
    "symbols.\n"
)

# ARC3_SCHEMA_PARSE: right after the core workflow bullet (the search push, the schema
# workflow or the default loop): read the world model's reports after every action
SCHEMA_PARSE_CHECK_LINE = (
    "- After every `action(...)` call, before anything else, read the `[model]` lines it "
    "printed, and fix the model before you plan again: (1) for an action MISPREDICTED, read "
    "`last_prediction['state_diff']` (step's state against parse of the real board) and fix "
    "step; (2) for a changed object that lies outside every object's bbox, or an object "
    "whose parsed value did not change, look at it with `frame_diff(previous_frame, "
    "current_frame)` and add the object or property to parse; (3) after you redefine parse "
    "or step, plan the next step only once the automatic backtest line shows every "
    "transition predicted and no board changes parse missed (`last_backtest['wrong']` and "
    "`last_backtest['uncovered']` hold the details).\n"
)
SCHEMA_PARSE_CHECK_ANCHORS = (
    "- Before each search, state the next subgoal.",  # ARC3_SEARCH_PUSH2
    "- Workflow: ",  # ARC3_SCHEMA without the search push
    "- A strong default loop is: ",
)

# The core workflow: one numbered section after the game overview, in the order the work is
# done. It replaces the level-transfer bullet (ARC3_LEVEL_TRANSFER_GUIDANCE /
# ARC3_NEW_ELEMENT_PROBE) and the search-push bullet (ARC3_SEARCH_PUSH2) whenever both are in
# the prompt. Steps 2 and 4 name the world model only when ARC3_SCHEMA is on (the parse
# version under ARC3_SCHEMA_PARSE); without it step 2 asks for the rules as Python.
CORE_WORKFLOW_HEAD = "\n\nWorkflow:\n"
CORE_WORKFLOW_EXPLORE = (
    "Explore: probe with `action(...)`, abstract the game's objects and state from what "
    "changes, and guess the goal."
)
CORE_WORKFLOW_MODEL = {
    "parse": "Model it: write the state (the objects and their state, abstracted) as `parse`, "
             "with `step` and `is_win_condition`, bundle them in a WorldModel, and build a "
             "block map (see World model below).",
    "schema": "Model it: write `step` and `is_win_condition` and bundle them in a WorldModel "
              "(see World model below).",
    # without ARC3_SCHEMA the later steps still speak of "your rules"
    "none": "Model it: write the rules you found as Python (a function from a state and an "
            "action to the next state) and search with it.",
}

# mode "none" with ARC3_GRID_MAP: the model step also asks for the block map, as the parse one does
CORE_WORKFLOW_BLOCK_MAP = ", and build a block map (see Block map below)."
# ARC3_GRID_MAP_AFTER_EXPLORE (with ARC3_GRID_MAP): the map is asked for once the objects are
# known, at the end of step 1, not with the rules in step 2 (sc25 never mapped its thin wire)
# (v10 asked "once you know the objects": ft09 mapped before it knew what a click did)
CORE_WORKFLOW_EXPLORE_MAP_HOOK = (
    " After you understand the objects and what the actions do, you can write a block-map "
    "function (with `cellstat`) and set `harness_grid_map` to it, if a simplified block map of "
    "the current frame, shown by the harness in every message, would help you understand the "
    "game (see Block map below).")
# ARC3_GRID_MAP_AFTER_EXPLORE=short: the same offer in one short sentence (demo10 v12)
CORE_WORKFLOW_EXPLORE_MAP_HOOK_SHORT = (
    " After that, if a simplified map would help you, you can write a `block_map` function and "
    "set `harness_grid_map = block_map` to have the harness show it (see Block map below).")
CORE_WORKFLOW_EXPLORE_MAP = (
    " After you understand the objects and what the actions do, you can write "
    "`grid_map(frame)` (with `cellstat`), if a simplified block map of the current frame, shown "
    "by the harness in every message, would help you understand the game (see Block map below).")
CORE_WORKFLOW_PARSE_MAP_CLAUSE = ", and build a block map (see World model below)."
# ARC3_BLOCK_MAP_HALVES: the cellstat example also names a block filled only in one half.
# The half checks go after the whole-wall check and before the floor check: a block with two
# wall rows over three floor rows would otherwise read as floor.
BLOCK_MAP_HALVES_ANCHOR = (
    "          if count.get('g', 0) >= 15: return '.'  # floor\n"
    "          if count.get('c', 0) >= 15: return '#'  # wall\n"
)
BLOCK_MAP_HALVES_LINES = (
    "          if count.get('c', 0) >= 15: return '#'  # wall\n"
    "          # a block filled only in part: name the part\n"
    "          def part(ch, r0, r1, c0, c1):  # cells of ch in rows row+r0..row+r1, cols col+c0..col+c1\n"
    "              return sum(rows[r][c] == ch for r in range(row + r0, row + r1 + 1)\n"
    "                         for c in range(col + c0, col + c1 + 1))\n"
    "          if part('c', -2, -1, -2, 2) >= 8: return '^'   # wall in the top half\n"
    "          if part('c', 1, 2, -2, 2) >= 8: return 'v'     # wall in the bottom half\n"
    "          if part('c', -2, 2, -2, -1) >= 8: return '['   # wall in the left half\n"
    "          if part('c', -2, 2, 1, 2) >= 8: return ']'     # wall in the right half\n"
    "          if count.get('g', 0) >= 15: return '.'  # floor\n"
)
# ARC3_GRID_MAP_STEP (with ARC3_GRID_MAP): the model step names the function that holds the map
CORE_WORKFLOW_GRID_MAP_OLD = "build a block map"
CORE_WORKFLOW_GRID_MAP_NEW = "write `grid_map(frame)` (with `cellstat`) to build the block map"
# ARC3_GRID_MAP_STEP=optional: the map is offered, not required (demo10 v20)
CORE_WORKFLOW_GRID_MAP_OPTIONAL = (
    "if it helps you, write `grid_map(frame)` (with `cellstat`) to build the block map")
CORE_WORKFLOW_PLAN = {
    "parse": "Plan: search with step for a plan that reaches the goal, or split the goal into "
             "subgoals and search for each one.",
    "schema": "Plan: search with step for a plan that reaches the goal, or split the goal into "
              "subgoals and search for each one.",
    "none": "Plan: search for a plan that reaches the goal, or split the goal into subgoals "
            "and search for each one.",
}
CORE_WORKFLOW_PLAN_TAIL = (
    " Search a whole level only when it fits in about {max_states:,} expansions within one "
    "Python call; when a goal or subgoal is simple, plan it directly."
)
# ARC3_USER_WORKFLOW_RECAP (mode "none"): step 2 writes the rules without searching, step 3 plans
# first and searches only when needed (user wording, 2026-10-25; arc3-kleea-low8 v21 s5i5 L2
# spent 16 minutes building and searching an unchecked simulator)
CORE_WORKFLOW_MODEL_PLAN_FIRST = (
    "Model it: write the rules you found as Python (a function from a state and an action to the "
    "next state).")
CORE_WORKFLOW_PLAN_FIRST = (
    "Plan: make a plan that reaches the goal, or split the goal into subgoals and plan for each "
    "subgoal; when a goal or subgoal is simple, plan it directly. Search with your rules only when "
    "the goal is complex or no clear solution is apparent, and search a whole level only when it "
    "fits in about {max_states:,} expansions within one Python call."
)
CORE_WORKFLOW_CHECK = {
    "parse": "Act and check: run the plan with `action(...)` and read the result it returns. "
             "After every `action(...)` call, look at the real change with "
             "`frame_diff(previous_frame, current_frame)`, and fix what the `[model]` lines "
             "report: a misprediction (`last_prediction['state_diff']`) in step, a change parse "
             "missed in parse. After a fix, the automatic backtest line should show every "
             "transition predicted and nothing missed (`last_backtest['wrong']`, "
             "`last_backtest['uncovered']`).",
    "schema": "Act and check: run the plan with `action(...)` and read the result it returns. "
              "After every `action(...)` call, look at the real change with "
              "`frame_diff(previous_frame, current_frame)`; when the `[model]` lines report a "
              "misprediction, `frame_diff(last_prediction['pred'], last_prediction['real'])` "
              "shows what it got wrong: fix step.",
    "none": "Act and check: run the plan with `action(...)` and read the result it returns. "
            "After every `action(...)` call, look at the real change with "
            "`frame_diff(previous_frame, current_frame)` and correct your rules.",
}
CORE_WORKFLOW_FAIL = (
    "When a search fails, do not just enlarge its depth, width, restarts or search space. "
    "First note the goal, the assumptions (anchor, coordinate bounds, rotations, placement "
    "order, pairing / connectivity, overlap / coverage, pruning), the budget used and whether "
    "the search was exhausted or cut off: conclude only 'no solution under these "
    "assumptions', never that the goal is impossible. Then change the subgoal, or act to get "
    "new evidence and revise rules that may be wrong."
)
CORE_WORKFLOW_NEXT_LEVEL = (
    "After you clear a level and enter the next: probe new elements (when the level shows an "
    "element you have not seen, try every applicable action on it, including ones you know: "
    "click it; move onto, next to or into it; and note what changes: such levels are usually "
    "built so that the goal can only be reached through the new mechanic), and re-check the "
    "goal (it usually stays the same, but your description of it is often incomplete or "
    "wrong: check it against the winning boards of earlier levels, using their labelled, "
    "fixed history references)."
)
# ARC3_CONST_PARSE_WARN above 0: appended to Workflow step 6 (exploring a new level), and the
# warning the host adds to the next user message after that many calls in a row with a
# constant parse. v56 tn36 explored with `{'dummy': 1}` 38 of 39 model calls and won 7 levels;
# v57 tn36 never did and won 1.
CONST_PARSE_EXPLORE = (
    " While you explore, a model whose parse returns a constant (its step then returns the "
    "state unchanged) lets one call run several actions past the limit on actions without "
    "a model; it checks nothing."
)
CONST_PARSE_WARNING = (
    "Your last {n} `action()` calls passed a model whose parse stayed the same while the board "
    "changed (a constant parse). Unless the game hinges on intricate details of mouse clicks "
    "or on objects that move on their own and are hard to predict (opponents, allies), stop "
    "using a constant parse: it weakens your understanding of the game. Write a parse that "
    "tracks the objects."
)
# ARC3_DEATH_SUSPECT: appended to the death notice's fatal-action line. bp35 (gridmap rerun)
# died twice just below a purple item with one wall row between and ruled the item out
# because the cells did not overlap.
DEATH_SUSPECT_NOTE = (
    " When an action causes a death, the objects near where it happened are the first "
    "suspects, even if a row or two of outline or wall seems to lie between them."
)
# ARC3_DEMO_HINT: display / demonstration parts of the board (tn36's code panel and cartridge
# icons, su15's goal icon in the HUD band), appended to step 1. ARC3_ANIM_ONLY_HINT:
# animation-only effects (tn36 rerun level 2 kept a run's end position that the board reset),
# see DEMO_HINT_CHECK. Until v57 (hard10 v40) ARC3_DEMO_HINT added both; there tn36
# called the code panel a goal preview and ls20 its dial a logo, so the two were split.
DEMO_HINT_EXPLORE = (
    " Some parts of the board only display or demonstrate: an icon, a legend, a HUD band, "
    "or a side panel that shows what a selection does (its code, or a small copy that runs "
    "it) or what the goal looks like. Read them as labels (this icon = this effect, this "
    "code = this instruction)."
)
# ARC3_ANIM_ONLY_HINT: closes Workflow step 4, as in hard10 v41-v43 (v42 = the baseline);
# in v44-v48 it was moved into the per-action report, now ARC3_ANIM_ONLY_REPORT
DEMO_HINT_CHECK = (
    " An action reported as `effect during its animation only` leaves the state as it was; "
    "learn what it does from its `.animation_frames` and `.animation_timeline`."
)
ANIMATED_ONLY_VERDICT = "effect during its animation only (the board area ended as it was)"
# ARC3_ANIM_ONLY_REPORT: the per-action report itself says where the effect is and, with
# ARC3_SCHEMA, what step does with it (v41 tn36 kept predicting a run's end position: 12 of
# its 19 runs mispredicted); hard10 v44-v48 ran it in place of the Workflow sentence.
ANIMATED_ONLY_VERDICT_HINT = (
    "effect during its animation only (the board area ended as it was; its "
    "`.animation_frames` show what it did)"
)
ANIMATED_ONLY_VERDICT_STEP = (
    "effect during its animation only (the board area ended as it was; its "
    "`.animation_frames` show what it did; step need not simulate it: return the state "
    "unchanged)"
)


def animated_only_verdict() -> str:
    """The per-action report of an action whose effect showed only during its animation."""
    truthy = ("1", "true", "yes", "on")
    if os.environ.get("ARC3_ANIM_ONLY_REPORT", "0").strip().lower() not in truthy:
        return ANIMATED_ONLY_VERDICT
    schema = os.environ.get("ARC3_SCHEMA", "0").strip().lower() in truthy
    return ANIMATED_ONLY_VERDICT_STEP if schema else ANIMATED_ONLY_VERDICT_HINT


def animated_only_with_ref(item: dict) -> str:
    """The animation-only verdict, naming the action's history entry when known.

    "effect during its animation only (the board area ended as it was;
    level_history[i][k].animation_timeline)": the entry stays readable after a
    later action replaces last_animation_*.
    """
    verdict = animated_only_verdict()
    ref = item.get("animation_ref")
    if ref and verdict.endswith(")"):
        verdict = f"{verdict[:-1]}; {ref}.animation_timeline)"
    return verdict


# the bullets the core workflow replaces, each removed with its whole line
CORE_WORKFLOW_REMOVED = (
    "- Levels usually build on mechanics learned in earlier levels",
    "- Before each search, state the next subgoal.",
)


def core_workflow_text(
    max_states: int,
    mode: str,
    demo: bool = False,
    const_parse: bool = False,
    block_map: bool = False,
    anim_only: bool = False,
    grid_map_step: bool | str = False,
    grid_map_hook: bool = False,
    map_after_explore: bool | str = False,
    plan_first: bool = False,
) -> str:
    """The numbered workflow section; mode is "parse", "schema" or "none".

    plan_first (ARC3_USER_WORKFLOW_RECAP, mode "none") rewords steps 2 and 3: the model step
    writes the rules without searching, the plan step searches only when no plan is apparent.

    demo adds the ARC3_DEMO_HINT sentence to the explore step, anim_only the
    ARC3_ANIM_ONLY_HINT sentence to the check step, const_parse the
    ARC3_CONST_PARSE_WARN sentence to the next-level step; block_map (mode "none" with
    ARC3_GRID_MAP, whose Block map section the model step then points to) asks for a block map.
    grid_map_step (ARC3_GRID_MAP_STEP, with block_map) has the model step ask for the map as
    `grid_map(frame)` written with `cellstat`; grid_map_hook (ARC3_GRID_MAP_HOOK) names
    `harness_grid_map` instead. map_after_explore (ARC3_GRID_MAP_AFTER_EXPLORE, with
    block_map) asks for the map at the end of the explore step instead of in the model step;
    "short" (with grid_map_hook) uses the one-sentence wording.
    """
    steps = [CORE_WORKFLOW_EXPLORE + (DEMO_HINT_EXPLORE if demo else "")]
    if block_map and map_after_explore:
        if grid_map_hook:
            steps[0] += (CORE_WORKFLOW_EXPLORE_MAP_HOOK_SHORT if map_after_explore == "short"
                         else CORE_WORKFLOW_EXPLORE_MAP_HOOK)
        else:
            steps[0] += CORE_WORKFLOW_EXPLORE_MAP
    plan_first = plan_first and mode == "none"
    if mode in CORE_WORKFLOW_MODEL:
        model_step = CORE_WORKFLOW_MODEL_PLAN_FIRST if plan_first else CORE_WORKFLOW_MODEL[mode]
        if block_map and map_after_explore:
            # the parse step's own block-map clause moves to the explore step too
            model_step = model_step.replace(CORE_WORKFLOW_PARSE_MAP_CLAUSE, ".")
        elif mode == "none" and block_map:
            model_step = model_step[:-1] + CORE_WORKFLOW_BLOCK_MAP
        if block_map and grid_map_step and not map_after_explore:
            if grid_map_step == "optional":
                new = CORE_WORKFLOW_GRID_MAP_OPTIONAL
            else:
                new = CORE_WORKFLOW_GRID_MAP_HOOK if grid_map_hook else CORE_WORKFLOW_GRID_MAP_NEW
            model_step = model_step.replace(CORE_WORKFLOW_GRID_MAP_OLD, new)
        steps.append(model_step)
    plan = (CORE_WORKFLOW_PLAN_FIRST.format(max_states=max_states) if plan_first
            else CORE_WORKFLOW_PLAN[mode] + CORE_WORKFLOW_PLAN_TAIL.format(max_states=max_states))
    steps += [plan,
              CORE_WORKFLOW_CHECK[mode] + (DEMO_HINT_CHECK if anim_only else ""),
              CORE_WORKFLOW_FAIL,
              CORE_WORKFLOW_NEXT_LEVEL + (CONST_PARSE_EXPLORE if const_parse else "")]
    return CORE_WORKFLOW_HEAD + "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))

# ARC3_SCHEMA_PARSE_MAP: the block map becomes a required part of the parsed state
SCHEMA_PARSE_MAP_ANCHOR = "  Build parse from `.segmentation['nodes']`"
SCHEMA_PARSE_MAP_RULE = (
    "  The state must also hold 'map': the block map (below) as a list of strings, one "
    "character per block, e.g. 'map': ['#####', '#.P.#', '#.K.#', '#####']; step predicts it "
    "too. A parse without a 'map' fails the prediction, so the rest of the plan is dropped "
    "(except in a level whose view has scrolled).\n"
)
SCHEMA_PARSE_MAP_REPLACED = "- A block map helps you read the board (it is not part of the state)."
SCHEMA_PARSE_MAP_REPLACEMENT = "- The block map (the state's 'map') helps you read the board."

# ARC3_GRID_MAP: the model's own block map, kept apart from the state and shown in every message
GRID_MAP_SYSTEM = (
    "- `grid_map(frame)` starts as a stub that returns None. Replace it with your block map: "
    "return a list of strings, one character per block, sampled at block centres spaced by "
    "the block size and named with your `cellstat` symbols. It is not part of the world "
    "model's state; every message shows `grid_map(current_frame)`.\n"
)
# without ARC3_SCHEMA there is no world model to set the block map apart from
GRID_MAP_SYSTEM_STATE = "It is not part of the world model's state; every message"
GRID_MAP_SYSTEM_NO_SCHEMA = "Every message"
# ARC3_GRID_MAP_IN_BLOCK (with ARC3_GRID_MAP): grid_map is explained inside the block-map
# guidance, right after the cellstat example, instead of as a separate Python-tool line;
# the reminder becomes conditional. Model maps were rare while the two were apart.
GRID_MAP_BLOCK_TEXT = (
    "  Then write or update `grid_map(frame)` with your `cellstat`: it returns the block map "
    "as a list of strings, one per row of blocks. For the 5*5 blocks above:\n"
    "      def grid_map(frame):\n"
    "          return [''.join(cellstat(frame, r, c) for c in range(2, 64, 5))\n"
    "                  for r in range(2, 64, 5)]\n"
    "  (5*5 tiles with 1-cell gaps between them, as in a maze, are 6 cells apart: "
    "range(3, 63, 6).) The harness finds the function by this name and shows "
    "grid_map(current_frame) in every message.\n"
)
# ARC3_GRID_MAP_HOOK (implies the in-block placement): the shown map is whatever function
# the model assigns to harness_grid_map; v8's maps lived under a dozen other names
GRID_MAP_HOOK_BLOCK_TEXT = (
    "  Then write or update a map function with your `cellstat`: it returns the block map as "
    "a list of strings, one per row of blocks. For the 5*5 blocks above:\n"
    "      def block_map(frame):\n"
    "          return [''.join(cellstat(frame, r, c) for c in range(2, 64, 5))\n"
    "                  for r in range(2, 64, 5)]\n"
    "      harness_grid_map = block_map\n"
    "  (5*5 tiles with 1-cell gaps between them, as in a maze, are 6 cells apart: "
    "range(3, 63, 6).) The harness shows harness_grid_map(current_frame) in every message: "
    "whenever you write a better map function, under any name, set `harness_grid_map` to it.\n"
)
GRID_MAP_HOOK_REMINDER = (
    "You have no harness_grid_map yet: if most objects can be drawn as n*n blocks or "
    "combinations of n*n blocks, write a map function and set `harness_grid_map = "
    "your_map_function` (see Block map); every message will then show it."
)
GRID_MAP_HOOK_HEADER = "Your harness_grid_map(current_frame), one row per line:"
# Workflow step 2 with ARC3_GRID_MAP_STEP and the hook
CORE_WORKFLOW_GRID_MAP_HOOK = (
    "write a block-map function (with `cellstat`) and set `harness_grid_map` to it")
GRID_MAP_REMINDER_SHORT = (
    "You have no grid_map yet: write it if most objects can be drawn as n*n blocks or "
    "combinations of n*n blocks (see Block map)."
)
GRID_MAP_HEADER = "Your grid_map(current_frame), one row per line:"
# ARC3_AUTO_MAP: the header of the harness's own map, which is grid_map until the model
# defines one (formatted with the detected lattice)
AUTO_MAP_HEADER = (
    "grid_map(current_frame), the harness's auto map (yours once you define grid_map; "
    "auto_grid_map(frame) stays available): one character per {s}*{s} tile, tiles every {p} "
    "cells, the first at row {r0}, col {c0}. A colour letter is the tile's main colour; '-' "
    "'|' a centre line, '+' 'x' a cross, 'o' a centred square. Details smaller than a tile "
    "are not shown."
)
GRID_MAP_REMINDER = (
    "You have no grid_map yet: write `cellstat(frame, row, col, h, w)` (one symbol for the "
    "h*w block centred at (row, col), from its color counts), then define `grid_map(frame)` "
    "to sample the board at block centres spaced by the block size and return one string of "
    "cellstat symbols per row of blocks. Every message will then show it."
)
# ARC3_GRID_MAP_OPTOUT: GRID_MAP_REMINDER with the block size bounded below (bp35 mapped the board in 2*2
# blocks) and a way to decline the map; `grid_map = False` silences it for the rest of the game.
GRID_MAP_REMINDER_OPTOUT = (
    "You have no grid_map yet: write `cellstat(frame, row, col, h, w)` (one symbol for the "
    "h*w block centred at (row, col), from its color counts), then define `grid_map(frame)` "
    "to sample the board at block centres spaced by the block size (n*n blocks, n of 3 or "
    "more) and return one string of cellstat symbols per row of blocks. Every message will "
    "then show it. If most objects cannot be drawn as n*n blocks with n of 3 or more, set "
    "`grid_map = False` instead: this note then stops for the rest of the game, and a "
    "grid_map you define later is shown again."
)
# ARC3_SCALE_NOTE: after the per-action trace and the animation note of a turn in which an
# object was scaled (the first two such turns of each level); {scaled} names the objects.
SCALE_NOTE = (
    "Note: an object changed size by an integer factor in that sequence ({scaled}). "
    "Scaling may be an important mechanic: it means an object's size carries information. "
    "The goal may need an object to shrink to pass through a narrow passage, or to be "
    "resized to match another object, and so on."
)
# ARC3_GRID_MAP_UNKNOWN: after a shown grid_map that holds '?' (the cellstat example's
# fallback); in v42 lf52's map was half '?' all game and 5 of 6 maps were never rewritten
GRID_MAP_UNKNOWN_NOTE = (
    "The map seems to hold unknown objects ('?'). If one is a key object that newly appeared, "
    "rewrite grid_map (and cellstat if needed) to map it to a new symbol; the grid_map the "
    "harness shows "
    "you will then update with it."
)

# ARC3_SCHEMA_PARSE: SCHEMA_SCROLL_NOTE becomes this
# ARC3_SCHEMA: every later user message of a game whose view has scrolled
SCROLL_GAME_LINE = (
    "This game scrolls: `action(actions)` runs every action in one call, with or without a "
    "model; you do not need to maintain a world model, just act."
)
SCHEMA_PARSE_SCROLL_NOTE = (
    "The view scrolls in this level, so screen positions change with the view and no world "
    "model is used here: in this level `action(actions)` runs all its actions in one call, "
    "with or without a model, and no prediction is judged. Plan the route on `world_map` "
    "above."
)

# ARC3_SCHEMA: once per level, once the view has scrolled in any level
SCHEMA_SCROLL_NOTE = (
    "The view scrolls in this level, so do not build a world model that predicts whole "
    "boards: use one with a constant key_state, e.g. `WorldModel(lambda f, a, row=None, "
    "col=None: board(f), key_state=lambda f: 0)`. Then action() checks only whether an "
    "action is fatal (and is_win_condition, if you give one) and runs your planned actions in "
    "one call; plan the route on `world_map` above."
)

# ARC3_GRID_NOTE: after the segmentation bullet of the runtime-state description. bp35's
# stalled runs spent many requests re-deriving pixel offsets; its 4-level run reasoned on
# the 6-pixel block grid.
GRID_NOTE_ADDENDUM = (
    "- Objects are drawn as blocks of n*n pixels: abstract the board into a grid aligned "
    "with those blocks and reason on it rather than pixel by pixel.\n"
)
# ARC3_THINK_BUDGET_NOTE: closes every user message without the scroll-level rules. In v40
# the games scoring 60+ thought 726 tokens a reply (median, p90 2.9K), those under 20 988
# (p90 4.1K); a softer form of the third scroll-level rule.
THINK_BUDGET_NOTE = (
    "Aim for about 2,500 tokens of thinking per feedback; think longer only when a step "
    "needs it."
)

# ARC3_HALF_ASCII: heads the half-resolution board added after each executed batch. A 64x64
# board's even rows and columns cost about 420-510 tokens (the image 146; the full board
# about 1,300-1,700), and give exact colours and coordinates.
HALF_ASCII_HEADER = (
    "Current board at half resolution (the same board as the image): every 2nd row and "
    "column, each line led by its board row number; the k-th letter of a line is column 2k. "
    "`current_frame.ascii` has every cell."
)
# ARC3_FULL_ASCII: the same, every row and column (about 1,300-1,800 tokens a 64x64 board).
FULL_ASCII_HEADER = (
    "Current board (the same board as the image): each line led by its board row number; "
    "the k-th letter of a line is column k."
)

# ARC3_SCROLL_RULES: closes every user message that carried the scroll note, after its
# images, where the model reads last. bp35's 4-level run (run 38) wrote `cellstat`, which
# names a 6x6 block of world_map by its colour counts, and planned on the printed block map;
# the runs that stalled printed every 2nd or 3rd row's colour runs and misread object
# outlines as walls.
SCHEMA_SCROLL_RULES = (
    "<scroll_level_rules>\n"
    "- Write a function `cellstat(row, col, h, w)`: given the center of a rectangle on "
    "`world_map` and its height and width, it returns one character for what object the "
    "rectangle holds, e.g. '#' wall, '.' open space, 'C' crate, 'X' goal, '?' unseen or "
    "unknown. Decide it from the cells inside: a color whose count passes a threshold, a "
    "particular shape, or any other feature you have found. Add a new character when you "
    "meet a new kind of object.\n"
    "- When you look over the map with a lattice of sample points (spaced by the player's "
    "step or the objects' size), print each point's `cellstat` character instead of its "
    "color: that shows the map's meaning in the fewest tokens.\n"
    "</scroll_level_rules>"
)

# ARC3_SCHEMA_MISS_BUDGET above 1: appended to the scroll note
SCHEMA_SCROLL_BUDGET = (
    " Each Python call may also run up to {budget} actions without a model."
)

# ARC3_SCHEMA_MISS_BUDGET above 1: the stop rule of SCHEMA_ADDENDUM becomes this
SCHEMA_STOP_RULE = (
    "once an action in a Python call ran without a model, or was mispredicted, any "
    "further `action()` in that call is refused with `ActionStoppedError`."
)
SCHEMA_STOP_RULE_BUDGET = (
    "a Python call may run up to {budget} actions that are unverified (run without a model) "
    "or mispredicted, counted together; after that, any further `action()` in that call is "
    "refused with `ActionStoppedError`. Within one `action(plan, model=model)`, the rest of "
    "the plan is still dropped at its first wrong prediction."
)

# ARC3_SCHEMA: the workflow that replaces the default loop; the _REPORT one under ARC3_SCHEMA_STOP=0
SCHEMA_WORKFLOW = (
    "- Workflow: (1) probe with single actions until you know the basic mechanism; (2) write "
    "step and is_win_condition and bundle them in a WorldModel (see World model below); "
    "(3) search for the goal, and when the search space is too large, split the goal into "
    "subgoals and search for them one at a time; (4) when the search reaches neither "
    "the win nor a subgoal, especially on a new level, the level likely has a new mechanic "
    "or element: keep acting with your current step (`action(actions, model=model)`) to "
    "explore it efficiently until a prediction is wrong, which is usually where the new "
    "mechanic shows, and add it to step; (5) run the plan with `action(plan, model=model)`; "
    "(6) when a prediction is wrong, fix step at that frame, then fix the errors the "
    "backtest function lists automatically, and search again.\n"
)
SCHEMA_WORKFLOW_REPORT = (
    "- Workflow: (1) probe with actions, single ones or short sequences, until you know the "
    "basic mechanism; (2) write step and is_win_condition and bundle them in a WorldModel "
    "(see World model below); (3) search for the goal, and when the search space is too "
    "large, split the goal into subgoals and search for them one at a time; (4) when the "
    "search reaches neither the win nor a subgoal, especially on a new level, the level "
    "likely has a new mechanic or element: explore it by acting with your current step "
    "(`action(actions, model=model)`); the actions it mispredicts, which the call lists, are "
    "usually where the new mechanic shows, so add it to step; (5) run the plan with "
    "`action(plan, model=model)`; (6) when predictions are wrong, fix step at those frames, "
    "then fix the errors the backtest function lists automatically, and search again.\n"
)

# ARC3_SCHEMA: the general guidance this changes, (old line, new line or "")
SCHEMA_REPLACEMENTS = (
    ("- A strong default loop is: summarize the board, infer the desired environment change, "
     "write a small scorer or search over candidate sequences, execute the best probe or plan "
     "with `action(...)`, then inspect again until you understand exactly what changed.\n",
     SCHEMA_WORKFLOW),
    ("- `action(...)` accepts an ordered list of one or more actions. Once your code has "
     "selected a reliable sequence, it is often useful to batch it.\n",
     "- `action(actions)` without a model runs only its first action; to run a sequence, pass "
     "`model=WorldModel(...)` (see World model below).\n"),
    ("- You can also call `action(...)` multiple times in one Python snippet, including inside "
     "loops. Each call updates the preloaded variables before execution continues.\n",
     ""),
)

# ARC3_SCHEMA_STOP=0: the batching line of SCHEMA_REPLACEMENTS becomes this
SCHEMA_BATCH_LINE = (
    "- `action(actions)` without a model runs only its first action; to run a sequence, pass "
    "`model=WorldModel(...)` (see World model below).\n"
)
SCHEMA_BATCH_LINE_REPORT = (
    "- `action(actions)` runs an ordered list of actions; pass `model=WorldModel(...)` so that "
    "each is checked against its prediction (see World model below).\n"
)
