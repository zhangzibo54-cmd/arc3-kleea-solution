"""Mechanism skills: short notes the model reads on demand (ARC3_SKILLS).

Each skill names one observable situation (its trigger) and gives guidance for
levels built on that mechanism: how they are usually solved, what the objects
involved can be, and common pitfalls. The system prompt lists only the
triggers; the text is read in a `python` call with `skill(name)`, which costs
no game action. Skills describe mechanisms, never the solution of a game.
"""

from __future__ import annotations

import os
import re

# name -> (trigger, text). The trigger is the condition the model checks against
# what it observed, so it must be concrete and observable.
SKILLS: dict[str, tuple[str, str]] = {
    "controlled-player": (
        "pressing UP, DOWN, LEFT or RIGHT moves an object on the board (your player), "
        "and you have not yet found how to complete the level",
        "Levels with a player you move with UP / DOWN / LEFT / RIGHT.\n"
        "How such levels are usually solved:\n"
        "- The player has to reach, enter, fill or line up with something. Objects that share your player's "
        "colors, shape or size are the first candidates: an exit, a socket, a container, a slot to fill, a copy "
        "to match.\n"
        "- Do not assume you control only one player. A common mechanic: clicking an object that resembles your "
        "player (similar colors, or a similar or different shape) switches control to that object or swaps it "
        "with your player. When MOUSE is available, click such objects to find out.\n"
        "Pitfalls:\n"
        "- An area you could not enter or reach earlier may become reachable once a mechanic has changed the "
        "board or your player (something moved, opened, shrank, ...). Try it again then, and when you reach it, "
        "move into it once more: entering, pushing, picking up or triggering usually happens on contact.\n"
        "Where this conflicts with what you observe, trust your observations.",
    ),
    "other-agent": (
        "an object on the board moves or acts on its own, in a way your action did not cause "
        "(another agent, controlled by the game)",
        "Levels with another agent that moves or acts by itself.\n"
        "How such levels are usually solved:\n"
        "- The agent is either a competitor or a teammate. A competitor races you for the same targets, blocks "
        "you or ends the attempt on contact. A teammate works toward the same goal as you, and is often an "
        "important helper for reaching the current goal: it carries, pushes, opens or delivers things that "
        "would cost you many actions or that you cannot do yourself.\n"
        "- Learn what it does before deciding which it is: watch it for a few actions in which you do not "
        "interfere, and note what it moves toward, what it picks up or changes, where it takes it, and whether "
        "it reacts to your player's position.\n"
        "- With a teammate, plan the level as a joint plan: let it do what it does well, do the rest yourself, "
        "and position yourself so it can work (or so it follows you where it is needed).\n"
        "Pitfalls:\n"
        "- An agent that takes or moves the things you need is not necessarily a competitor: it may be "
        "delivering them to the goal.\n"
        "Where this conflicts with what you observe, trust your observations.",
    ),
    "search": (
        "you are about to write, or are writing, a simulator of the game's rules or a search over action "
        "sequences",
        "Simulating the rules and searching over action sequences.\n"
        "Checking the simulator:\n"
        "- Check it against the recorded transitions on what decides the outcome: which actions are accepted "
        "or refused, and the positions and shapes of the objects the goal and the mechanics depend on. You "
        "should understand most of the changes you see, but do not spend turns matching every pixel of "
        "decorations that do not affect the outcome. A mismatch on what decides the outcome means a rule is "
        "wrong; fix that rule.\n"
        "Searching by subgoals:\n"
        "- A search over the whole level at once is often too large. Split the goal into subgoals that build "
        "on each other: state A (for example, one object moved or reshaped into a certain position), from which "
        "state B can be reached, from which the final goal can be reached.\n"
        "- Search only for a way to reach A, play it in the game and check that A was reached. Then search "
        "from A for B, play it and check B. Then search from B for the goal. Each search is much smaller than "
        "the whole, and each check in the game catches a wrong rule early.\n"
        "- A classic case: a maze whose exit opens only after a series of switches is triggered in order. "
        "Search for the path to the first switch, take it and check that it triggered, then search from there "
        "for the path to the next switch, and so on, and finally for the path to the exit.\n"
        "- If a subgoal turns out to be unreachable, choose a different split rather than enlarging the search.\n"
        "Where this conflicts with what you observe, trust your observations.",
    ),
}


# Code that searches over action sequences: a frontier (BFS / priority queue / beam) together
# with game moves. A frontier alone is often a flood fill over cells. Models rarely read a skill
# by themselves (hard10 v17 / v18: sk48 wrote eleven searches and never read `search`); this
# detection triggers ARC3_SEARCH_PUSH instead.
_FRONTIER = re.compile(r"\b(deque|popleft|heappush|heappop|beam[_ ]?(search|width|size)|BFS|bfs)\b", re.IGNORECASE)
_MOVES = re.compile(r"'(UP|DOWN|LEFT|RIGHT|SPACE|MOUSE|UNDO)'|\b(actions?|moves?|sim\w*|step)\s*\(|\bfor\s+\w+\s+in\s+(ACTIONS|MOVES|moves|actions)\b")


def skill_for_code(code: str) -> str | None:
    """The skill that applies to what `code` is doing, or None."""
    if _FRONTIER.search(code) and _MOVES.search(code):
        return "search"
    return None


def skills_enabled() -> bool:
    """Whether skills are offered (ARC3_SKILLS, default off)."""
    return os.environ.get("ARC3_SKILLS", "0").strip().lower() in ("1", "true", "yes", "on")


def skill_texts() -> dict[str, str]:
    """The skill texts handed to the Python worker, empty when skills are off."""
    return {name: text for name, (_, text) in SKILLS.items()} if skills_enabled() else {}


def skills_system_section() -> str:
    """The system prompt section listing each skill and when to read it."""
    lines = [
        "\n\nSkills:",
        "- Skills are short notes on kinds of game mechanics. Each entry below names a skill and the observation "
        "that makes it relevant. When what you observed matches an entry, read the skill in a `python` call with "
        "`print(skill('<name>'))`; it uses no game action. `skill()` lists the names.",
        "- A skill gives guidance for levels built on that mechanism: how they are usually solved, what the "
        "objects involved can be, and common pitfalls. It is general guidance, not a description of this game; "
        "where it conflicts with what you observe, trust your observations.",
    ]
    lines += [f"- `{name}`: read when {trigger}." for name, (trigger, _) in SKILLS.items()]
    return "\n".join(lines) + "\n"
