# Harness switches in the submitted run

Every experiment in the harness is an environment variable, off by default (the same convention as Franzen's M2
code). `config/v25_settings.json` holds the exact values of the submitted run (version 25, public 39.22).
This page lists the switches that run turns on, with one line each. Anything not listed is off / at its default.

## Our switches (34)

### Stuck levels and trims
| Switch | Value | What it does |
|---|---|---|
| `ARC3_LEVEL_TOKEN_NUDGE` | 30000 | After 30K tokens on one level, the next user turn says so |
| `ARC3_LEVEL_TOKEN_NUDGE_EVERY` | 30000 | Repeat that note every further 30K tokens on the same level |
| `ARC3_LEVEL_NUDGE_CAUSES` | 1 | The note lists the usual reasons a level stays unsolved, as a checklist |
| `ARC3_ACTION_INDEX` | 1 | Checklist entries cite the action numbers `[k<a>-<b>]` they refer to |
| `ARC3_COMPACT_NOTE_STYLE` | experiments | At a context trim, the summary is a lab notebook (what was tried, where, why, conclusion) |
| `ARC3_COMPACT_NOTE_MAX_TOKENS` | 2000 | Length cap of that note |
| `ARC3_COMPACT_NOTE_IN_SYSTEM` | 1 | The note goes at the end of the system prompt, so it survives later trims |
| `ARC3_KV_PRESSURE_MIN_TOKENS` | 98000 | Never force-trim a game below 98K context when KV runs short |
| `ARC3_KV_PREMATURE_TOKENS` | 119000 | Forced trims below 119K are logged as premature (logging only) |

### Perception
| Switch | Value | What it does |
|---|---|---|
| `ARC3_FRAME_DIFF_V2` | 1 | Per-step `frame_diff` with object-level changes (bbox lists) instead of raw cell lists |
| `ARC3_LONG_ANIMATION_NOTE` | 1 | An action whose animation is long is named as such in the trace |
| `ARC3_LONG_ANIMATION_FRAMES` | 8 | Frame count from which an animation counts as long |
| `ARC3_ANIMATION_REPLAY_FRAMES` | 2 | Show up to 2 intermediate images of a long animation |
| `ARC3_ANIMATION_REPLAY_MIN` | 15 | ...only when it had at least 15 intermediate frames |
| `ARC3_ANIM_ONLY_HINT` | 1 | Hint that parts that only move in animations are displays, not controls |
| `ARC3_SCALE_NOTE` | 1 | Note when an object was scaled by the last action |
| `ARC3_TRACE_GROUPED` | 1 | Consecutive identical actions are grouped in the per-action trace |
| `ARC3_LEVEL_FEEDBACK` | 1 | On a level change: goal review and the terminal image of the finished level |
| `ARC3_DEATH_SUSPECT` | 1 | The death notice names the action most likely to have been fatal |

### World model / block map
| Switch | Value | What it does |
|---|---|---|
| `ARC3_SCHEMA_PARSE` | 1 | Prompt section asking for a world model: state as a dict of objects and values |
| `ARC3_SCHEMA_STOP` | call | `action()` stops a plan at the first call whose prediction was wrong |
| `ARC3_SCHEMA_FREE_ACTIONS` | 4 | Up to 4 actions per call without a prediction |
| `ARC3_SCHEMA_TOLERANCE` | 16 | Board differences smaller than 16 connected cells don't count as wrong |
| `ARC3_SCHEMA_REPORT_DIFF` | 1 | Also report what a correctly predicted action changed beyond the prediction |
| `ARC3_GRID_MAP` | 1 | Ask for a coarse block map of the board (`grid_map(frame)` with `cellstat`) |
| `ARC3_GRID_MAP_STEP` | 1 | The workflow step names the function that holds the map |
| `ARC3_GRID_MAP_OPTOUT` | 1 | Reminder with a lower bound on block size; the model may opt out |

### Search and exploration prompts
| Switch | Value | What it does |
|---|---|---|
| `ARC3_NEW_ELEMENT_PROBE` | 1 | On a new level, probe new elements first (replaces the level-transfer guidance) |
| `ARC3_SEARCH_DECOMPOSE` | 1 | Subgoal guidance: split a long search into subgoals |
| `ARC3_SEARCH_PUSH2` | 1 | One search policy, stated in the system prompt and in a short reminder |

### Cost per turn
| Switch | Value | What it does |
|---|---|---|
| `ARC3_STATE_DELTA` | 1 | History carries deltas of the runtime state instead of rewriting it each turn |
| `ARC3_SNIPPET_NOOP_LIMIT` | 4 | After 4 no-op actions in a row, the snippet may not act any more |
| `ARC3_PYTHON_PROCESS_TIMEOUT` | 30 | A `python` tool call may run 30 s (default 60) |
| `ARC3_FUNCTION_SOURCE` | 1 | After a trim, the namespace listing shows the source of the model's functions |

## Upstream switches (Franzen M2 / Duck) set by the run (41)
These keep Franzen's meaning; see his repository (`upstream/arc-agi-3-solution`). Values are in `config/v25_settings.json`.

| Switch | Value |
|---|---|
| `ARC3_ACTION_ECHO` | `1` |
| `ARC3_ACTION_INFO` | `1` |
| `ARC3_ANIMATION` | `1` |
| `ARC3_BATCH_NOOP_BLOCK` | `1` |
| `ARC3_CONTEXT_DRAIN_TOKENS` | `67000` |
| `ARC3_DEDUPE_MULTICALL_LINE` | `1` |
| `ARC3_DIFF_IMAGE` | `1` |
| `ARC3_EXPLAIN_GAMEPLAY_CHANGED` | `1` |
| `ARC3_FRAME_DIFF_HINT` | `1` |
| `ARC3_GAMEOVER_DIFF_IMAGE` | `1` |
| `ARC3_GUARDS_FROM_LEVEL` | `2` |
| `ARC3_HISTORY_ASSISTANT_TURNS` | `170` |
| `ARC3_HISTORY_DRAIN_COALESCE` | `1` |
| `ARC3_HISTORY_TURN_DRAIN` | `34` |
| `ARC3_LEVEL_TRANSFER_GUIDANCE` | `1` |
| `ARC3_MAX_ACTIVE_STREAMS` | `20` |
| `ARC3_MEMORY_SECTIONS` | `off` |
| `ARC3_MIDDLE_TRUNCATION` | `1` |
| `ARC3_NEW_CHANGED_PROMPTS` | `1` |
| `ARC3_PERSISTENT_FUNCTIONS` | `1` |
| `ARC3_PERSISTENT_FUNCTIONS_IMPORTS` | `1` |
| `ARC3_PERSISTENT_FUNCTIONS_SCOPE` | `game` |
| `ARC3_PRIORITY_SCORE_NORMALIZATION` | `1` |
| `ARC3_PRIORITY_TAIL_FADE` | `1` |
| `ARC3_PRIORITY_TAIL_FADE_FRACTION` | `0.2` |
| `ARC3_PRIORITY_TAIL_LOOKUP` | `remaining` |
| `ARC3_REASONING_HISTORY_KEY` | `reasoning_content` |
| `EXPOSE_RESET` | `on` |
| `EXPOSE_UNDO` | `on` |
| `INFERENCE_ANALYZER_MODEL` | `<expr> SERVED_MODEL_NAME` |
| `LOCAL_ANALYZER_BASE_URL` | `<expr> SERVER_BASE_URL` |
| `LOCAL_ANALYZER_CONTEXT_WINDOW` | `<expr> (128 + 12) * 1024` |
| `LOCAL_ANALYZER_MAX_OUTPUT` | `<expr> 12 * 1024` |
| `LOCAL_ANALYZER_MODEL_ID` | `<expr> SERVED_MODEL_NAME` |
| `LOCAL_ANALYZER_TEMPERATURE` | `0.7` |
| `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS` | `3072` |
| `LOCAL_ANALYZER_TOOL_STEPS` | `0` |
| `LOCAL_ANALYZER_YIELD_TOKENS` | `2048` |
| `MULTIMODAL_CONTEXT` | `current_grid` |
| `MULTIMODAL_UPSCALE` | `8` |
| `OPENAI_BASE_URL` | `<expr> SERVER_BASE_URL` |
