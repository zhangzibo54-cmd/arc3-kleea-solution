import json
import os
import pickle
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

# Kaggle might cancel the run in no action is done in some timeframe.
# To prevent this, harness will start up even if backend is not yet ready,
# and the first started game will perform a single reset action.
NOTEBOOK_START_TIME = time.time()
SERVER_STARTUP_TIMEOUT = 12 * 60
os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1'
os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'

# model and dataset paths
WHEELHOUSE_DIR    = '/kaggle/input/datasets/dfranzen/pennyroyal-v253'
MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'
DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'
ORIG_BUNDLE_DIR   = '/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'
SERVED_MODEL_NAME = 'flashnext'

# Apply harness patch
BUNDLE_DIR = Path('/kaggle/taaf-kaggle-source-share')
!rm -Rf $BUNDLE_DIR
!cp -a $ORIG_BUNDLE_DIR $BUNDLE_DIR
subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)
print('harness patch applied successfully')

# True only inside a real competition rerun; switches diagnostics + soft deadline.
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
NOTEBOOK_START_EPOCH = time.time()

# Non-interactive matplotlib backend: diagnostics render plots with no display attached.
os.environ["MPLBACKEND"] = "Agg"
# Marks the run as a (real or emulated) submission so the framework + solver can adjust.
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
# In submission, disable the periodic JSON/HTML diagnostics writes and per-frame logging.
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
# Pin arc_agi's cached level_reset_only before its client is built (RESET keeps the level).
os.environ["ONLY_RESET_LEVELS"] = "true"

# Everything the run produces is written here.
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")
SERVED_MODEL_HOST = '127.0.0.1'
SERVED_MODEL_PORT = 8001
SERVER_BASE_URL = f'http://{SERVED_MODEL_HOST}:{SERVED_MODEL_PORT}/v1'
USE_PRIORITY_SCHEDULING = True

setup_env = {
    'ARC3_LEVEL_FEEDBACK': '1',
    'ARC3_SEARCH_PUSH2': '1',  # baseline search push2
    ########## transformers config ##########
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',

    ########## serving config ##########
    'LOCAL_ANALYZER_BASE_URL': SERVER_BASE_URL,
    'OPENAI_BASE_URL': SERVER_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': 'vllm',
    'OPENAI_PROVIDER': 'vllm',
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
    'LOCAL_ANALYZER_APP_NAME': 'ARC3 Agent Harness',
    
    ########## analyzer config ##########
    'ARC3_REASONING_HISTORY_KEY': 'reasoning_content',
    'LOCAL_ANALYZER_ENABLE_THINKING': 'true',
    'LOCAL_ANALYZER_TOOL_STEPS': '0',
    'LOCAL_ANALYZER_TOOL_TIMEOUT': '30',
    'LOCAL_ANALYZER_YIELD_SECONDS': '0',
    'LOCAL_ANALYZER_YIELD_TOKENS': '2048',
    'LOCAL_ANALYZER_TEMPERATURE': '0.7',
    'LOCAL_ANALYZER_TOP_P': '0.95',
    'LOCAL_ANALYZER_TOP_K': '20',

    ########## context pruning ##########
    # 200K context (2026-10-06): window 188K + 12K reply = 200K, 8K under the server's 208K limit;
    # trim above 192,000 input tokens, drain to 96,000; turn cap scaled with the window.
    'LOCAL_ANALYZER_CONTEXT_WINDOW': (128+12)*1024,  # ctx 140K variant
    'LOCAL_ANALYZER_MAX_OUTPUT': 12*1024,
    'ARC3_CONTEXT_DRAIN_TOKENS': 67000,
    'ARC3_HISTORY_ASSISTANT_TURNS': 170,
    'ARC3_HISTORY_TURN_DRAIN': 34,
    'ARC3_HISTORY_DRAIN_COALESCE': '1',

    ########## tool output ##########
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '3072',
    'ARC3_MIDDLE_TRUNCATION': '1',
    'ARC3_ACTION_ECHO': '1',
    
    ########## prompt changes ##########
    'ARC3_EXPLAIN_GAMEPLAY_CHANGED': '1',
    'ARC3_NEW_CHANGED_PROMPTS': '1',
    'ARC3_LEVEL_TRANSFER_GUIDANCE': '1',
    'ARC3_SEARCH_DECOMPOSE': '1',  # subgoal bullet + note on timed-out / out-of-memory calls (since 2026-10-10)
    'ARC3_NEW_ELEMENT_PROBE': '1',  # try every action on unseen elements; goal rarely changes (since 2026-10-10)
    'ARC3_ACTION_INFO': '1',
    'ARC3_DEDUPE_MULTICALL_LINE': '1',
    'ARC3_FRAME_DIFF_HINT': '1',

    ########## vision & animations ##########
    'MULTIMODAL_CONTEXT': 'current_grid',
    'ARC3_PRECACHE_PLE_AHEAD_GB': '40',  # notebook: PLE files read ahead of the loader, at most this many GB
    'ARC3_PRECACHE_RESERVE_GB': '16',  # notebook: memory kept free while reading PLE files ahead
    'ARC3_PRECACHE_PLE_THREADS': '32',  # notebook: threads of the PLE reader (shared by its files)
    'ARC3_PRECACHE_PLE_FILES_IN_FLIGHT': '8',  # notebook: PLE files read at once
    'ARC3_PRECACHE_DROP': '1',  # notebook: drop model files the loader has finished from the page cache
    'MULTIMODAL_UPSCALE': '8',  # 512 x 512 px boards, ~258 vision tokens
    'ARC3_DIFF_IMAGE': '1',
    'ARC3_GAMEOVER_DIFF_IMAGE': '1',
    'ARC3_ANIMATION': '1',

    ########## guards ##########
    'ARC3_GUARDS_FROM_LEVEL': 2,
    'ARC3_BATCH_NOOP_BLOCK': '1',
    'ARC3_SNIPPET_NOOP_LIMIT': '4',
    'ARC3_QSA_FP4_LAYERS': '15-35',  # NVFP4 KV on these full-attention layers  # a snippet stops acting after 4 no-ops in a row
    'ARC3_STALE_STATE_BLOCK': '0',  # off since 2026-10-10: it refused probe loops after a no-op (sp80 red-piece click)

    ########## world model ##########
    'ARC3_MEMORY_SECTIONS': 'off',

    ########## persistent functions ##########
    'ARC3_PERSISTENT_FUNCTIONS': '1',
    'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game',
    'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
    # hard5 experiment: one persistent Python process per game (supersedes retained functions)
    'ARC3_PERSISTENT_PYTHON_PROCESS': '1',
    'ARC3_PYPROC_INVENTORY_ON_TRIM_ONLY': '1',  # list the namespace only at start and after trims
    'ARC3_OPENER_STATIC_ON_TRIM_ONLY': '1',  # opener's fixed instructions only at start and after trims
    'ARC3_PROMPT_NO_SEARCH_URGING': '1',  # drop the 'write a BFS / stop probing and search' wording
    'ARC3_VIEW_SHIFT': '1',  # detect camera scrolls: compensated frame_diff / diff image, echo note
    'ARC3_COMPACT_NOTE': '1',  # goal / action / rule note written at each context trim
    'ARC3_COMPACT_NOTE_STYLE': 'experiments',  # lab notebook: purpose / actions / observed per entry, observations only
    'ARC3_COMPACT_NOTE_MAX_TOKENS': '2000',
    'ARC3_COMPACT_NOTE_IN_SYSTEM': '1',  # the notebook ends the system prompt
    'ARC3_SCHEMA': '0',  # no world model (demo10 no-schema control)
    'ARC3_SCHEMA_STOP': 'call',  # a wrong prediction drops only the rest of its call (v27)
    'ARC3_SCHEMA_TOLERANCE': '16',  # a board difference is wrong from 16 connected cells
    'ARC3_SCHEMA_REPORT_DIFF': '1',  # print frame_diff objects a right prediction missed
    'ARC3_SCHEMA_PARSE': '1',  # WorldModel(parse, step, ...): predictions judged on parsed states
    'ARC3_GRID_MAP': '1',  # grid_map(frame) stub; every message shows grid_map(current_frame)
    'ARC3_GRID_MAP_STEP': '1',  # Workflow step 2: write grid_map(frame) (with cellstat) to build the block map
    'ARC3_SCALE_ECHO': '0',  # off since 2026-10-21: per-action flood fill held the GIL (all25 slowdown)
    'ARC3_PYTHON_PROCESS_TIMEOUT': '30',  # python call limit 30 s (since 2026-10-21; was 90)
    'ARC3_FRAME_DIFF_V2': '1',  # frame_diff without background; frame_diff timelines
    'ARC3_LONG_ANIMATION_NOTE': '1',  # 10+ frame animations name their entry in the echo
    'ARC3_LONG_ANIMATION_FRAMES': '8',  # long-animation note from 8 frames
    'ARC3_MROPE_DEVICE_CACHE': '1',  # SGLang: mrope deltas cached on the GPU (sglang_patches/mrope_cache.py; back on since 2026-10-22)
    'ARC3_STATE_DELTA': '1',  # runtime state in memory, history deltas to the worker
    'ARC3_QSA_INDEXER_BUDGET': '2048',  # SGLang: tokens each QSA query attends (model default 2048; 8192 is the only other)
    'EXPOSE_RESET': 'on',  # RESET offered to the model (system prompt line, RESET-first guard)
    'ARC3_MM_IMAGE_CACHE': '1',  # SGLang: features sent for new images only (sglang_patches/mm_image_cache.py)
    'ARC3_TRACE_GROUPED': '1',  # per-action trace: one line per run of one verdict
    'ARC3_FUNCTION_SOURCE': '1',  # fn_source['name']: source of the model's latest definition
    'ARC3_ACTION_INDEX': '1',  # level_history indices of each batch; per-entry lab note ranges
    'ARC3_LEVEL_TOKEN_NUDGE': '30000',  # stuck-level note from 30K tokens on the level
    'ARC3_LEVEL_TOKEN_NUDGE_EVERY': '30000',  # and every 30K after it
    'ARC3_LEVEL_NUDGE_CAUSES': '1',  # the note lists the usual reasons a level stays unsolved
    'ARC3_GRID_MAP_OPTOUT': '1',  # grid_map reminder: n >= 3 blocks; grid_map = False stops it
    'ARC3_SCALE_NOTE': '1',  # opener notes a scaled object (twice per level); scale echo on
    'ARC3_LOWM_GEMM': '1',  # SGLang: split-K BF16 GEMMs at 33-128 rows (sglang_patches/arc3_lowm_patch.py)
    'ARC3_LOWM_HC_MIX': '1',  # SGLang: HC mix in two launches
    'ARC3_LOWM_HC_COMBINE_NORM': '1',  # SGLang: HC combine fused with the next hc_norm
    'ARC3_ANIM_ONLY_HINT': '1',  # Workflow step 4: animation-only effects keep the state
    'ARC3_DEATH_SUSPECT': '1',  # death notice: objects near the fatal position are first suspects
    'ARC3_SCHEMA_FREE_ACTIONS': '4',  # up to 4 unchecked actions per Python call without a model
    'ARC3_PROMPT_PROBE_ACTIONS': '1',  # system prompt: probe untried SPACE / MOUSE / UNDO early and when stuck
    'ARC3_ACTION_FRAME_IMAGES': '10',  # board after each action of a batch, in order (default since 2026-10-09)
    'ARC3_ANIMATION_REPLAY_FRAMES': '2',  # long animations only: 2 states + final board (since 2026-10-10)
    'ARC3_ANIMATION_REPLAY_MIN': '15',  # ... with >= 15 intermediate states
    'ARC3_DEATH_REPLAY_FRAMES': '5',  # fatal action's animation frames before the death frame (default since 2026-10-09)

    ########## controls ##########
    'EXPOSE_UNDO': 'on',
}

if USE_PRIORITY_SCHEDULING: setup_env.update({
    'ARC3_MAX_ACTIVE_STREAMS': 20,  # 20 concurrent games (MAMBA_CACHE 100 = 20 x 5 slots)
    # KV-budget admission: ARC3_KV_ADMISSION_TOKENS / ARC3_KV_PRESSURE_TOKENS are set in the customization
    # hook from the server's real pool size (90% / 95% of max_total_num_tokens). Pressure valve floor and the
    # premature-trim warning scaled to the 192K trigger; [kv] report every 200 s.
    'ARC3_KV_PRESSURE_MIN_TOKENS': 98000,
    'ARC3_KV_PREMATURE_TOKENS': 119000,
    'ARC3_PRIORITY_REFRESH_QUEUE': '1',
    'ARC3_PRIORITY_PACE': '0',
    'ARC3_PRIORITY_TAIL_FADE': '1',
    'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.2',
    'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining',
    'ARC3_PRIORITY_SCORE_NORMALIZATION': '1',
})

os.environ.update({k: str(v) for k,v in setup_env.items()})