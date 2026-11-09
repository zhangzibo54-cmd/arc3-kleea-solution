# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.
bm.solver.max_runtime_s_per_game = 48*60 #7920.0
bm.solver.analyzer_timeout = 900.0
bm.solver.concurrency = 10
bm.solver.max_actions_per_game = None
bm.solver.save_request_logs = True
bm.n_passes = 1
# Offline test set. HARD10 = the 10 lowest-scoring games of w3all-r25-1pass-120min (default harness test,
# 10 games x 60 min, all concurrent); None = all 25 games. Ignored in a real submission.
HARD10 = ['g50t', 'cn04', 'lf52', 'tn36', 'bp35', 'sk48', 'wa30', 'sp80', 'su15', 'ls20']  # ar25 dropped (its low score was noise)
DEMO10 = ['ar25', 'ft09', 'lp85', 'r11l', 're86', 'sb26', 'sc25', 'tr87', 'tu93', 'vc33']  # Franzen's original offline set
OFFLINE_GAMES = DEMO10  # hard10 x 60 min, 140K context  # build_ista_notebook --games;  # KV admission test: all 25 games, at most 20 active
OFFLINE_MINUTES = 5
bm.game_weights = None

# KV-budget admission from the server's actual pool (max_total_num_tokens in serve.log): admit under 90%,
# pressure valve above 95%. Read by the harness when the priority gate is first built (first game start).
import re as _re
_pool = int(_re.search(r'max_total_num_tokens=(\d+)', (WORKING_DIR / 'serve.log').read_text(errors='ignore')).group(1))
os.environ['ARC3_KV_ADMISSION_TOKENS'] = str(int(0.90 * _pool))
os.environ['ARC3_KV_ADMISSION_PEAK_TOKENS'] = '0'  # build_ista_notebook --kv-peak
os.environ['ARC3_KV_PRESSURE_TOKENS'] = str(int(0.95 * _pool))
print(f"KV pool {_pool} tokens -> admission {os.environ['ARC3_KV_ADMISSION_TOKENS']}, pressure {os.environ['ARC3_KV_PRESSURE_TOKENS']}")

demo_excluded_games = [] if TRUE_SUBMISSION else []  # KLEE offline check: 25 games

if TRUE_SUBMISSION:  # overrides for contest run
    bm.solver.save_request_logs = False
    bm.n_passes = 1
    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = 120  # game count should be 110
        bm.solver.max_runtime_s_per_game = 532*60
else:
    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = (len(OFFLINE_GAMES) if OFFLINE_GAMES else 25 - len(demo_excluded_games)) * bm.n_passes
        bm.solver.max_runtime_s_per_game = OFFLINE_MINUTES*60
    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:
        raise RuntimeError(
            f'Expected the 32400-second notebook budget, got {target.max_runtime_s!r}.'
        )

# show config
print('benchmark.solver:', bm.solver)