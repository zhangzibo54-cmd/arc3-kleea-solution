# Online submissions (public leaderboard)

All 33 submissions with their original descriptions: `submissions.csv` (exported from Kaggle).
The table below keeps the ones that changed what we did next. Public scores are on the semi-private
hidden set; one run of the same notebook varies by about ±2–3 points (v25: 39.22, resubmitted: 36.94).

| Date | Ref | Public | What changed |
|---|---|---|---|
| 09-04 → 09-19 | 56006389 … 56355089 | 0 – 3.3 | Pipeline checks (random agent, CPU), first Duck runs |
| 10-01 | 56760681 | **28.17** | Franzen M2 baseline as published: Intel W4A16, FP8 KV, 10 streams |
| 10-04 | 56829194 | 26.61 | Hybrid weights (klee W3 + Intel W4 in layers 12–35), NVFP4 KV everywhere, 18 streams |
| 10-05 | 56839525 | 25.00 | Only the KV change (NVFP4 on all layers): full-NVFP4 KV costs quality |
| 10-10 → 10-15 | 57033973 … 57217850 | 22.3 – 29.1 | Harness experiments (level feedback, search push, probing), W3 vs W4 |
| 10-18 / 10-19 | 57340085, 57378944 | 25.7, 25.1 | Schema-free baseline at 14 streams, 128K context |
| 10-20 | 57429991 | 28.76 | Compact note at trims, frame_diff v2, long-animation note, 140K context |
| 10-21 | 57470198 | **32.56** | History deltas (`ARC3_STATE_DELTA`), 30 s Python, CUDA graph sizes |
| 10-22 / 10-23 | 57522838, 57546645 | 29.07, 26.06 | Same notebook rerun: variance check (32.56 → 26–29) |
| 10-24 | 57596781 | **34.66** | Hybrid A packed at serve time (layers 12–35 W4), 18 streams, 8 px/cell |
| 10-25 | 57634121 | 36.14 | Without the explore note |
| 10-26 | 57667680 | **37.96** | Stuck-level note 30K/30K with causes, grid_map opt-out |
| 10-27 | 57714940 | 31.58 | 6 px/cell, 10 action frames, lowm kernels (image change hurt) |
| 10-28 | 57745091 | **39.22** | NVFP4 KV only on QSA layers 15–35, FP8 elsewhere; 20 streams — **best** |
| 10-29 | 57780246 | 37.41 | Full FP4 KV, 24 streams, scroll notes |
| 10-30 | 57811098 | 32.42 | KV deletion + time-budget scheduling (ended after ~7 h) |
| 10-31 | 57843534 | 23.41 | Same + HiCache + `show_image` |
| 11-01 | 57870604 | 36.94 | v25 resubmitted unchanged (variance check) |
| 11-02 | 57891485 | 37.25 | v25 + shared notes across games |

Takeaways:
- The steps that held up online were serving changes (hybrid A, mixed-precision KV, more streams) and the
  stuck-level note. NVFP4 KV on *all* layers lost points; FP8 on the first/last QSA layers fixed that.
- Scheduling changes that looked neutral offline (KV deletion, time budgets) lost 5–15 points online.
- Offline runs (25 public games, 2 h) did not predict online differences of 2–4 points; see `LESSONS.md`.
