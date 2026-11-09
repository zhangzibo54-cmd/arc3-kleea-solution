# ARC Prize 2026 · ARC-AGI-3 — "klee A" solution (solo, public LB 39.22)

Solo entry by **zhangzibo54**. Best public-leaderboard submission: **39.22**, Kaggle notebook
`zhangzibo54/arc3-kleea-v25rerun`, **version 25** (submission ref 57745091, 2026-10-28). The same notebook resubmitted
later scored 36.94, so a single online run varies by about ±2–3 points.

The agent is **The Duck** (Tufa Labs) with **Daniel Franzen's Milestone-2 patch**, served on one RTX PRO 6000 (96 GB)
by SGLang (Pennyroyal v2.5.3). Our work went mostly into **serving throughput** — more concurrent games and more tokens
per game inside the 9-hour limit — plus a set of harness switches on top of Franzen's.

> 中文简介：单人参赛，公开榜最好成绩 39.22（第 25 版笔记本）。在 Tufa Labs 的 Duck 和 Franzen 的 M2 方案上，主要做了
> 推理加速（3/4 bit 混合专家量化、部分层 NVFP4 KV、20 路并发、若干 SGLang 内核补丁）和一些 harness 开关。
> 实验记录和经验教训见 `experiments/`。

## What made the difference

### 1. Throughput (largest effect)
| Change | What it does | Effect |
|---|---|---|
| **"Hybrid A" weights** | Routed experts from `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` (3-bit) in layers 0–11 and 36–47, Intel W4A16 experts in layers 12–35 (the 2-bit middle layers lost too much), served by the **Humming** W3/W4 kernels (`serving/klee/`) | Main model ~70 GB → ~55 GB; W3 MoE GEMM 21–29 % faster than Marlin W4. Frees memory for KV → more concurrent games |
| **NVFP4 KV on the middle full-attention layers** | QSA layers 15–35 store KV in NVFP4, the first/last QSA layers stay FP8 (`serving/arc3_kvmix_patch.py`, `serving/nvfp4_kv/` ported from cookys/sglang-qsa-nvfp4-kv) | Larger KV pool at 148K context; allows 20 streams instead of 10–18 |
| **20 concurrent games** | `MAXREQ 20`, CUDA graphs up to bs 20, 100 Mamba slots | ~1,000 tok/s aggregate decode (vs ~600 for the original 10-stream setup) |
| **Small-kernel SGLang patches** | split-K BF16 GEMM for decode-sized M, fused HC mix/combine-norm (`serving/lowm/`), M-RoPE device cache, multimodal image cache | Shorter decode step at verify batch sizes |
| **MTP speculative decoding** | albucino MTP drafter, 3 draft tokens (from Franzen) | ~2.6 tokens per step |

Online progression (public LB) as these landed — see `experiments/SUBMISSIONS.md`:
28.17 (Franzen M2 baseline, 10 streams) → 32.56 (harness fixes) → 34.66 (hybrid A, 18 streams) → 37.96 (+ stuck-level
notes) → **39.22 (+ NVFP4 KV on layers 15–35, 20 streams)**.

### 2. Harness (on top of Franzen M2)
The submitted version turns on 34 of our own switches (full list with one-line descriptions: `docs/SWITCHES.md`).
The ones we believe mattered most:
- **Stuck-level notes** (`ARC3_LEVEL_TOKEN_NUDGE` 30K, `ARC3_LEVEL_NUDGE_CAUSES`): after 30K tokens on one level the
  model is told so and given the usual reasons a level stays unsolved.
- **Compact lab notebook in the system prompt** (`ARC3_COMPACT_NOTE_*`): at each context trim the model's experiments
  are condensed into a note appended to the system prompt, so a trim keeps what was learned.
- **Perception**: object-level `frame_diff` v2, long-animation notes, animation replay frames, level feedback with
  terminal images.
- **No wasted actions**: a snippet that makes 4 no-op actions in a row may not act any more (`ARC3_SNIPPET_NOOP_LIMIT`).
- **Cheaper turns**: history deltas instead of rewriting the runtime state (`ARC3_STATE_DELTA`), 30 s Python limit.

Individual harness switches were tested offline on 10–25 public games; single runs vary by ±4–6 points, so most
of these effects are not individually significant (see `experiments/LESSONS.md`).

## Layout
| Path | Contents |
|---|---|
| `harness/` | The harness exactly as submitted (Tufa Labs Duck + Franzen M2 + our changes), our commit `dcceaeb` |
| `upstream/arc-agi-3-solution/` | Git submodule: Franzen's published M2 solution (Apache-2.0), which our harness started from |
| `patches/kleea-vs-franzen-m2.patch` | Our harness changes relative to Franzen's M2 code (17 files) |
| `patches/harness-changes.v25.patch` | The full harness diff the notebook ships (relative to the Duck base), decoded |
| `serving/` | SGLang patches and model-assembly code embedded in the notebook, decoded into plain files |
| `notebook/arc3-kleea-v25.ipynb` | The submitted notebook, byte for byte (version 25) |
| `notebook/cells/` | Every code cell of that notebook as a `.py` file, with the base64 blobs replaced by pointers into `serving/` and `patches/` |
| `config/v25_settings.json` | All harness environment settings of the submitted run |
| `docs/SWITCHES.md` | The switches the submission turns on, grouped, one line each |
| `experiments/` | Online submissions, key offline runs, and lessons learned |

The harness keeps every experiment as an environment switch (default off), the same convention as Franzen's
code; `config/v25_settings.json` is the exact configuration that ran. Switches not listed there are off.

## Running it
The notebook is self-contained on Kaggle (RTX PRO 6000, no internet): attach the models and datasets listed in its
metadata (`zhangzibo54/arc3-kleea` model for hybrid A, Intel W4A16, the albucino drafter, Franzen's TAAF bundle and
Pennyroyal wheels) and run. Offline it plays the bundled public games (`OFFLINE_GAMES`, `OFFLINE_MINUTES` in the
customization cell); in a competition rerun it plays the hidden set for 9 hours.

## Credits
- **Tufa Labs** — The Duck harness and TAAF (`github.com/Tufalabs/duck-harness`).
- **Daniel Franzen** — Milestone-2 solution (`github.com/da-fr/arc-agi-3-solution`, Apache-2.0), Pennyroyal SGLang build,
  the albucino MTP drafter and the TAAF Kaggle bundle.
- **cookys** — NVFP4 KV cache for SGLang's QSA path (`github.com/cookys/sglang-qsa-nvfp4-kv`, Apache-2.0; see `serving/nvfp4_kv/NOTICE`).
- **klee100** — AutoRound 3bpw checkpoint; **Intel** — W4A16 AutoRound checkpoint.

## License
Apache-2.0 for our code (see `LICENSE`). Upstream code keeps its own terms; see `NOTICE`.
