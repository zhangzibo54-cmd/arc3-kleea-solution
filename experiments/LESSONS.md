# Lessons

## 1. Throughput was worth more than we measured
The 8th-place public entry (Grigoriy Zatravkin) used Franzen's M2 harness with byte-identical prompts and
scored ~50 vs our ~39; almost the whole gap was serving:
- full KV in pinned host RAM, a GPU hot-group cache, and a per-step gather of the QSA top-k groups (HiSparse-style);
- int4 KV, fp8 indexer keys, experts pruned to 400 at load time;
- 110 concurrent games at 200K context, roughly 2.5–3× our aggregate throughput.

A community entry (vLLM, NVFP4 experts pruned to 320, NVFP4 KV, 32 streams, a retrained MTP head) reached 45.

Why we missed it: our offline test (25 public games × 2 h) sits where throughput gains don't show.
- The easy games are solved by ~300–400K tokens, so more tokens add nothing.
- The hard games are noisy enough to hide a few points.

So we spent the last weeks on prompt and notes changes whose effects were also lost in noise.

What we would do instead:
1. Measure throughput and tokens per game directly with a fixed-load speed test, separate from scores.
2. Value throughput with a score-vs-tokens curve on a hard subset, with several copies per game.
3. Test at the online load (~110 games at once), not 20–25.
4. Build or collect more hard games instead of relying on 25 saturated public ones.

## 2. Precision: where the bits go matters more than how many
- 2-bit experts in the middle layers lost too much; 3-bit at the ends + 4-bit in layers 12–35 kept quality.
- NVFP4 KV on all QSA layers cost points online. FP8 on the first and last QSA layers with NVFP4 in between
  kept the quality and most of the memory saving.

## 3. Offline noise
- One 2 h all25 run varies by ±4–6 points, and one online run by ±2–3.
- Use level counts and repeated runs. Treat single-run harness A/B results as hints, not evidence.

## 4. Scheduling changes are risky
KV deletion and time-budget scheduling looked neutral offline and lost 5–15 points online. A change to how
games share the GPU should be tested at the online load and length (9 h, ~110 games), not in a 2 h offline run.
