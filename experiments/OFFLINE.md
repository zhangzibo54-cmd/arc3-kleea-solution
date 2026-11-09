# Key offline runs

Offline = the 25 public games bundled with the competition, on Kaggle (RTX PRO 6000) or a rented
RTX PRO 6000 (vast.ai). Score = competition metric averaged over games. A 2 h all25 run varies by ±4–6 points.

| Run | Setting | Games × time | Mean | Levels |
|---|---|---|---|---|
| Speed bench v1 | Franzen M2 serving, 10 streams | trace replay | 599 tok/s aggregate, accept len 2.64 | — |
| vast "65.70" | v25 settings | all25 × 2 h | 65.70 | 147 |
| Kaggle v4 | v25 settings | all25 × 2 h | 61.05 | 133 |
| Kaggle v25 rerun (v38) | exact v25 notebook | all25 × 2 h | 58.19 | — |
| Kaggle notes v3 | v25 + shared notes | all25 × 2 h | 65.27 | 140 |
| Kaggle notes v6 | v25 + shared notes (mechanics-first prompt) | all25 × 2 h | 54.36 | 129 |
| vast all25×2 | shared notes, KV-booked compression | 50 games × 4 h | 63.13 | — |
| Kaggle all25×4 | shared notes, per-game views | 100 games × 8 h | 57.07 | — |

Throughput of the v25 configuration: about 1,000 tok/s aggregate decode at 20 streams, falling to ~900 after
the first hour in long runs as contexts grow.

The same v25 settings scored 58–66 offline across four runs, so offline differences under ~6 points between
two configurations are not meaningful at this sample size.
