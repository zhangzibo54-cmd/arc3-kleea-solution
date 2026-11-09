# ARC Prize 2026 · ARC-AGI-3 —— "klee A" 方案（单人，公开榜 39.22）

[English](README.md) | **中文**

**zhangzibo54** 单人参赛。公开榜最好成绩 **39.22**，对应 Kaggle notebook `zhangzibo54/arc3-kleea-v25rerun` 的**第 25 版**
（提交号 57745091，2026-10-28）。同一个 notebook 之后重新提交得了 36.94，所以单次线上运行大约有 ±2–3 分的波动。

Agent 是 Tufa Labs 的 **The Duck**，加上 **Daniel Franzen 的 Milestone-2 补丁**，用 SGLang（Pennyroyal v2.5.3）部署在
一张 RTX PRO 6000（96 GB）上。我们的工作主要在**推理吞吐**上：在 9 小时限制内同时跑更多局游戏、每局用上更多 token；
另外在 Franzen 的基础上加了一组 harness 开关。

## 起作用的改动

### 1. 吞吐（影响最大）
| 改动 | 做法 | 效果 |
|---|---|---|
| **"Hybrid A" 权重** | 第 0–11 层和 36–47 层用 `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` 的 3-bit 路由专家，第 12–35 层用 Intel W4A16 专家（中间层用 2-bit 损失太大），由 **Humming** W3/W4 内核执行（`serving/klee/`） | 主模型约 70 GB → 约 55 GB；W3 MoE GEMM 比 Marlin W4 快 21–29%。省下的显存给 KV → 更多并发局 |
| **中间全注意力层的 NVFP4 KV** | QSA 第 15–35 层的 KV 用 NVFP4 存储，首尾的 QSA 层保持 FP8（`serving/arc3_kvmix_patch.py`；`serving/nvfp4_kv/` 移植自 cookys/sglang-qsa-nvfp4-kv） | 148K 上下文下 KV 池更大；并发从 10–18 路提到 20 路 |
| **20 局并发** | `MAXREQ 20`，CUDA graph 到 bs 20，100 个 Mamba 槽 | 总解码吞吐约 1,000 tok/s（原 10 路方案约 600） |
| **SGLang 小算子补丁** | decode 尺寸 M 的 split-K BF16 GEMM、融合的 HC mix/combine-norm（`serving/lowm/`）、M-RoPE 设备端缓存、多模态图片缓存 | 验证批大小下的 decode 单步更短 |
| **MTP 投机解码** | albucino MTP 草稿模型，每次推测 3 个 token（来自 Franzen） | 每步约 2.6 个 token |

这些改动依次上线时的线上（公开榜）分数，详见 `experiments/SUBMISSIONS.md`：
28.17（Franzen M2 基线，10 路）→ 32.56（harness 修正）→ 34.66（hybrid A，18 路）→ 37.96（+ 卡关提示）→
**39.22（+ 第 15–35 层 NVFP4 KV，20 路）**。

### 2. Harness（在 Franzen M2 之上）
**执行范式：一个常驻 Python 进程，而不是一组 tool call。** Agent 只有一个工具 `python`，代码在一个长期存活的进程里运行，
进程里保存着游戏（`action()`、画面、关卡历史）以及模型至今写过的所有函数。模型不是每次从固定工具集里选一个动作调用，
而是写程序：把画面解析成对象，编写并复用辅助函数（`grid_map`、世界模型预测），在状态上跑 BFS/搜索，一次调用就执行
一整串动作。这比常见的 tool call 范式给了模型大得多的自由度，一轮模型输出能推进很多步游戏。这个范式来自 The Duck；
我们的开关建立在它之上（跨裁剪保留的函数、无效动作上限、每次调用 30 秒时限、进程状态的增量历史）。

提交版本打开了我们自己的 34 个开关（完整列表和一句话说明见 `docs/SWITCHES.md`）。我们认为最有用的几个：
- **卡关提示**（`ARC3_LEVEL_TOKEN_NUDGE` 30K、`ARC3_LEVEL_NUDGE_CAUSES`）：同一关用掉 30K token 后告诉模型，并列出关卡
  过不去的常见原因。
- **系统提示里的精简实验笔记**（`ARC3_COMPACT_NOTE_*`）：每次裁剪上下文时，把模型做过的实验压缩成一段笔记附在系统提示
  末尾，裁剪后学到的东西不会丢。
- **感知**：对象级 `frame_diff` v2、长动画提示、动画回放帧、带终局图片的关卡反馈。
- **不浪费动作**：一段代码连续 4 次无效动作后不能再执行动作（`ARC3_SNIPPET_NOOP_LIMIT`）。
- **更省的轮次**：历史里记录运行状态的增量，而不是每次重写（`ARC3_STATE_DELTA`）；Python 每次 30 秒时限。

单个 harness 开关是在 10–25 个公开游戏上离线测的；单次运行有 ±4–6 分的波动，所以多数开关的效果单独看并不显著
（见 `experiments/LESSONS.md`）。

## 目录结构
| 路径 | 内容 |
|---|---|
| `harness/` | 提交时的 harness 原样（Tufa Labs Duck + Franzen M2 + 我们的改动），我们的提交 `dcceaeb` |
| `upstream/arc-agi-3-solution/` | Git 子模块：Franzen 公开的 M2 方案（Apache-2.0），我们的 harness 由此开始 |
| `patches/kleea-vs-franzen-m2.patch` | 我们相对 Franzen M2 代码的 harness 改动（17 个文件） |
| `patches/harness-changes.v25.patch` | notebook 里携带的完整 harness diff（相对 Duck 基础版），已解码 |
| `serving/` | notebook 中嵌入的 SGLang 补丁和模型组装代码，解码成普通文件 |
| `notebook/arc3-kleea-v25.ipynb` | 提交的 notebook，逐字节相同（第 25 版） |
| `notebook/arc3-kleea-readable.ipynb` | 相同代码和设置，去掉 base64：从 Kaggle 数据集 `zhangzibo54/arc3-kleea-solution-code`（= `serving/` + `patches/` 平铺）读取文件。由 `notebook/build_readable.py` 生成；即 Kaggle `arc3-kleea-v25rerun` 第 39 版 |
| `notebook/cells/` | 可读版 notebook 的每个代码 cell，各存为一个 `.py` |
| `config/v25_settings.json` | 提交那次运行的全部 harness 环境变量 |
| `docs/SWITCHES.md` | 提交版本打开的开关，分组，每个一行 |
| `experiments/` | 线上提交记录、关键离线运行、经验教训 |

harness 里每个实验都是一个环境变量开关（默认关闭），和 Franzen 代码的约定一致；`config/v25_settings.json` 就是实际运行
的配置，没列出的开关都是关闭的。

## 运行
notebook 在 Kaggle 上是自包含的（RTX PRO 6000，无网络）：挂上元数据里列出的模型和数据集（hybrid A 用的
`zhangzibo54/arc3-kleea` 模型、Intel W4A16、albucino 草稿模型、Franzen 的 TAAF bundle 和 Pennyroyal wheels）后运行即可。
离线时它玩自带的公开游戏（customization cell 里的 `OFFLINE_GAMES`、`OFFLINE_MINUTES`）；比赛重跑时玩隐藏集 9 小时。

## 致谢
- **Tufa Labs**：The Duck harness 和 TAAF（`github.com/Tufalabs/duck-harness`）。
- **Daniel Franzen**：Milestone-2 方案（`github.com/da-fr/arc-agi-3-solution`，Apache-2.0）、Pennyroyal SGLang 构建、
  albucino MTP 草稿模型和 TAAF Kaggle bundle。
- **cookys**：SGLang QSA 路径的 NVFP4 KV cache（`github.com/cookys/sglang-qsa-nvfp4-kv`，Apache-2.0；见 `serving/nvfp4_kv/NOTICE`）。
- **klee100**：AutoRound 3bpw 检查点；**Intel**：W4A16 AutoRound 检查点。

## 许可证
我们的代码使用 Apache-2.0（见 `LICENSE`）。上游代码保留各自的条款，见 `NOTICE`。
