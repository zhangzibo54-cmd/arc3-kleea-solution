"""Build the readable notebook from the submitted one (version 25).

    python notebook/build_readable.py

Same code and settings as notebook/arc3-kleea-v25.ipynb. Two changes only:
  * every base64 blob is replaced by reading the same file from the Kaggle dataset
    zhangzibo54/arc3-kleea-solution-code (= serving/ + patches/ of this repo, flattened);
  * a first markdown cell with a summary and links.
The offline demo runs demo10 x 10 min.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "arc3-kleea-v25.ipynb"
OUT = HERE / "arc3-kleea-readable.ipynb"
REPO = "https://github.com/zhangzibo54-cmd/arc3-kleea-solution"

INTRO = f"""# ARC-AGI-3 · klee A — solo solution, public LB 39.22

**Scored version: version 25 of this notebook** (submission 57745091, public 39.22). This latest version is the
same code and settings in readable form: the base64 blobs of v25 are now plain files in the dataset
[`zhangzibo54/arc3-kleea-solution-code`](https://www.kaggle.com/datasets/zhangzibo54/arc3-kleea-solution-code).
Offline it plays 10 public games for 10 minutes each.

**Code, experiment log and lessons: [{REPO}]({REPO})**
(`experiments/SUBMISSIONS.md`, `experiments/LESSONS.md`, `docs/SWITCHES.md`).

### What we changed
Agent: [The Duck](https://github.com/Tufalabs/duck-harness) (Tufa Labs) with
[Daniel Franzen's Milestone-2 solution](https://github.com/da-fr/arc-agi-3-solution) (public 28.17 as our baseline).

**1. Throughput (most of the gain).** One RTX PRO 6000, SGLang (Pennyroyal v2.5.3):
- *Hybrid A* weights: 3-bit routed experts (klee100 AutoRound 3bpw) in layers 0–11 and 36–47, Intel W4A16 in
  layers 12–35, Humming W3/W4 kernels → model ~70 → ~55 GB.
- KV in NVFP4 on the QSA layers 15–35, FP8 on the first/last QSA layers (NVFP4 everywhere lost points).
- 20 concurrent games at 148K context (Franzen: 10); ~1,000 tok/s aggregate decode vs ~600.
- Small decode kernels (split-K BF16 GEMM, fused HC mix/combine-norm), M-RoPE device cache, image cache.

**2. Harness** (34 switches on top of Franzen's, all listed in `docs/SWITCHES.md`):
stuck-level note after 30K tokens on a level with a checklist of causes; a lab-notebook summary kept in the
system prompt across context trims; object-level frame diffs and long-animation notes; history deltas instead of
full runtime state; no-op action limit; 30 s Python limit.

Online: 28.17 → 32.56 (history deltas) → 34.66 (hybrid A, 18 streams) → 37.96 (stuck-level note) →
**39.22** (mixed NVFP4/FP8 KV, 20 streams).
"""

CODE_CELL = '''# Plain-file sources (formerly base64 blobs): the dataset zhangzibo54/arc3-kleea-solution-code holds
# serving/ and patches/ of the GitHub repo, flattened.
from pathlib import Path as _P

_CODE_DIRS = [_P('/kaggle/input/datasets/zhangzibo54/arc3-kleea-solution-code'),
              _P('/kaggle/input/arc3-kleea-solution-code')]


def code_file(name: str) -> bytes:
    for d in _CODE_DIRS:
        hits = sorted(d.rglob(name)) if d.is_dir() else []
        if hits:
            return hits[0].read_bytes()
    raise FileNotFoundError(f'{name} not found in the arc3-kleea-solution-code dataset')


# The harness diff (Duck base -> our harness): patches/harness-changes.v25.patch
_P('/kaggle/harness-changes.patch').write_bytes(code_file('harness-changes.v25.patch'))
assert __import__('hashlib').sha256(open('/kaggle/harness-changes.patch', 'rb').read()).hexdigest() == '174a9163b51eab6ab28ae3ebe7684baa964761aa7f22a1235ecd7a34b904c963'
'''

B64 = r"[A-Za-z0-9+/=]{200,}"
REPLACE = [  # (cell marker, regex, replacement)
    ("KLEE 3BPW MODEL", rf'_base64\.b64decode\("{B64}"\)', None),  # by content, below
    ("NVFP4 KV PATCH", rf'json\.loads\(base64\.b64decode\("{B64}"\)\.decode\(\)\)',
     "{n: code_file(n).decode() for n in ['apply_qsa_nvfp4_patches.py', 'qsa_nvfp4_kv.py', "
     "'test_qsa_nvfp4_gather.py', 'test_concurrent_needle.py', 'NOTICE']}"),
    ("NVFP4 KV PATCH", rf"_b64\.b64decode\('{B64}'\)", "code_file('int8_ckpt_ple.py')"),
    ("tensor_filter hook", rf"base64\.b64decode\('{B64}'\)", "code_file('tensor_filter.py')"),
    ("Paste this entire file", rf'json\.loads\(_b64lw\.b64decode\("{B64}"\)\.decode\(\)\)',
     "{n: code_file(n).decode() for n in ['lowm_splitk_gemm.py', 'hc_mix_2k.py', 'hc_combine_norm.py', 'patch.py']}"),
    ("Paste this entire file", rf'_b64km\.b64decode\("{B64}"\)', "code_file('arc3_kvmix_patch.py')"),
]
KLEE_FILES = ["assemble.py", "pack_hybrid.py", "hybrid.py"]  # in order of appearance
EDITS = [("OFFLINE_MINUTES = 5\n", "OFFLINE_MINUTES = 10\n")]


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]
    src = lambda c: "".join(c["source"])
    assert src(cells[0]).startswith("# hybrid A baseline") and "gzip+base64" in src(cells[2])
    cells[0]["source"] = INTRO
    cells[1]["source"] = "## 0. Harness patch\n\nOur changes to the Tufa harness, from the code dataset."
    cells[2]["source"] = CODE_CELL
    for marker, rx, new in REPLACE:
        hit = [c for c in cells if c["cell_type"] == "code" and marker in src(c)]
        assert len(hit) == 1, marker
        s = src(hit[0])
        if new is None:
            parts = re.split(rx, s)
            assert len(parts) == len(KLEE_FILES) + 1
            s = parts[0] + "".join(f"code_file('{f}')" + p for f, p in zip(KLEE_FILES, parts[1:]))
        else:
            s, n = re.subn(rx, lambda m: new, s)
            assert n == 1, (marker, n)
        hit[0]["source"] = s
    for old, new in EDITS:
        hit = [c for c in cells if old in src(c)]
        assert len(hit) == 1, old
        hit[0]["source"] = src(hit[0]).replace(old, new, 1)
    for c in cells:
        c["source"] = src(c)
        assert not re.search(r"[A-Za-z0-9+/=]{2000,}", c["source"]), c["source"][:80]
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
