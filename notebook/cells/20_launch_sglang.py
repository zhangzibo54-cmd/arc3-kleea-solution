# Paste this entire file into the launcher cell.
# Required notebook variables: MODEL_DIR, DRAFT_MODEL_DIR, WHEELHOUSE_DIR,
# WORKING_DIR, SERVED_MODEL_NAME, SERVED_MODEL_PORT.
# MODEL_DIR: Intel AutoRound INT4 target, or the assembled klee Humming model.
# DRAFT_MODEL_DIR: downloaded albucino runtime/mtp-int4-g32, or its dataset root.
import os, sys, glob, time, shutil, subprocess, urllib.request, urllib.error
import copy, hashlib, json, re, shlex, socket, zipfile
from pathlib import Path

PREFIX = "/tmp/sgl-intel"       # Separate from the previous fork's /tmp/sgl venv.
LOG = str(Path(WORKING_DIR) / "serve.log")
CFG = dict(
    CTX=(116+12+8)*1024,
    MEMFRAC=0.96,
    MAXREQ=10,
    CUDAGRAPH_MAXBS=10,
    MAMBA_CACHE=60,
    CHUNK=8192,
    MAX_PREFILL=16384,
    SPEC=True,
    FRSPEC=True,
    AUTOTUNE=True,
    # Preserve the original Kaggle checkpoint prefetch/cache behavior.
    PREFETCH_CHECKPOINTS=True,
    DROP_CACHE_AFTER_LOAD=True,
    LINEAR_BACKEND="flashinfer",
    KVDTYPE="fp8_e4m3",
    SSM_DTYPE="bfloat16",
    MAMBA_RADIX="extra_buffer",
    GDN_MTP_CACHE_MODE="none",
    SERVED_NAME=SERVED_MODEL_NAME,
    SPEC_ACCEPT_SINGLE=1.0,
    SPEC_ACCEPT_ACC=1.0,
    SPEC_STEPS=3,
)
# >>> bench hook: variant overrides (empty for baseline)
CFG.update({k: v for k, v in CFG_OVERRIDES.items() if k != "GRAPH_BS"})
print("CFG =", CFG)
VENV = f"{PREFIX}/venv"
PYTHON = f"{VENV}/bin/python"
SGLANG = f"{VENV}/bin/sglang"
TOKEN_MAP_SHA = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
EXPERT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"
DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"


def run(cmd, *, check=True, env=None):
    cmd = list(map(str, cmd))
    print("$", shlex.join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if result.stdout.strip(): print(result.stdout.strip()[-6000:])
    if result.stderr.strip() and result.returncode: print(result.stderr.strip()[-6000:])
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(cmd)}")
    return result


def find_unique(root, name, *, directory=False, required=True):
    root = Path(root)
    direct = root if root.name == name else root / name
    valid = lambda p: p.is_dir() if directory else p.is_file()
    if valid(direct): return direct
    hits = sorted(p for p in root.rglob(name) if valid(p))
    if len(hits) > 1:
        raise RuntimeError(f"Multiple {name} matches under {root}; use a more specific directory: {hits}")
    if hits: return hits[0]
    if required: raise FileNotFoundError(f"{name} not found under {root}")
    return None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def indexed_shards(root):
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if not names: raise RuntimeError(f"Empty weight index: {root}")
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Invalid shard path: {name}")
        p = root / rel
        if not p.is_file() or p.stat().st_size < 8:
            raise RuntimeError(f"Missing/empty shard: {p}")
    return index, names


def prepare_draft_view(target, source_root, work_root):
    """Adapt metadata only; link original dataset weights without copying them."""
    target, source_root, work_root = map(Path, (target, source_root, work_root))
    configs = [source_root / "config.json"] if (source_root / "config.json").is_file() else []
    configs += sorted(p for p in source_root.rglob("config.json") if p not in configs)
    candidates = []
    for path in configs:
        cfg = json.loads(path.read_text())
        q = cfg.get("quantization_config", {})
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            candidates.append((path.parent, cfg))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one INT4 g32 MTP checkpoint under DRAFT_MODEL_DIR={source_root}; found {len(candidates)}")
    source, cfg = candidates[0]
    index, shards = indexed_shards(source)
    if not any(k.startswith("mtp.layers.0.mlp.experts.") for k in index["weight_map"]):
        raise RuntimeError("Draft index does not contain the expected MTP experts")
    cfg = copy.deepcopy(cfg)
    q = cfg["quantization_config"]
    group = q["config_groups"]["mtp_routed_experts"]
    w = group["weights"]
    if (w["num_bits"], w["group_size"], w["symmetric"]) != (4, 32, True):
        raise RuntimeError("Expected symmetric INT4 group32 draft experts")
    if group["targets"] not in (["RoutedExperts"], [EXPERT_TARGET]):
        raise RuntimeError(f"Unexpected draft target rules: {group['targets']}")
    if q.get("ignore", []) not in ([], [DENSE_IGNORE]):
        raise RuntimeError("Unexpected draft ignore rules")
    group["targets"] = [EXPERT_TARGET]
    q["ignore"] = [DENSE_IGNORE]
    # Content/path-specific directory makes notebook reruns idempotent without
    # overwriting an unrelated view. We never modify the read-only datasets.
    identity = json.dumps({"source": str(source.resolve()), "target": str(target.resolve()), "config": cfg}, sort_keys=True)
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    view = work_root / f"draft-view-{suffix}"
    view.mkdir(parents=True, exist_ok=True)
    config_path = view / "config.json"
    config_text = json.dumps(cfg, indent=2) + "\n"
    if config_path.exists() or config_path.is_symlink():
        if config_path.is_symlink() or config_path.read_text() != config_text:
            raise RuntimeError(f"Unexpected existing draft-view config: {config_path}")
    else:
        config_path.write_text(config_text)
    links = {name: source / name for name in shards}
    links["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
        if (target / name).is_file(): links[name] = target / name
    for name, src in links.items():
        dest = view / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink() or dest.resolve() != src.resolve():
                raise RuntimeError(f"Unexpected existing draft-view file: {dest}")
        else:
            dest.symlink_to(src.resolve())
    print(f"Draft source: {source}\nSGLang draft view: {view} ({len(shards)} linked weight shards)")
    return view


# ---- validate notebook inputs / identify the offline bundle ----
MODEL_DIR = str(Path(MODEL_DIR).expanduser().resolve())
assert Path(MODEL_DIR).is_dir(), f"model dir not found: {MODEL_DIR}"
assert CFG["CUDAGRAPH_MAXBS"] >= CFG["MAXREQ"], "CUDAGRAPH_MAXBS must cover MAXREQ"
assert CFG["CHUNK"] > 0 and CFG["CHUNK"] % 64 == 0
# Refuse to start a second server on an occupied port; do not kill other jobs.
with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", int(SERVED_MODEL_PORT))) == 0:
        raise RuntimeError(f"Port {SERVED_MODEL_PORT} is occupied; stop the existing server before rerunning")
Path(PREFIX).mkdir(parents=True, exist_ok=True)
Path(LOG).parent.mkdir(parents=True, exist_ok=True)
WHEELS = find_unique(WHEELHOUSE_DIR, "wheels", directory=True)
LOCK = find_unique(WHEELHOUSE_DIR, "requirements.lock")
sglang_wheels = sorted(WHEELS.glob("sglang-*.whl"))
assert len(sglang_wheels) == 1, f"Expected one SGLang wheel, found {sglang_wheels}"
wheel = sglang_wheels[0]
install_marker = Path(PREFIX) / "installed-bundle.txt"

# ---- one-time offline install, isolated from the old mratsim environment ----
install_env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1")
install_env.pop("PYTHONPATH", None)
if not Path(SGLANG).exists() or not install_marker.is_file():
    print("=== installing Pennyroyal (offline) ===")
    # Bootstrap uv directly from its wheel; no global pip changes/ensurepip.
    uv_wheels = sorted(WHEELS.glob("uv-*.whl"))
    assert len(uv_wheels) == 1, "Expected one bundled uv wheel"
    uv_binary = Path(PREFIX) / "bin/uv"
    uv_binary.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(uv_wheels[0]) as archive:
        members = [n for n in archive.namelist() if Path(n).name == "uv" and not n.endswith("/")]
        assert len(members) == 1, "Cannot identify the Linux uv executable in its wheel"
        binary = archive.read(members[0])
        assert binary.startswith(b"\x7fELF"), "Bundled uv executable is not a Linux ELF binary"
        uv_binary.write_bytes(binary)
    uv_binary.chmod(0o755)
    uv = [str(uv_binary)]
    if not Path(PYTHON).exists():
        run(uv + ["venv", "--python", sys.executable, VENV], env=install_env)
    install = uv + ["pip", "install", "--python", PYTHON, "--no-index", "--find-links", str(WHEELS)]
    run(install + ["-r", str(LOCK)], env=install_env)
    run(install + ["--reinstall", "--no-deps", str(wheel)], env=install_env)
    install_marker.write_text('install')
else:
    print(f"Matching offline bundle already installed at {VENV}")

# ---- bundled CUDA toolkit and linker symlinks ----
cu13 = sorted(Path(VENV).glob("lib/python*/site-packages/nvidia/cu13"))
if not cu13 or not (cu13[0] / "bin/nvcc").is_file():
    raise RuntimeError("Bundled CUDA 13 nvcc missing; check that the complete Pennyroyal wheelhouse was uploaded")
CUDA_HOME = str(cu13[0])
libdir = Path(CUDA_HOME) / "lib"
lib64 = Path(CUDA_HOME) / "lib64"
if not lib64.exists() and not lib64.is_symlink(): lib64.symlink_to("lib")
for so in sorted(libdir.glob("*.so.*")):
    link = libdir / re.sub(r"\.so\..*$", ".so", so.name)
    if not link.exists() and not link.is_symlink(): link.symlink_to(so.name)
assert (libdir / "libcudart.so").exists(), "libcudart.so missing: JIT linker cannot find CUDA runtime"
if not (libdir / "libcuda.so").exists():
    candidates = []
    for directory in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
        candidates += sorted(p for p in Path(directory).glob("libcuda.so*") if p.is_file())
    if not candidates:
        code = 'import torch; torch.cuda.init(); print(open("/proc/self/maps").read())'
        maps = run([PYTHON, "-c", code], env=install_env).stdout
        candidates = [Path(p) for p in re.findall(r"(/[^\s]*libcuda\.so[^\s]*)", maps)]
    if not candidates: raise RuntimeError("GPU driver libcuda.so not found")
    link = libdir / "libcuda.so"
    if link.is_symlink(): link.unlink()
    link.symlink_to(candidates[0].resolve())
CXX = next((c for c in ("g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++") if shutil.which(c)), None)
assert CXX, "Host C++ compiler not found"
CC = CXX.replace("g++", "gcc")
assert shutil.which(CC), f"Matching C compiler missing: {CC}"

# ---- runtime environment ----
env = dict(os.environ)
env.pop("PYTHONPATH", None)
cache = f"{PREFIX}/cache"
for name in ("huggingface", "torch", "torchinductor", "triton", "flashinfer", "cuda", "sglang/jit"):
    Path(cache, name).mkdir(parents=True, exist_ok=True)
env.update({
    "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_HOME": CUDA_HOME, "CUDACXX": f"{CUDA_HOME}/bin/nvcc",
    "PATH": f"{CUDA_HOME}/bin:{VENV}/bin:" + env.get("PATH", ""),
    "LD_LIBRARY_PATH": f"{CUDA_HOME}/lib64:{CUDA_HOME}/lib:" + env.get("LD_LIBRARY_PATH", ""),
    "CC": CC, "CXX": CXX, "CUDAHOSTCXX": CXX,
    "TORCH_CUDA_ARCH_LIST": "12.0",
    "HF_HOME": f"{cache}/huggingface", "XDG_CACHE_HOME": cache,
    "TORCH_HOME": f"{cache}/torch", "CUDA_CACHE_PATH": f"{cache}/cuda",
    "TORCHINDUCTOR_CACHE_DIR": f"{cache}/torchinductor", "TRITON_CACHE_DIR": f"{cache}/triton",
    "FLASHINFER_WORKSPACE_BASE": f"{cache}/flashinfer",
    "SGLANG_CACHE_DIR": f"{cache}/sglang", "SGLANG_JIT_CACHE_DIR": f"{cache}/sglang/jit",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
    "MAX_JOBS": "8", "CMAKE_BUILD_PARALLEL_LEVEL": "8",
    "FLASHINFER_NINJA_JOBS": "8", "FLASHINFER_NVCC_THREADS": "2", "TORCHINDUCTOR_COMPILE_THREADS": "8",
    "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "1", "SGLANG_SM120_ONLINE_MXFP8": "0",
    "SGLANG_SM120_LOWM_FP8_WEIGHT": "0", "SGLANG_SM120_LM_HEAD_FP8": "0",
    "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_NUMA_BIND_V2": "false",
    "SGLANG_MM_PREPROCESS_DEVICE": "cpu", "NUMPY_MADVISE_HUGEPAGE": "0",
})
env.update(ENV_OVERRIDES)  # bench hook
print("ENV overrides:", ENV_OVERRIDES)
run([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], env=env)
print("Host compiler:", CXX)
# Report the optional later caching patch. This cell does not modify wheel code.
run([PYTHON, "-c", '''import pathlib, sglang, torch
print("sglang", sglang.__version__, "| torch", torch.__version__)
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
'''], env=env)

# ---- validate target and create the read-only draft's SGLang view ----
target_cfg = json.loads(Path(MODEL_DIR, "config.json").read_text())
q = target_cfg.get("quantization_config", {})
if KLEE:
    assert q.get("dynamic"), "KLEE MODEL_DIR must contain Humming dynamic rules"
    assert q.get("quant_method") == "auto-round", "unexpected KLEE quantization method"
    if not KLEE_TOKENIZER_MATCH:
        CFG["FRSPEC"] = False
        print("KLEE tokenizer differs from the pinned FR-Spec map; disabling FR-Spec")
    klee_humming_preflight(PYTHON, env, MODEL_DIR)  # bench hook: SGLang's Humming precision per module, before the load
else:
    assert q.get("quant_method") == "auto-round" and q.get("bits") == 4, "MODEL_DIR must be the Intel/albucino AutoRound INT4 checkpoint"
_, target_shards = indexed_shards(MODEL_DIR)
print("Target:", MODEL_DIR, "| shards:", len(target_shards), "| PLE dtype:", target_cfg.get("text_config", {}).get("ple_embedding_dtype", "checkpoint default (typically BF16)"))
if KLEE:
    ple_source = json.loads(Path(MODEL_DIR, "PLE_SOURCE.json").read_text())
    print("PLE offload remains enabled; linked source files:", ple_source["files"])
# ARC3_MROPE_DEVICE_CACHE=1: per-request mrope deltas cached on the GPU, so speculative decode steps
# with image requests do not block on a host-to-device copy (game/sglang_patches/mrope_cache.py).
MROPE_CACHE_OLD = '            mrope_deltas = [\n                (\n                    torch.zeros(1, dtype=torch.int64)\n                    if mm_inputs[i] is None\n                    else mm_inputs[i].mrope_position_delta.squeeze(0)\n                )\n                for i in range(batch_size)\n            ]\n            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)\n'
MROPE_CACHE_NEW = '            # ARC3 patch: keep each request\'s delta on the GPU (copied once per\n            # request) so this per-step path does no host-to-device copy. The\n            # pageable copy blocked the CPU until the GPU drained the verify,\n            # which stalled the overlap scheduler on every decode step.\n            mrope_deltas = []\n            for mm_input in mm_inputs:\n                if mm_input is None:\n                    mrope_deltas.append(torch.zeros(1, dtype=torch.int64, device=device))\n                    continue\n                cache = mm_input.__dict__.setdefault("_arc3_mrope_delta_by_device", {})\n                cached = cache.get(str(device))\n                if cached is None:\n                    cached = mm_input.mrope_position_delta.squeeze(0).to(device=device)\n                    cache[str(device)] = cached\n                mrope_deltas.append(cached)\n            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0)\n'
_fb = sorted(Path(VENV).glob("lib/python*/site-packages/sglang/srt/model_executor/forward_batch_info.py"))
assert len(_fb) == 1, _fb
_src = _fb[0].read_text(encoding="utf-8")
_want_new = os.environ.get("ARC3_MROPE_DEVICE_CACHE", "0") == "1"
_from, _to = (MROPE_CACHE_OLD, MROPE_CACHE_NEW) if _want_new else (MROPE_CACHE_NEW, MROPE_CACHE_OLD)
if _to not in _src:
    assert _src.count(_from) == 1, "forward_batch_info.py changed: the mrope hunk does not match once"
    _fb[0].write_text(_src.replace(_from, _to, 1), encoding="utf-8")
    for _pyc in _fb[0].parent.glob("__pycache__/forward_batch_info.*.pyc"):
        _pyc.unlink()
print("mrope device cache:", "on" if _want_new else "off")
# ARC3_MM_IMAGE_CACHE=1: SGLang processes and sends features only for new images; history images go by content
# hash and are rebuilt in the scheduler only if re-encoded (game/sglang_patches/mm_image_cache.py).
MM_IMAGE_CACHE_TARGETS = [('lib/python*/site-packages/sglang/srt/multimodal/processors/base_processor.py', '# ===== ARC3 mm lazy image features (game/sglang_patches/mm_image_cache.py) BEGIN =====\n', '# ===== ARC3 mm lazy image features END =====\n', '# ===== ARC3 mm lazy image features (game/sglang_patches/mm_image_cache.py) BEGIN =====\n\ndef _arc3_install_mm_lazy_features(base_cls):\n    """Tokenizer side: process and send only new images; items of seen images carry no feature."""\n    import collections\n    import hashlib\n    import io\n    import os\n    import pickle\n    import threading\n\n    import numpy as np\n    import torch\n    from PIL import Image\n    from transformers import BatchFeature\n\n    store = os.environ.get("ARC3_MM_IMAGE_STORE", "/dev/shm/arc3_mm_images")\n    os.makedirs(store, exist_ok=True)\n    lock = threading.Lock()\n    seen = collections.OrderedDict()  # key -> (hash, grid row, rows, stored)\n    meta = {}  # feature width and dtype of this processor\'s output\n    local = threading.local()\n    stats = collections.Counter()\n\n    def image_key(image, kwargs_key):\n        if isinstance(image, Image.Image):\n            data, desc = image.tobytes(), (image.mode, image.size)\n        elif isinstance(image, torch.Tensor):\n            data, desc = image.detach().cpu().contiguous().numpy().tobytes(), (str(image.dtype), tuple(image.shape))\n        elif isinstance(image, np.ndarray):\n            data, desc = np.ascontiguousarray(image).tobytes(), (str(image.dtype), image.shape)\n        else:\n            return None\n        digest = hashlib.blake2b(digest_size=20)\n        digest.update(repr((desc, kwargs_key)).encode())\n        digest.update(data)\n        return digest.digest()\n\n    def store_image(item_hash, inner, image, kwargs):\n        """Write the image (PNG) and the call\'s kwargs for the scheduler; False if not possible."""\n        if not isinstance(image, Image.Image):\n            return False\n        try:\n            proc_dir = os.path.join(store, "processor")\n            if not os.path.exists(os.path.join(proc_dir, "CLASS")):\n                os.makedirs(proc_dir, exist_ok=True)\n                inner.save_pretrained(proc_dir)\n                with open(os.path.join(proc_dir, "CLASS.tmp"), "w") as f:\n                    f.write(type(inner).__name__)\n                os.replace(os.path.join(proc_dir, "CLASS.tmp"), os.path.join(proc_dir, "CLASS"))\n            png = io.BytesIO()\n            image.save(png, format="PNG")\n            path = os.path.join(store, f"{item_hash:016x}.pkl")\n            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"\n            with open(tmp, "wb") as f:\n                pickle.dump((type(inner).__name__, kwargs, png.getvalue()), f)\n            os.replace(tmp, path)\n            return True\n        except Exception as e:  # the image is then always sent with features\n            print(f"[arc3 mm lazy features] store failed: {e!r}", flush=True)\n            return False\n\n    class LazyImageProcessor:\n        """The wrapped image processor: features are computed for new images only."""\n\n        def __init__(self, inner):\n            self.__dict__["_arc3_inner"] = inner\n\n        def __getattr__(self, name):\n            return getattr(self.__dict__["_arc3_inner"], name)\n\n        def __setattr__(self, name, value):\n            setattr(self.__dict__["_arc3_inner"], name, value)\n\n        def __call__(self, images=None, **kwargs):\n            inner = self.__dict__["_arc3_inner"]\n            local.info = None\n            items = list(images) if isinstance(images, (list, tuple)) else None\n            if not items or any(isinstance(x, (list, tuple)) for x in items):\n                stats["bypass"] += 1\n                return inner(images=images, **kwargs)\n            kwargs_key = repr(sorted((k, repr(v)) for k, v in kwargs.items()))\n            keys = [image_key(x, kwargs_key) for x in items]\n            if any(k is None for k in keys):\n                stats["bypass"] += 1\n                return inner(images=images, **kwargs)\n            with lock:\n                known = {k: seen[k] for k in set(keys) if k in seen and seen[k][3]}\n            todo = []  # first index of each image that needs features\n            for i, k in enumerate(keys):\n                if k not in known and all(keys[j] != k for j in todo):\n                    todo.append(i)\n            fresh = {}  # key -> (pixel rows, grid row)\n            if todo:\n                out = inner(images=[items[i] for i in todo], **kwargs)\n                pv, grid = out.get("pixel_values"), out.get("image_grid_thw")\n                if (set(out.keys()) != {"pixel_values", "image_grid_thw"} or not isinstance(pv, torch.Tensor)\n                        or not isinstance(grid, torch.Tensor) or pv.device.type != "cpu" or pv.dim() != 2\n                        or len(grid) != len(todo)):\n                    stats["bypass"] += 1\n                    print("[arc3 mm lazy features] unexpected image processor output; this call unpatched", flush=True)\n                    return inner(images=images, **kwargs)\n                meta["width"], meta["dtype"] = pv.shape[1], pv.dtype\n                for i, rows, g in zip(todo, pv.split(grid.prod(-1).tolist()), grid):\n                    fresh[keys[i]] = (rows, g)\n                    h = int.from_bytes(keys[i][:8], "big")\n                    stored = store_image(h, inner, items[i], kwargs)\n                    with lock:\n                        seen[keys[i]] = (h, g.clone(), rows.shape[0], stored)\n                        while len(seen) > 200_000:\n                            seen.popitem(last=False)\n            with lock:\n                info = [seen[k] for k in keys]\n                for k in set(keys):\n                    seen.move_to_end(k)\n            total = sum(x[2] for x in info)\n            pv = torch.empty(total, meta["width"], dtype=meta["dtype"])  # rows of seen images stay unwritten\n            start = 0\n            for k, (_, _, rows, _) in zip(keys, info):\n                if k in fresh:\n                    pv[start:start + rows].copy_(fresh[k][0])\n                start += rows\n            grid = torch.stack([x[1] for x in info])\n            local.info = [(x[0], k in fresh, x[2]) for k, x in zip(keys, info)]\n            stats["calls"] += 1\n            stats["images"] += len(keys)\n            stats["with_features"] += sum(1 for k in keys if k in fresh)\n            if stats["calls"] % 200 == 0:\n                print(f"[arc3 mm lazy features] calls {stats[\'calls\']}, images {stats[\'images\']}, "\n                      f"sent with features {stats[\'with_features\']}, bypass {stats[\'bypass\']}, "\n                      f"seen {len(seen)}", flush=True)\n            return BatchFeature(data={"pixel_values": pv, "image_grid_thw": grid})\n\n    orig_resolve = base_cls._resolve_processor\n\n    def _resolve_processor(self, processor=None):\n        processor, tokenizer = orig_resolve(self, processor)\n        image_processor = getattr(processor, "image_processor", None)\n        if image_processor is not None and not isinstance(image_processor, LazyImageProcessor):\n            processor.image_processor = LazyImageProcessor(image_processor)\n        return processor, tokenizer\n\n    orig_finalize = base_cls._finalize_mm_items\n\n    def _finalize_mm_items(self, mm_items, *, images):\n        info, local.info = getattr(local, "info", None), None\n        if info is not None:\n            image_items = [item for item in mm_items if item.is_image()]\n            if len(image_items) != len(info):\n                raise RuntimeError(f"[arc3 mm lazy features] {len(image_items)} image items for {len(info)} images")\n            for item, (item_hash, has_features, rows) in zip(image_items, info):\n                if not isinstance(item.feature, torch.Tensor) or item.feature.shape[0] != rows:\n                    raise RuntimeError("[arc3 mm lazy features] image item does not match its image")\n                item.hash = item_hash\n                if not has_features:\n                    item.feature = None\n        return orig_finalize(self, mm_items, images=images)\n\n    base_cls._resolve_processor = _resolve_processor\n    base_cls._finalize_mm_items = _finalize_mm_items\n    print(f"[arc3 mm lazy features] tokenizer side installed, store {store}", flush=True)\n\n\n_arc3_install_mm_lazy_features(BaseMultimodalProcessor)\n# ===== ARC3 mm lazy image features END =====\n'), ('lib/python*/site-packages/sglang/srt/managers/mm_schedule.py', '# ===== ARC3 mm lazy image features, scheduler side (game/sglang_patches/mm_image_cache.py) BEGIN =====\n', '# ===== ARC3 mm lazy image features, scheduler side END =====\n', '# ===== ARC3 mm lazy image features, scheduler side (game/sglang_patches/mm_image_cache.py) BEGIN =====\n\ndef _arc3_install_feature_rebuild(module):\n    """Scheduler side: rebuild the features of an image sent without them right before it is encoded."""\n    import io\n    import os\n    import pickle\n\n    import torch\n    import transformers\n    from PIL import Image\n\n    store = os.environ.get("ARC3_MM_IMAGE_STORE", "/dev/shm/arc3_mm_images")\n    processors = {}\n    stats = {"rebuilt": 0}\n\n    def rebuild(item):\n        path = os.path.join(store, f"{item.hash & ((1 << 64) - 1):016x}.pkl")\n        with open(path, "rb") as f:\n            cls_name, kwargs, png = pickle.load(f)\n        if cls_name not in processors:\n            processors[cls_name] = getattr(transformers, cls_name).from_pretrained(os.path.join(store, "processor"))\n        image = Image.open(io.BytesIO(png))\n        image.load()\n        out = processors[cls_name](images=[image], **kwargs)\n        grid = out["image_grid_thw"].reshape(-1)\n        if not torch.equal(grid.to(item.image_grid_thw.device), item.image_grid_thw.reshape(-1)):\n            raise RuntimeError(f"[arc3 mm lazy features] rebuilt grid {grid.tolist()} != item grid")\n        item.feature = out["pixel_values"]\n        stats["rebuilt"] += 1\n        if stats["rebuilt"] % 100 == 1:\n            print(f"[arc3 mm lazy features] rebuilt {stats[\'rebuilt\']} image features in the scheduler", flush=True)\n\n    orig_move = module._move_items_to_device\n\n    def _move_items_to_device(items, device):\n        for item in items:\n            if item.feature is None and item.precomputed_embeddings is None and item.is_image():\n                rebuild(item)\n        return orig_move(items, device)\n\n    module._move_items_to_device = _move_items_to_device\n    # always take the path that moves (and so rebuilds) features before the encoder call\n    module._can_skip_pre_embed_feature_move = lambda data_embedding_func: False\n\n\nimport sys as _a3_sys  # noqa: E402\n\n_arc3_install_feature_rebuild(_a3_sys.modules[__name__])\n# ===== ARC3 mm lazy image features, scheduler side END =====\n')]
for _pattern, _begin, _end, _block in MM_IMAGE_CACHE_TARGETS:
    _paths = sorted(Path(VENV).glob(_pattern))
    assert len(_paths) == 1, _paths
    _src = _paths[0].read_text(encoding="utf-8")
    if _begin in _src:
        _src = _src[:_src.index(_begin)].rstrip("\n") + "\n" + _src[_src.index(_end) + len(_end):]
    if os.environ.get("ARC3_MM_IMAGE_CACHE", "0") == "1":
        _src = _src.rstrip("\n") + "\n\n\n" + _block
    _paths[0].write_text(_src, encoding="utf-8")
    for _pyc in _paths[0].parent.glob(f"__pycache__/{_paths[0].stem}.*.pyc"):
        _pyc.unlink()
print("mm lazy image features:", "on" if os.environ.get("ARC3_MM_IMAGE_CACHE", "0") == "1" else "off")
# ARC3_LOWM_GEMM / _HC_MIX / _HC_COMBINE_NORM: split-K BF16 GEMMs at 33-128 rows, two-launch HC mix, HC combine
# fused with the next hc_norm (game/sglang_patches/arc3_lowm_patch.py, installed as the package arc3_lowm).
import base64 as _b64lw
_LOWM_FILES = {n: code_file(n).decode() for n in ['lowm_splitk_gemm.py', 'hc_mix_2k.py', 'hc_combine_norm.py', 'patch.py']}
_LOWM_ON = any(os.environ.get(_k, "0") == "1" for _k in ("ARC3_LOWM_GEMM", "ARC3_LOWM_HC_MIX", "ARC3_LOWM_HC_COMBINE_NORM"))
_LOWM_HOOK = "\n# ARC3_LOWM_HOOK\ntry:\n    import arc3_lowm.patch as _arc3_lowm_patch\n    _arc3_lowm_patch.{fn}({cls})\nexcept ImportError:\n    pass\n"
_site = sorted(Path(VENV).glob("lib/python*/site-packages"))
assert len(_site) == 1, _site
shutil.rmtree(_site[0] / "arc3_lowm", ignore_errors=True)
if _LOWM_ON:
    (_site[0] / "arc3_lowm").mkdir()
    (_site[0] / "arc3_lowm" / "__init__.py").write_text("")
    for _name, _text in _LOWM_FILES.items():
        (_site[0] / "arc3_lowm" / _name).write_text(
            _text.replace("\nfrom lowm_splitk_gemm import", "\nfrom .lowm_splitk_gemm import"), encoding="utf-8")
for _rel, _fn, _cls in (("srt/layers/quantization/unquant.py", "install_linear", "UnquantizedLinearMethod"),
                        ("srt/layers/hyperconnection.py", "install_hc", "GatedResidual")):
    _path = _site[0] / "sglang" / _rel
    _src = _path.read_text(encoding="utf-8")
    _cut = _src.find("\n# ARC3_LOWM_HOOK\n")
    if _cut >= 0:
        _src = _src[:_cut] + "\n"
    if _LOWM_ON:
        _src = _src.rstrip("\n") + "\n" + _LOWM_HOOK.format(fn=_fn, cls=_cls)
    _path.write_text(_src, encoding="utf-8")
    for _pyc in _path.parent.glob(f"__pycache__/{_path.stem}.*.pyc"):
        _pyc.unlink()
print("arc3 lowm patch:", "on" if _LOWM_ON else "off",
      {_k: os.environ.get(_k, "0") for _k in ("ARC3_LOWM_GEMM", "ARC3_LOWM_HC_MIX", "ARC3_LOWM_HC_COMBINE_NORM")})
DRAFT_VIEW = None
if CFG["SPEC"]:
    assert Path(DRAFT_MODEL_DIR).is_dir(), f"draft dir not found: {DRAFT_MODEL_DIR}"
    DRAFT_VIEW = prepare_draft_view(MODEL_DIR, DRAFT_MODEL_DIR, PREFIX)

# ---- assemble Pennyroyal / AutoRound launch arguments ----
graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})
graph_bs = [n for n in graph_bs if n <= CFG["CUDAGRAPH_MAXBS"]]
if "GRAPH_BS" in CFG_OVERRIDES: graph_bs = sorted(set(CFG_OVERRIDES["GRAPH_BS"]) | {CFG["MAXREQ"]})  # bench hook
args = [SGLANG, "serve", "--model-path", MODEL_DIR, "--load-format", "safetensors",
    "--model-loader-extra-config", '{"enable_multithread_load":true,"num_threads":8}',
    "--served-model-name", str(CFG["SERVED_NAME"]), "--host", "127.0.0.1",
    "--port", str(SERVED_MODEL_PORT), "--tensor-parallel-size", "1",
    "--dtype", "bfloat16", "--quantization", "humming" if KLEE else "auto-round",
    "--kv-cache-dtype", CFG["KVDTYPE"], "--mem-fraction-static", str(CFG["MEMFRAC"]),
    "--context-length", str(CFG["CTX"]), "--page-size", "64",
    "--max-running-requests", str(CFG["MAXREQ"]), "--chunked-prefill-size", str(CFG["CHUNK"]),
    "--max-prefill-tokens", str(CFG["MAX_PREFILL"]),
    "--cuda-graph-max-bs-decode", str(CFG["CUDAGRAPH_MAXBS"]),
    "--cuda-graph-bs-decode", *map(str, graph_bs),
    "--mamba-ssm-dtype", CFG["SSM_DTYPE"], "--max-mamba-cache-size", str(CFG["MAMBA_CACHE"]),
    "--mamba-radix-cache-strategy", CFG["MAMBA_RADIX"], "--mamba-track-interval", "64",
    "--mamba-backend", CFG["LINEAR_BACKEND"],
    "--linear-attn-decode-backend", CFG["LINEAR_BACKEND"],
    "--linear-attn-prefill-backend", CFG["LINEAR_BACKEND"],
    "--moe-runner-backend", "auto", "--ple-offload-embedding", "--trust-remote-code",
    "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
    "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
    "--default-chat-template-kwargs", '{"preserve_thinking":true}',
    "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
    "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging"]
if CFG["PREFETCH_CHECKPOINTS"]: args += ["--weight-loader-prefetch-checkpoints"]
if CFG["DROP_CACHE_AFTER_LOAD"]: args += ["--weight-loader-drop-cache-after-load"]
tmpl = Path(MODEL_DIR, "chat_template.jinja")
if tmpl.is_file():
    # Keep the "Reasoning effort ..." line in the system prompt whether or not thinking is on, so a
    # thinking-off request (rolling summaries) shares the cached prefix of the thinking-on game requests.
    # Thinking-on renders are byte-identical to the stock template (serving/check_chat_template.py).
    _t = tmpl.read_text(encoding="utf-8")
    _cond = "{%- if enable_thinking is undefined or enable_thinking is true %}\n    {%- set resolved_reasoning_effort"
    assert _t.count(_cond) == 1, "chat template changed: reasoning-effort condition not found"
    tmpl = Path("/kaggle/working/chat_template_reasoning_line_always.jinja")
    tmpl.write_text(_t.replace(_cond, "{%- if true %}\n    {%- set resolved_reasoning_effort"), encoding="utf-8")
    args += ["--chat-template", str(tmpl)]
if not CFG["AUTOTUNE"]: args += ["--disable-flashinfer-autotune"]
if CFG["GDN_MTP_CACHE_MODE"]: args += ["--gdn-mtp-cache-mode", CFG["GDN_MTP_CACHE_MODE"]]
if CFG["SPEC"]:
    args += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(CFG["SPEC_STEPS"]),
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(CFG["SPEC_STEPS"]+1),
        "--speculative-draft-model-path", str(DRAFT_VIEW),
        "--speculative-draft-model-quantization", "compressed-tensors", "--speculative-moe-runner-backend", "auto",
        "--speculative-draft-kv-cache-dtype", CFG.get("DRAFT_KVDTYPE", CFG["KVDTYPE"]),  # bench hook
        "--speculative-accept-threshold-single", str(CFG["SPEC_ACCEPT_SINGLE"]),
        "--speculative-accept-threshold-acc", str(CFG["SPEC_ACCEPT_ACC"])]
    if CFG["FRSPEC"]:
        tok = find_unique(WHEELHOUSE_DIR, "hot_tokens_64k.pt", required=False)
        if tok is None: tok = find_unique(WHEELHOUSE_DIR, "flash-next-64k.pt")
        assert sha256(tok) == TOKEN_MAP_SHA, "Wrong FR-Spec map; use the map from the pinned Pennyroyal bundle"
        assert sha256(Path(MODEL_DIR, "tokenizer.json")) == TOKENIZER_SHA, "Tokenizer differs from the FR-Spec map; set CFG['FRSPEC']=False"
        args += ["--speculative-token-map", str(tok)]

# ARC3_QSA_INDEXER_BUDGET: tokens each QSA query attends (model config indexer_budget = 2048, in
# blocks of 4). SGLang's QSA top-k kernel takes 512 or 2048 blocks, i.e. 2048 or 8192 tokens; the
# override also reaches the MTP draft model (no speculative_draft_model_override_args is set).
_qsa_budget = int(os.environ.get("ARC3_QSA_INDEXER_BUDGET", "2048"))
assert _qsa_budget in (2048, 8192), f"ARC3_QSA_INDEXER_BUDGET must be 2048 or 8192, got {_qsa_budget}"
if _qsa_budget != 2048:
    args += ["--json-model-override-args", json.dumps({"text_config": {"indexer_budget": _qsa_budget}})]
print("QSA indexer budget:", _qsa_budget)
args += list(EXTRA_ARGS)  # bench hook
if PATCH_NVFP4: apply_nvfp4_patch(VENV, PYTHON, env)
else:  # fp8 KV variant: the hooks the NVFP4 patch chain would install
    if KLEE_HYBRID:
        _install_tensor_filter(VENV)
        env['SGLANG_INDEX_FILTER_DIR'] = MODEL_DIR
    if ISTA: _install_ista(VENV, PYTHON, env)  # bench hook: NVFP4 KV patch + GPU unit test
# ARC3_QSA_FP4_LAYERS: NVFP4 KV for the listed QSA full-attention layers, fp8 for the rest
# (game/sglang_patches/arc3_kvmix_patch.py, installed as the package arc3_kvmix; needs the NVFP4 KV patch).
import base64 as _b64km
_KVMIX_SRC = code_file('arc3_kvmix_patch.py').decode()
_KVMIX_ON = bool(os.environ.get("ARC3_QSA_FP4_LAYERS", "").strip())
_KVCALIB_ON = bool(os.environ.get("ARC3_KV_CALIB_DIR", "").strip())  # fp8 calibration statistics
_site = sorted(Path(VENV).glob("lib/python*/site-packages"))
assert len(_site) == 1, _site
shutil.rmtree(_site[0] / "arc3_kvmix", ignore_errors=True)
if _KVMIX_ON:
    assert PATCH_NVFP4, "ARC3_QSA_FP4_LAYERS needs the NVFP4 KV patch (V['PATCH_NVFP4'])"
if _KVMIX_ON or _KVCALIB_ON:
    (_site[0] / "arc3_kvmix").mkdir()
    (_site[0] / "arc3_kvmix" / "__init__.py").write_text("")
    (_site[0] / "arc3_kvmix" / "patch.py").write_text(_KVMIX_SRC, encoding="utf-8")
_KVMIX_HOOK = ("\n# ARC3_KVMIX_HOOK\ntry:\n    import sys as _arc3_sys\n    import arc3_kvmix.patch as _arc3_kvmix\n"
               "    _arc3_kvmix.{fn}(_arc3_sys.modules[__name__])\nexcept ImportError:\n    pass\n")
for _rel, _fn, _on in (("srt/mem_cache/memory_pool.py", "install_pool", _KVMIX_ON),
                       ("srt/model_executor/pool_configurator.py", "install_cell_size", _KVMIX_ON),
                       ("srt/layers/attention/qsa_nvfp4_kv.py", "install_view", _KVMIX_ON),
                       ("srt/layers/attention/qwen_sparse_attn_backend.py", "install_calib", _KVCALIB_ON)):
    _path = _site[0] / "sglang" / _rel
    if not _path.exists():
        assert not _on, f"{_rel} missing (NVFP4 KV patch not applied?)"
        continue
    _src = _path.read_text(encoding="utf-8")
    _cut = _src.find("\n# ARC3_KVMIX_HOOK\n")
    if _cut >= 0:
        _src = _src[:_cut] + "\n"
    if _on:
        _src = _src.rstrip("\n") + "\n" + _KVMIX_HOOK.format(fn=_fn)
    _path.write_text(_src, encoding="utf-8")
    for _pyc in _path.parent.glob(f"__pycache__/{_path.stem}.*.pyc"):
        _pyc.unlink()
print("arc3 kvmix patch:", "on" if _KVMIX_ON else "off", "ARC3_QSA_FP4_LAYERS =",
      repr(os.environ.get("ARC3_QSA_FP4_LAYERS", "")), "| kvcalib:", "on" if _KVCALIB_ON else "off")
# ---- launch detached and wait for health ----
precache_model_thread.start()
print("\n" + shlex.join(args) + "\n")
log_offset = Path(LOG).stat().st_size if Path(LOG).exists() else 0
with open(LOG, "ab", buffering=0) as logf:
    proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
print(f"pid {proc.pid} -> {LOG}")

def show_log_tail():
    with open(LOG, "rb") as f:
        f.seek(0, 2)
        f.seek(max(log_offset, f.tell()-20000))
        print(f.read().decode(errors="replace"))

url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", time.time()
last_report = -30
def _startup_status(elapsed):
    mem = {}
    try:
        for line in open('/proc/meminfo'):
            k, v = line.split(':', 1)
            mem[k] = int(v.split()[0]) / 1e6
    except Exception:
        pass
    tail = ''
    try:
        with open(LOG, 'rb') as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 4000))
            t = f.read().decode(errors='replace').replace('\r', '\n').strip().splitlines()
            tail = t[-1][-110:] if t else ''
    except Exception:
        pass
    pre = globals().get('_PRE') or {}
    cached = pre.get('ple_cached', {}) if pre else {}
    ahead = sum(1 for n in cached if n not in pre.get('gone', ()))
    ple = (f"PLE read {len(cached)} (ahead of loader {ahead}), dropped {pre.get('dropped_gb', 0):.0f} GB, "
           f"loader {pre.get('progress', 0)}/{pre.get('total', 0)} | ") if pre else ''
    return (f"  {elapsed}s | precache {_PRECACHE_DONE[0] / 1e9:.1f} GB | {ple}"
            f"MemAvailable {mem.get('MemAvailable', 0):.1f} GB, Cached {mem.get('Cached', 0):.1f} GB | {tail}")
while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT:
    if proc.poll() is not None:
        show_log_tail()
        break
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status == 200:
                print(f"\nREADY after {int(time.time()-t0)}s -> {url.replace('/health','/v1')}")
                break
    except Exception:
        pass
    elapsed = int(time.time()-t0)
    if elapsed-last_report >= 30:
        print(_startup_status(elapsed), flush=True); last_report=elapsed
    time.sleep(3)
else:
    print(f"\nDEADLINE at {int(time.time()-NOTEBOOK_START_TIME)}s from notebook start; "
          f"releasing the benchmark with the server still loading")
