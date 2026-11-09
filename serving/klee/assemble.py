"""Assemble the klee100 3bpw model directory from the mirror parts on Kaggle.

Inputs (attached to the notebook): the outputs of zhangzibo54/arc3-klee3bpw-p1..p4
(each holds `klee3bpw/`), and the Intel W4A16 model, whose BF16 n-gram/PLE table
replaces klee100 shard 1 (102 GB, not mirrored).

The result is a directory of symlinks plus three rewritten files:
  * config.json  <- config.humming.json (Humming `dynamic` rules, see
    tools/humming_config.py);
  * model.safetensors.index.json: PLE table keys point at the Intel files;
  * PLE_SOURCE.json: which Intel files were linked.

It refuses to link an Intel file that also holds tensors other than the PLE
table, because the loader would read those too and overwrite klee weights.

Usage (in a notebook cell):
    from assemble import assemble
    MODEL_DIR = assemble(intel_dir=..., out='/tmp/klee3bpw-model')
"""

import json
from pathlib import Path

PLE_SHARD = 'model-00001-of-00011.safetensors'


def find_parts(root: str = '/kaggle/input') -> list[Path]:
    """Return the `klee3bpw` directories of the attached mirror parts.

    A part is a `klee3bpw/` directory (notebook outputs, datasets) or a directory
    holding a `klee3bpw.part` marker (Kaggle Model instances, whose files are flat).
    """
    parts = sorted({p.parent for p in Path(root).rglob('klee3bpw/*.safetensors')}
                   | {p.parent for p in Path(root).rglob('klee3bpw/config.json')}
                   | {p.parent for p in Path(root).rglob('klee3bpw.part')})
    return parts


def _tensor_names(path: Path) -> list[str]:
    import struct
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop('__metadata__', None)
    return list(header)


def assemble(intel_dir: str, out: str = '/tmp/klee3bpw-model', root: str = '/kaggle/input') -> str:
    """Build the model directory and return its path.

    Raises:
        FileNotFoundError: A part, a shard or the PLE table is missing.
        RuntimeError: An Intel PLE file holds other tensors as well.
    """
    parts = find_parts(root)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for part in parts:
        for p in part.iterdir():
            if p.is_file():
                files.setdefault(p.name, p)
    for name in ('config.humming.json', 'model.safetensors.index.json'):
        if name not in files:
            raise FileNotFoundError(f'{name} missing; attach arc3-klee3bpw-p4 (parts found: {parts})')
    index = json.loads(files['model.safetensors.index.json'].read_text())
    wmap = index['weight_map']
    needed = sorted(set(wmap.values()) - {PLE_SHARD})
    missing = [f for f in needed if f not in files]
    if missing:
        raise FileNotFoundError(f'shards missing: {missing}; parts found: {parts}')

    # PLE table from the Intel model
    intel = Path(intel_dir)
    iwmap = json.loads((intel / 'model.safetensors.index.json').read_text())['weight_map']
    ple_keys = [k for k, f in wmap.items() if f == PLE_SHARD]
    absent = [k for k in ple_keys if k not in iwmap]
    if absent:
        raise FileNotFoundError(f'{len(absent)} PLE tensors not in the Intel model, e.g. {absent[:3]}')
    intel_files = sorted({iwmap[k] for k in ple_keys})
    ple_set = set(ple_keys)
    for f in intel_files:
        extra = [k for k in _tensor_names(intel / f) if k not in ple_set]
        if extra:
            raise RuntimeError(f'Intel file {f} also holds {len(extra)} non-PLE tensors, e.g. {extra[:3]}')
        if f in files:
            raise RuntimeError(f'Intel file name {f} collides with a klee file')
    for k in ple_keys:
        wmap[k] = iwmap[k]

    def link(dst_name, src):
        dst = out_dir / dst_name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)

    for name, p in files.items():
        if name in ('config.json', 'model.safetensors.index.json', 'config.humming.json'):
            continue
        link(name, p)
    for f in intel_files:
        link(f, intel / f)
    (out_dir / 'config.json').write_text(files['config.humming.json'].read_text())
    (out_dir / 'model.safetensors.index.json').write_text(json.dumps(index))
    (out_dir / 'PLE_SOURCE.json').write_text(json.dumps({'intel_dir': str(intel), 'files': intel_files}))
    print(f'assembled {out_dir}: {len(needed)} klee shards from {len(parts)} parts, '
          f'PLE from {len(intel_files)} Intel files')
    return str(out_dir)
