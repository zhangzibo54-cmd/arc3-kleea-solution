# Plain-file sources (formerly base64 blobs): the dataset zhangzibo54/arc3-kleea-solution-code holds
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
