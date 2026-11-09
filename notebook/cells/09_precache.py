import os
import threading
import time

from concurrent.futures import as_completed, ThreadPoolExecutor

_PRECACHE_DONE = [0]  # bytes read by precache(), for the startup report
def precache(*paths, times=1, delay=0, threads=16) -> None:
    time.sleep(delay)
    print(f'Precaching: Started...')
    start_time = time.time()

    CHUNK_SIZE = 32 * 1024 * 1024  # 128 MB

    chunks = []  # list of (path, offset, length)
    total_size = 0
    file_count = 0

    def _add_file(i: int, path: str) -> None:
        nonlocal total_size, file_count
        size = os.path.getsize(path)
        total_size += size
        file_count += 1
        for offset in range(0, max(size, 1), CHUNK_SIZE):
            length = min(CHUNK_SIZE, size - offset)
            chunks.append((i, path, offset, length))

    for i, path in enumerate(paths):
        try:
            if os.path.isfile(path):
                _add_file(i, path)
                continue
            for root, _, files in os.walk(path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if os.path.isfile(file_path):
                        _add_file(i, file_path)
        except: pass

    chunks.sort()

    _local = threading.local()

    def _read_chunk(chunk: tuple) -> None:
        try:
            buf = _local.buf
        except AttributeError:
            _local.buf = buf = bytearray(CHUNK_SIZE)
        i, path, offset, _ = chunk
        try:
            with open(path, 'rb') as f:
                f.seek(offset)
                remaining = CHUNK_SIZE
                for i in range(1024):
                    n = f.readinto(buf)
                    if not n:
                        break
                    remaining -= n
                    _PRECACHE_DONE[0] += n
                    if remaining <= 0: break
        except: pass

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in range(times):
            list(pool.map(_read_chunk, chunks))

    elapsed = time.time() - start_time
    print(f'Precaching: loaded {file_count} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')

precache_wheels_thread = threading.Thread(target=precache, args=(WHEELHOUSE_DIR,))
precache_wheels_thread.start()
precache_wheels_thread.join()

# model precache, staged (2026-10-24), started with the server; two readers run side by side:
#  1. the non-PLE model files and the draft model, 32 threads (they go to the GPU);
#  2. the PLE files in name order, a shared pool of ARC3_PRECACHE_PLE_THREADS (32) threads over up to
#     ARC3_PRECACHE_PLE_FILES_IN_FLIGHT (3) files at once (v53: one file at a time with 16 threads read 350 MB/s
#     against 470 MB/s for the non-PLE files at 32). A PLE file starts once (a) the admitted PLE files the loader
#     has not finished stay within ARC3_PRECACHE_PLE_AHEAD_GB (40) and (b) MemAvailable, less our cached files the
#     loader still needs and the non-PLE bytes still to be read, stays above ARC3_PRECACHE_RESERVE_GB (16): the PLE
#     table is allocated as ~100 GB of pinned host memory at model init, so the page cache has ~75 GB in all;
#     the PLE reader starts as soon as the loader's progress frees room, not after the non-PLE reader ends
#     (v53: the PLE files waited until 146 s while dropping began at ~60 s), but not before the loader's first
#     progress line, i.e. after that allocation (2026-10-25: files read at launch were evicted by it);
#  3. meanwhile every model file the server has finished reading is dropped from the page cache
#     (posix_fadvise DONTNEED): non-PLE files sit on the GPU, PLE files in pinned memory. A file counts as
#     finished when serve.log's shard progress is at least its place in the order the server first opened the
#     files plus 2 (prefetch threads open files ahead of the loader) and no server process has it open or
#     mapped. Without /proc access nothing is dropped and step 2 stops at the window.
# ARC3_PRECACHE_DROP='0' keeps the cache; ARC3_PRECACHE_MODEL='0' skips the precache.
import json as _pj
import re as _pre
from pathlib import Path as _PP

_PRE_AHEAD = float(os.environ.get('ARC3_PRECACHE_PLE_AHEAD_GB', '40')) * 1e9
_PRE_RESERVE = float(os.environ.get('ARC3_PRECACHE_RESERVE_GB', '16')) * 1e9
_PRE_DROP = os.environ.get('ARC3_PRECACHE_DROP', '1') != '0'
_PRE_PLE_THREADS = int(os.environ.get('ARC3_PRECACHE_PLE_THREADS', '32'))
_PRE_PLE_FILES = int(os.environ.get('ARC3_PRECACHE_PLE_FILES_IN_FLIGHT', '3'))
_PRE_MARGIN = 2  # shards between a file's open-order place and the progress before it counts as finished
_PRE_LIMIT_S = 1500  # give up the loop after this long (server failed or never reports progress)
_ple_files = []
try:
    _ple_files = sorted(_pj.loads((_PP(MODEL_DIR) / 'PLE_SOURCE.json').read_text())['files'])
except Exception:
    pass
_ple_set = set(_ple_files)
_model_files = sorted(p.name for p in _PP(MODEL_DIR).iterdir() if p.name.endswith('.safetensors'))
_model_size = {n: os.path.getsize(os.path.join(MODEL_DIR, n)) for n in _model_files}
_model_real = {}  # real path -> name (the assembled directory holds symlinks)
for _n in _model_files:
    _model_real[os.path.realpath(os.path.join(MODEL_DIR, _n))] = _n
_precache_paths = [str(p) for p in sorted(_PP(MODEL_DIR).iterdir()) if p.name not in _ple_set] + [DRAFT_MODEL_DIR]


def _pre_tree_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs
               if os.path.isfile(os.path.join(r, f)))


_nonple_total = sum(_pre_tree_size(p) for p in _precache_paths)
_PRE = {'order': [], 'gone': set(), 'progress': 0, 'total': 0, 'ple_cached': {}, 'waits': 0.0,
        'dropped_gb': 0.0, 'dropped_nonple': 0, 'ple_read': 0, 'nonple_base': None, 'nonple_done': False,
        'ple_first_s': None, 'tracking': None}


def _pre_server_files():
    """Model files open or mapped by any process but this one (None without /proc access)."""
    used, seen_any = set(), False
    me = os.getpid()
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == me:
            continue
        d = f'/proc/{pid}'
        try:
            for fd in os.listdir(d + '/fd'):
                seen_any = True
                try:
                    target = os.readlink(f'{d}/fd/{fd}')
                except OSError:
                    continue
                if target in _model_real:
                    used.add(_model_real[target])
        except OSError:
            pass
        try:
            with open(d + '/maps') as f:
                for line in f:
                    parts = line.split(None, 5)
                    if len(parts) == 6 and parts[5].strip() in _model_real:
                        used.add(_model_real[parts[5].strip()])
        except OSError:
            pass
    return used if seen_any else None


def _pre_progress():
    """(shards loaded, shards in all) from serve.log's last progress line."""
    try:
        with open(LOG, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 20000))
            text = f.read().decode(errors='replace')
        got = _pre.findall(r'checkpoint shards:\s*\d+% Completed \| (\d+)/(\d+)', text)
        return (int(got[-1][0]), int(got[-1][1])) if got else (0, 0)
    except Exception:
        return 0, 0


def _pre_drop(name):
    path = os.path.join(MODEL_DIR, name)
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        _PRE['dropped_gb'] += _model_size[name] / 1e9
        if name not in _ple_set:
            _PRE['dropped_nonple'] += _model_size[name]
    except Exception:
        pass


def _pre_track(stop):
    """Follow the loader and drop finished files until all shards are loaded (or stop is set)."""
    start = time.time()
    while not stop.is_set() and time.time() - start < _PRE_LIMIT_S:
        used = _pre_server_files()
        _PRE['tracking'] = used is not None
        progress, total = _pre_progress()
        if total == len(_model_files):  # the main model's loader (the draft model has its own count)
            _PRE['progress'], _PRE['total'] = max(_PRE['progress'], progress), total
        if used and len(used) <= 24:  # more at once would be a header scan, not the loader's order
            for name in sorted(used):
                if name not in _PRE['order']:
                    _PRE['order'].append(name)
        if _PRE_DROP and used is not None:
            for place, name in enumerate(_PRE['order']):
                if name in _PRE['gone'] or name in used:
                    continue
                if place + 1 + _PRE_MARGIN <= _PRE['progress'] or _PRE['progress'] >= len(_model_files):
                    _pre_drop(name)
                    _PRE['gone'].add(name)
        if _PRE['total'] and _PRE['progress'] >= _PRE['total'] and len(_PRE['gone']) >= len(_PRE['order']):
            break
        time.sleep(1)


def _pre_room(size):
    """Whether a PLE file of `size` bytes may be read now (window and memory)."""
    ahead = sum(v for n, v in _PRE['ple_cached'].items() if n not in _PRE['gone'])
    nonple_read = max(0, _PRECACHE_DONE[0] - (_PRE['nonple_base'] or 0) - _PRE['ple_read'])
    nonple_cached = max(0, min(nonple_read, _nonple_total) - _PRE['dropped_nonple'])
    nonple_unread = 0 if _PRE['nonple_done'] else max(0, _nonple_total - nonple_read)
    try:
        avail = int(next(l for l in open('/proc/meminfo') if l.startswith('MemAvailable')).split()[1]) * 1024
    except Exception:
        avail = 0
    # MemAvailable counts our own cached files as reclaimable: leave out what the loader still needs
    return (ahead + size <= _PRE_AHEAD
            and avail - ahead - nonple_cached - nonple_unread - size >= _PRE_RESERVE)


def _pre_ple(start):
    """Read the PLE files ahead of the loader (shared pool, several files in flight)."""
    chunk = 32 * 1024 * 1024

    def part(path, offset):
        buf = bytearray(chunk)
        try:
            with open(path, 'rb') as f:
                f.seek(offset)
                n = f.readinto(buf) or 0
            _PRECACHE_DONE[0] += n
            _PRE['ple_read'] += n
        except Exception:
            pass

    def whole(path):
        """One PLE file start to end in 64 MB blocks on one handle (sequential read for the mount)."""
        buf = bytearray(2 * chunk)
        try:
            with open(path, 'rb', buffering=0) as f:
                try:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
                except Exception:
                    pass
                while True:
                    n = f.readinto(buf)
                    if not n:
                        break
                    _PRECACHE_DONE[0] += n
                    _PRE['ple_read'] += n
        except Exception:
            pass

    waiting = list(_ple_files)
    inflight = {}
    with ThreadPoolExecutor(max_workers=_PRE_PLE_THREADS) as pool:
        while (waiting or inflight) and time.time() - start < _PRE_LIMIT_S:
            for name in [n for n, futs in inflight.items() if all(f.done() for f in futs)]:
                del inflight[name]
            if _PRE['total'] and _PRE['progress'] >= _PRE['total']:
                break  # the load is over
            if waiting and waiting[0] in _PRE['gone']:
                waiting.pop(0)  # the loader is past it
                continue
            if not _PRE['total']:
                # not before the loader's first progress line: the ~100 GB pinned PLE table is allocated at
                # model init and evicts what was cached before it (v63 / kleea-low8 v19-v21: 8 PLE files read
                # at launch were evicted, still counted as cached, filled the window, and the loader read
                # them cold at 10-20 s per shard)
                time.sleep(0.5)
                continue
            if waiting and len(inflight) < _PRE_PLE_FILES and _pre_room(_model_size[waiting[0]]):
                name = waiting.pop(0)
                path = os.path.join(MODEL_DIR, name)
                _PRE['ple_cached'][name] = _model_size[name]
                if _PRE['ple_first_s'] is None:
                    _PRE['ple_first_s'] = time.time() - start
                inflight[name] = [pool.submit(whole, path)]
                continue
            if waiting:
                _PRE['waits'] += 0.5
            time.sleep(0.5)


def _precache_seq(*paths, files=12, block=64 * 1024 * 1024):
    """Read every file once, start to end in large blocks on one handle, `files` files at a time, in order."""
    print(f'Precaching: Started (sequential, {files} files at a time)...')
    start_time = time.time()
    names = []
    for path in paths:
        if os.path.isfile(path):
            names.append(path)
        else:
            for root, _, fs in os.walk(path):
                names += [os.path.join(root, f) for f in sorted(fs) if os.path.isfile(os.path.join(root, f))]
    total = sum(os.path.getsize(p) for p in names)

    def whole(path):
        buf = bytearray(block)
        try:
            with open(path, 'rb', buffering=0) as f:
                try:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
                except Exception:
                    pass
                while True:
                    n = f.readinto(buf)
                    if not n:
                        break
                    _PRECACHE_DONE[0] += n
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=files) as pool:
        list(pool.map(whole, names))
    print(f'Precaching: loaded {len(names)} files ({total / 1e9:.2f} GB) in {time.time() - start_time:.2f} seconds.\n')


def _pre_staged():
    stop = threading.Event()
    start = time.time()
    _PRE['nonple_base'] = _PRECACHE_DONE[0]
    tracker = threading.Thread(target=_pre_track, args=(stop,), daemon=True)
    tracker.start()
    ple = threading.Thread(target=_pre_ple, args=(start,), daemon=True)
    ple.start()
    _precache_seq(*_precache_paths, files=24)
    _PRE['nonple_done'] = True
    ple.join(_PRE_LIMIT_S)
    first = _PRE['ple_first_s']
    print(f"Precaching: PLE {len(_PRE['ple_cached'])}/{len(_ple_files)} files read ahead "
          f"({_PRE['ple_read'] / 1e9:.1f} GB, first at {first if first is None else round(first)} s, done at "
          f"{time.time() - start:.0f} s, waited {_PRE['waits']:.0f} s for room); tracking "
          f"{'on' if _PRE['tracking'] else 'OFF'}, dropped {len(_PRE['gone'])} files "
          f"({_PRE['dropped_gb']:.1f} GB) so far", flush=True)
    tracker.join(_PRE_LIMIT_S)  # it keeps dropping finished files until the load ends
    print(f"Precaching: load followed to {_PRE['progress']}/{_PRE['total']} shards; dropped {len(_PRE['gone'])} "
          f"files ({_PRE['dropped_gb']:.1f} GB) from the page cache", flush=True)


if os.environ.get('ARC3_PRECACHE_MODEL', '1') == '0':
    precache_model_thread = threading.Thread(target=lambda: print('Precaching: model directory skipped (ARC3_PRECACHE_MODEL=0)'))
else:
    precache_model_thread = threading.Thread(target=_pre_staged, daemon=True)