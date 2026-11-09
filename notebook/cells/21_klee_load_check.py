# ============================== KLEE HYBRID LOAD CHECK + SANITY ==============================
# 1. serve.log must show the tensor filter ran for every shard of MODEL_DIR with no missing tensor;
# 2. two greedy prompts with known answers (a wrong per-module precision can load and produce garbage).
import json as _json, urllib.request as _ur, time as _time
klee_hybrid_check_load(LOG)
ista_check_load(LOG)  # ISTA variant: every layer loaded
_url = f'http://127.0.0.1:{SERVED_MODEL_PORT}'
_t0 = _time.time()
while _time.time() - _t0 < 45 * 60:
    try:
        if _ur.urlopen(_url + '/health', timeout=5).status == 200: break
    except Exception: pass
    _time.sleep(5)
_qs = [('What is 17*23? Reply with the number only.', '391'),
       ('What is the capital of France? Reply with one word.', 'Paris')]
_ok = []
for _p, _want in _qs:
    _body = _json.dumps(dict(model=SERVED_MODEL_NAME, messages=[{'role': 'user', 'content': _p}],
                             max_tokens=4096, temperature=0.0)).encode()
    try:
        _m = _json.loads(_ur.urlopen(_ur.Request(_url + '/v1/chat/completions', _body,
                                                 {'Content-Type': 'application/json'}), timeout=600).read())['choices'][0]['message']
        _ans = (_m.get('content') or '').strip()
        _ok.append(_want.lower() in (_ans or (_m.get('reasoning_content') or '')[-300:]).lower())
        print(f'Q: {_p}\n  answer: {_ans[:200]!r}  ok={_ok[-1]}')
    except Exception as _e:
        _ok.append(False); print('sanity request failed:', repr(_e)[:300])
print('KLEE sanity:', f'{sum(_ok)}/{len(_ok)}')
if not any(_ok):
    raise RuntimeError('hybrid model answers no sanity prompt correctly; check serve.log / Humming rules')
