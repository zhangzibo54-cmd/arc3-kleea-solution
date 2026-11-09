"""Concurrent needle test — correctness under batching, which a bs=1 evaluation cannot cover.

Everything that checked this patch's CORRECTNESS elsewhere ran at batch size 1: a smoke
test, a single long-context needle run, and a 34-task agentic exam whose harness iterates
sequentially (the server log for that window shows #running-req never above 1). Throughput
arms did reach 16 concurrent, but they only measured tok/s — they never looked at the output.

That gap is not hypothetical here. sglang issue #31641 was "NVFP4 KV cache batched decode
(trtllm_mha / SM120 XQA) produces corrupted output", with single-request traffic correct.
Its fix (#31667) predates the build this patch targets and its path was trtllm_mha, which
this configuration does not use — but the failure SHAPE, right at bs=1 and wrong under
batching, is exactly what bs=1 evidence cannot rule out.

This fires N requests simultaneously, each carrying its own distinct needles, and checks
both that every request retrieves its own values and that none retrieves ANOTHER
request's. Cross-contamination between concurrent sequences is the signature of a KV or
scale indexing bug under batching, and it is invisible to a single-stream test.

Usage:  python test_concurrent_needle.py [port] [n_requests] [filler_words]
"""
import json, sys, urllib.request, concurrent.futures, re

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
WORDS = int(sys.argv[3]) if len(sys.argv) > 3 else 9000

FILLER = ("The maintenance log records routine inspection of the auxiliary coolant loop. "
          "Pressure readings remained inside the nominal band and no operator action was required. ")

def build(i):
    # Per-request unique tokens, so a swap between concurrent sequences is detectable.
    code = f"ZR{1000+i*7}-{chr(65+i%26)}{chr(90-i%26)}"
    depth_word = f"{4100+i*3}"
    body = (FILLER.split() * (WORDS // 20))[:WORDS]
    body.insert(int(len(body)*0.12), f"\n\nThe bay-{i} access code is {code}.\n\n")
    body.insert(int(len(body)*0.83), f"\n\nThe verified torque setting for bay {i} is {depth_word} newton-metres.\n\n")
    prompt = ("Read this maintenance log, then answer using only facts stated in it.\n\n"
              "=== LOG START ===\n" + " ".join(body) + "\n=== LOG END ===\n\n"
              f"1. What is the bay-{i} access code?\n"
              f"2. What is the verified torque setting for bay {i}?\n\n"
              "Answer on two numbered lines, quoting the exact values.")
    return prompt, code, depth_word

def ask(i):
    prompt, code, torque = build(i)
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({"model": "flashnext", "chat_template_kwargs": {"enable_thinking": False},
                         "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 500, "temperature": 0.7, "top_p": 0.95}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
        txt = d["choices"][0]["message"].get("content") or ""
    except Exception as e:
        return i, code, torque, f"<ERROR {type(e).__name__}: {e}>", False, False
    return i, code, torque, txt, code.lower() in txt.lower(), torque in txt

with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
    results = list(ex.map(ask, range(N)))

all_codes = {c for _, c, _, _, _, _ in results}
ok = 0
print(f"{'req':>4} {'code':>12} {'torque':>7}  verdict")
for i, code, torque, txt, hit_c, hit_t in sorted(results):
    # A concurrent KV/indexing bug shows up as ANOTHER request's needle in this answer.
    foreign = sorted(c for c in all_codes if c != code and c.lower() in txt.lower())
    tag = "OK" if (hit_c and hit_t) else ("PARTIAL" if (hit_c or hit_t) else "MISS")
    if foreign:
        tag += f"  ⚠ CROSS-CONTAMINATION: {foreign}"
    if txt.startswith("<ERROR"):
        tag = txt[:80]
    ok += (hit_c and hit_t and not foreign)
    print(f"{i:>4} {code:>12} {torque:>7}  {tag}")
print(f"\nclean: {ok}/{len(results)} requests retrieved BOTH of their own needles with no foreign needle")
