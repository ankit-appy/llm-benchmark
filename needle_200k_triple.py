#!/usr/bin/env python3
"""Rerun the 200k/0.50 needle case 3x on turbo3 to check reproducibility of the miss."""
from needle_test import tokenize, build_haystack, CORPUS, score, BASE_URL, MODEL, QUESTION
import time
import requests

RUNS = 3
DEPTH = 200_000
POS = 0.50
MAX_TOKENS = 4096


def query(haystack: str):
    user_content = (
        "Below is a long text. Somewhere in it, a secret passphrase is stated. "
        "Read carefully and then answer the question at the end.\n\n"
        f"<text>\n{haystack}\n</text>\n\n"
        f"{QUESTION}"
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=900)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    elapsed = time.time() - t0
    data = r.json()
    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    combined = (content + "\n\n[REASONING]\n" + reasoning).strip()
    return combined, {
        "elapsed_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": data.get("usage", {}).get("completion_tokens"),
        "content_len": len(content),
        "reasoning_len": len(reasoning),
    }


def main():
    corpus_tokens = tokenize(CORPUS.read_text())
    print(f"Corpus tokens: {len(corpus_tokens):,}")
    print(f"Testing depth={DEPTH}, pos={POS}, {RUNS} runs, max_tokens={MAX_TOKENS}")
    print()

    haystack, actual = build_haystack(corpus_tokens, DEPTH, POS)
    print(f"Haystack built: {actual:,} tokens")
    print()

    results = []
    for i in range(1, RUNS + 1):
        print(f"[Run {i}/{RUNS}] querying...", end=" ", flush=True)
        try:
            resp, meta = query(haystack)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append(("ERR", 0, str(e)))
            continue
        verdict = score(resp)
        print(f"{verdict} ({meta['elapsed_s']:.1f}s, finish={meta['finish_reason']}, completion_tokens={meta['completion_tokens']}, content_len={meta['content_len']}, reasoning_len={meta['reasoning_len']})")
        print(f"  first 250 chars: {resp[:250]!r}")
        print()
        results.append((verdict, meta["elapsed_s"], resp[:500]))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passes = sum(1 for r in results if r[0] == "PASS")
    print(f"{passes}/{RUNS} PASS")
    for i, (verdict, elapsed, _) in enumerate(results, 1):
        print(f"  Run {i}: {verdict} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
