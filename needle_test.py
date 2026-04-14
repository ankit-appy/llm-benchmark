#!/usr/bin/env python3
"""
Needle-in-haystack recall test for gemma-4-31b on ik_llama.cpp server.

Protocol:
- Load a long corpus (reuses benchy's cached Sherlock Holmes text)
- At each target depth, insert a secret passphrase at 3 positions (5%, 50%, 95%)
- Ask the model to recall it
- Score: PASS (exact passphrase found), PARTIAL (some tokens), FAIL

Uses the server's own /tokenize endpoint for accurate token accounting.
"""

import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://192.168.68.62:8080"
MODEL = "gemma-4-31b"
CORPUS = Path.home() / ".cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt"

NEEDLE = "The secret passphrase is `purple-elephant-7294`. Remember this passphrase exactly."
PASSPHRASE = "purple-elephant-7294"

QUESTION = "What is the secret passphrase mentioned somewhere in the text above? Reply with just the passphrase, nothing else."

DEPTHS = [32_000, 65_000, 125_000]
POSITIONS = [0.05, 0.50, 0.95]  # needle insertion point as fraction of depth

TIMEOUT = 600  # seconds per request


def tokenize(text: str) -> list[int]:
    r = requests.post(f"{BASE_URL}/tokenize", json={"content": text}, timeout=60)
    r.raise_for_status()
    return r.json()["tokens"]


def detokenize(tokens: list[int]) -> str:
    r = requests.post(f"{BASE_URL}/detokenize", json={"tokens": tokens}, timeout=60)
    r.raise_for_status()
    return r.json()["content"]


def build_haystack(corpus_tokens: list[int], target_depth: int, needle_pos_frac: float) -> tuple[str, int]:
    needle_tokens = tokenize(NEEDLE)
    question_tokens = tokenize(QUESTION)
    # Reserve tokens for needle + question + some slack for chat template
    slack = 200
    body_budget = target_depth - len(needle_tokens) - len(question_tokens) - slack
    if body_budget <= 0:
        raise ValueError(f"target_depth {target_depth} too small")
    if body_budget > len(corpus_tokens):
        # Tile the corpus if needed
        reps = (body_budget // len(corpus_tokens)) + 1
        body_toks = (corpus_tokens * reps)[:body_budget]
    else:
        body_toks = corpus_tokens[:body_budget]

    insert_at = int(len(body_toks) * needle_pos_frac)
    haystack_toks = body_toks[:insert_at] + needle_tokens + body_toks[insert_at:]
    haystack_text = detokenize(haystack_toks)
    return haystack_text, len(haystack_toks)


def query(haystack: str) -> tuple[str, dict]:
    user_content = (
        "Below is a long text. Somewhere in it, a secret passphrase is stated. "
        "Read carefully and then answer the question at the end.\n\n"
        f"<text>\n{haystack}\n</text>\n\n"
        f"{QUESTION}"
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT
    )
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    elapsed = time.time() - t0
    data = r.json()
    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    combined = (content + "\n\n[REASONING]\n" + reasoning).strip()
    usage = data.get("usage", {})
    return combined, {
        "elapsed_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "content_len": len(content),
        "reasoning_len": len(reasoning),
    }


def score(response: str) -> str:
    r = response.lower()
    if PASSPHRASE.lower() in r:
        return "PASS"
    # partial: any of the three words
    parts = PASSPHRASE.lower().split("-")
    hits = sum(1 for p in parts if p in r)
    if hits >= 2:
        return "PARTIAL"
    if hits == 1:
        return "WEAK"
    return "FAIL"


def main():
    if not CORPUS.exists():
        print(f"ERROR: corpus not found at {CORPUS}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading corpus from {CORPUS}")
    corpus_text = CORPUS.read_text()
    print(f"Tokenizing corpus via server /tokenize...")
    corpus_tokens = tokenize(corpus_text)
    print(f"Corpus tokens: {len(corpus_tokens):,}")
    print(f"Needle: {NEEDLE!r}")
    print(f"Passphrase: {PASSPHRASE}")
    print()

    results = []
    total = len(DEPTHS) * len(POSITIONS)
    idx = 0
    for depth in DEPTHS:
        for pos in POSITIONS:
            idx += 1
            label = f"[{idx}/{total}] depth={depth:>6} pos={pos:>4.2f}"
            print(f"{label}  building...", end=" ", flush=True)
            try:
                haystack, actual_tokens = build_haystack(corpus_tokens, depth, pos)
            except Exception as e:
                print(f"BUILD ERROR: {e}")
                results.append((depth, pos, None, "BUILD_ERR", 0, str(e)))
                continue
            print(f"({actual_tokens:,} tok) querying...", end=" ", flush=True)
            try:
                response, meta = query(haystack)
            except Exception as e:
                print(f"QUERY ERROR: {e}")
                results.append((depth, pos, actual_tokens, "QUERY_ERR", 0, str(e)))
                continue
            verdict = score(response)
            print(f"{verdict} ({meta['elapsed_s']:.1f}s)  resp={response.strip()[:80]!r}")
            results.append((depth, pos, actual_tokens, verdict, meta["elapsed_s"], response.strip()[:200]))

    print()
    print("=" * 80)
    print("NEEDLE-IN-HAYSTACK RESULTS")
    print("=" * 80)
    print(f"{'depth':>8} {'pos':>6} {'actual_tok':>12} {'verdict':>10} {'elapsed':>10}  response")
    print("-" * 80)
    for depth, pos, actual, verdict, elapsed, resp in results:
        actual_s = f"{actual:,}" if actual else "-"
        print(f"{depth:>8} {pos:>6.2f} {actual_s:>12} {verdict:>10} {elapsed:>9.1f}s  {resp!r}")

    # Summary grid
    print()
    print("Summary grid (rows=depth, cols=position):")
    print(f"{'depth':>8}  " + "  ".join(f"{p:>8.2f}" for p in POSITIONS))
    for depth in DEPTHS:
        row = [r for r in results if r[0] == depth]
        cells = []
        for pos in POSITIONS:
            match = next((r for r in row if r[1] == pos), None)
            cells.append(f"{match[3]:>8}" if match else f"{'--':>8}")
        print(f"{depth:>8}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
