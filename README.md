# LLM Benchmark Scripts

This repository contains the test scripts used to evaluate KV cache performance (`q8_0` vs `turbo3`) and needle-in-a-haystack recall on long-context models (like Gemma-4-31B) to determine if context is actually attended to at depths up to 250k tokens.

## Scripts Included

### 1. `needle_test.py`
The main needle-in-a-haystack benchmark script. 
- Reuses the Gutenberg corpus (The Adventures of Sherlock Holmes) to act as the haystack.
- Inserts a target passphrase at the **5%**, **50%**, and **95%** positions at your chosen context depths.
- Uses the `/v1/chat/completions` endpoint.
- Reads both `content` and `reasoning_content` (for reasoning models) and scores via exact substring matching. 

*Note: Budget `max_tokens=4096` or higher at long context, and leave headroom in your target depth so input + output fits within the server's `--ctx-size`.*

### 2. `needle_200k_triple.py`
A verification script used to check for deterministic reasoning traps at specific context depths (e.g., the 200k middle position pathology).
- Runs the exact same single query three times back-to-back at `temperature=0`.

## Clean Benchmark Command

If you're benchmarking `llama-server`, the default flags will result in distorted latency numbers and potential OOM-kills due to checkpointing overhead. To get honest numbers, use the following benchmark-safe flags:

```bash
--no-cache-prompt
--ctx-checkpoints 0
--checkpoint-every-n-tokens -1
--cache-ram 0
--parallel 1
```

**Full sample command for a clean benchmark run:**

```bash
./build/bin/llama-server -m gemma-4-31B-it-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 131072 --n-gpu-layers 99 \
  --cache-type-k <q8_0|turbo3> --cache-type-v <q8_0|turbo3> \
  --flash-attn on \
  --no-cache-prompt --ctx-checkpoints 0 \
  --checkpoint-every-n-tokens -1 --cache-ram 0 --parallel 1
```

*Always confirm the server is actually running at the requested context size by checking `curl http://host:port/props` for the `n_ctx` value, as launch flags are not guaranteed.*
