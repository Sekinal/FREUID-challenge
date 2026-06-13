# FREUID 2026 — Agent Findings & Working Notes

Knowledge base for anyone (human or agent) picking up this repo. Written from the
full-dataset work on the GPU box (`/root/freuid`, A100-80GB).

## TL;DR

- **The competition is won on out-of-distribution generalization, not in-distribution accuracy.**
  The host confirmed the private test contains **2 document types unseen in train/public**,
  plus an emphasis on **captured / print-and-capture** images. In-distribution scores
  (and the public leaderboard) are misleading.
- Our training data is **99.97% digital** (`is_digital=True` for all but 20 of 69,352).
  The private set emphasizes *captured* images → a large **digital→captured domain shift**
  stacked on top of the unseen-type shift.
- A vanilla EfficientNet-b2 gets **FREUID ≈ 0.0001 in-distribution** (near-perfect) but only
  **0.291 on the public LB** (same 5 types, held-out, even *with* ~19% leaked twins) — a ~0.29
  gap that is the whole game (the digital→captured shift, confirmed by a real submission).
- Validation is now **leakage-safe** (group-stratified by near-duplicate component) and the
  FREUID metric is **vectorized (~1130× faster, bit-identical)**.

## Key numbers

| Thing | Value |
|---|---|
| Train images | 69,352 (5 types, ~0.42 fraud) |
| Public test | 7,821 (= public LB set) |
| Private test | 134,997 (hidden; **2 unseen types**, captured emphasis) |
| `is_digital=True` | 69,332 / 69,352 (99.97%) |
| Near-dup pairs (256-bit pHash, thr 10) | 10,240 / 1,124 components |
| Train↔public_test near-dup leaks | ~1,505 (~19% of public test) |
| Split groups | 64,135 (all partitions fraud-rate 0.423) |
| Baseline in-dist val/test FREUID | ~0.0001 |
| Baseline **public LB** (same types, held-out) | **0.291** |
| → in-dist vs public gap | ~0.29 (the digital→captured shift, made real) |

## Documents

- [`01-dataset-and-leakage.md`](01-dataset-and-leakage.md) — data shape, near-dup detection, leakage.
- [`02-validation.md`](02-validation.md) — leakage-safe splits, LOTO, metric vectorization.
- [`03-competition-and-strategy.md`](03-competition-and-strategy.md) — host's OOD design, public/private, what to build.
- [`04-results-and-open-questions.md`](04-results-and-open-questions.md) — runs, scores, TODO.

## Environment

- Box: `ssh root@216.81.248.172 -p 40299`, repo + full data at `/root/freuid`.
- Run with `.venv/bin/python` (no `uv` on the box). GPU: A100-80GB.
- Repo: `github.com/Sekinal/FREUID-challenge` (push from local with the user's creds).
