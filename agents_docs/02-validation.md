# Validation Design

## The problem with naive validation here

- Only **5 document types** → holding out whole types gives 5 brittle, high-variance configs.
- **Near-duplicate twins** leak across folds unless pinned together.
- The original pipeline read **13-image-sample** duplicate pairs whose ids don't exist in the
  69k set → leakage protection was effectively **off** on the real data.

## What we built (`freuid/splits.py`, `scripts/09`)

- **Group key = near-duplicate connected component** (singletons = own id). 64,135 groups.
- **`StratifiedGroupKFold`** assigns *whole groups* to train/val/test and CV folds, balancing
  the `type × label` distribution.
- **Hard `assert_no_leakage`**: fails if any group or id crosses a partition; CV folds checked
  for group disjointness. (build_splits raises on violation.)
- **Leave-one-type-out (LOTO)** is also emitted — group-aware (a dup component touching the
  held type is removed from train, never split across) — as the OOD proxy.

**Resulting split (full data):** train 49,537 / val 9,908 / test 9,907, **all at fraud rate
0.423**, CV fold fraud-rate **std = 0.0001**, zero duplicate leakage.

### Two regimes, deliberately kept separate
- **In-distribution** = StratifiedGroupKFold / the val,test partitions (same 5 types in all).
  Mirrors the *public* LB. **Optimistic** for the final eval.
- **Out-of-distribution** = LOTO (train on 4 types, score the 5th). Mirrors the *private*
  test's unseen-type axis. **This is the number to optimize** (see strategy doc).
  Caveat: LOTO still can't capture the digital→captured shift (almost no captured train data).

## Metric (`freuid/metrics.py`)

- `freuid_score` = harmonic combo of AuDET (area under DET curve) and APCER@1%BPCER; lower better.
- **Best-checkpoint selection uses AuDET, not FREUID** — FREUID folds in a single
  near-threshold operating point that can jump to 1.0 between epochs (observed). AuDET is smooth.
- **DET curve vectorized** via `searchsorted`: **bit-identical** to the old per-threshold loop
  (verified exact on 80 edge cases) and **~1130× faster** (1371 ms → 1.21 ms per call, n=9908).
  `freuid_score` builds the curve once. Validation smoke: minutes → 1.5 s.

## Validation tooling (`freuid/validation.py`)

- `ValidationPipeline`: holdout scoring, group CV, LOTO, stratified **bootstrap CIs**,
  per-type breakdown, and a leakage-failing `sanity_check`.

## Tests (`tests/`)

20 tests, all green: metrics (incl. exact-equivalence lock + bootstrap), dedup (LSH
completeness vs brute force, union-find), splits (no group/id leakage, fraud-rate balance,
deterministic, LOTO cross-type safety). Run: `.venv/bin/python tests/test_*.py`.
