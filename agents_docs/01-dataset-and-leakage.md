# Dataset & Leakage

## Shape (full release)

| Split | Images | Notes |
|---|---|---|
| `train/train/` | 69,352 | labeled (`train_labels.csv`) |
| `public_test/` | 7,821 | the **public leaderboard** set |
| (private) | 134,997 | hidden; images not provided yet |
| `sample_submission.csv` | 142,818 ids | = 7,821 public + 134,997 private |

- **5 document types only:** EGYPT/DL (15,867), GUINEA/DL (13,389), BENIN/DL (13,369),
  MOZAMBIQUE/DL (13,365), MAURITIUS/ID (13,362).
- **Label balance:** 40,005 genuine / 29,347 fraud (≈42% fraud). Per-type fraud rate ≈0.40,
  except EGYPT/DL ≈0.50.
- **`is_digital`:** 69,332 True / **20 False**. The training set is effectively all-digital.
  This is the single most important data fact for generalization (see strategy doc).

## Near-duplicate detection (`freuid/dedup.py`, `scripts/05`)

Why it matters: genuine↔tampered twins and shared blank templates must stay in the same
CV partition or validation leaks. Also reveals train↔test leakage.

**Calibration (this was non-obvious):**
- **64-bit pHash is useless here** — these are same-template ID docs, so *distinct* documents
  collide even at Hamming distance 0 (78 distinct docs shared one hash in a 3k sample).
- **256-bit pHash (`hash_size=16`) separates cleanly:** 0 components up to distance 6, then
  small isolated pairs appear. Chosen operating point: **threshold = 10**.
- Method: parallel pHash (draft-decoded JPEGs, ~1,700 img/s on 16 cores), **LSH banding**
  (pigeonhole-complete for Hamming ≤ threshold), union-find components. **Fails closed**
  (`incomplete` flag) if any bucket exceeds the hard cap — never silently drops candidates.

**Findings on full train (threshold 10, 256-bit):**
- 10,240 near-duplicate pairs → **1,124 connected components**, largest = 1,585.
- **~35% of near-dup pairs have conflicting labels** (genuine↔tampered twins) — the exact
  leakage hazard. All near-dups are **within-type** (cross-type = 0).
- **~1,505 train↔public_test near-duplicate pairs** (~19% of public test has a train twin).

## Consequences

1. Splits **must** group by near-dup component (done — see validation doc).
2. The **public LB is leakage-inflated**: ~19% of public-test images have a near-identical
   training image, so a model that memorizes them scores well there. Don't over-trust it.
3. The full `artifacts/duplicates.json` (~2 MB, all pairs) is **git-ignored** (regenerable via
   `scripts/05`); `artifacts/duplicates_summary.json` keeps the counts + examples.

## Cheap-leak probes — ALL NEGATIVE (2026-07-01)

Direct empirical checks (on the box, train vs public_test) that the public
leaderboard is **not** trivially winnable. Every cheap explanation for the
top-of-LB 0.0005 scores is dead:

- **Container / metadata:** genuine and fraud share the *same* JPEG quantization
  table (md5 `36670c99dd`), identical dimensions, same `APP0+APP1` marker shell,
  stripped EXIF, baseline (non-progressive). `public_test` is byte-format
  identical to `train`. → no container leak, and public_test is **not** "captured"
  at the container level.
- **Retrieval:** nearest-train-neighbor (256-bit pHash) label agreement ≈ **0.70**
  — the genuine↔tampered twins are pHash-near-identical but oppositely labeled,
  so "copy the neighbor's label" is barely above base rate. Only ~**7%** of
  public_test has a train twin within Hamming 10 (median NN distance 19). kNN
  label-copy is useless.
- **id ordering:** `corr(int(id[:8],16), label) = +0.003` — ids are not generated
  in fraud/genuine blocks.
- **JPEG marker structure:** uniform (`APP0+APP1`, baseline, no comment segment)
  across genuine / fraud / public.

**Conclusion:** the organizers scrubbed the trivial leaks. The 0.0005 top scores
are a *real* pixel-level manipulation artifact extracted well, not a cheat. See
`agents_docs/04` for the (rejected) OCR-semantic follow-up and the forgery-method
teardown.
