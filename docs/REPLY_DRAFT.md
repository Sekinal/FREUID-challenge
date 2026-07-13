# Draft — reproducibility reply (post ONE reply on pinned thread by July 15, 23:59 AoE)

Fill the official REPLY_TEMPLATE.txt from
https://freuid2026.microblink.com/reproducibility.html with:

- **Team name**: <team name as on Kaggle leaderboard>
- **Kaggle usernames**: EliasTSJ, Sekinal
- **Repository URL**: https://github.com/Sekinal/FREUID-challenge
- **Frozen commit SHA**: <git rev-parse origin/main on July 13, after final CSVs>
- **Technical report**: report/technical_report.pdf (in repo at frozen commit)
- **Final Kaggle submission(s)** (per host guidance, thread 723991 — list both):
  - Pick 1 — FINAL_public_bm1024.csv, 2026-07-13 18:50:38 UTC — reproduced via:
    `docker run --rm --network none --gpus all --shm-size 2g -e FREUID_MODEL=public -v <test>:/data:ro -v <out>:/submissions freuid2026-eliastsj`
    — submitted-file sha256: `5f83120d681bcfa555b3c8604aa23fe77ef4c776c4968ef94966a3207536ddb6`
  - Pick 2 — FINAL_robust_slot2v3.csv, 2026-07-13 18:50:40 UTC — reproduced via:
    same command with `-e FREUID_MODEL=robust` (default)
    — submitted-file sha256: `05d694281f17f8cfc9383534540cafbfbbe802d40bbe645c80727a9544abd28a`
  - Note: re-runs reproduce scores numerically, not bitwise (bf16 + cuDNN
    autotune; measured per-image drift ≤ ~1e-3, metric-neutral) — see
    docker/README.md "Reproducibility tolerance".
- **Confirmation**: the repository at the frozen commit, with the released
  weights (`final-models`) embedded per docker/README.md, reproduces both
  selected final submissions under `--network none`.
- **Signature**: <team captain>, <date>
