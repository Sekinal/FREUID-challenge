# Draft — reproducibility reply (post ONE reply on pinned thread by July 15, 23:59 AoE)

Fill the official REPLY_TEMPLATE.txt from
https://freuid2026.microblink.com/reproducibility.html with:

- **Team name**: <team name as on Kaggle leaderboard>
- **Kaggle usernames**: EliasTSJ, Sekinal
- **Repository URL**: https://github.com/Sekinal/FREUID-challenge
- **Frozen commit SHA**: <git rev-parse origin/main on July 13, after final CSVs>
- **Technical report**: report/technical_report.pdf (in repo at frozen commit)
- **Final Kaggle submission(s)** (per host guidance, thread 723991 — list both):
  - Pick 1 — FINAL_public_bm1024.csv, <timestamp shown on Kaggle> — reproduced via:
    `docker run --rm --network none --gpus all --shm-size 2g -e FREUID_MODEL=public -v <test>:/data:ro -v <out>:/submissions freuid2026-eliastsj`
    — output sha256: `<from day13 script>`
  - Pick 2 — FINAL_robust_slot2v3.csv, <timestamp> — reproduced via:
    same command with `-e FREUID_MODEL=robust` (default)
    — output sha256: `<from day13 script>`
- **Confirmation**: the repository at the frozen commit, with the released
  weights (`final-models`) embedded per docker/README.md, reproduces both
  selected final submissions under `--network none`.
- **Signature**: <team captain>, <date>
