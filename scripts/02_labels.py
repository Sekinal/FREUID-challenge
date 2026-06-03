#!/usr/bin/env python
"""Analyse the label table: class balance and the fraud signal across strata.

Produces distributions for ``label`` (genuine/fraud), ``is_digital``, country,
and document type, plus cross-tabs of fraud-rate by each stratum. Writes
``artifacts/label_stats.json`` and several figures.

    uv run scripts/02_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io, viz  # noqa: E402


def value_counts(df: pd.DataFrame, col: str) -> dict:
    if col not in df.columns:
        return {}
    vc = df[col].value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def fraud_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, dropna=False)[config.LABEL_COL].agg(["mean", "count"])
    return g.rename(columns={"mean": "fraud_rate"}).reset_index()


def main() -> None:
    viz.set_style()
    df = io.load_labels()

    n = len(df)
    pos = int((df[config.LABEL_COL] == config.POSITIVE_LABEL).sum())
    stats = {
        "n_rows": n,
        "n_fraud": pos,
        "n_genuine": n - pos,
        "fraud_rate": pos / n if n else None,
        "label_counts": value_counts(df, config.LABEL_COL),
        "is_digital_counts": value_counts(df, config.IS_DIGITAL_COL),
        "country_counts": value_counts(df, "country"),
        "doc_type_counts": value_counts(df, "doc_type"),
        "type_counts": value_counts(df, config.TYPE_COL),
        "ids_unique": int(df[config.ID_COL].is_unique),
        "missing_images": int((~df["path_exists"]).sum()) if "path_exists" in df else None,
    }

    # fraud-rate cross-tabs
    for col in ("country", "doc_type", config.IS_DIGITAL_COL):
        if col in df.columns:
            stats[f"fraud_rate_by_{col}"] = fraud_rate_by(df, col).to_dict("records")

    out = io.save_json("label_stats.json", stats)

    # --- figure: label balance ---
    fig, ax = viz.plt.subplots(figsize=(4, 3.5))
    counts = df[config.LABEL_COL].map(config.LABEL_NAMES).value_counts()
    ax.bar(counts.index, counts.values,
           color=[viz.LABEL_PALETTE[k] for k in counts.index])
    viz.annotate_bars(ax)
    ax.set(title=f"Label balance (fraud rate {stats['fraud_rate']:.0%})", ylabel="count")
    viz.save_fig(fig, "labels_balance.png")

    # --- figure: is_digital split by label ---
    if config.IS_DIGITAL_COL in df.columns:
        ct = pd.crosstab(df[config.IS_DIGITAL_COL],
                         df[config.LABEL_COL].map(config.LABEL_NAMES))
        fig, ax = viz.plt.subplots(figsize=(5, 3.5))
        ct.plot(kind="bar", stacked=True, ax=ax,
                color=[viz.LABEL_PALETTE[c] for c in ct.columns])
        ax.set(title="is_digital vs label", xlabel="is_digital", ylabel="count")
        ax.legend(title="label")
        viz.save_fig(fig, "labels_is_digital.png")

    # --- figure: counts by country, colored by fraud share ---
    if "country" in df.columns:
        ct = pd.crosstab(df["country"], df[config.LABEL_COL].map(config.LABEL_NAMES))
        ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
        fig, ax = viz.plt.subplots(figsize=(7, 3.8))
        ct.plot(kind="bar", stacked=True, ax=ax,
                color=[viz.LABEL_PALETTE[c] for c in ct.columns])
        ax.set(title="Documents per country (by label)", xlabel="country", ylabel="count")
        ax.legend(title="label")
        viz.save_fig(fig, "labels_by_country.png")

    # --- figure: counts by doc_type ---
    if "doc_type" in df.columns:
        ct = pd.crosstab(df["doc_type"], df[config.LABEL_COL].map(config.LABEL_NAMES))
        fig, ax = viz.plt.subplots(figsize=(4.5, 3.5))
        ct.plot(kind="bar", stacked=True, ax=ax,
                color=[viz.LABEL_PALETTE[c] for c in ct.columns])
        ax.set(title="Documents per type", xlabel="doc_type", ylabel="count")
        ax.legend(title="label")
        viz.save_fig(fig, "labels_by_doctype.png")

    print(f"rows={n}  fraud={pos}  genuine={n - pos}  fraud_rate={stats['fraud_rate']:.2%}")
    print(f"countries: {stats['country_counts']}")
    print(f"doc_types: {stats['doc_type_counts']}")
    print(f"is_digital: {stats['is_digital_counts']}")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
