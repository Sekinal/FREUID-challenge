#!/usr/bin/env python
"""Project embeddings to 2-D and probe structure: clusters, separability, drift.

- 2-D projection: UMAP (cosine) with a PCA fallback for tiny n.
- Unsupervised clusters: KMeans (k chosen by silhouette) and HDBSCAN.
- Label separability: leave-one-out kNN accuracy + silhouette of the label
  grouping in the embedding space — a quick read on how visually separable
  fraud is before any modelling.
- Colour the projection by label / is_digital / country / doc_type to expose
  whether the encoder organises the data by stratum (a hint at distribution shift).

Writes ``artifacts/cluster_stats.json`` and projection figures.

    uv run scripts/07_clustering_umap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io, viz  # noqa: E402


def project_2d(x: np.ndarray) -> tuple[np.ndarray, str]:
    n = len(x)
    if n >= 4:
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, max(2, n - 1)),
                min_dist=0.1,
                metric="cosine",
                init="random",
                random_state=42,
            )
            return reducer.fit_transform(x), "UMAP"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] UMAP failed ({exc}); falling back to PCA")
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=42).fit_transform(x), "PCA"


def loo_knn_accuracy(x: np.ndarray, y: np.ndarray, k: int = 3) -> float | None:
    from sklearn.model_selection import LeaveOneOut
    from sklearn.neighbors import KNeighborsClassifier

    if len(np.unique(y)) < 2 or len(x) <= k:
        return None
    preds = []
    loo = LeaveOneOut()
    for tr, te in loo.split(x):
        knn = KNeighborsClassifier(n_neighbors=min(k, len(tr)), metric="cosine")
        knn.fit(x[tr], y[tr])
        preds.append(knn.predict(x[te])[0])
    return float((np.array(preds) == y).mean())


def best_kmeans(x: np.ndarray) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(x)
    best = {"k": None, "silhouette": None, "labels": None}
    for k in range(2, min(8, n)):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(x)
        if len(np.unique(km.labels_)) < 2:
            continue
        sil = silhouette_score(x, km.labels_, metric="cosine")
        if best["silhouette"] is None or sil > best["silhouette"]:
            best = {"k": k, "silhouette": float(sil), "labels": km.labels_.tolist()}
    return best


def scatter(xy, series, name, title):
    fig, ax = viz.plt.subplots(figsize=(6, 5))
    cats = pd.Series(series).astype("object").fillna("NA")
    for val, color in zip(sorted(cats.unique(), key=str),
                          viz.sns.color_palette("husl", cats.nunique())):
        m = (cats == val).values
        c = viz.LABEL_PALETTE.get(val, color)
        ax.scatter(xy[m, 0], xy[m, 1], s=55, alpha=.85, label=str(val), color=c)
    ax.set(title=title, xlabel="dim 1", ylabel="dim 2")
    ax.legend(title=name, fontsize=8, loc="best")
    viz.save_fig(fig, f"embed_proj_{name}.png")


def main() -> None:
    viz.set_style()
    emb_path = config.EMB_DIR / "embeddings.npy"
    if not emb_path.exists():
        sys.exit("No embeddings. Run scripts/06_embeddings_gpu.py first.")
    x = np.load(emb_path)
    ids = json.loads((config.EMB_DIR / "ids.json").read_text())

    labels = io.load_labels().set_index(config.ID_COL)
    df = pd.DataFrame({"id": ids})
    for col in (config.LABEL_COL, config.IS_DIGITAL_COL, "country", "doc_type"):
        if col in labels.columns:
            df[col] = df["id"].map(labels[col])
    df["label_name"] = df[config.LABEL_COL].map(config.LABEL_NAMES)

    xy, method = project_2d(x)

    # separability / clustering
    from sklearn.metrics import silhouette_score

    y = df[config.LABEL_COL].to_numpy()
    stats = {
        "n": int(len(x)),
        "dim": int(x.shape[1]),
        "projection": method,
        "loo_knn_label_accuracy": loo_knn_accuracy(x, y),
        "label_silhouette": (
            float(silhouette_score(x, y, metric="cosine"))
            if len(np.unique(y)) > 1 else None
        ),
        "kmeans": best_kmeans(x),
    }
    try:
        import hdbscan

        if len(x) >= 5:
            cl = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean").fit(x)
            stats["hdbscan"] = {
                "n_clusters": int(len(set(cl.labels_)) - (1 if -1 in cl.labels_ else 0)),
                "n_noise": int((cl.labels_ == -1).sum()),
                "labels": cl.labels_.tolist(),
            }
    except Exception as exc:  # noqa: BLE001
        stats["hdbscan_error"] = str(exc)

    out = io.save_json("cluster_stats.json", stats)

    # projection plots
    color_col = "label_name" if "label_name" in df.columns else config.LABEL_COL
    scatter(xy, df[color_col], "label", f"{method}: colored by label")
    for col, nm in [(config.IS_DIGITAL_COL, "is_digital"),
                    ("country", "country"), ("doc_type", "doc_type")]:
        if col in df.columns:
            scatter(xy, df[col], nm, f"{method}: colored by {nm}")

    print(f"projection={method}  loo_knn_acc={stats['loo_knn_label_accuracy']}  "
          f"label_silhouette={stats['label_silhouette']}  kmeans_k={stats['kmeans']['k']}")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
