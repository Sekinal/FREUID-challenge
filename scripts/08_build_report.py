#!/usr/bin/env python
"""Assemble all artifacts + figures into a Typst report and compile to PDF.

Reads ``artifacts/*.json``, references figures in ``figures/``, writes
``report/report.typ`` and runs ``typst compile`` -> ``report/report.pdf``.
Sections are emitted only when their backing artifact exists, so the report
degrades gracefully on partial runs.

    uv run scripts/08_build_report.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io  # noqa: E402

FIG = "../figures"


def esc(s) -> str:
    return str(s).replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def fig(name: str, caption: str, width: str = "85%") -> str:
    if not (config.FIGURES_DIR / name).exists():
        return ""
    return (f'#figure(image("{FIG}/{name}", width: {width}), '
            f'caption: [{esc(caption)}])\n\n')


def table(headers, rows, columns=None) -> str:
    columns = columns or len(headers)
    cells = ["*" + esc(h) + "*" for h in headers]
    for r in rows:
        cells += [esc(c) for c in r]
    body = ", ".join("[" + c + "]" for c in cells)
    return f"#table(columns: {columns}, {body})\n\n"


def kv_table(d: dict, key_hdr="metric", val_hdr="value") -> str:
    return table([key_hdr, val_hdr], [[k, v] for k, v in d.items()])


def main() -> None:
    inv = io.load_json("inventory.json")
    lab = io.load_json("label_stats.json")
    img = io.load_json("image_stats_summary.json")
    dup = io.load_json("duplicates.json")
    clu = io.load_json("cluster_stats.json")
    emb = io.load_json("embeddings_meta.json")

    p: list[str] = []
    p.append('#set page(paper: "a4", margin: 2cm, numbering: "1")\n')
    p.append('#set text(size: 10pt)\n')
    p.append('#set heading(numbering: "1.1")\n')
    p.append('#show heading.where(level: 1): it => [#pagebreak(weak: true) #it]\n')
    p.append("#align(center)[#text(20pt, weight: \"bold\")[FREUID Challenge 2026]\\ "
             "#text(13pt)[Identity-Document Fraud Detection — Exploratory Data Analysis]\\ "
             "#text(9pt, style: \"italic\")[IJCAI-ECAI 2026 · generated from the public data sample]]\n")
    p.append("#v(0.5cm)\n#outline(indent: auto)\n")

    # ---- Overview ----
    p.append("= Overview\n")
    p.append(
        "The FREUID challenge targets *next-generation identity-document fraud "
        "detection* across physical manipulations, GenAI-driven multimodal edits, "
        "and print-and-capture forgeries. The task is a *binary classification* "
        "(`label` 0 = genuine, 1 = fraud) scored as a probability per image id. "
        "This report profiles the publicly released sample to ground later modelling.\n\n")

    # ---- Inventory ----
    if inv:
        p.append("= Data inventory\n")
        p.append(f"Extracted footprint: *{inv['total_files']} files* "
                 f"({esc(inv['total_human'])}), *{inv['n_images_indexed']} images* indexed.\n\n")
        p.append(table(["extension", "count", "size"],
                       [[e, m["count"], m["human"]] for e, m in inv["by_extension"].items()]))
        p.append(fig("inventory_file_types.png", "File count by extension.", "60%"))
        for name, meta in inv.get("csvs", {}).items():
            p.append(f"== `{esc(name)}`\n")
            p.append(f"{meta['rows']} rows · columns: "
                     + ", ".join(f"`{esc(c)}`" for c in meta["columns"]) + "\n\n")

    # ---- Labels ----
    if lab:
        p.append("= Labels & class balance\n")
        p.append(kv_table({
            "rows": lab["n_rows"], "fraud": lab["n_fraud"], "genuine": lab["n_genuine"],
            "fraud rate": f"{lab['fraud_rate']:.1%}" if lab["fraud_rate"] is not None else "n/a",
            "unique ids": bool(lab["ids_unique"]),
            "missing images": lab.get("missing_images"),
        }))
        p.append(fig("labels_balance.png", "Genuine vs fraud counts.", "50%"))
        p.append(fig("labels_is_digital.png", "Digital vs physical capture, split by label.", "55%"))
        p.append(fig("labels_by_country.png", "Documents per country, stacked by label."))
        p.append(fig("labels_by_doctype.png", "Documents per document type.", "55%"))
        p.append("The sample spans multiple countries and document types (DL/ID); "
                 "`is_digital` flags born-digital vs print-and-capture acquisition — "
                 "an axis the threat model explicitly cares about.\n\n")

    # ---- Image properties ----
    if img:
        p.append("= Image properties & integrity\n")
        p.append(f"*{img['n_images']} images*, *{img['n_corrupt']} corrupt*. "
                 f"Formats: {esc(img['formats'])}; modes: {esc(img['modes'])}.\n\n")
        dims = {k: img[k] for k in ("width", "height", "aspect", "megapixels", "bytes")
                if k in img}
        if dims:
            p.append(table(
                ["property", "min", "median", "max", "mean"],
                [[k, f"{v['min']:.2f}", f"{v['median']:.2f}", f"{v['max']:.2f}",
                  f"{v['mean']:.2f}"] for k, v in dims.items()]))
        p.append(fig("image_property_hists.png", "Distributions of size & aspect."))
        p.append(fig("image_resolution_scatter.png", "Resolution coloured by label.", "55%"))

    # ---- Sample montages ----
    if any((config.FIGURES_DIR / f"grid_label_{n}.png").exists()
           for n in config.LABEL_NAMES.values()):
        p.append("= Visual samples\n")
        p.append(fig("grid_overall.png", "All sample documents.", "95%"))
        for v, n in config.LABEL_NAMES.items():
            p.append(fig(f"grid_label_{n}.png", f"Label = {n}.", "95%"))

    # ---- Duplicates / leakage ----
    if dup:
        p.append("= Duplicates & leakage\n")
        p.append(kv_table({
            "exact duplicate groups": dup["n_exact_duplicate_groups"],
            "near-duplicate pairs (pHash<=%d)" % dup["phash_threshold"]:
                dup["n_near_duplicate_pairs"],
            "label conflicts on near-dups": dup["n_label_conflicts"],
            "test set present": dup["leakage"]["test_images_present"],
        }))
        p.append(fig("duplicate_pairs.png", "Closest near-duplicate pairs.", "60%"))
        if dup["n_label_conflicts"]:
            p.append(
                f"*Key finding:* {dup['n_label_conflicts']} of "
                f"{dup['n_near_duplicate_pairs']} near-duplicate pairs carry "
                "*conflicting labels*. This is consistent with the data containing "
                "matched genuine↔tampered pairs of the same underlying document, "
                "rather than annotation noise — so cross-validation must group "
                "near-duplicates into the same fold to avoid optimistic leakage.\n\n")
        p.append("_" + esc(dup["leakage"]["note"]) + "_\n\n")

    # ---- Embeddings / structure ----
    if clu:
        p.append("= Embedding structure\n")
        if emb:
            p.append(f"Encoder: *{esc(emb['model'])}* (`{esc(emb['pretrained'])}`) on "
                     f"*{esc(emb['device'])}* — {emb['n']}×{emb['dim']} features in {emb['seconds']}s.\n\n")
        p.append(kv_table({
            "projection": clu["projection"],
            "LOO kNN label accuracy": clu.get("loo_knn_label_accuracy"),
            "label silhouette (cosine)": clu.get("label_silhouette"),
            "best KMeans k": clu.get("kmeans", {}).get("k"),
            "KMeans silhouette": clu.get("kmeans", {}).get("silhouette"),
        }))
        p.append(fig("embed_proj_label.png", "Projection coloured by label.", "60%"))
        p.append(fig("embed_proj_is_digital.png", "Projection coloured by is_digital.", "60%"))
        p.append(fig("embed_proj_country.png", "Projection coloured by country.", "60%"))
        p.append(fig("embed_proj_doc_type.png", "Projection coloured by doc_type.", "60%"))

    # ---- Takeaways ----
    p.append("= Takeaways & modelling implications\n")
    p.append(
        "- *Target*: predict P(fraud) per id; optimise a ranking metric (AUC-style). "
        "Calibrate probabilities for the submission.\n"
        "- *Stratified validation*: split by `country`/`doc_type` and `is_digital` to "
        "avoid leakage and to measure generalisation to unseen issuers and capture modes.\n"
        "- *Heterogeneous resolution/aspect*: standardise preprocessing; preserve aspect "
        "to keep document-edge and micro-print cues that betray manipulation.\n"
        "- *Group near-duplicates before splitting*: label-conflicting near-duplicates "
        "look like genuine↔tampered pairs of one base document; keep each pair in a "
        "single fold (group/stratified CV) or folds leak.\n"
        "- *Pretrained features already cluster by document type/country*: a strong vision "
        "backbone is a sensible starting encoder; fraud cues are subtle and likely need "
        "high-resolution / forensic features beyond generic CLIP embeddings.\n\n"
        "_The public sample is tiny (development aid); all statistics above scale "
        "automatically when the full release is downloaded._\n")

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    typ = config.REPORT_DIR / "report.typ"
    typ.write_text("".join(p))
    print(f"[typst] wrote {typ}")

    typst_bin = shutil.which("typst")
    if not typst_bin:
        print("[warn] `typst` not on PATH — wrote .typ only. Install typst to compile.")
        return
    pdf = config.REPORT_DIR / "report.pdf"
    subprocess.run(
        [typst_bin, "compile", "--root", str(config.REPO_ROOT), str(typ), str(pdf)],
        check=True,
    )
    print(f"[done] compiled -> {pdf} ({pdf.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
