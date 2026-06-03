#set page(paper: "a4", margin: 2cm, numbering: "1")
#set text(size: 10pt)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => [#pagebreak(weak: true) #it]
#align(center)[#text(20pt, weight: "bold")[FREUID Challenge 2026]\ #text(13pt)[Identity-Document Fraud Detection — Exploratory Data Analysis]\ #text(9pt, style: "italic")[IJCAI-ECAI 2026 · generated from the public data sample]]
#v(0.5cm)
#outline(indent: auto)
= Overview
The FREUID challenge targets *next-generation identity-document fraud detection* across physical manipulations, GenAI-driven multimodal edits, and print-and-capture forgeries. The task is a *binary classification* (`label` 0 = genuine, 1 = fraud) scored as a probability per image id. This report profiles the publicly released sample to ground later modelling.

= Data inventory
Extracted footprint: *15 files* (3.2 MB), *13 images* indexed.

#table(columns: 3, [*extension*], [*count*], [*size*], [.jpeg], [13], [3.2 MB], [.csv], [2], [1.8 KB])

#figure(image("../figures/inventory_file_types.png", width: 60%), caption: [File count by extension.])

== `sample_submission.csv`
13 rows · columns: `id`, `label`

== `train_sample_labels.csv`
13 rows · columns: `id`, `image_path`, `label`, `is_digital`, `type`

= Labels & class balance
#table(columns: 2, [*metric*], [*value*], [rows], [13], [fraud], [3], [genuine], [10], [fraud rate], [23.1%], [unique ids], [True], [missing images], [0])

#figure(image("../figures/labels_balance.png", width: 50%), caption: [Genuine vs fraud counts.])

#figure(image("../figures/labels_is_digital.png", width: 55%), caption: [Digital vs physical capture, split by label.])

#figure(image("../figures/labels_by_country.png", width: 85%), caption: [Documents per country, stacked by label.])

#figure(image("../figures/labels_by_doctype.png", width: 55%), caption: [Documents per document type.])

The sample spans multiple countries and document types (DL/ID); `is_digital` flags born-digital vs print-and-capture acquisition — an axis the threat model explicitly cares about.

= Image properties & integrity
*13 images*, *0 corrupt*. Formats: {'JPEG': 13}; modes: {'RGB': 13}.

#table(columns: 5, [*property*], [*min*], [*median*], [*max*], [*mean*], [width], [840.00], [1000.00], [1585.00], [1135.77], [height], [530.00], [630.00], [1000.00], [716.54], [aspect], [1.58], [1.58], [1.59], [1.59], [megapixels], [0.45], [0.63], [1.58], [0.88], [bytes], [122096.00], [275929.00], [338191.00], [257084.23])

#figure(image("../figures/image_property_hists.png", width: 85%), caption: [Distributions of size & aspect.])

#figure(image("../figures/image_resolution_scatter.png", width: 55%), caption: [Resolution coloured by label.])

= Visual samples
#figure(image("../figures/grid_overall.png", width: 95%), caption: [All sample documents.])

#figure(image("../figures/grid_label_genuine.png", width: 95%), caption: [Label = genuine.])

#figure(image("../figures/grid_label_fraud.png", width: 95%), caption: [Label = fraud.])

= Duplicates & leakage
#table(columns: 2, [*metric*], [*value*], [exact duplicate groups], [0], [near-duplicate pairs (pHash<=8)], [5], [label conflicts on near-dups], [3], [test set present], [False])

#figure(image("../figures/duplicate_pairs.png", width: 60%), caption: [Closest near-duplicate pairs.])

*Key finding:* 3 of 5 near-duplicate pairs carry *conflicting labels*. This is consistent with the data containing matched genuine↔tampered pairs of the same underlying document, rather than annotation noise — so cross-validation must group near-duplicates into the same fold to avoid optimistic leakage.

_Public release ships only train_sample; the hidden test set cannot be checked locally. Re-run when test images are available._

= Embedding structure
Encoder: *ViT-B-32* (`laion2b_s34b_b79k`) on *cuda* — 13×512 features in 0.28s.

#table(columns: 2, [*metric*], [*value*], [projection], [UMAP], [LOO kNN label accuracy], [0.7692307692307693], [label silhouette (cosine)], [-0.028390683233737946], [best KMeans k], [5], [KMeans silhouette], [0.5361450910568237])

#figure(image("../figures/embed_proj_label.png", width: 60%), caption: [Projection coloured by label.])

#figure(image("../figures/embed_proj_is_digital.png", width: 60%), caption: [Projection coloured by is_digital.])

#figure(image("../figures/embed_proj_country.png", width: 60%), caption: [Projection coloured by country.])

#figure(image("../figures/embed_proj_doc_type.png", width: 60%), caption: [Projection coloured by doc_type.])

= Takeaways & modelling implications
- *Target*: predict P(fraud) per id; optimise a ranking metric (AUC-style). Calibrate probabilities for the submission.
- *Stratified validation*: split by `country`/`doc_type` and `is_digital` to avoid leakage and to measure generalisation to unseen issuers and capture modes.
- *Heterogeneous resolution/aspect*: standardise preprocessing; preserve aspect to keep document-edge and micro-print cues that betray manipulation.
- *Group near-duplicates before splitting*: label-conflicting near-duplicates look like genuine↔tampered pairs of one base document; keep each pair in a single fold (group/stratified CV) or folds leak.
- *Pretrained features already cluster by document type/country*: a strong vision backbone is a sensible starting encoder; fraud cues are subtle and likely need high-resolution / forensic features beyond generic CLIP embeddings.

_The public sample is tiny (development aid); all statistics above scale automatically when the full release is downloaded._
