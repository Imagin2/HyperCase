# HyperCase — End-to-End Hyperbolic Retrieval for Long Legal Documents

This repository contains the official implementation accompanying the following accepted paper:

HyperCase — End-to-End Hyperbolic Retrieval for Long Legal Documents

Authors: Akram Benamar, Mohamed Djallel Dilmi, and Faten Chaieb Chakchouk.

Venue: International Conference on Agents and Artificial Intelligence (ICAART 2027).

Status: Accepted for publication. The camera-ready version is currently in preparation.

This repository provides the code and resources required to reproduce the experiments reported in the paper.
HyperCase is a single end-to-end retrieval framework: a segment-aware
hierarchical encoder (SegCaseBERT) and a learnable projection into the
Poincaré ball are trained jointly under one hyperbolic contrastive objective.
No citation links, no charge annotations and no graph stage are used.

---

## 1. Layout

```
hypercase/
├── data/LegalBert/
│   ├── Hdata_load.py                 corpus loading, chunking, (query, positive) pairs
│   ├── data_load.py                  labels and pre-computed embeddings
│   └── precompute_embs.py            encode a corpus with a frozen backbone
├── models/
│   ├── SegCaseBert/SegCaseBert.py     segment-aware encoder + attention pooling
│   └── HyperCase/
│       ├── hypercaseret_model.py     Poincaré ball operations, HyperbolicHead
│       ├── hypercase_e2e.py          training entry point
│       └── train_hypercase_from_hier_embs.py
│                                     hyperbolic losses and retrieval evaluation
├── utils/
│   ├── metrics.py                    P@k, R@k, Mi/Ma-F1, MRR@k, MAP, nDCG@k
│   ├── hyperbolic_utils.py           standalone manifold utilities
│   └── HyperCase/geometry_analysis/
│       ├── build_levenshtein_matrices.py    reference distances between queries
│       ├── levenshtein_distortions.py       distortion diagnostic
│       └── hypercase_embeddings/            pre-computed embeddings (train splits)
└── requirements.txt
```

All scripts resolve imports relative to the package root, so **run them from
this directory** (or export `PYTHONPATH=$(pwd)`).

## 2. Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10 and CUDA 12. A GPU with 24 GB is enough for the
default configuration; `--chunk_batch_size` is the main memory knob.

## 3. Data

The COLIEE Task 1 collections are obtained from the organisers and are not
redistributed here. The expected layout is:

```
data/raw/COLIEE2023/task1_train_files_2023/*.txt
data/raw/COLIEE2023/task1_test_files_2023/*.txt
data/processed/label/task1_train_labels_2023.json
data/processed/label/task1_test_labels_2023.json
data/processed/label/test_2023_candidate_with_yearfilter.json   (optional)
```

Label files map a query file name to the list of its relevant case file names,
for instance `{"008447.txt": ["072495.txt", "082291.txt"]}`. Replace `2023` by
`2022` or `2024` for the other collections.

## 4. Training HyperCase

```bash
python models/HyperCase/hypercase_e2e.py --data 2023
```

Encoder and hyperbolic projection are optimised jointly under the symmetric
hyperbolic InfoNCE of the paper, with in-batch negatives and no negative
mining. Useful arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--data` | `2023` | collection: `2022`, `2023` or `2024` |
| `--batch_size` / `--grad_accum` | `32` / `2` | pairs per batch; negatives come from the batch, not from accumulation |
| `--lr` / `--head_lr` | `1e-5` / `1e-5` | encoder and projection learning rates |
| `--c` | `1.0` | curvature of the ball |
| `--temp` | `0.05` | temperature of the contrastive objective |
| `--r_init` | `0.1` | initial radial scale of the head |
| `--max_len` / `--overlap` | `512` / `64` | chunking of a case |
| `--chunk_batch_size` | `16` | chunks per forward pass (VRAM) |
| `--epoch` / `--patience` | `100` / `100` | training length and early stopping |
| `--seed` | `42` | random seed |
| `--output_dir` | `./results/hypercase` | run directory |

Each run creates `results/hypercase/hypercase_<data>_<pool>_.../` containing
TensorBoard scalars and `best_model.pt`, written whenever test nDCG@5
improves. The checkpoint holds `model_state`, `head_state`, `epoch`, `ndcg`
and the full argument set. Retrieval metrics at rank 5 are printed at every
epoch by `utils/metrics.py`.

Select the GPU with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=1 python models/HyperCase/hypercase_e2e.py --data 2024
```

## 5. Geometry diagnostic

The diagnostic asks how faithfully a space reproduces the relational structure
of the corpus. The reference distance between two queries is the Levenshtein
distance between their sorted lists of relevant cases: queries answered by the
same precedents are at distance zero, queries sharing none at distance one
under the `max` normalisation.

### 5.1 Reference matrices

```bash
python utils/HyperCase/geometry_analysis/build_levenshtein_matrices.py \
    --labels data/processed/label/task1_train_labels_2023.json \
    --out    results/levenshtein/lev_2023_train.npz
```

The `.npz` holds the query names, the raw matrix and its two normalised
variants (`max`, `min`), the number of shared cases per pair and the
relevance-set sizes. Add `--plots` for heat maps and histograms, `--mds` for a
non-metric MDS projection.

### 5.2 Distortion between the two geometries

```bash
python utils/HyperCase/geometry_analysis/levenshtein_distortions.py \
    --lev     results/levenshtein/lev_2023_train.npz \
    --emb_euc utils/HyperCase/geometry_analysis/hypercase_embeddings/coliee2023_train_hierarchical_attention_embeddings.pt \
    --emb_hyp utils/HyperCase/geometry_analysis/hypercase_embeddings/coliee2023_train_hyperbolic.pt \
    --out_dir results/levenshtein/2023_train --plots
```

This writes `levenshtein_distortion.json` with, for each normalisation
(`raw`, `max`, `min`) and for both spaces, the Spearman correlation against
the reference, Kruskal stress, average and worst-case distortion, the robust
p95/p05 ratio, and the per-query comparison with its Wilcoxon signed-rank
test, over all pairs and over the pairs sharing at least one relevant case.
The paper reports the `max` normalisation, whose reference distance is bounded
by 1.

The curvature is read from the hyperbolic embedding file and can be overridden
with `--c`; `--euc_metric` switches the flat baseline between `l2` and
`cosine`.

### 5.3 Provided embeddings

`utils/HyperCase/geometry_analysis/hypercase_embeddings/` contains the
representations of the three training corpora used in the analysis:
`coliee<year>_train_hierarchical_attention_embeddings.pt` for the flat variant
and `coliee<year>_train_hyperbolic.pt` for HyperCase. Each file holds
`embeddings`, `names`, and the relevance judgements in `label_dict`; the
hyperbolic files also record `space` and `c`. The diagnostic of Section 5.2
therefore runs without retraining, once the reference matrices of Section 5.1
have been built.

To export embeddings from a corpus with a frozen backbone instead, use
`data/LegalBert/precompute_embs.py`.

## 6. Reproducing the reported numbers

Results in the paper are averaged over three seeds (`--seed`). Effectiveness
tables can be read from a run log or from the `ndcg` field of `best_model.pt`.