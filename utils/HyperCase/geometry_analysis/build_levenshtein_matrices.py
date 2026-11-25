import os
import json
import argparse

import numpy as np

try:
    from Levenshtein import distance as _lev
except ImportError:
    _lev = None

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--labels", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--mode", type=str, default="sorted", choices=["sorted", "sequence", "set"])
    p.add_argument("--keep_empty", action="store_true")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--mds", action="store_true")
    p.add_argument("--mds_matrix", type=str, default="max", choices=["raw", "max", "min"])
    p.add_argument("--highlight", type=int, nargs="*", default=[])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def norm_name(n):
    return n[:-4] if isinstance(n, str) and n.endswith(".txt") else n

def levenshtein(a, b):
    if _lev is not None:
        return _lev(a, b)
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]

def load_lists(path, mode, keep_empty):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    names, lists = [], []
    for q, refs in raw.items():
        docs = [norm_name(r) for r in refs if norm_name(r) != norm_name(q)]
        if not docs and not keep_empty:
            continue
        if mode == "sorted":
            docs = sorted(set(docs))
        elif mode == "set":
            docs = sorted(set(docs))
        names.append(norm_name(q))
        lists.append(docs)
    return names, lists

def shared_counts(lists):
    vocab = {}
    rows, cols = [], []
    for i, docs in enumerate(lists):
        for d in set(docs):
            rows.append(i)
            cols.append(vocab.setdefault(d, len(vocab)))
    from scipy.sparse import csr_matrix
    B = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                   shape=(len(lists), max(len(vocab), 1)))
    inter = np.asarray((B @ B.T).todense(), dtype=np.float64)
    np.fill_diagonal(inter, 0)
    return inter

def raw_matrix(lists, mode, inter):
    n = len(lists)
    sizes = np.array([len(x) for x in lists], dtype=np.float64)
    if mode == "set":
        D = sizes[:, None] + sizes[None, :] - 2.0 * inter
        np.fill_diagonal(D, 0.0)
        return D
    D = np.zeros((n, n), dtype=np.float64)
    step = max(n // 10, 1)
    for i in range(n):
        a = lists[i]
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = levenshtein(a, lists[j])
        if (i + 1) % step == 0:
            print(f"  {i + 1}/{n} rows")
    return D

def normalized(D, sizes):
    mx = np.maximum(sizes[:, None], sizes[None, :])
    mn = np.minimum(sizes[:, None], sizes[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        D_max = np.where(mx > 0, D / mx, np.nan)
        D_min = np.where(mn > 0, D / mn, np.nan)
    np.fill_diagonal(D_max, 0.0)
    np.fill_diagonal(D_min, 0.0)
    return D_max, D_min

def describe(name, D, overlap):
    iu = np.triu_indices(D.shape[0], k=1)
    v = D[iu]
    o = overlap[iu]
    v_ok = v[np.isfinite(v)]
    vo = v[o & np.isfinite(v)]
    print(f"  {name:<6} all pairs: min {v_ok.min():.3f} | median {np.median(v_ok):.3f} | "
          f"max {v_ok.max():.3f}   overlapping: "
          + (f"min {vo.min():.3f} | median {np.median(vo):.3f} | max {vo.max():.3f}" if vo.size else "none"))

def make_plots(out_base, mats, overlap, names, sizes, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iu = np.triu_indices(len(names), k=1)
    o = overlap[iu]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (key, D) in zip(axes, mats.items()):
        im = ax.imshow(D, cmap="viridis", interpolation="nearest")
        ax.set_title(f"Levenshtein ({key})")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_base + "_heatmaps.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, D) in zip(axes, mats.items()):
        v = D[iu]
        ax.hist(v[o & np.isfinite(v)], bins=40, alpha=0.8, label="overlapping pairs")
        ax.set_title(f"{key}: {int(o.sum())} overlapping / {o.size:,} pairs")
        ax.set_xlabel("distance")
        ax.set_ylabel("count")
        ax.legend()
    fig.suptitle("Levenshtein distances among queries sharing at least one relevant case")
    fig.tight_layout()
    fig.savefig(out_base + "_histograms.png", dpi=150)
    plt.close(fig)

    if args.mds:
        import warnings
        from sklearn.manifold import MDS
        warnings.filterwarnings("ignore", category=FutureWarning)
        D = np.nan_to_num(mats[args.mds_matrix], nan=np.nanmax(mats[args.mds_matrix]))
        print(f"Running non-metric MDS on '{args.mds_matrix}' matrix...")
        mds = MDS(n_components=2, metric=False, n_init=1, dissimilarity="precomputed",
                  random_state=args.seed)
        X = mds.fit_transform(D)
        colors = np.zeros(len(names))
        for i in args.highlight:
            if 0 <= i < len(names):
                colors[i] = 1
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=10, cmap="coolwarm")
        ax.set_title(f"Non-metric MDS of queries (Levenshtein, {args.mds_matrix})")
        fig.tight_layout()
        fig.savefig(out_base + f"_mds_{args.mds_matrix}.png", dpi=150)
        plt.close(fig)

def main():
    args = parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    if _lev is None and args.mode != "set":
        print("[Info] python-Levenshtein not installed: using pure-Python fallback (slower).")

    names, lists = load_lists(args.labels, args.mode, args.keep_empty)
    n = len(names)
    sizes = np.array([len(x) for x in lists], dtype=np.float64)
    print(f"Labels   : {args.labels}")
    print(f"Mode     : {args.mode}")
    print(f"Queries  : {n} | relevant set size mean {sizes.mean():.2f} | max {int(sizes.max())}")

    inter = shared_counts(lists)
    overlap = inter > 0
    n_pairs = n * (n - 1) // 2
    n_overlap = int(np.triu(overlap, k=1).sum())
    print(f"Pairs    : {n_pairs:,} | sharing >=1 case: {n_overlap:,} ({100 * n_overlap / max(n_pairs, 1):.3f}%)")

    print("Computing raw Levenshtein matrix...")
    D_raw = raw_matrix(lists, args.mode, inter)
    D_max, D_min = normalized(D_raw, sizes)

    identical = int(np.triu((D_raw == 0), k=1).sum())
    print(f"Pairs with distance 0 (identical sets): {identical}")
    print("\nSummary:")
    mats = {"raw": D_raw, "max": D_max, "min": D_min}
    for key, D in mats.items():
        describe(key, D, overlap)

    np.savez_compressed(
        args.out,
        names=np.array(names),
        raw=D_raw.astype(np.float32),
        max=D_max.astype(np.float32),
        min=D_min.astype(np.float32),
        shared=inter.astype(np.int16),
        sizes=sizes.astype(np.int16),
        mode=np.array(args.mode),
        labels=np.array(args.labels),
    )
    print(f"\nMatrices -> {args.out}")

    if args.plots or args.mds:
        base = os.path.splitext(args.out)[0]
        make_plots(base, mats, overlap, names, sizes, args)
        print(f"Figures  -> {base}_*.png")

if __name__ == "__main__":
    main()
