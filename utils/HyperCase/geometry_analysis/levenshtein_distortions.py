import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from models.HyperCase.hypercaseret_model import poincare_distance_matrix as _proj_poincare
except Exception:
    _proj_poincare = None

VARIANTS = ["raw", "max", "min"]

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lev", type=str, required=True)
    p.add_argument("--emb_euc", type=str, required=True)
    p.add_argument("--emb_hyp", type=str, required=True)
    p.add_argument("--c", type=float, default=None)
    p.add_argument("--euc_metric", type=str, default="l2", choices=["l2", "cosine"])
    p.add_argument("--min_pairs_per_query", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="./results/levenshtein")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    return p.parse_args()

def norm_name(n):
    n = str(n)
    return n[:-4] if n.endswith(".txt") else n

def read_emb(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected format in {path}")
    emb = names = None
    for k in ("embeddings", "emb", "features", "vectors"):
        if k in obj and torch.is_tensor(obj[k]):
            emb = obj[k]
            break
    if emb is None:
        emb = next((v for v in obj.values() if torch.is_tensor(v) and v.dim() == 2), None)
    for k in ("names", "ids", "filenames", "keys"):
        if k in obj and isinstance(obj[k], (list, tuple)):
            names = list(obj[k])
            break
    if emb is None or names is None:
        raise ValueError(f"Cannot extract (embeddings, names) from {path}")
    return emb, names, obj

def poincare(A, B, c):
    if _proj_poincare is not None:
        return _proj_poincare(A, B, c)
    A2 = (A * A).sum(dim=1, keepdim=True)
    B2 = (B * B).sum(dim=1, keepdim=True).T
    sq = (A2 + B2 - 2 * A @ B.T).clamp_min(0)
    den = (1 - c * A2).clamp_min(1e-10) * (1 - c * B2).clamp_min(1e-10)
    return torch.acosh((1 + 2 * c * sq / den).clamp_min(1.0 + 1e-7)) / (c ** 0.5)

def pairwise(X, kind, c, device, chunk=512):
    X = X.to(device).float()
    n = X.shape[0]
    out = torch.empty((n, n), dtype=torch.float32)
    for i in range(0, n, chunk):
        blk = X[i:i + chunk]
        d = poincare(blk, X, c) if kind == "poincare" else torch.cdist(blk, X, p=2)
        out[i:i + chunk] = d.float().cpu()
    out.fill_diagonal_(0.0)
    return out.numpy().astype(np.float64)

def distortion_metrics(d, delta):
    keep = (delta > 0) & np.isfinite(delta) & (d > 0)
    d, delta = d[keep], delta[keep]
    r = d / delta
    alpha = float(r.sum() / max((r * r).sum(), 1e-12))
    ratio = alpha * r
    lr = np.log(r)
    a_st = float((d * delta).sum() / max((d * d).sum(), 1e-12))
    stress = float(np.sqrt(((a_st * d - delta) ** 2).sum() / max((delta ** 2).sum(), 1e-12)))
    rho = spearmanr(d, delta)[0] if d.size > 2 else float("nan")
    return {
        "n_pairs": int(d.size),
        "spearman": float(rho),
        "kruskal_stress": stress,
        "avg_distortion": float(np.abs(ratio - 1.0).mean()),
        "log_ratio_std": float(lr.std()),
        "worst_case_distortion": float(r.max() / r.min()),
        "robust_distortion_p95_p05": float(np.percentile(r, 95) / np.percentile(r, 5)),
    }

def per_query_log_std(D, delta, mask, min_pairs):
    n = D.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        m = mask[i].copy()
        m[i] = False
        m &= (delta[i] > 0) & np.isfinite(delta[i]) & (D[i] > 0)
        if m.sum() >= min_pairs:
            out[i] = np.log(D[i, m] / delta[i, m]).std()
    return out

def identical_pair_stats(D, delta_raw, iu):
    same = delta_raw[iu] == 0
    if not same.any():
        return None
    med_all = float(np.median(D[iu]))
    return float(np.median(D[iu][same]) / max(med_all, 1e-12)), int(same.sum())

def make_plots(out_dir, D_e, D_h, lev, iu, overlap_iu, pq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    se = float(np.median(D_e[iu]))
    sh = float(np.median(D_h[iu]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharey="row")
    for col, key in enumerate(VARIANTS):
        delta = lev[key][iu]
        ok = overlap_iu & np.isfinite(delta)
        for row, (D, s, name) in enumerate([(D_e, se, "Euclidean"), (D_h, sh, "Hyperbolic")]):
            ax = axes[row, col]
            x, y = delta[ok], D[iu][ok] / s
            ax.scatter(x + np.random.default_rng(0).uniform(-0.02, 0.02, x.size) * max(x.max(), 1e-9),
                       y, s=6, alpha=0.35)
            if x.size > 2:
                rho = spearmanr(x, y)[0]
                ax.set_title(f"{name} | Levenshtein {key} | Spearman {rho:+.3f}", fontsize=9)
            ax.set_xlabel(f"Levenshtein ({key})")
            if col == 0:
                ax.set_ylabel("normalized embedding distance")
    fig.suptitle("Embedding distance vs Levenshtein distance (queries sharing >=1 relevant case)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distance_vs_levenshtein.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key in zip(axes, VARIANTS):
        delta = lev[key][iu]
        ok = overlap_iu & np.isfinite(delta) & (delta > 0)
        le = np.log(D_e[iu][ok] / delta[ok])
        lh = np.log(D_h[iu][ok] / delta[ok])
        le -= np.median(le)
        lh -= np.median(lh)
        bins = np.linspace(min(le.min(), lh.min()), max(le.max(), lh.max()), 60)
        ax.hist(le, bins=bins, alpha=0.55, density=True, label=f"Euclidean (std {le.std():.3f})")
        ax.hist(lh, bins=bins, alpha=0.55, density=True, label=f"Hyperbolic (std {lh.std():.3f})")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("log(d / Levenshtein), centered")
        ax.set_title(f"Levenshtein ({key})")
        ax.legend(fontsize=8)
    fig.suptitle("Distortion distribution: narrower = distances better preserved")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "log_ratio_distribution.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key in zip(axes, VARIANTS):
        e, h = pq[key]
        ok = np.isfinite(e) & np.isfinite(h)
        ax.scatter(e[ok], h[ok], s=8, alpha=0.4)
        lim = [0, max(e[ok].max(), h[ok].max()) * 1.05] if ok.any() else [0, 1]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        frac = float((h[ok] < e[ok]).mean()) if ok.any() else float("nan")
        ax.set_title(f"{key}: hyperbolic better for {100 * frac:.1f}% of queries", fontsize=9)
        ax.set_xlabel("Euclidean per-query distortion")
        ax.set_ylabel("Hyperbolic per-query distortion")
    fig.suptitle("Per-query distortion (points below the diagonal favor hyperbolic)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_query_distortion.png"), dpi=150)
    plt.close(fig)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    z = np.load(args.lev, allow_pickle=False)
    lev_names = [norm_name(n) for n in z["names"]]
    lev_full = {k: z[k].astype(np.float64) for k in VARIANTS}
    shared_full = z["shared"].astype(np.int64)
    mode = str(z["mode"]) if "mode" in z else "unknown"

    emb_e, names_e, _ = read_emb(args.emb_euc)
    emb_h, names_h, extra_h = read_emb(args.emb_hyp)
    c = args.c
    if c is None and isinstance(extra_h, dict):
        c = extra_h.get("c")
    if c is None:
        raise SystemExit("Curvature not found: pass --c explicitly.")

    map_e = {norm_name(n): i for i, n in enumerate(names_e)}
    map_h = {norm_name(n): i for i, n in enumerate(names_h)}
    keep = [i for i, n in enumerate(lev_names) if n in map_e and n in map_h]
    if len(keep) < 4:
        raise SystemExit("Fewer than 4 queries found in both embedding files.")
    q_names = [lev_names[i] for i in keep]
    ix = np.ix_(keep, keep)
    lev = {k: v[ix] for k, v in lev_full.items()}
    shared = shared_full[ix]
    n = len(q_names)

    Xe = emb_e[[map_e[q] for q in q_names]].float()
    Xh = emb_h[[map_h[q] for q in q_names]].float()
    if args.euc_metric == "cosine":
        Xe = torch.nn.functional.normalize(Xe, dim=1)

    iu = np.triu_indices(n, k=1)
    overlap_iu = shared[iu] > 0
    n_pairs = iu[0].size

    print(f"Levenshtein file : {args.lev}  (mode: {mode})")
    print(f"Euclidean        : {args.emb_euc}")
    print(f"Hyperbolic       : {args.emb_hyp}  (c = {c})")
    print(f"Queries          : {n} of {len(lev_names)} found in both embedding files")
    print(f"Query pairs      : {n_pairs:,} | sharing >=1 case: {int(overlap_iu.sum()):,} "
          f"({100 * overlap_iu.mean():.3f}%)")

    norms = Xh.norm(dim=1)
    print(f"Hyperbolic radius: max {norms.max():.4f} / ball radius {1 / np.sqrt(c):.4f}")

    print("\nComputing embedding distances...")
    D_e = pairwise(Xe, "l2", c, device)
    D_h = pairwise(Xh, "poincare", c, device)

    results = {"config": {
        "lev_file": args.lev, "lev_mode": mode, "emb_euc": args.emb_euc,
        "emb_hyp": args.emb_hyp, "c": float(c), "n_queries": n,
        "n_pairs": int(n_pairs), "n_overlapping_pairs": int(overlap_iu.sum()),
        "euc_metric": args.euc_metric,
    }}

    idm = identical_pair_stats(D_e, lev["raw"], iu)
    if idm:
        idh = identical_pair_stats(D_h, lev["raw"], iu)
        print(f"Identical-set pairs: {idm[1]} | median distance relative to overall median: "
              f"Euclidean {idm[0]:.3f}  Hyperbolic {idh[0]:.3f}  (lower = better)")
        results["identical_pairs"] = {"n": idm[1], "euclidean_rel_median": idm[0],
                                      "hyperbolic_rel_median": idh[0]}

    mask_overlap = shared > 0
    pq_plot = {}

    for key in VARIANTS:
        delta_iu = lev[key][iu]
        print(f"\n{'=' * 74}\nLevenshtein / {key}\n{'=' * 74}")
        block = {}
        for scope, sel in [("overlapping_pairs", overlap_iu), ("all_pairs", np.ones(n_pairs, bool))]:
            m_e = distortion_metrics(D_e[iu][sel], delta_iu[sel])
            m_h = distortion_metrics(D_h[iu][sel], delta_iu[sel])
            block[scope] = {"euclidean": m_e, "hyperbolic": m_h}
            print(f"\n  [{scope.replace('_', ' ')}]  n = {m_e['n_pairs']:,}")
            print(f"  {'metric':<34}{'Euclidean':>12}{'Hyperbolic':>13}{'better':>11}")
            for mk, label, higher in [
                ("spearman", "Spearman (+)", True),
                ("kruskal_stress", "Kruskal stress (-)", False),
                ("avg_distortion", "Average distortion (-)", False),
                ("log_ratio_std", "Std of log ratio (-)", False),
                ("robust_distortion_p95_p05", "Robust distortion p95/p05 (-)", False),
                ("worst_case_distortion", "Worst-case distortion (-)", False),
            ]:
                ve, vh = m_e[mk], m_h[mk]
                win = "hyperbolic" if (vh > ve) == higher else "euclidean"
                print(f"  {label:<34}{ve:>12.4f}{vh:>13.4f}{win:>11}")

        pq_e = per_query_log_std(D_e, lev[key], mask_overlap, args.min_pairs_per_query)
        pq_h = per_query_log_std(D_h, lev[key], mask_overlap, args.min_pairs_per_query)
        pq_plot[key] = (pq_e, pq_h)
        ok = np.isfinite(pq_e) & np.isfinite(pq_h)
        if ok.sum() <= 10:
            print(f"\n  Per-query distortion: only {int(ok.sum())} queries with >= "
                  f"{args.min_pairs_per_query} overlapping pairs, test skipped.")
        else:
            stat, pval = wilcoxon(pq_h[ok], pq_e[ok], alternative="less")
            frac = float((pq_h[ok] < pq_e[ok]).mean())
            print(f"\n  Per-query distortion (std of log ratio over each query's overlapping pairs):")
            print(f"    median Euclidean {np.median(pq_e[ok]):.4f} | median Hyperbolic {np.median(pq_h[ok]):.4f}")
            print(f"    hyperbolic lower for {100 * frac:.1f}% of {int(ok.sum())} queries | "
                  f"paired Wilcoxon (H < E) p = {pval:.2e}")
            block["per_query"] = {
                "n_queries": int(ok.sum()),
                "median_euclidean": float(np.median(pq_e[ok])),
                "median_hyperbolic": float(np.median(pq_h[ok])),
                "frac_hyperbolic_lower": frac,
                "wilcoxon_H_lt_E_pvalue": float(pval),
            }
        results[f"levenshtein_{key}"] = block

    print(f"\n{'=' * 74}\nSUMMARY  (overlapping pairs, scale-invariant metrics)\n{'=' * 74}")
    print(f"{'Levenshtein':<12}{'metric':<26}{'Euclidean':>12}{'Hyperbolic':>13}")
    for key in VARIANTS:
        b = results[f"levenshtein_{key}"]["overlapping_pairs"]
        for mk, label in [("spearman", "Spearman (+)"), ("log_ratio_std", "Std log ratio (-)"),
                          ("avg_distortion", "Avg distortion (-)")]:
            print(f"{key:<12}{label:<26}{b['euclidean'][mk]:>12.4f}{b['hyperbolic'][mk]:>13.4f}")
    print("\nReading guide:")
    print("  - 'overlapping pairs' is the informative scope: for disjoint pairs the Levenshtein")
    print("    distance only reflects list lengths, not relatedness.")
    print("  - Std of log ratio and Spearman are fully scale-invariant: they are the fairest")
    print("    comparison between spaces whose distances live on different scales.")
    print("  - 'min' normalization is not bounded by 1 and can exaggerate small-set pairs.")

    out_json = os.path.join(args.out_dir, "levenshtein_distortion.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics -> {out_json}")

    if args.plots:
        make_plots(args.out_dir, D_e, D_h, lev, iu, overlap_iu, pq_plot)
        print(f"Figures -> {args.out_dir}/*.png")

if __name__ == "__main__":
    main()
