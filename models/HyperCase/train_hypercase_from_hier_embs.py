import os
import sys
import time
import argparse
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.LegalBert.data_load import load_labels, load_embeddings, EmbeddingDataset
from models.HyperCase.hypercaseret_model import (
    HyperbolicHead,
    poincare_distance_matrix,
    early_stopping_hyp,
)
from utils.metrics import rank, t_metrics, metric, yf_metric

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--emb_train",    type=str,
                   default="results/segcasebert/embeddings/coliee2024_train_hierarchical_attention_embeddings.pt")

    p.add_argument("--emb_test",     type=str,
                   default="results/segcasebert/embeddings/coliee2024_test_hierarchical_attention_embeddings.pt")
    p.add_argument("--data",         type=str,  default="2024")
    p.add_argument("--label_dir",    type=str,  default="./data/processed/label")
    p.add_argument("--yf_path",      type=str,  default=None)
    p.add_argument("--output_dir",   type=str,  default="./results/hypercase_from_embs")
    p.add_argument("--topk",         type=int,  default=5)
    p.add_argument("--top_k_hard",         type=int,  default=2)
    p.add_argument("--use_hard_neg",         type=bool,  default=False)
    p.add_argument("--use_triplet_loss",         type=bool,  default=False)
    p.add_argument("--use_noise",         type=bool,  default=False)

    p.add_argument("--hyp_h_dim",    type=int,   default=768)
    p.add_argument("--hyp_out_dim",  type=int,   default=768)
    p.add_argument("--c",            type=float, default=1.0)
    p.add_argument("--dropout",      type=float, default=0.5)
    p.add_argument("--r_init",      type=float, default=0.1)
    p.add_argument("--use_residual",      type=bool, default=True)

    p.add_argument("--epoch",        type=int,   default=15000)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--wd",           type=float, default=1e-5)
    p.add_argument("--temp",         type=float, default=0.025)
    p.add_argument("--patience",     type=int,   default=15000)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--num_workers",  type=int,   default=0)

    return p.parse_args()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

def collate_fn(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    q_embs   = torch.stack([item["query_emb"] for item in batch], dim=0)
    pos_embs = torch.stack([item["pos_emb"]   for item in batch], dim=0)
    return q_embs, pos_embs

def hyp_info_nce_loss(
    q:    torch.Tensor,
    pos:  torch.Tensor,
    c:    float,
    temp: float = 0.05,
) -> torch.Tensor:
    dist_qp   = poincare_distance_matrix(q, pos, c)
    dist_pq   = poincare_distance_matrix(pos, q, c)
    logits_qp = -dist_qp / temp
    logits_pq = -dist_pq / temp
    labels    = torch.arange(q.shape[0], device=q.device)
    return (
        F.cross_entropy(logits_qp, labels) +
        F.cross_entropy(logits_pq, labels)
    ) / 2.0
def hyp_info_nce_loss_with_hard_negatives(q, pos, corpus_hyp, c, temp, top_k_hard=10):
    dist_all = poincare_distance_matrix(q, corpus_hyp, c)

    for i in range(q.shape[0]):
        dist_all[i, i] = 1e9

    hard_neg_idx = dist_all.topk(top_k_hard, dim=1, largest=False).indices

    dist_pos = poincare_distance_matrix(q, pos, c).diag()

    hard_neg_embs = corpus_hyp[hard_neg_idx]
    dist_hard = torch.stack([
        poincare_distance_matrix(q, hard_neg_embs[:, k, :], c).diag()
        for k in range(top_k_hard)
    ], dim=1)

    logits = -torch.cat([dist_pos.unsqueeze(1), dist_hard], dim=1) / temp
    labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)
def hyp_triplet_loss(
    q:      torch.Tensor,
    pos:    torch.Tensor,
    c:      float,
    margin: float = 0.5,
) -> torch.Tensor:
    B = q.shape[0]

    dist_qp = poincare_distance_matrix(q, pos, c).diag()

    dist_qn = poincare_distance_matrix(q, pos, c)

    mask = torch.eye(B, dtype=torch.bool, device=q.device)
    dist_qn = dist_qn.masked_fill(mask, 1e9)

    hardest_neg = dist_qn.min(dim=1).values

    loss = F.relu(dist_qp - hardest_neg + margin)

    violated = loss > 0
    if violated.sum() == 0:
        return loss.mean()
    return loss[violated].mean()

@torch.no_grad()
def evaluate(
    hyp_head:        HyperbolicHead,
    emb_matrix_test: torch.Tensor,
    test_names:      list[str],
    test_labels:     dict,
    test_query_list: list[str],
    device:          torch.device,
    c:               float,
    topk:            int = 5,
    yf_path:         str = None,
    writer               = None,
    epoch:           int = 0,
) -> float:

    hyp_head.eval()
    name2idx_test = {n: i for i, n in enumerate(test_names)}

    BATCH = 512
    all_hyp_parts = []
    for i in range(0, len(test_names), BATCH):
        batch = emb_matrix_test[i: i + BATCH].to(device)
        all_hyp_parts.append(hyp_head(batch).cpu())
    corpus_hyp = torch.cat(all_hyp_parts, dim=0)

    valid_queries = [q for q in test_query_list if q in name2idx_test]
    if not valid_queries:
        print("[Eval] WARNING: 0 valid queries - check name alignment.")
        return 0.0

    query_euc       = emb_matrix_test[[name2idx_test[q] for q in valid_queries]]
    query_hyp_parts = []
    for i in range(0, len(valid_queries), BATCH):
        batch = query_euc[i: i + BATCH].to(device)
        query_hyp_parts.append(hyp_head(batch).cpu())
    query_hyp = torch.cat(query_hyp_parts, dim=0)

    dist_matrix = poincare_distance_matrix(
        query_hyp.to(device), corpus_hyp.to(device), c
    ).cpu()
    sim_matrix = -dist_matrix

    for qi, q in enumerate(valid_queries):
        if q in name2idx_test:
            sim_matrix[qi, name2idx_test[q]] = -1e9

    corpus_stripped = [n[:-4] if n.endswith(".txt") else n for n in test_names]
    final_pre_dict  = rank(
        score_matrix   = sim_matrix,
        query_list     = valid_queries,
        candidate_list = corpus_stripped,
        topk           = topk,
    )

    label_dict = {}
    for q, refs in test_labels.items():
        label_dict[q] = [r if r.endswith(".txt") else r + ".txt" for r in refs]

    ndcg_score, mrr_score, map_score, p_score = t_metrics(
        label_dict=label_dict, pred_dict=final_pre_dict, topk=topk)
    (correct_pred, retri_cases, relevant_cases,
     micro_pre, micro_rec, micro_f,
     macro_pre, macro_rec, macro_f) = metric(
        topk=topk, pred_dict=final_pre_dict, label_dict=label_dict)

    yf_str = ""
    if yf_path and os.path.exists(yf_path):
        try:
            (_, _, _, _,
             yf_micro_pre, yf_micro_rec, yf_micro_f,
             yf_macro_pre, yf_macro_rec, yf_macro_f) = yf_metric(
                topk=topk, yf_path=yf_path,
                pred_dict=final_pre_dict, label_dict=label_dict)
            yf_str = (
                f"\n  [YF] Micro  P={yf_micro_pre:.4f}  R={yf_micro_rec:.4f}  F1={yf_micro_f:.4f}"
                f"\n  [YF] Macro  P={yf_macro_pre:.4f}  R={yf_macro_rec:.4f}  F1={yf_macro_f:.4f}"
            )
            if writer:
                writer.add_scalar("eval_yf/micro_f1", yf_micro_f, epoch)
                writer.add_scalar("eval_yf/macro_f1", yf_macro_f, epoch)
        except Exception as e:
            print(f"[Warning] yf_metric failed: {e}")

    print(
        f"\n{'='*60}"
        f"\n[Eval epoch {epoch}]  @{topk}"
        f"\n  NDCG      = {ndcg_score:.4f}"
        f"\n  MRR       = {mrr_score:.4f}"
        f"\n  MAP       = {map_score:.4f}"
        f"\n  Precision = {p_score:.4f}"
        f"\n  Micro  P={micro_pre:.4f}  R={micro_rec:.4f}  F1={micro_f:.4f}"
        f"\n  Macro  P={macro_pre:.4f}  R={macro_rec:.4f}  F1={macro_f:.4f}"
        f"  (correct={correct_pred}, retrieved={retri_cases}, relevant={relevant_cases})"
        f"{yf_str}"
        f"\n{'='*60}"
    )

    if writer:
        for k, v in [
            ("eval/ndcg", ndcg_score), ("eval/mrr", mrr_score),
            ("eval/map", map_score),   ("eval/precision", p_score),
            ("eval/micro_f1", micro_f), ("eval/macro_f1", macro_f),
        ]:
            writer.add_scalar(k, v, epoch)

    return ndcg_score.item() if hasattr(ndcg_score, "item") else float(ndcg_score)

def main():
    args = parse_args()
    set_seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(message)s",
    )
    logging.info(args)

    log_dir = os.path.join(
        args.output_dir,
        f"HyperCase_{args.data}"
        f"_c{args.c}_outdim{args.hyp_out_dim}"
        f"_bs{args.batch_size}_lr{args.lr}"
        f"_{time.strftime('%m%d-%H%M%S')}"
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"Logging to: {log_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading train embeddings: {args.emb_train}")
    emb_train, names_train, _ = load_embeddings(args.emb_train)
    print(f"  shape: {emb_train.shape}")

    print(f"Loading test embeddings:  {args.emb_test}")
    emb_test, names_test, _   = load_embeddings(args.emb_test)
    print(f"  shape: {emb_test.shape}")

    in_dim = emb_train.shape[1]

    train_labels = load_labels(
        os.path.join(args.label_dir, f"task1_train_labels_{args.data}.json"))
    test_labels  = load_labels(
        os.path.join(args.label_dir, f"task1_test_labels_{args.data}.json"))

    yf_path = args.yf_path
    if yf_path is None:
        yf_path = os.path.join(
            args.label_dir, f"test_{args.data}_candidate_with_yearfilter.json")
    if not os.path.exists(yf_path):
        yf_path = None
        print("No year-filter file found — skipping yf metrics.")

    train_dataset = EmbeddingDataset(
        emb_matrix = emb_train,
        names      = names_train,
        label_dict = train_labels,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )

    test_query_list = list(test_labels.keys())
    print(f"\nTrain dataset: {len(train_dataset)} queries, "
          f"{len(train_loader)} batches/epoch")
    print(f"Test corpus  : {len(names_test)} docs  |  queries: {len(test_query_list)}")

    hyp_head = HyperbolicHead(
        in_dim  = in_dim,
        h_dim   = args.hyp_h_dim,
        out_dim = args.hyp_out_dim,
        dropout = args.dropout,
        c       = args.c,
        r_init  = args.r_init,
        use_residual=args.use_residual
    ).to(device)
    print(f"\nHyperbolicHead params: "
          f"{sum(p.numel() for p in hyp_head.parameters() if p.requires_grad):,}")

    optimizer    = torch.optim.AdamW(hyp_head.parameters(),
                                     lr=args.lr, weight_decay=args.wd)
    total_steps  = len(train_loader) * args.epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"Scheduler: {warmup_steps} warmup / {total_steps} total steps\n")

    highest_ndcg = 0.0
    counter      = 0
    corpus_hyp_cache = None
    REFRESH_EVERY = 1

    for epoch in tqdm(range(args.epoch), desc="Epochs"):

        hyp_head.train()
        epoch_loss, n_steps = 0.0, 0
        print(f"LR: {optimizer.param_groups[0]['lr']}")

        for q_euc, pos_euc in tqdm(train_loader,
                                   desc=f"Train epoch {epoch}", leave=False):

            q_euc   = q_euc.to(device)
            pos_euc = pos_euc.to(device)
            if args.use_noise:
                noise_std = 0.2
                q_euc   = q_euc   + torch.randn_like(q_euc)   * noise_std
                pos_euc = pos_euc + torch.randn_like(pos_euc) * noise_std
            q_hyp   = hyp_head(q_euc)
            pos_hyp = hyp_head(pos_euc)
            if args.use_triplet_loss:
                loss = hyp_triplet_loss(q_hyp, pos_hyp, c=args.c, margin=0.5)

            elif args.use_hard_neg:
                if corpus_hyp_cache is None or n_steps % REFRESH_EVERY == 0:
                    with torch.no_grad():
                        parts = []
                        for i in range(0, len(emb_train), 512):
                            parts.append(hyp_head(emb_train[i:i+512].to(device)))
                        corpus_hyp_cache = torch.cat(parts, dim=0)
                loss = hyp_info_nce_loss_with_hard_negatives(
                                                    q          = q_hyp,
                                                    pos        = pos_hyp,
                                                    corpus_hyp = corpus_hyp_cache,
                                                    c          = args.c,
                                                    temp       = args.temp,
                                                    top_k_hard = args.top_k_hard,
                                                )

            else:

                loss = hyp_info_nce_loss(q_hyp, pos_hyp, c=args.c, temp=args.temp)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hyp_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_steps    += 1

        mean_loss = epoch_loss / max(n_steps, 1)
        print(f"[Train epoch {epoch}] loss = {mean_loss:.4f}")
        if writer:
            writer.add_scalar("train/loss", mean_loss, epoch)

        ndcg = evaluate(
            hyp_head        = hyp_head,
            emb_matrix_test = emb_test,
            test_names      = names_test,
            test_labels     = test_labels,
            test_query_list = test_query_list,
            device          = device,
            c               = args.c,
            topk            = args.topk,
            yf_path         = yf_path,
            writer          = writer,
            epoch           = epoch,
        )

        if ndcg > highest_ndcg:
            ckpt_path = os.path.join(log_dir, "best_hyperhead.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": hyp_head.state_dict(),
                "ndcg":        ndcg,
                "args":        vars(args),
                "emb_train":   args.emb_train,
                "emb_test":    args.emb_test,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint: {ckpt_path}")

        highest_ndcg, should_stop, counter = early_stopping_hyp(
            highest_ndcg, ndcg, epoch, counter, patience=args.patience)
        print('best_ndcg :',highest_ndcg)
        if should_stop:
            break

    writer.close()
    print("Done.")

if __name__ == "__main__":
    main()
