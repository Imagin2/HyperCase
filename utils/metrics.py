import torch
import json

import os
import sys
from pathlib import Path
import math
def rank(score_matrix, query_list, candidate_list, topk=5):
    pred = {}
    k    = min(topk, len(candidate_list))
    for qi, qname in enumerate(query_list):
        top_idx      = torch.topk(score_matrix[qi], k=k).indices.tolist()
        pred[qname]  = [candidate_list[i] for i in top_idx]
    return pred

def metric(topk, pred_dict, label_dict):
    correct_total = retrieved_total = relevant_total = 0
    macro_p, macro_r, macro_f = [], [], []

    for qname, preds in pred_dict.items():
        positives = set(label_dict.get(qname, []))
        preds_set = set(preds[:topk])
        tp        = len(preds_set & positives)

        correct_total   += tp
        retrieved_total += len(preds_set)
        relevant_total  += len(positives)

        p = tp / len(preds_set) if preds_set else 0.0
        r = tp / len(positives) if positives else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        macro_p.append(p); macro_r.append(r); macro_f.append(f)

    mi_p = correct_total / retrieved_total if retrieved_total else 0.0
    mi_r = correct_total / relevant_total  if relevant_total  else 0.0
    mi_f = 2 * mi_p * mi_r / (mi_p + mi_r) if (mi_p + mi_r) else 0.0

    avg  = lambda lst: sum(lst) / len(lst) if lst else 0.0
    return (correct_total, retrieved_total, relevant_total,
            mi_p, mi_r, mi_f,
            avg(macro_p), avg(macro_r), avg(macro_f))

def _dcg(rel):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel))

def t_metrics(label_dict, pred_dict, topk=5):
    ndcg_l, mrr_l, map_l, p_l = [], [], [], []

    for qname, preds in pred_dict.items():
        positives = set(label_dict.get(qname, []))
        preds_k   = preds[:topk]
        rel       = [1 if p in positives else 0 for p in preds_k]

        ideal = sorted(rel, reverse=True)
        dcg   = _dcg(rel); idcg = _dcg(ideal)
        ndcg_l.append(dcg / idcg if idcg else 0.0)

        mrr = next((1.0 / (i + 1) for i, p in enumerate(preds_k) if p in positives), 0.0)
        mrr_l.append(mrr)

        hits, sp = 0, 0.0
        for i, p in enumerate(preds_k):
            if p in positives:
                hits += 1; sp += hits / (i + 1)
        map_l.append(sp / len(positives) if positives else 0.0)
        p_l.append(sum(rel) / topk)

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
    return avg(ndcg_l), avg(mrr_l), avg(map_l), avg(p_l)

def yf_metric(topk, yf_path, pred_dict, label_dict):
    with open(yf_path, 'r') as f:
        yf_candidates = json.load(f)

    yf_pred = {
        q: [p for p in preds if p in set(yf_candidates.get(q, []))][:topk]
        for q, preds in pred_dict.items()
    }
    return (yf_pred,) + metric(topk, yf_pred, label_dict)
