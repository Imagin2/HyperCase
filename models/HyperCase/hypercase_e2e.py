import os
import sys
import math
import time
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.LegalBert.data_load import load_labels
from data.LegalBert.Hdata_load import (
    load_corpus, pretokenise_corpus, HierTextDataset, hier_collate_fn
)
from models.SegCaseBert.SegCaseBert import SegCaseBert as SegCaseBert, early_stopping
from models.HyperCase.hypercaseret_model import HyperbolicHead
from models.HyperCase.train_hypercase_from_hier_embs import hyp_info_nce_loss, evaluate

def str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "t")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model_name",         type=str,   default="nlpaueb/legal-bert-base-uncased")
parser.add_argument("--max_len",            type=int,   default=512)
parser.add_argument("--overlap",            type=int,   default=64)
parser.add_argument("--pool",               type=str,   default="attention",
                    choices=["mean", "max", "attention"])
parser.add_argument("--freeze_until_layer", type=int,   default=0)
parser.add_argument("--chunk_batch_size",   type=int,   default=16)

parser.add_argument("--hyp_h_dim",    type=int,      default=768)
parser.add_argument("--hyp_out_dim",  type=int,      default=768)
parser.add_argument("--c",            type=float,    default=1.0)
parser.add_argument("--dropout",      type=float,    default=0.2)
parser.add_argument("--r_init",       type=float,    default=0.1)
parser.add_argument("--use_residual", type=str2bool, default=True)

parser.add_argument("--epoch",        type=int,   default=200)
parser.add_argument("--lr",           type=float, default=2e-6)
parser.add_argument("--wd",           type=float, default=1e-2)
parser.add_argument("--head_lr",      type=float, default=1e-5)
parser.add_argument("--head_wd",      type=float, default=1e-5)
parser.add_argument("--batch_size",   type=int,   default=32)
parser.add_argument("--grad_accum",   type=int,   default=2)
parser.add_argument("--temp",         type=float, default=0.025)
parser.add_argument("--patience",     type=int,   default=200)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--seed",         type=int,   default=42)

parser.add_argument("--data",        type=str,  default="2023")
parser.add_argument("--data_dir",    type=str,  default="./data/raw")
parser.add_argument("--label_dir",   type=str,  default="./data/processed/label")
parser.add_argument("--yf_path",     type=str,  default=None)
parser.add_argument("--output_dir",  type=str,  default="./results/hypercase")
parser.add_argument("--num_workers", type=int,  default=0)
parser.add_argument("--topk",        type=int,  default=5)

args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s[line:%(lineno)d] %(message)s"
)
logging.info(args)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

@torch.no_grad()
def encode_corpus(model, tokenised, names, device, batch_size):
    model.eval()
    use_amp = device.type == "cuda"
    embs = []
    for start in range(0, len(names), batch_size):
        batch_names = names[start:start + batch_size]
        ids   = [tokenised[n][0] for n in batch_names]
        masks = [tokenised[n][1] for n in batch_names]
        offsets = [0]
        for x in ids:
            offsets.append(offsets[-1] + x.shape[0])
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(torch.cat(ids, dim=0).to(device),
                        torch.cat(masks, dim=0).to(device),
                        offsets)
        embs.append(out.float().cpu())
    return torch.cat(embs, dim=0)

def train_epoch(model, hyp_head, dataloader, device, optimizer, scheduler,
                scaler, c, temp, writer, epoch, grad_accum=1):
    model.train()
    hyp_head.train()
    optimizer.zero_grad()
    params     = [p for g in optimizer.param_groups for p in g["params"]]
    total_loss = 0.0
    n_steps    = 0
    use_amp    = device.type == "cuda"

    def optimizer_step():
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc=f"Train epoch {epoch}", leave=False)):

        with torch.cuda.amp.autocast(enabled=use_amp):
            q_emb = model(
                batch["q_ids"].to(device),
                batch["q_masks"].to(device),
                batch["q_offsets"],
            )
            pos_emb = model(
                batch["pos_ids"].to(device),
                batch["pos_masks"].to(device),
                batch["pos_offsets"],
            )

        q_hyp   = hyp_head(q_emb.float())
        pos_hyp = hyp_head(pos_emb.float())
        loss    = hyp_info_nce_loss(q_hyp, pos_hyp, c=c, temp=temp) / grad_accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * grad_accum
        n_steps    += 1

        if (step + 1) % grad_accum == 0:
            optimizer_step()

    if n_steps % grad_accum != 0:
        optimizer_step()

    mean_loss = total_loss / max(n_steps, 1)
    print(f"[Train epoch {epoch}] loss = {mean_loss:.4f}")
    if writer:
        writer.add_scalar("train/loss", mean_loss, epoch)
    return mean_loss

def main():

    set_seed(args.seed)
    effective_batch = args.batch_size * args.grad_accum
    log_dir = os.path.join(
        args.output_dir,
        f"hypercase_{args.data}_{args.pool}"
        f"_freeze{args.freeze_until_layer}"
        f"_bs{effective_batch}"
        f"_lr{args.lr}_hlr{args.head_lr}"
        f"_c{args.c}_t{args.temp}"
        f"_{time.strftime('%m%d-%H%M%S')}"
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"Logging to: {log_dir}")
    print(f"Effective batch size: {effective_batch} "
          f"({args.batch_size} × {args.grad_accum} grad_accum)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_case_dir = os.path.join(
        args.data_dir, f"COLIEE{args.data}",
        f"task1_train_files_{args.data}")
    test_case_dir  = os.path.join(
        args.data_dir, f"COLIEE{args.data}",
        f"task1_test_files_{args.data}")

    print(f"Loading train corpus from: {train_case_dir}")
    train_corpus = load_corpus(train_case_dir)
    print(f"  {len(train_corpus)} cases. Tokenising...")
    tokenised_train = pretokenise_corpus(
        train_corpus, tokenizer, args.max_len, args.overlap)

    print(f"Loading test corpus from: {test_case_dir}")
    test_corpus = load_corpus(test_case_dir)
    print(f"  {len(test_corpus)} cases. Tokenising...")
    tokenised_test = pretokenise_corpus(
        test_corpus, tokenizer, args.max_len, args.overlap)

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

    train_dataset = HierTextDataset(
        tokenised  = tokenised_train,
        label_dict = train_labels,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = hier_collate_fn,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )

    test_names      = list(tokenised_test.keys())
    test_query_list = list(test_labels.keys())

    print(f"Train dataset: {len(train_dataset)} queries, "
          f"{len(train_dataloader)} batches/epoch")

    model = SegCaseBert(
        model_name         = args.model_name,
        max_len            = args.max_len,
        overlap            = args.overlap,
        pool               = args.pool,
        freeze_until_layer = args.freeze_until_layer,
        chunk_batch_size   = args.chunk_batch_size,
    ).to(device)

    hyp_head = HyperbolicHead(
        in_dim       = model.out_dim,
        h_dim        = args.hyp_h_dim,
        out_dim      = args.hyp_out_dim,
        dropout      = args.dropout,
        c            = args.c,
        r_init       = args.r_init,
        use_residual = args.use_residual,
    ).to(device)

    n_bert = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in hyp_head.parameters() if p.requires_grad)
    print(f"Parameters: SegCaseBert {n_bert:,} trainable | HyperbolicHead {n_head:,}")

    bert_params  = [p for p in model.bert.parameters() if p.requires_grad]
    other_params = (
        list(model.chunk_attn.parameters()) if model.chunk_attn is not None else []
    )

    param_groups = [
        {"params": bert_params, "lr": args.lr, "weight_decay": args.wd},
    ]
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr * 5,
                             "weight_decay": args.wd})
    param_groups.append({"params": list(hyp_head.parameters()), "lr": args.head_lr,
                         "weight_decay": args.head_wd})

    optimizer = torch.optim.AdamW(param_groups)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum)
    total_steps     = steps_per_epoch * args.epoch
    warmup_steps    = int(total_steps * args.warmup_ratio)
    scheduler       = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    print(f"Scheduler: {warmup_steps} warmup steps / {total_steps} total steps")

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    highest_ndcg  = 0.0
    con_epoch_num = 0

    for epoch in tqdm(range(args.epoch), desc="Epochs"):

        train_epoch(
            model      = model,
            hyp_head   = hyp_head,
            dataloader = train_dataloader,
            device     = device,
            optimizer  = optimizer,
            scheduler  = scheduler,
            scaler     = scaler,
            c          = args.c,
            temp       = args.temp,
            writer     = writer,
            epoch      = epoch,
            grad_accum = args.grad_accum,
        )

        emb_test = encode_corpus(model, tokenised_test, test_names, device,
                                 args.batch_size)

        ndcg = evaluate(
            hyp_head        = hyp_head,
            emb_matrix_test = emb_test,
            test_names      = test_names,
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
            ckpt_path = os.path.join(log_dir, "best_model.pt")
            torch.save({
                "epoch":       epoch,
                "data":        args.data,
                "model_state": model.state_dict(),
                "head_state":  hyp_head.state_dict(),
                "ndcg":        ndcg,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  → Saved best checkpoint: {ckpt_path}")

        highest_ndcg, should_stop, con_epoch_num = early_stopping(
            highest_ndcg, ndcg, epoch, con_epoch_num, patience=args.patience)
        if should_stop:
            break

    writer.close()
    print("Done.")

if __name__ == "__main__":
    main()
