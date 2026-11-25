import os
import random
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

def load_corpus(case_dir: str) -> dict:
    corpus = {}
    for fname in os.listdir(case_dir):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(case_dir, fname)
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            corpus[fname] = f.read()
    return corpus

def _chunk_token_ids(token_ids: list, max_len: int, overlap: int) -> list:
    window = max_len - 2
    stride = window - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be < max_len-2 ({window})")
    chunks, start = [], 0
    while start < len(token_ids):
        chunks.append(token_ids[start: start + window])
        start += stride
    return chunks

def pretokenise_corpus(
    corpus:     dict,
    tokenizer:  PreTrainedTokenizer,
    max_len:    int = 512,
    overlap:    int = 64,
) -> dict:
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    tokenised = {}
    for name, text in corpus.items():
        raw_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks  = _chunk_token_ids(raw_ids, max_len, overlap)
        if not chunks:
            chunks = [[]]

        n = len(chunks)
        input_ids      = torch.full((n, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len),         dtype=torch.long)

        for i, chunk in enumerate(chunks):
            ids    = [cls_id] + chunk + [sep_id]
            length = min(len(ids), max_len)
            input_ids[i, :length]      = torch.tensor(ids[:length], dtype=torch.long)
            attention_mask[i, :length] = 1

        tokenised[name] = (input_ids, attention_mask)

    return tokenised

class HierTextDataset(Dataset):

    def __init__(
        self,
        tokenised:     dict,
        label_dict:    dict,
    ):
        self.tokenised  = tokenised
        self.label_dict = label_dict

        self.queries = [
            q for q in label_dict
            if q in tokenised
            and any(p in tokenised for p in label_dict[q])
        ]
        if not self.queries:
            raise ValueError("No valid queries found — check label_dict and tokenised keys match.")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q_name = self.queries[idx]

        valid_pos = [p for p in self.label_dict[q_name] if p in self.tokenised]
        pos_name  = random.choice(valid_pos)

        q_ids,   q_masks   = self.tokenised[q_name]
        pos_ids, pos_masks = self.tokenised[pos_name]

        return {
            'q_name':    q_name,
            'pos_name':  pos_name,
            'q_ids':     q_ids,
            'q_masks':   q_masks,
            'pos_ids':   pos_ids,
            'pos_masks': pos_masks,
        }

def hier_collate_fn(batch: list) -> dict:

    def pack(ids_list, masks_list):
        offsets = [0]
        for ids in ids_list:
            offsets.append(offsets[-1] + ids.shape[0])
        flat_ids   = torch.cat(ids_list,   dim=0)
        flat_masks = torch.cat(masks_list, dim=0)
        return flat_ids, flat_masks, offsets

    q_ids,   q_masks,   q_off   = pack(
        [b['q_ids']    for b in batch],
        [b['q_masks']  for b in batch],
    )
    pos_ids, pos_masks, pos_off = pack(
        [b['pos_ids']   for b in batch],
        [b['pos_masks'] for b in batch],
    )

    return {
        'q_ids':       q_ids,
        'q_masks':     q_masks,
        'q_offsets':   q_off,
        'pos_ids':     pos_ids,
        'pos_masks':   pos_masks,
        'pos_offsets': pos_off,
        'q_names':     [b['q_name']   for b in batch],
        'pos_names':   [b['pos_name'] for b in batch],
    }
