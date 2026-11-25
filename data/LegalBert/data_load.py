import random
import json
import torch
from torch.utils.data import Dataset

def load_labels(path: str) -> dict:
    with open(path, 'r') as f:
        raw = json.load(f)
    return raw

def load_embeddings(path: str):
    data = torch.load(path, map_location='cpu')

    if isinstance(data, dict):
        emb_matrix = data['embeddings']
        names      = data['names']
        label_dict = data.get('label_dict', None)
    else:
        emb_matrix, names, label_dict = data

    if not isinstance(emb_matrix, torch.Tensor):
        emb_matrix = torch.tensor(emb_matrix, dtype=torch.float32)

    return emb_matrix.float(), names, label_dict

class EmbeddingDataset(Dataset):

    def __init__(
        self,
        emb_matrix:  torch.Tensor,
        names:       list,
        label_dict:  dict,
        hard_neg_dict=None, hard_neg_num=5
    ):
        self.hard_neg_dict = hard_neg_dict
        self.hard_neg_num  = hard_neg_num
        self.emb_matrix = emb_matrix
        self.name2idx   = {n: i for i, n in enumerate(names)}

        self.queries = [
            q for q in label_dict
            if q in self.name2idx
            and any(p in self.name2idx for p in label_dict[q])
        ]
        self.label_dict = label_dict

        if not self.queries:
            raise ValueError(
                "No valid queries found — check that label_dict keys "
                "and names share the same filename format (.txt or not)."
            )

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q_name    = self.queries[idx]
        valid_pos = [p for p in self.label_dict[q_name] if p in self.name2idx]
        pos_name  = random.choice(valid_pos)
        with torch.no_grad():
            q_emb    = self.emb_matrix[self.name2idx[q_name]]
            pos_embs = torch.stack([self.emb_matrix[self.name2idx[p]]
                                    for p in valid_pos])
            sims     = torch.mv(pos_embs, q_emb)
            hardest  = valid_pos[sims.argmin().item()]

        return {
            'query_emb': self.emb_matrix[self.name2idx[q_name]],
            'pos_emb':   self.emb_matrix[self.name2idx[pos_name]],
        }
