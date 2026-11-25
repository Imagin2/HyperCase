import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class ChunkAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, chunk_embs: torch.Tensor) -> torch.Tensor:
        scores  = self.attn(chunk_embs).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(-1) * chunk_embs).sum(dim=0)

class SegCaseBert(nn.Module):

    def __init__(
        self,
        model_name:          str   = "nlpaueb/legal-bert-base-uncased",
        max_len:             int   = 512,
        overlap:             int   = 64,
        pool:                str   = "attention",
        freeze_until_layer:  int   = 6,
        chunk_batch_size:    int   = 16,
    ):
        super().__init__()

        self.max_len          = max_len
        self.overlap          = overlap
        self.pool_mode        = pool
        self.chunk_batch_size = chunk_batch_size

        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.gradient_checkpointing_enable()

        if freeze_until_layer > 0:
            self._freeze_layers(freeze_until_layer)

        bert_hidden = self.bert.config.hidden_size
        self.out_dim = bert_hidden

        if pool == "attention":
            self.chunk_attn = ChunkAttentionPooling(bert_hidden)
        else:
            self.chunk_attn = None

    def _freeze_layers(self, n: int):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def _pool_chunks(self, chunk_embs: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "mean":
            return chunk_embs.mean(dim=0)
        elif self.pool_mode == "max":
            return chunk_embs.max(dim=0).values
        elif self.pool_mode == "attention":
            return self.chunk_attn(chunk_embs)
        else:
            raise ValueError(f"Unknown pool mode: {self.pool_mode}")

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        doc_offsets:    list,
    ) -> torch.Tensor:

        device       = input_ids.device
        total_chunks = input_ids.size(0)
        cls_outputs  = []

        for i in range(0, total_chunks, self.chunk_batch_size):
            batch_ids  = input_ids[i:i + self.chunk_batch_size]
            batch_mask = attention_mask[i:i + self.chunk_batch_size]

            with torch.cuda.amp.autocast(enabled=batch_ids.is_cuda):
                out = self.bert(input_ids=batch_ids, attention_mask=batch_mask)
                cls = out.last_hidden_state[:, 0, :]

            cls_outputs.append(cls)
            del out
            if batch_ids.is_cuda:
                torch.cuda.empty_cache()

        all_cls = torch.cat(cls_outputs, dim=0)

        B        = len(doc_offsets) - 1
        doc_embs = []
        for i in range(B):
            s, e       = doc_offsets[i], doc_offsets[i + 1]
            pooled     = self._pool_chunks(all_cls[s:e])
            doc_embs.append(pooled)

        out = torch.stack(doc_embs, dim=0)
        return F.normalize(out, dim=-1)

def early_stopping(
    highest_ndcg:  float,
    current_ndcg:  float,
    epoch:         int,
    con_epoch_num: int,
    patience:      int = 5,
):
    if current_ndcg > highest_ndcg:
        highest_ndcg  = current_ndcg
        con_epoch_num = 0
        print(f"[Early stopping] New best NDCG: {highest_ndcg:.4f} at epoch {epoch}")
    else:
        con_epoch_num += 1
        print(f"[Early stopping] No improvement for {con_epoch_num}/{patience} epochs.")

    should_stop = con_epoch_num >= patience
    if should_stop:
        print(f"[Early stopping] Stopping at epoch {epoch}.")
    return highest_ndcg, should_stop, con_epoch_num
