import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data",       type=str, default="2024",  help="2022 or 2023")
parser.add_argument("--split",      type=str, default="test", help="train or test")
parser.add_argument("--model_name", type=str, default="nlpaueb/legal-bert-base-uncased")
parser.add_argument("--max_len",    type=int, default=512,     help="tokens per chunk (incl. special tokens)")
parser.add_argument("--overlap",    type=int, default=64,      help="overlap in tokens between consecutive chunks")
parser.add_argument("--pool",       type=str, default="mean",  choices=["mean", "max"], help="chunk pooling strategy")
parser.add_argument("--batch_size", type=int, default=32,      help="chunks per GPU batch")
parser.add_argument("--data_dir",   type=str, default="./data/raw")
parser.add_argument("--output_dir", type=str, default="./data/processed/LegalBert/embeddings")
parser.add_argument("--num_workers",type=int, default=4)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def clean_text(text: str) -> str:
    for ch in ('[', ']', '"', ',', '\\'):
        text = text.replace(ch, '')
    return text.strip()

def load_cases(case_dir: str) -> dict:
    corpus = {}
    for fname in sorted(os.listdir(case_dir)):
        fpath = os.path.join(case_dir, fname)
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        key = fname if fname.endswith('.txt') else fname.split('.')[0] + '.txt'
        corpus[key] = clean_text(raw)
    return corpus

def chunk_token_ids(token_ids: list, max_len: int, overlap: int) -> list:
    window = max_len - 2
    stride = window - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be < max_len-2 ({window})")

    chunks = []
    start = 0
    while start < len(token_ids):
        chunks.append(token_ids[start : start + window])
        start += stride
    return chunks

def encode_chunks(chunks: list, tokenizer, cls_id: int, sep_id: int, pad_id: int, max_len: int):
    num_chunks = len(chunks)
    input_ids      = torch.full((num_chunks, max_len), pad_id,  dtype=torch.long)
    attention_mask = torch.zeros((num_chunks, max_len),          dtype=torch.long)

    for i, chunk in enumerate(chunks):
        ids = [cls_id] + chunk + [sep_id]
        length = len(ids)
        input_ids[i, :length]      = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :length] = 1

    return input_ids, attention_mask

def pool_chunks(chunk_embs: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'mean':
        return chunk_embs.mean(dim=0)
    elif mode == 'max':
        return chunk_embs.max(dim=0).values
    else:
        raise ValueError(f"Unknown pool mode: {mode}")

@torch.no_grad()
def encode_case(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_len: int,
    overlap: int,
    pool: str,
    batch_size: int,
    cls_id: int,
    sep_id: int,
    pad_id: int,
) -> torch.Tensor:
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) == 0:
        hidden = model.config.hidden_size
        return torch.zeros(hidden)

    chunks = chunk_token_ids(token_ids, max_len=max_len, overlap=overlap)

    input_ids, attention_mask = encode_chunks(
        chunks, tokenizer, cls_id, sep_id, pad_id, max_len)

    all_cls = []
    for start in range(0, len(chunks), batch_size):
        ids  = input_ids[start : start + batch_size].to(device)
        mask = attention_mask[start : start + batch_size].to(device)
        out  = model(input_ids=ids, attention_mask=mask)
        cls  = out.last_hidden_state[:, 0, :]
        all_cls.append(cls.cpu())

    chunk_embs = torch.cat(all_cls, dim=0)
    global_emb = pool_chunks(chunk_embs, mode=pool)

    global_emb = F.normalize(global_emb.unsqueeze(0), dim=1).squeeze(0)
    return global_emb

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading tokenizer and model: {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert_model = AutoModel.from_pretrained(args.model_name)
    bert_model.to(device)
    bert_model.eval()
    for param in bert_model.parameters():
        param.requires_grad = False

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    case_dir = os.path.join(
        args.data_dir, f'COLIEE{args.data}', f'task1_{args.split}_files_{args.data}')
    print(f'Loading cases from: {case_dir}')
    corpus = load_cases(case_dir)
    print(f'  {len(corpus)} cases found.')

    all_emb   = []
    all_names = []

    for fname, text in tqdm(corpus.items(), desc=f'Encoding ({args.pool} pool)'):
        emb = encode_case(
            text=text,
            tokenizer=tokenizer,
            model=bert_model,
            device=device,
            max_len=args.max_len,
            overlap=args.overlap,
            pool=args.pool,
            batch_size=args.batch_size,
            cls_id=cls_id,
            sep_id=sep_id,
            pad_id=pad_id,
        )
        all_emb.append(emb)
        all_names.append(fname)

    emb_matrix = torch.stack(all_emb, dim=0)
    print(f'Embedding matrix shape: {emb_matrix.shape}')

    out_path = os.path.join(
        args.output_dir,
        f'coliee{args.data}_{args.split}_hierarchical_{args.pool}_embeddings.pt'
    )
    torch.save({'embeddings': emb_matrix, 'names': all_names}, out_path)
    print(f'Saved to: {out_path}')

if __name__ == '__main__':
    main()
