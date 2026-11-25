from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

def _clamp_norm(x: torch.Tensor, c: float, eps: float = EPS) -> torch.Tensor:
    max_norm = (1.0 / math.sqrt(c)) - eps
    norms    = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    scale    = (norms.clamp(max=max_norm) / norms)
    return x * scale

def expmap0(v: torch.Tensor, c: float) -> torch.Tensor:
    sqrt_c = math.sqrt(c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    direction = v / v_norm
    mapped = torch.tanh(sqrt_c * v_norm) * direction / sqrt_c
    return _clamp_norm(mapped, c)

def logmap0(x: torch.Tensor, c: float) -> torch.Tensor:
    sqrt_c = math.sqrt(c)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    direction = x / x_norm
    arg = (sqrt_c * x_norm).clamp(max=1.0 - EPS)
    return torch.atanh(arg) * direction / sqrt_c

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = (1 + 2 * c * xy + c * c * x2 * y2).clamp(min=EPS)
    return _clamp_norm(num / den, c)

def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    sqrt_c = math.sqrt(c)
    diff   = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1).clamp(max=1.0 / sqrt_c - EPS)
    return (2.0 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)

def poincare_distance_matrix(
    A: torch.Tensor,
    B: torch.Tensor,
    c: float,
) -> torch.Tensor:
    A_exp = A.unsqueeze(1)
    B_exp = B.unsqueeze(0)
    sqrt_c = math.sqrt(c)
    diff   = mobius_add(-A_exp, B_exp, c)
    diff_norm = diff.norm(dim=-1).clamp(max=1.0 / sqrt_c - EPS)
    return (2.0 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)

class LearnableHyperbolicProj(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, c=0.01, dropout=0.7):

        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-1:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.c = c

    def forward(self, x):
        h = self.net(x)
        norm_h = h.norm(dim=1, keepdim=True)
        max_norm = 1.0 / (self.c ** 0.5) - 1e-5
        scale = torch.clamp(norm_h, max=max_norm) / (norm_h + 1e-6)
        h = h * scale
        return h
class HyperbolicReparam(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.phi_dir = nn.Linear(in_dim, hid_dim)
        self.phi_norm = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        v = self.phi_dir(h)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-6)

        p = self.sigmoid(self.phi_norm(h))

        theta = p * v
        return theta

class HyperbolicHead(nn.Module):

    def __init__(
        self,
        in_dim:  int   = 768,
        h_dim:   int   = 256,
        out_dim: int   = 64,
        dropout: float = 0.1,
        c:       float = 1.0,
        r_init:   float=0.1,
        use_residual:  bool=False,
    ):
        super().__init__()

        self.c = c
        self.r_init=r_init
        self.use_residual=use_residual
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, out_dim),
        )

        nn.init.xavier_uniform_(self.net[-1].weight, gain=self.r_init)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        v = self.net(x)
        if self.use_residual:
            h=expmap0(x,self.c)
            o=mobius_add(expmap0(v,self.c),h,self.c)
            return o
        else:
            return expmap0(v, self.c)

class HyperbolicHead_n(nn.Module):

    def __init__(
        self,
        in_dim:  int   = 768,
        h_dim:   int   = 256,
        out_dim: int   = 64,
        dropout: float = 0.1,
        c:       float = 1.0,
    ):
        super().__init__()
        print('######################################################################')
        print('######################################################################')
        print('######################################################################')
        print('######################################################################')
        print('######################################################################')
        print('######################################################################')
        print('Hypercase version hyoercaselink')
        print('######################################################################')
        self.c = c

        self.learnable_proj_h= LearnableHyperbolicProj(in_dim,[], out_dim, c,dropout)

        self.reparam = HyperbolicReparam(out_dim, out_dim)

        nn.init.xavier_uniform_(self.learnable_proj_h.net[-3].weight, gain=0.1)
        nn.init.zeros_(self.learnable_proj_h.net[-3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h=self.learnable_proj_h(x)
        feat_hyp = expmap0(x,self.c)
        h = expmap0(h,self.c)
        h = mobius_add(h, feat_hyp,self.c)
        h = self.reparam(h)
        return expmap0(h, self.c)

def early_stopping_hyp(
    highest: float,
    current: float,
    epoch:   int,
    counter: int,
    patience: int = 100,
) -> tuple[float, bool, int]:
    if current > highest:
        print(f'  [ES] NDCG improved {highest:.4f} → {current:.4f}')
        return current, False, 0
    counter += 1
    print(f'  [ES] No improvement ({counter}/{patience})')
    if counter >= patience:
        print(f'  [ES] Early stopping at epoch {epoch}.')
        return highest, True, counter
    return highest, False, counter
