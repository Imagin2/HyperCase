import torch
import torch.nn as nn
from torch.optim import Optimizer

class PoincareManifold:
    def __init__(self, c=1.0, eps=1e-5):
        self.c = c
        self.eps = eps

    def mobius_add(self, x, y):
        c = self.c
        c=torch.tensor(self.c, dtype=x.dtype, device=x.device)
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c**2 * x2 * y2
        return num / (denom.clamp_min(self.eps))

    def exp_map_zero(self, v):
        v_norm = torch.clamp(torch.norm(v, p=2, dim=-1, keepdim=True), min=self.eps)
        c = torch.tensor(self.c, dtype=v.dtype, device=v.device)
        return torch.tanh(torch.sqrt(c) * v_norm) * v / (torch.sqrt(c) * v_norm)

    def log_map_zero(self, y):
        y_norm = torch.clamp(torch.norm(y, p=2, dim=-1, keepdim=True), min=self.eps)
        c = torch.tensor(self.c, dtype=y.dtype, device=y.device)
        return (1.0 / torch.sqrt(c)) * torch.atanh(torch.sqrt(c) * y_norm) * y / y_norm

    def distance(self, x, y):

        c = self.c
        c = torch.tensor(self.c, dtype=x.dtype, device=x.device)

        x2 = torch.sum(x**2, dim=-1)
        y2 = torch.sum(y**2, dim=-1)

        num = 2 * c * torch.sum((x - y)**2, dim=-1)
        denom = (1 - c * x2) * (1 - c * y2)

        argument = 1 + num / denom
        argument = torch.clamp(argument, min=1+1e-5)
        dist = (1 / torch.sqrt(c)) * torch.acosh(argument)
        return dist

    def proj(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = (1 - self.eps) / torch.sqrt(torch.tensor(self.c))
        cond = norm >= max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)

class PoincareDistance(nn.Module):
    def __init__(self, c=1.0):
        super().__init__()
        self.manifold = PoincareManifold(c)

    def forward(self, x, y):
        return self.manifold.distance(x, y)

import torch

class LorentzManifold:
    def __init__(self, c=1.0):
        self.c = c

    def lorentz_inner(self, x, y):
        res = -x[..., :1] * y[..., :1] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1, keepdim=True)
        return res

    def exp_map_zero(self, v):
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-8)
        sqrt_c = self.c ** 0.5
        time = torch.cosh(sqrt_c * v_norm)
        space = torch.sinh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
        return torch.cat([time, space], dim=-1)

    def log_map_zero(self, x):
        sqrt_c = self.c ** 0.5
        space = x[..., 1:]
        v_norm = torch.clamp(torch.norm(space, dim=-1, keepdim=True), min=1e-8)
        time = x[..., :1]
        alpha = torch.acosh(time)
        return alpha * space / (sqrt_c * v_norm)

    def project(self, x):
        d = x.size(-1) - 1
        x_t = torch.sqrt(1.0 + torch.sum(x[..., 1:] ** 2, dim=-1, keepdim=True))
        return torch.cat([x_t, x[..., 1:]], dim=-1)
