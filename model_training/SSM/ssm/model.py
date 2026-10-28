"""TrackerSSM: Mamba-2 SSD (chunk-form) backbone for track finding.

Design constraints baked in:
- ONNX/CPU exportable: every op is matmul/cumsum/exp/softplus/elementwise.
  No python loops, no data-dependent control flow; the only dynamic axis is
  the (padded) sequence length.
- Two-level chunked SSD: intra-chunk quadratic term in chunk_size (=64),
  inter-chunk in n_chunks. No sequential scan anywhere => the training graph
  IS the deployment graph.
- Causality contract: sequences are padded AT THE END (to a multiple of
  chunk_size). Because the SSM recurrence is causal, trailing padding cannot
  contaminate outputs at real positions; no attention-style mask is needed.
- fp32 state paths: dt/cumsum/exp are forced float32 even under bf16
  autocast (selective-scan state numerics).

Inputs:  x  (B, N_pad, n_features) float32, N_pad % chunk_size == 0
Outputs: embedding (B, N_pad, embedding_dim), beta_logit (B, N_pad, 1)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y.clamp(min=1e-6)))


class SSDCore(nn.Module):
    """Chunked state-space-duality mixer (Mamba-2), export-safe two-level form."""

    def __init__(self, d_inner: int, n_heads: int, d_state: int,
                 chunk_size: int = 64, dt_min: float = 1e-3, dt_max: float = 0.1):
        super().__init__()
        assert d_inner % n_heads == 0
        self.n_heads = n_heads
        self.headdim = d_inner // n_heads
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.A_log = nn.Parameter(torch.empty(n_heads).uniform_(math.log(dt_min), math.log(dt_max)))
        self.dt_bias = nn.Parameter(_inv_softplus(torch.empty(n_heads).uniform_(dt_min, dt_max)))
        self.D = nn.Parameter(torch.ones(n_heads))

    def forward(self, x: torch.Tensor, dt_raw: torch.Tensor,
                Bmat: torch.Tensor, Cmat: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, S, P, c = self.n_heads, self.d_state, self.headdim, self.chunk_size
        if N % c != 0:
            raise ValueError(f"sequence length {N} must be a multiple of chunk_size {c}")
        C = N // c

        # ---- discretisation (fp32) ----
        dt = F.softplus(dt_raw.float() + self.dt_bias.float())            # (B,N,H)
        a = -torch.exp(self.A_log.float())                                # (H,)
        dA = dt * a                                                       # (B,N,H)

        # ---- chunk reshape ----
        dAc = dA.reshape(B, C, c, H)                                      # (B,C,c,H)
        Xc = x.reshape(B, C, c, H, P)                                     # (B,C,c,H,P)
        Bc = Bmat.reshape(B, C, c, S)                                     # (B,C,c,S)
        Cc = Cmat.reshape(B, C, c, S)                                     # (B,C,c,S)
        cum = torch.cumsum(dAc, dim=2)                                    # (B,C,c,H)
        dA_sum = cum[:, :, -1, :]                                         # (B,C,H)

        # exponentials on tiny (B,C,c,H)/(B,C,H) tensors only; the c x c /
        # C x C structure is carried by constant masks + GEMMs (ORT-friendly).
        e_cum = torch.exp(cum.clamp(max=0.0))                             # (B,C,c,H)
        e_ncum = torch.exp((-cum).clamp(max=20.0))                        # (B,C,c,H)

        idx = torch.arange(c, device=x.device)
        lower = (idx.unsqueeze(1) >= idx.unsqueeze(0))                    # (c_t,c_s) const
        CB = torch.matmul(Cc, Bc.transpose(-1, -2))                       # (B,C,c_t,c_s)
        # per-head scaling: M[b,ci,h,t,s] = CB[t,s]*triu[t,s]*e_cum[t]*e_ncum[s]
        M = (CB * lower.to(CB.dtype)).unsqueeze(-1)                       # (B,C,c_t,c_s,1)
        M = M * e_cum.unsqueeze(3)                                        # * e_cum_t  -> (B,C,c,c,H)
        M = M * e_ncum.unsqueeze(2)                                       # * e_ncum_s
        M = M.permute(0, 1, 4, 2, 3)                                      # (B,C,H,c,c)
        Xp = Xc.permute(0, 1, 3, 2, 4)                                    # (B,C,H,c,P)
        Y1 = torch.matmul(M, Xp)                                          # (B,C,H,c,P)

        # ---- per-chunk state produced: state_k[h] = sum_s exp(dA_sum - cum_s) B_s^T X_s
        w_out = torch.exp((dA_sum.unsqueeze(2) - cum).clamp(max=0.0))     # (B,C,c,H)
        Bw = w_out.unsqueeze(-1) * Bc.unsqueeze(3)                        # (B,C,c,H,S)
        state_k = torch.matmul(Bw.permute(0, 1, 3, 4, 2), Xp)             # (B,C,H,S,P)

        # ---- inter-chunk carry (vectorised, quadratic in C) ----
        Dcum = torch.cumsum(dA_sum, dim=1)                                # (B,C,H)
        Dcarry = torch.zeros_like(Dcum)
        Dcarry[:, 1:, :] = Dcum[:, :-1, :]
        e_carry = torch.exp(Dcarry.clamp(max=0.0))                        # (B,C,H)
        e_dcum = torch.exp((-Dcum).clamp(max=20.0))                       # (B,C,H)
        cidx = torch.arange(C, device=x.device)
        strict = (cidx.unsqueeze(1) > cidx.unsqueeze(0)).to(x.dtype)      # [ci,k]
        # W[b,ci,k,h] = strict[ci,k] * e_carry[ci] * e_dcum[k]
        W = strict.unsqueeze(-1) * e_carry.unsqueeze(2) * e_dcum.unsqueeze(1)  # (B,C,C,H)
        h_in = torch.matmul(
            W.permute(0, 3, 1, 2),                                        # (B,H,Cq,Ck)
            state_k.permute(0, 2, 1, 3, 4).reshape(B, H, C, S * P),
        ).reshape(B, H, C, S, P).permute(0, 2, 1, 3, 4)                   # (B,C,H,S,P)

        # ---- inter-chunk output: Y2[...,t,:] = exp(cum_t) * (C_t . h_in)
        Y2 = torch.matmul(Cc.unsqueeze(2),                                # (B,C,1,c,S)
                          h_in)                                           # (B,C,H,S,P) -> bcast
        Y2 = Y2 * e_cum.permute(0, 1, 3, 2).unsqueeze(-1)                 # (B,C,H,c,P)

        Y = Y1 + Y2                                                       # (B,C,H,c,P)
        Y = Y.permute(0, 1, 3, 2, 4)                                      # (B,C,c,H,P)
        Y = Y + self.D.float().reshape(1, 1, 1, H, 1) * Xc
        return Y.reshape(B, N, H * P)

    @torch.no_grad()
    def reference_scan(self, x, dt_raw, Bmat, Cmat):
        """Exact stepwise recurrence; unit tests only."""
        B, N, _ = x.shape
        H, S, P = self.n_heads, self.d_state, self.headdim
        X = x.reshape(B, N, H, P)
        dt = F.softplus(dt_raw.float() + self.dt_bias.float())
        decay = torch.exp(dt * (-torch.exp(self.A_log.float())))          # (B,N,H)
        hs = x.new_zeros(B, H, S, P)
        outs = []
        for t in range(N):
            hs = decay[:, t, :, None, None] * hs + Bmat[:, t, None, :, None] * X[:, t, :, None, :]
            yt = torch.einsum("bs,bhsp->bhp", Cmat[:, t], hs)
            outs.append(yt + self.D.float().reshape(1, H, 1) * X[:, t])
        return torch.stack(outs, dim=1).reshape(B, N, H * P)


class Block(nn.Module):
    """Pre-norm SSD block: LayerNorm -> [z|x,B,C,dt] proj -> causal depthwise conv -> SSD -> gate -> out proj."""

    def __init__(self, d_model: int, expand: int, n_heads: int, d_state: int,
                 chunk_size: int, d_conv: int = 4, bidirectional: bool = False):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.n_heads = n_heads
        self.bidirectional = bidirectional
        # bidirectional => bias-free on the x path, so end-padding stays exact
        # zeros through the network: the padded FRONT of the reversed scan then
        # carries an exact-zero state and cannot contaminate real-hit outputs.
        self.norm = nn.LayerNorm(d_model, elementwise_affine=not bidirectional)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + n_heads, bias=False)
        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, padding=d_conv - 1,
                                groups=conv_dim, bias=not bidirectional)
        self.core = SSDCore(self.d_inner, n_heads, d_state, chunk_size)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        zxbcdt = self.in_proj(self.norm(h))
        z, xBC, dt_raw = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.n_heads], dim=-1)
        xBC = self.conv1d(xBC.transpose(1, 2))[:, :, :N].transpose(1, 2)  # causal
        xBC = self.act(xBC)
        x, Bmat, Cmat = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        y = self.core(x, dt_raw, Bmat, Cmat)
        if self.bidirectional:
            xf, bf, cf, dtf = (torch.flip(t, [1]) for t in (x, Bmat, Cmat, dt_raw))
            y = y + torch.flip(self.core(xf, dtf, bf, cf), [1])
        y = y * torch.rsqrt(y.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(y.dtype)
        y = y * self.act(z)
        return h + self.out_proj(y)


class TrackerSSM(nn.Module):
    """SSD backbone -> per-hit [embedding(embedding_dim) | beta_logit(1)] head."""

    def __init__(self, n_features: int = 12, d_model: int = 256, n_layer: int = 6,
                 expand: int = 1, n_heads: int = 4, d_state: int = 64,
                 chunk_size: int = 64, embedding_dim: int = 4,
                 bidirectional: bool = False):
        super().__init__()
        self.chunk_size = chunk_size
        self.embedding_dim = embedding_dim
        self.input_proj = nn.Linear(n_features, d_model, bias=not bidirectional)
        self.blocks = nn.ModuleList(
            Block(d_model, expand, n_heads, d_state, chunk_size,
                  bidirectional=bidirectional) for _ in range(n_layer))
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, embedding_dim + 1)

    def forward(self, x: torch.Tensor):
        if x.shape[1] % self.chunk_size != 0:
            raise ValueError(f"input length {x.shape[1]} must be a multiple of {self.chunk_size}")
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        out = self.head(self.norm_out(h))
        return out[..., : self.embedding_dim], out[..., self.embedding_dim:]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
