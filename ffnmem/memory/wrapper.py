"""Adapter around the vendored Lngram v2 module.

The vendored ``RouteNgramMemoryV2`` does not own the router: the caller supplies
route logits and codes.  This wrapper owns the router (``d_model -> R * bits``),
mirrors the v1 integration (router reads the pre-attention RMSNorm output, the
memory module reads the raw residual stream, output is added to the residual
before attention), applies the Lngram v2 Appendix E initialisation, keeps the
routing logits in fp32, and exposes diagnostics.

Document-boundary masking is deliberately off: training windows are plain
contiguous chunks, so ``segment_ids`` is always ``None`` (one segment/window).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .lngram_v2 import RouteNgramMemoryV2

# Appendix E recipe (Lngram v2) -- overrides the code defaults, which are the
# opposite initialisation (table std 0, output projection std 1).
APPENDIX_E = dict(
    bits_per_route=4,
    ngrams=(2, 3),
    table_init_std_scale=1.0,
    output_proj_init_std_scale=0.0,
    conv_zero_init=True,
    conv_kernel_size=4,
    q_surrogate_enable=True,
    q_surrogate_mode="approximate",
    sink_initial_active_mass=0.5,
)


class LngramMemory(nn.Module):
    def __init__(self, d_model: int, mem_cfg: dict, init_std: float = 0.02, norm_eps: float = 1e-5):
        super().__init__()
        cfg = dict(APPENDIX_E)
        cfg.update({k: v for k, v in mem_cfg.items() if k not in ("routes", "mem_dim", "kv_heads", "q_heads", "head_dim", "router_init_std")})
        self.routes = int(mem_cfg["routes"])
        self.bits = int(cfg["bits_per_route"])
        head_dim = int(mem_cfg.get("head_dim", 64))
        q_heads = int(mem_cfg.get("q_heads", d_model // head_dim))
        if "ngrams" in cfg:
            cfg["ngrams"] = tuple(int(n) for n in cfg["ngrams"])
        self.module = RouteNgramMemoryV2(
            hidden_size=d_model,
            num_routes=self.routes,
            memory_dim=int(mem_cfg["mem_dim"]),
            readout_num_heads=q_heads,
            readout_num_kv_heads=int(mem_cfg.get("kv_heads", 4)),
            readout_head_dim=head_dim,
            initializer_range=init_std,
            rmsnorm_eps=norm_eps,
            **cfg,
        )
        self.router = nn.Linear(d_model, self.routes * self.bits, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=float(mem_cfg.get("router_init_std", init_std)))
        # Diagnostics (filled only while ``collect_diagnostics`` is True, in eval).
        self.collect_diagnostics = False
        self.reset_diagnostics()

    # ------------------------------------------------------------------ forward
    def route_logits(self, h_norm: Tensor) -> Tensor:
        """Router in fp32 regardless of autocast (sign thresholding is precision-sensitive)."""
        with torch.autocast(device_type=h_norm.device.type, enabled=False):
            return nn.functional.linear(h_norm.float(), self.router.weight.float())

    def forward(self, h: Tensor, h_norm: Tensor) -> Tensor:
        q_logits = self.route_logits(h_norm)
        codes = self.module.quantize_route_logits(q_logits)
        if self.collect_diagnostics and not self.training:
            self._accumulate_diagnostics(h, codes)
        return self.module(h, codes, q_logits=q_logits if self.training else None)

    # -------------------------------------------------------------- diagnostics
    def reset_diagnostics(self) -> None:
        self._code_hist = None  # [R, 2^bits] counts
        self._sink_sum = 0.0
        self._sink_count = 0

    @torch.no_grad()
    def _accumulate_diagnostics(self, h: Tensor, codes: Tensor) -> None:
        m = self.module
        n_codes = 1 << self.bits
        flat = codes.reshape(-1, self.routes).long()
        onehot_counts = torch.zeros(self.routes, n_codes, device=codes.device, dtype=torch.float64)
        onehot_counts.scatter_add_(
            1, flat.t(), torch.ones_like(flat.t(), dtype=torch.float64)
        )
        self._code_hist = onehot_counts if self._code_hist is None else self._code_hist + onehot_counts
        # Sink mass: recompute the readout scores (same path as the module's full mode).
        query = m._project_query(h)
        tokens = m._lookup_route_chunk(codes, None, 0, m.num_routes, None, None)
        key, _ = m._project_memory(tokens)
        b, t = query.shape[:2]
        grouped = query.view(b, t, m.readout_num_kv_heads, m.readout_groups, m.readout_head_dim)
        scores = torch.matmul(grouped, key.transpose(-1, -2)).float() * m.softmax_scale
        real = m._real_route_probabilities(scores)  # [B,T,Hkv,G,R]
        sink = 1.0 - real.sum(-1)
        self._sink_sum += float(sink.sum())
        self._sink_count += sink.numel()

    def diagnostics(self) -> dict:
        out: dict = {}
        if self._sink_count:
            out["sink_mass"] = self._sink_sum / self._sink_count
        if self._code_hist is not None:
            p = self._code_hist / self._code_hist.sum(-1, keepdim=True).clamp_min(1)
            entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
            eff = entropy.exp()  # effective number of codes per route
            out["eff_codes_mean"] = float(eff.mean())
            out["frac_routes_eff_lt4"] = float((eff < 4).double().mean())
            out["frac_routes_eff_lt2"] = float((eff < 2).double().mean())
            out["dead_code_frac"] = float((self._code_hist == 0).double().mean())
            out["top_code_share"] = float(p.max(-1).values.mean())
        return out

    # ------------------------------------------------------------ bookkeeping
    @property
    def sink_initial_mass(self) -> float:
        # The module's sink logit gives (1 - mass) to the sink at zero real logits.
        return 1.0 - self.module.sink_initial_active_mass

    def table_parameters(self):
        return list(self.module.tables.parameters())

    def active_table_params_per_token(self) -> int:
        return self.routes * len(self.module.ngrams) * self.module.memory_dim


def param_group_of(name: str, param: nn.Parameter) -> str:
    """Optimizer group for a parameter name of the full model.

    tables -> 4x LR, wd 0; router -> base LR, wd 0; other memory params (readout
    Q/KV/O, conv) -> base LR, wd 0.1; backbone matrices -> wd 0.1; every 1-D
    parameter (norm weights) -> wd 0.
    """
    if ".memory.module.tables." in name:
        return "tables"
    if ".memory.router." in name:
        return "router"
    if param.ndim < 2:
        return "no_decay"
    return "decay"

