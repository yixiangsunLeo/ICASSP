"""AdaHGNN-MI: ``EEG_AdaMSHyper`` adapted to cross-subject motor imagery (D-039 / D-040).

Design record: ``docs/reports/PROPOSAL_adahgnn_mi_adaptation_2026-09-20.md`` (T0 + T1 + T2 accepted
by the user 2026-09-20 17:38; the unadapted "F+" original is not run; a same-class mixup arm is on;
DANN and the NHC constraint are off). The lab file ``_official/models.py`` is untouched; this module
is the adapted network.

Forward, for ``C`` electrodes and ``T = 1000`` samples at 250 Hz (per-trial z-scored input)::

    x[B, C, T]
      -> S  = BN( conv_a(x) + conv_b(x) )          shared temporal filter bank (F maps, k = 125 / 63)   [B, F, C, T]
      -> S~ = S - gamma_c * (A1 @ S)                learned hypergraph-Laplacian spatial filter          [B, F, C, T]
      -> GELU -> AvgPool(patch) -> Dropout -> softmax time weights -> per-node affine                   [B, C, F]
      -> scale 1: HGNNP(A1)                      -> mean over electrodes                                [B, F]
      -> anatomical pooling (regions)  -> scale 2: HGNNP(A2) -> flatten                                 [B, R2*F]
      -> anatomical pooling (SM-L/SM-R/Mid/Rest) -> scale 3: HGNNP(A3) -> flatten                       [B, 4*F]
      -> concat -> Linear(hidden) -> BN -> ReLU -> Dropout -> Linear(n_classes)

What is kept from the source: two-kernel temporal front end summed then GELU / AvgPool / Dropout(0.5);
node/hyperedge embeddings -> relu -> softmax(alpha) -> top-k incidence per scale; ``HypergraphConvDHG``
= Linear -> HGNNPConv(Linear -> mean v2v aggregation -> ReLU -> Dropout); mean pooling between scales
(``AnatomicalSpatialPooling``); Linear-BN-ReLU-Dropout-Linear head.

What is changed and why (audit items A1-A9 of the proposal):
  * filters shared across electrodes (A5), width 32 instead of 128 (A1), kernels re-scaled to 250 Hz
    (A4), no LearnableTemporalShift (A4);
  * the incidence matrix is differentiable (soft top-k weights, embeddings init N(0, d^-1/2)) so the
    structure is actually learned (A2);
  * the first hypergraph aggregation acts on the band-passed *signals* before the non-linearity as a
    learnable Laplacian reference, ``(I - gamma A1) S`` (A6);
  * scale hierarchy is anatomical and hemisphere-aware instead of consecutive-index pairs (A3);
  * the read-out keeps node identity at scales 2 and 3 (A7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._common import canonical_channel_name, count_params, dataset_channel_names, squeeze_input

# --------------------------------------------------------------------------- #
# anatomy (hemisphere-aware; names canonicalised with the shared helper, so SHU's T3/T4/T5/T6
# are matched as T7/T8/P7/P8)
# --------------------------------------------------------------------------- #
REGIONS: Dict[str, List[Tuple[str, List[str]]]] = {
    "wbci_shu": [
        ("PF",   ["FPZ", "FP1", "FP2", "AF3", "AF4", "AF7", "AF8"]),
        ("F_L",  ["F1", "F3", "F5", "F7"]),
        ("F_R",  ["F2", "F4", "F6", "F8"]),
        ("FC_L", ["FC1", "FC3", "FC5", "FT7"]),
        ("FC_R", ["FC2", "FC4", "FC6", "FT8"]),
        ("C_L",  ["C1", "C3", "C5", "T7"]),
        ("C_R",  ["C2", "C4", "C6", "T8"]),
        ("CP_L", ["CP1", "CP3", "CP5", "TP7"]),
        ("CP_R", ["CP2", "CP4", "CP6", "TP8"]),
        ("P_L",  ["P3", "P5", "P7", "PO3", "PO5", "PO7"]),
        ("P_R",  ["P4", "P6", "P8", "PO4", "PO6", "PO8"]),
        ("MID",  ["FZ", "FCZ", "CZ"]),
        ("OCC",  ["POZ", "OZ", "O1", "O2"]),
    ],
    "shu": [
        ("PF",   ["FP1", "FP2"]),
        ("F_L",  ["F3", "F7"]),
        ("F_R",  ["F4", "F8"]),
        ("FC_L", ["FC1", "FC5"]),
        ("FC_R", ["FC2", "FC6"]),
        ("C_L",  ["C3", "T7"]),
        ("C_R",  ["C4", "T8"]),
        ("CP_L", ["CP1", "CP5"]),
        ("CP_R", ["CP2", "CP6"]),
        ("P_L",  ["P3", "P7", "PO3"]),
        ("P_R",  ["P4", "P8", "PO4"]),
        ("MID",  ["FZ", "CZ", "PZ"]),
        ("OCC",  ["OZ", "O1", "O2"]),
        ("EAR",  ["A1", "A2"]),
    ],
}

#: Scale-3 super-regions, in this fixed order (the read-out flattens them, so order matters).
SUPER_REGIONS: List[Tuple[str, List[str]]] = [
    ("SM_L", ["FC_L", "C_L", "CP_L"]),
    ("SM_R", ["FC_R", "C_R", "CP_R"]),
    ("MID",  ["MID"]),
    ("REST", ["PF", "F_L", "F_R", "P_L", "P_R", "OCC", "EAR"]),
]

#: Source ``Config`` hyperedge counts at the electrode scale (64-ch row reused for 58 ch).
HYPER_NUM_SCALE1: Dict[int, int] = {32: 30, 58: 40, 64: 40}


def anatomical_pooling(channel_set: str, channel_names: Sequence[str]
                       ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Return ``(P12 [R2, C], P23 [R3, R2], region_names, super_names)`` for a cohort.

    Every electrode of the cohort must be assigned to exactly one region and every table entry
    must exist in the cohort, otherwise this raises (a mismatch means the wrong ``channel_set``).
    Pooling is the source's rule (uniform mean over member nodes), only the membership differs.
    """
    if channel_set not in REGIONS:
        raise ValueError(f"adahgnn-mi: no anatomical table for channel_set '{channel_set}' "
                         f"(known: {sorted(REGIONS)})")
    canon = [canonical_channel_name(c) for c in channel_names]
    if len(set(canon)) != len(canon):
        raise ValueError(f"adahgnn-mi: duplicate electrode names after canonicalisation: {canon}")
    pos = {c: i for i, c in enumerate(canon)}
    table = REGIONS[channel_set]
    assigned: Dict[str, str] = {}
    for region, members in table:
        for m in members:
            m = canonical_channel_name(m)
            if m not in pos:
                raise ValueError(f"adahgnn-mi: electrode {m} of region {region} is not in the "
                                 f"{channel_set} montage ({len(canon)} channels)")
            if m in assigned:
                raise ValueError(f"adahgnn-mi: electrode {m} assigned twice ({assigned[m]}, {region})")
            assigned[m] = region
    missing = [c for c in canon if c not in assigned]
    if missing:
        raise ValueError(f"adahgnn-mi: electrodes without a region in '{channel_set}': {missing}")

    C = len(canon)
    region_names = [r for r, _ in table]
    P12 = torch.zeros(len(table), C)
    for r_idx, (_, members) in enumerate(table):
        idx = [pos[canonical_channel_name(m)] for m in members]
        P12[r_idx, idx] = 1.0 / len(idx)

    r_pos = {r: i for i, r in enumerate(region_names)}
    super_names = [s for s, _ in SUPER_REGIONS]
    P23 = torch.zeros(len(SUPER_REGIONS), len(table))
    seen_regions: List[str] = []
    for s_idx, (_, members) in enumerate(SUPER_REGIONS):
        idx = [r_pos[m] for m in members if m in r_pos]   # EAR only exists for SHU
        if not idx:
            raise ValueError(f"adahgnn-mi: super-region {super_names[s_idx]} is empty for {channel_set}")
        P23[s_idx, idx] = 1.0 / len(idx)
        seen_regions += [m for m in members if m in r_pos]
    if sorted(seen_regions) != sorted(region_names):
        raise ValueError("adahgnn-mi: super-regions do not partition the regions exactly")
    return P12, P23, region_names, super_names


# --------------------------------------------------------------------------- #
# adaptive (differentiable) hypergraph structure
# --------------------------------------------------------------------------- #
def aggregation_from_incidence(H: torch.Tensor) -> torch.Tensor:
    """``A = D_v^{-1} H D_e^{-1} H^T`` for a non-negative incidence ``H [N, E]``.

    With a 0/1 ``H`` this is exactly ``dhg.Hypergraph.v2v(X, aggr="mean")`` (mean vertex->edge
    followed by mean edge->vertex), the aggregation inside ``dhg.nn.HGNNPConv``. Rows of ``A``
    sum to one; empty hyperedges contribute nothing.
    """
    d_v = H.sum(dim=1, keepdim=True)
    Hn = torch.where(d_v > 0, H / d_v.clamp_min(1e-12), torch.zeros_like(H))
    d_e = Hn.sum(dim=0)
    inv_de = torch.where(d_e > 0, 1.0 / d_e.clamp_min(1e-12), torch.zeros_like(d_e))
    return Hn @ (Hn.t() * inv_de[:, None])


class AdaptiveIncidence(nn.Module):
    """Source ``MultiScale_EEG_Hypergraph`` for one scale, made differentiable.

    Same recipe -- node/hyperedge embeddings, ``softmax(alpha * relu(sim))`` over hyperedges, top-k
    per node -- but (i) the kept soft weights are used as the incidence instead of a 0/1 threshold,
    so gradients reach the embeddings, and (ii) embeddings start at ``N(0, d^{-1/2})`` so the
    similarities are O(1) and the softmax is not saturated (the source's ``N(0, 1)`` at d = 128
    gives one-hot rows, i.e. a fixed random partition).
    """

    def __init__(self, n_nodes: int, n_edges: int, dim: int, k: int, alpha: float):
        super().__init__()
        if n_nodes < 1 or n_edges < 1:
            raise ValueError("adahgnn-mi: hypergraph needs >= 1 node and >= 1 hyperedge")
        std = float(dim) ** -0.25
        self.node_emb = nn.Parameter(torch.randn(n_nodes, dim) * std)
        self.edge_emb = nn.Parameter(torch.randn(n_edges, dim) * std)
        self.k = int(min(k, n_edges))
        self.alpha = float(alpha)

    def incidence(self) -> torch.Tensor:
        logits = self.alpha * F.relu(self.node_emb @ self.edge_emb.t())     # [N, E]
        soft = F.softmax(logits, dim=1)
        if self.k < soft.shape[1]:
            _, top = soft.topk(self.k, dim=1)
            mask = torch.zeros_like(soft).scatter_(1, top, 1.0)
            soft = soft * mask
            soft = soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return soft

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        H = self.incidence()
        return aggregation_from_incidence(H), H


class HypergraphBlock(nn.Module):
    """Source ``HypergraphConvDHG`` without the NHC term: Linear -> [HGNNPConv: Linear -> mean
    aggregation -> ReLU -> Dropout]. ``A`` is the shared (batch-independent) aggregation matrix."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(dim, dim)      # HypergraphConvDHG.lin
        self.theta = nn.Linear(dim, dim)    # HGNNPConv.theta
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:   # x [B, N, D], A [N, N]
        h = self.theta(self.lin(x))
        h = torch.matmul(A, h)              # broadcast over the batch
        return self.drop(F.relu(h))


# --------------------------------------------------------------------------- #
# the network
# --------------------------------------------------------------------------- #
@dataclass
class MIConfig:
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: float = 250.0
    channel_set: Optional[str] = None       # 'wbci_shu' | 'shu' (per_dataset in adahgnn.yaml)
    n_filters: int = 32                     # F: shared temporal filters (= node feature dim)
    kernel_sizes: Tuple[int, int] = (125, 63)   # 0.5 s / 0.25 s at 250 Hz (source: 63/31 at 128 Hz)
    patch_size: int = 16                    # 64 ms (source: 8 at 128 Hz = 62.5 ms)
    embed_dim: int = 32                     # structure embeddings
    k: int = 5                              # hyperedges kept per node
    alpha: float = 3.0                      # softmax temperature (source)
    hyper_num_scale1: Optional[int] = None  # None -> HYPER_NUM_SCALE1[n_channels]
    hyper_num_scale2: int = 8
    hyper_num_scale3: int = 3
    dropout: float = 0.5
    laplacian: bool = True                  # signal-domain (I - gamma A1) stage
    temporal_attention: bool = True         # softmax weights over patches (init uniform = mean)
    hidden: int = 128
    mixup: bool = False                     # same-class cross-trial Dirichlet mixup (training only)
    mixup_prob: float = 0.5
    mixup_alpha: float = 2.0
    mixup_k: int = 3
    channel_names: Optional[List[str]] = field(default=None, repr=False)   # override for tests


MI_FIELDS: Tuple[str, ...] = (
    "channel_set", "n_filters", "kernel_sizes", "patch_size", "embed_dim", "k", "alpha",
    "hyper_num_scale1", "hyper_num_scale2", "hyper_num_scale3", "dropout", "laplacian",
    "temporal_attention", "hidden", "mixup", "mixup_prob", "mixup_alpha", "mixup_k", "channel_names",
)


class AdaHGNNMI(nn.Module):
    """See the module docstring. Contract: ``forward(x[B,C,T]) -> {logits, features, confidence}``."""

    def __init__(self, cfg: MIConfig):
        super().__init__()
        self.cfg = cfg
        C, T, Fd = int(cfg.n_channels), int(cfg.n_times), int(cfg.n_filters)
        if len(cfg.kernel_sizes) != 2:
            raise ValueError("adahgnn-mi: kernel_sizes must have two entries")
        k1, k2 = (int(k) for k in cfg.kernel_sizes)
        if k1 % 2 == 0 or k2 % 2 == 0:
            raise ValueError("adahgnn-mi: kernel sizes must be odd ('same' padding)")
        if cfg.patch_size < 1 or T // int(cfg.patch_size) < 1:
            raise ValueError("adahgnn-mi: patch_size out of range")
        if not cfg.channel_set and not cfg.channel_names:
            raise ValueError("adahgnn-mi needs params['channel_set'] ('wbci_shu' or 'shu'); it is "
                             "injected by per_dataset.<data.name>.channel_set in adahgnn.yaml")
        names = list(cfg.channel_names) if cfg.channel_names else dataset_channel_names(cfg.channel_set)
        if len(names) != C:
            raise ValueError(f"adahgnn-mi: channel_set '{cfg.channel_set}' lists {len(names)} "
                             f"electrodes but data has n_channels={C}")
        cs = cfg.channel_set or ("wbci_shu" if C == 58 else "shu")
        P12, P23, self.region_names, self.super_names = anatomical_pooling(cs, names)
        self.register_buffer("pool_12", P12)
        self.register_buffer("pool_23", P23)
        R2, R3 = P12.shape[0], P23.shape[0]
        self.n_patches = T // int(cfg.patch_size)

        # 1. shared temporal filter bank (source: two depthwise branches summed -> GELU)
        self.conv_a = nn.Conv2d(1, Fd, (1, k1), padding=(0, k1 // 2), bias=False)
        self.conv_b = nn.Conv2d(1, Fd, (1, k2), padding=(0, k2 // 2), bias=False)
        self.bn_in = nn.BatchNorm2d(Fd)

        # 2. structures (one per scale, as in the source)
        e1 = int(cfg.hyper_num_scale1 or HYPER_NUM_SCALE1.get(C, max(8, round(0.7 * C))))
        self.hg1 = AdaptiveIncidence(C, e1, cfg.embed_dim, cfg.k, cfg.alpha)
        self.hg2 = AdaptiveIncidence(R2, int(cfg.hyper_num_scale2), cfg.embed_dim, cfg.k, cfg.alpha)
        self.hg3 = AdaptiveIncidence(R3, int(cfg.hyper_num_scale3), cfg.embed_dim, cfg.k, cfg.alpha)
        self.hyper_nums = [e1, int(cfg.hyper_num_scale2), int(cfg.hyper_num_scale3)]

        # 3. Laplacian reference gain per electrode, gamma = sigmoid(0) = 0.5 at init
        self.lap_logit = nn.Parameter(torch.zeros(C)) if cfg.laplacian else None

        # 4. power read-out
        self.pool_t = nn.AvgPool2d((1, int(cfg.patch_size)), stride=(1, int(cfg.patch_size)))
        self.drop_in = nn.Dropout(cfg.dropout)
        self.time_logits = nn.Parameter(torch.zeros(self.n_patches)) if cfg.temporal_attention else None
        self.node_scale = nn.Parameter(torch.ones(C, Fd))
        self.node_bias = nn.Parameter(torch.zeros(C, Fd))

        # 5. multi-scale hypergraph convolutions
        self.block1 = HypergraphBlock(Fd, cfg.dropout)
        self.block2 = HypergraphBlock(Fd, cfg.dropout)
        self.block3 = HypergraphBlock(Fd, cfg.dropout)

        # 6. head (source: Linear -> BN -> ReLU -> Dropout -> Linear)
        self.feature_dim = Fd + R2 * Fd + R3 * Fd
        self.head_lin = nn.Linear(self.feature_dim, int(cfg.hidden))
        self.head_bn = nn.BatchNorm1d(int(cfg.hidden))
        self.head_drop = nn.Dropout(cfg.dropout)
        self.head_out = nn.Linear(int(cfg.hidden), int(cfg.n_classes))

        self.uses_custom_loss = bool(cfg.mixup)

    # ---- helpers --------------------------------------------------------------------------
    def structures(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return {"scale1": self.hg1(), "scale2": self.hg2(), "scale3": self.hg3()}

    def gamma(self) -> Optional[torch.Tensor]:
        return torch.sigmoid(self.lap_logit) if self.lap_logit is not None else None

    def _head_norm(self, h: torch.Tensor) -> torch.Tensor:
        # drop_last=False in the protocol loader: a ragged final batch of one trial would crash
        # BatchNorm1d in training mode; fall back to running statistics for that single batch.
        bn = self.head_bn
        if self.training and h.shape[0] < 2:
            return F.batch_norm(h, bn.running_mean, bn.running_var, bn.weight, bn.bias,
                                False, 0.0, bn.eps)
        return bn(h)

    # ---- forward --------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        x = squeeze_input(x)
        B, C, T = x.shape
        if C != self.cfg.n_channels or T != self.cfg.n_times:
            raise ValueError(f"adahgnn-mi was built for [B, {self.cfg.n_channels}, {self.cfg.n_times}], "
                             f"got {tuple(x.shape)}")
        A1, _ = self.hg1()
        A2, _ = self.hg2()
        A3, _ = self.hg3()

        x4 = x.unsqueeze(1)                                             # [B, 1, C, T]
        s = self.bn_in(self.conv_a(x4) + self.conv_b(x4))               # [B, F, C, T]
        if self.lap_logit is not None:
            Fd = s.shape[1]
            flat = s.reshape(B * Fd, C, T)
            smooth = torch.matmul(A1, flat).reshape(B, Fd, C, T)        # A1 @ S over electrodes
            s = s - torch.sigmoid(self.lap_logit).view(1, 1, C, 1) * smooth
        p = self.drop_in(self.pool_t(F.gelu(s)))                        # [B, F, C, P]
        if self.time_logits is not None:
            w = torch.softmax(self.time_logits, dim=0)
            feat = torch.einsum("bfcp,p->bfc", p, w)
        else:
            feat = p.mean(dim=-1)
        feat = feat.permute(0, 2, 1) * self.node_scale + self.node_bias   # [B, C, F]

        x1 = self.block1(feat, A1)                                       # [B, C, F]
        r1 = x1.mean(dim=1)                                              # [B, F]
        x2 = self.block2(torch.matmul(self.pool_12, x1), A2)             # [B, R2, F]
        x3 = self.block3(torch.matmul(self.pool_23, x2), A3)             # [B, R3, F]
        z = torch.cat([r1, x2.flatten(1), x3.flatten(1)], dim=1)         # [B, feature_dim]

        h = self.head_drop(F.relu(self._head_norm(self.head_lin(z))))
        logits = self.head_out(h)
        return {"logits": logits, "features": z, "confidence": None}

    # ---- optional same-class mixup (training only; trainer hook) ----------------------------
    @staticmethod
    def _same_class_partners(y: torch.Tensor, n_partners: int) -> torch.Tensor:
        """``[n_partners, B]`` indices of random *other* trials with the same label (self if the
        class has a single trial in the batch)."""
        B = y.numel()
        partners = torch.arange(B, device=y.device).unsqueeze(0).repeat(n_partners, 1)
        for c in torch.unique(y):
            idx = (y == c).nonzero(as_tuple=True)[0]
            n_c = idx.numel()
            if n_c < 2:
                continue
            pos = torch.arange(n_c, device=y.device)
            for j in range(n_partners):
                offs = torch.randint(1, n_c, (n_c,), device=y.device)   # never 0 -> never self
                partners[j, idx] = idx[(pos + offs) % n_c]
        return partners

    def mix_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Source ``CrossSubjectMixupDataset`` (Dirichlet(alpha) over ``mixup_k`` trials, applied
        with ``mixup_prob``) restricted to **same-label** partners, then re-z-scored per
        (trial, channel) so mixed trials keep the input statistics of real ones."""
        cfg = self.cfg
        B = x.shape[0]
        n_part = max(int(cfg.mixup_k) - 1, 1)
        apply = torch.rand(B, device=x.device) < float(cfg.mixup_prob)
        if not bool(apply.any()):
            return x, 0.0
        partners = self._same_class_partners(y, n_part)
        conc = torch.full((int(cfg.mixup_k),), float(cfg.mixup_alpha), device=x.device)
        lam = torch.distributions.Dirichlet(conc).sample((B,))            # [B, k]
        one_hot = torch.zeros_like(lam)
        one_hot[:, 0] = 1.0
        lam = torch.where(apply.unsqueeze(1), lam, one_hot)
        mixed = lam[:, 0].view(B, 1, 1) * x
        for j in range(n_part):
            mixed = mixed + lam[:, j + 1].view(B, 1, 1) * x[partners[j]]
        mu = mixed.mean(dim=-1, keepdim=True)
        sd = mixed.std(dim=-1, keepdim=True).clamp_min(1e-6)
        mixed = (mixed - mu) / sd
        mixed = torch.where(apply.view(B, 1, 1), mixed, x)
        return mixed, float(apply.float().mean().item())

    def training_step(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 0, **_) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = squeeze_input(x)
        frac = 0.0
        if self.cfg.mixup and self.training:
            with torch.no_grad():
                x, frac = self.mix_batch(x, y)
        loss = F.cross_entropy(self(x)["logits"], y)
        return loss, {"ce": float(loss.detach().item()), "mix_frac": frac}

    # ---- description -------------------------------------------------------------------------
    def describe(self) -> Dict[str, object]:
        cfg = self.cfg
        g = self.gamma()
        return {
            "name": "adahgnn-mi",
            "arch": "mi",
            "source": "EEG_AdaMSHyper (DeepHypergraph project_V4, 2026-01-16) adapted for MI; "
                      "docs/reports/PROPOSAL_adahgnn_mi_adaptation_2026-09-20.md",
            "n_params": count_params(self),
            "n_trainable_params": count_params(self, trainable_only=True),
            "feature_dim": int(self.feature_dim),
            "n_filters": int(cfg.n_filters),
            "kernel_sizes": [int(k) for k in cfg.kernel_sizes],
            "patch_size": int(cfg.patch_size),
            "n_patches": int(self.n_patches),
            "embed_dim": int(cfg.embed_dim),
            "k": int(cfg.k),
            "alpha": float(cfg.alpha),
            "hyper_nums": [int(e) for e in self.hyper_nums],
            "scale_node_nums": [int(cfg.n_channels), len(self.region_names), len(self.super_names)],
            "regions": list(self.region_names),
            "super_regions": list(self.super_names),
            "laplacian": bool(cfg.laplacian),
            "gamma_mean": (float(g.mean().item()) if g is not None else None),
            "temporal_attention": bool(cfg.temporal_attention),
            "dropout": float(cfg.dropout),
            "hidden": int(cfg.hidden),
            "mixup": bool(cfg.mixup),
            "mixup_prob": float(cfg.mixup_prob),
            "mixup_alpha": float(cfg.mixup_alpha),
            "mixup_k": int(cfg.mixup_k),
            "dann": False,
            "nhc_constraint": False,
            "pretrained_weights": None,
            "channel_set": cfg.channel_set,
        }


def build_mi(*, n_channels: int, n_times: int, n_classes: int, sfreq: float = 250.0,
             params: Optional[Dict] = None) -> AdaHGNNMI:
    p = dict(params or {})
    if "kernel_sizes" in p and p["kernel_sizes"] is not None:
        p["kernel_sizes"] = tuple(int(k) for k in p["kernel_sizes"])
    cfg = MIConfig(n_channels=int(n_channels), n_times=int(n_times), n_classes=int(n_classes),
                   sfreq=float(sfreq), **p)
    return AdaHGNNMI(cfg)


__all__ = ["REGIONS", "SUPER_REGIONS", "HYPER_NUM_SCALE1", "anatomical_pooling",
           "aggregation_from_incidence", "AdaptiveIncidence", "HypergraphBlock",
           "MIConfig", "MI_FIELDS", "AdaHGNNMI", "build_mi"]
