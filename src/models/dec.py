"""Deep Embedded Clustering components (Xie et al., 2016; Enguehard et al., 2019).

Notation (matches docs/phase1_planning.md):
    z_i   embedding of sample i            (N, D)
    mu_j  cluster centre j                 (K, D)
    q_ij  soft assignment, Student's t     (N, K)
    p_ij  sharpened target distribution    (N, K)
"""
from __future__ import annotations

import torch
from torch import nn


class ClusteringLayer(nn.Module):
    """Learnable centres + Student's t soft assignment:

        q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2)
               / sum_j' (1 + ||z_i - mu_j'||^2 / alpha)^(-(alpha+1)/2)

    In SSLDEC there is one centre per class, so column j of q is the
    probability of class j. Centres must be initialised with
    `init_centers` (class means of labelled embeddings) before training.
    """

    def __init__(self, num_clusters: int, embedding_dim: int, alpha: float = 1.0):
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_clusters, embedding_dim) * 0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.centers.shape[1]:
            raise ValueError(f"expected z of shape (N, {self.centers.shape[1]}), got {tuple(z.shape)}")
        z = z.float()                                   # distances in fp32 even under AMP
        # explicit difference (not cdist): cdist's gradient is NaN at zero distance
        d2 = (z.unsqueeze(1) - self.centers.float().unsqueeze(0)).pow(2).sum(dim=2)   # (N, K)
        kernel = (1.0 + d2 / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        return kernel / kernel.sum(dim=1, keepdim=True)

    @torch.no_grad()
    def init_centers(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """mu_j = mean of labelled embeddings of class j. This anchors cluster j
        to class j, so cluster IDs are never arbitrary."""
        k = self.centers.shape[0]
        missing = [j for j in range(k) if not (y == j).any()]
        if missing:
            raise ValueError(f"no labelled samples for classes {missing}; cannot initialise centres")
        means = torch.stack([z[y == j].float().mean(dim=0) for j in range(k)])
        self.centers.copy_(means.to(self.centers.device))


@torch.no_grad()
def target_distribution(q: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
    """Sharpened target:

        f_j  = sum_i q_ij
        p_ij = (q_ij^2 / f_j * pi_j) / sum_j' (q_ij'^2 / f_j' * pi_j')

    With prior=None (pi uniform) this is the original DEC target, which pushes
    clusters towards equal size. Project modification: passing the class prior
    pi estimated from the labelled set makes the target cluster sizes follow
    pi instead, which matters for imbalanced medical data.

    q must cover the whole (sub)set used for training so f_j is a dataset-level
    frequency, not a batch statistic.
    """
    if q.ndim != 2:
        raise ValueError(f"q must be (N, K), got {tuple(q.shape)}")
    weight = q.pow(2) / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
    if prior is not None:
        if prior.shape != (q.shape[1],):
            raise ValueError(f"prior must have shape ({q.shape[1]},)")
        weight = weight * prior.to(q)
    return weight / weight.sum(dim=1, keepdim=True)
