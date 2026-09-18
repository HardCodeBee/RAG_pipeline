"""Frozen-term attention for pre-retrieval utility routing.

The model receives query term states and frozen corpus statistics only.
It neither scores documents nor changes either downstream retriever.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TermRouter(nn.Module):
    def __init__(self, arm, dimension=384, rank=8):
        super().__init__()
        if arm not in ('U', 'F', 'N', 'S', 'T'):
            raise ValueError(arm)
        self.arm = arm
        self.head = nn.Linear(2 * dimension, 1)
        if arm in ('N', 'S', 'T'):
            self.project = nn.Linear(dimension, rank)
            self.semantic = nn.Parameter(torch.empty(rank))
            self.interaction = nn.Parameter(torch.empty(rank))
            nn.init.xavier_uniform_(self.project.weight)
            nn.init.zeros_(self.project.bias)
            nn.init.normal_(self.semantic, std=.02)
            nn.init.normal_(self.interaction, std=.02)

    def pool(self, batch, return_attention=False):
        mask, hidden = batch['mask'], batch['hidden']
        if self.arm in ('U', 'F'):
            weights = mask.float() if self.arm == 'U' else batch['idf'] * mask
            attention = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        else:
            state = torch.tanh(self.project(batch['normalized']))
            z = torch.zeros_like(batch['z']) if self.arm == 'N' else batch['z']
            energy = state @ self.semantic + z * (state @ self.interaction)
            attention = energy.masked_fill(~mask, -torch.inf).softmax(-1)
        raw = (attention.unsqueeze(-1) * hidden).sum(1)
        pooled = F.normalize(raw, dim=-1, eps=1e-12)
        # The shared fallback is based on input availability, never on outcomes.
        pooled = torch.where(batch['fallback'][:, None], batch['M6'], pooled)
        if not torch.isfinite(pooled).all():
            raise FloatingPointError('Nonfinite pooled representation')
        if return_attention:
            return pooled, attention
        return pooled

    def features(self, batch):
        return torch.cat((batch['M6'], self.pool(batch)), dim=-1)

    def forward(self, batch):
        return self.head(self.features(batch)).squeeze(-1)

    def penalty(self, head_l2, attention_l2):
        head = self.head.weight.square().sum() * head_l2 / 2
        if self.arm in ('N', 'S', 'T'):
            gate = self.project.weight.square().sum() + self.semantic.square().sum()
            if self.arm != 'N':
                gate = gate + self.interaction.square().sum()
            return head + attention_l2 * gate / 2
        return head


def make_batch(cache, indices, scaled_idf, *, shuffled=False, device='cuda'):
    """Pad only the selected queries; term states stay immutable in the cache."""
    indices = np.asarray(indices, dtype=np.int64)
    ptr = cache['term_indptr']
    starts, ends = ptr[indices], ptr[indices + 1]
    lengths = ends - starts
    if not len(indices) or (lengths <= 0).any():
        raise ValueError('Expected nonempty queries with the shared fallback row')
    positions = np.arange(int(lengths.max()))[None, :]
    mask = positions < lengths[:, None]
    rows = starts[:, None] + np.minimum(positions, lengths[:, None] - 1)
    idf_rows = cache['permutation'][rows] if shuffled else rows
    hidden = torch.tensor(np.asarray(cache['term_hidden'][rows]), device=device)
    hidden = hidden * torch.tensor(mask, device=device).unsqueeze(-1)
    return {
        'hidden': hidden,
        'normalized': F.normalize(hidden, dim=-1, eps=1e-12),
        'mask': torch.tensor(mask, device=device),
        'idf': torch.tensor(np.asarray(cache['term_idf'][idf_rows]), device=device),
        'z': torch.tensor(np.asarray(scaled_idf[idf_rows]), device=device),
        'M6': torch.tensor(np.asarray(cache['M6'][indices]), device=device),
        'fallback': torch.tensor(np.asarray(cache['fallback'][indices]), device=device),
    }
