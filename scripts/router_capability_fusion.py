"""Query-only capability-fusion prototype; no data loading or training runner.

The caller supplies a local BERT and the M6 head from its OWN training split.
This module takes ownership of the embeddings and first ``depth`` BERT layers.
It freezes their weights and disables their dropout, but retains the gradient
path through those layers when learning adapters. It does not mutate the
original BERT's layer list or download weights.

This is custom representation-level fusion inspired by staged task composition,
not a reproduction of AdapterFusion's layerwise attention or R3AG. Each branch
runs its own path through shared frozen BERT weights. Fusion therefore requires
three encoder passes (two for utility-only), not one pass at the cost of M6.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def ordinary_token_mean(hidden, attention_mask, special_tokens_mask):
    """Historical M6 pooling: FP32 ordinary-token mean, CLS fallback, L2 norm."""
    mask = attention_mask.bool() & ~special_tokens_mask.bool()
    count = mask.sum(dim=1)
    pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / count.clamp_min(1).float().unsqueeze(1)
    pooled = torch.where((count == 0).unsqueeze(1), hidden[:, 0].float(), pooled)
    return F.normalize(pooled, p=2, dim=1)


class ResidualAdapter(nn.Module):
    """A tokenwise bottleneck after a BERT block, initially the identity."""

    def __init__(self, width, bottleneck):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden):
        return hidden + self.up(F.gelu(self.down(hidden)))


class UtilityFusion(nn.Module):
    """Residual utility logits; coverage-head logits never enter this module.

    Static fusion uses equal weights and separately learned utility readouts.
    Conditional fusion changes only the gate: sigmoid(linear(base_query)).
    Both residual readouts start at zero, preserving the M6 prediction initially.
    The gate has small random weights: a zero gate plus identical input features
    would keep the two zero-initialized readouts symmetric and prevent gating.
    Gate weights describe expert mixing, not BM25/Dense action probabilities.
    """

    def __init__(self, width, mode):
        super().__init__()
        if mode not in {"utility_only", "static", "conditional"}:
            raise ValueError(f"Unknown fusion mode: {mode}")
        self.mode = mode
        names = ("utility",) if mode == "utility_only" else ("auxiliary", "utility")
        self.readouts = nn.ModuleDict({name: nn.Linear(width, 1) for name in names})
        for head in self.readouts.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.gate = nn.Linear(width, 1) if mode == "conditional" else None
        if self.gate is not None:
            nn.init.normal_(self.gate.weight, std=0.02)
            nn.init.zeros_(self.gate.bias)

    def auxiliary_weight(self, base_query):
        if self.mode == "utility_only":
            return base_query.new_zeros(base_query.shape[0])
        if self.gate is None:
            return base_query.new_full((base_query.shape[0],), 0.5)
        return self.gate(base_query).squeeze(-1).sigmoid()

    def forward(self, base_query, utility_query, auxiliary_query=None):
        utility = self.readouts["utility"](utility_query).squeeze(-1)
        if self.mode == "utility_only":
            return utility
        if auxiliary_query is None:
            raise ValueError("Two-branch fusion requires the auxiliary representation")
        auxiliary = self.readouts["auxiliary"](auxiliary_query).squeeze(-1)
        weight = self.auxiliary_weight(base_query)
        return weight * auxiliary + (1 - weight) * utility


class CapabilityRouter(nn.Module):
    """Separate task adapters followed by a frozen-branch fusion stage.

    ``branch='auxiliary'`` returns two logits in [BM25, Dense] order. The planned
    retrieval task fits top-5 supporting-page coverage as two soft BCE targets.
    An architecture-matched two-utility control can instead fit their difference
    with the SAME utility-weighted BCE used for the single utility logit.
    ``branch='utility'`` returns one logit; positive means switch to BM25.
    ``branch=None`` returns the fused utility logit with the same action sign.

    Labels and losses deliberately belong to the future experiment runner, not
    this prediction interface. Loading weights does not restore training stage:
    call set_stage explicitly and construct a NEW optimizer for that stage.
    A resumable runner must also save optimizer/scheduler/RNG state and split IDs.
    """

    def __init__(self, bert, m6_head, *, depth=6, bottleneck=16, fusion="conditional"):
        super().__init__()
        config = bert.config
        if (config.model_type != "bert" or config.is_decoder
                or config.add_cross_attention or config.position_embedding_type != "absolute"):
            raise ValueError("This prototype supports absolute-position BERT encoders only")
        if not 1 <= depth <= len(bert.encoder.layer) or bottleneck < 1:
            raise ValueError("Invalid encoder depth or adapter bottleneck")
        width = config.hidden_size
        if (m6_head.in_features, m6_head.out_features) != (width, 1):
            raise ValueError("M6 head must map the BERT width to one utility logit")

        self.embeddings = bert.embeddings
        self.layers = nn.ModuleList(list(bert.encoder.layer)[:depth])
        self.baseline_head = deepcopy(m6_head)
        self.adapters = nn.ModuleDict({
            name: nn.ModuleList([ResidualAdapter(width, bottleneck) for _ in range(depth)])
            for name in ("auxiliary", "utility")
        })
        self.auxiliary_head = nn.Linear(width, 2)
        self.utility_head = deepcopy(m6_head)
        self.fusion = UtilityFusion(width, fusion)
        self.set_stage("inference")
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        # eval() closes pretrained dropout; it does NOT cut adapter gradients.
        self.embeddings.eval()
        self.layers.eval()
        return self

    def set_stage(self, stage):
        if stage not in {"auxiliary", "utility", "fusion", "inference"}:
            raise ValueError(f"Unknown training stage: {stage}")
        self.requires_grad_(False)
        self.zero_grad(set_to_none=True)
        if stage in {"auxiliary", "utility"}:
            self.adapters[stage].requires_grad_(True)
            getattr(self, f"{stage}_head").requires_grad_(True)
        elif stage == "fusion":
            self.fusion.requires_grad_(True)
        self.stage = stage
        return tuple(parameter for parameter in self.parameters() if parameter.requires_grad)

    def encode(self, input_ids, attention_mask, special_tokens_mask,
               token_type_ids=None, *, branch=None):
        if branch is not None and branch not in self.adapters:
            raise ValueError(f"Unknown capability branch: {branch}")
        hidden = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        # Broadcastable additive padding mask, accepted by eager and SDPA BERT.
        extended_mask = attention_mask[:, None, None, :].to(dtype=hidden.dtype)
        extended_mask = (1 - extended_mask) * torch.finfo(hidden.dtype).min
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, attention_mask=extended_mask)[0]
            if branch is not None:
                hidden = self.adapters[branch][index](hidden)
        return ordinary_token_mean(hidden, attention_mask, special_tokens_mask)

    def forward(self, input_ids, attention_mask, special_tokens_mask,
                token_type_ids=None, *, branch=None):
        tokens = (input_ids, attention_mask, special_tokens_mask, token_type_ids)
        if branch is not None:
            query = self.encode(*tokens, branch=branch)
            logits = getattr(self, f"{branch}_head")(query)
            return logits.squeeze(-1) if branch == "utility" else logits

        base_query = self.encode(*tokens)
        utility_query = self.encode(*tokens, branch="utility")
        auxiliary_query = None
        if self.fusion.mode != "utility_only":
            auxiliary_query = self.encode(*tokens, branch="auxiliary")
        residual = self.fusion(base_query, utility_query, auxiliary_query)
        return self.baseline_head(base_query).squeeze(-1) + residual
