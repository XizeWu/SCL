import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional
import copy



def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, padding_mask: torch.Tensor, tau: float = 1,  hard: bool = False, dim: int = -1) -> torch.Tensor:
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)

    if padding_mask is not None:
        gumbels_new = gumbels.masked_fill(padding_mask, -float("inf"))  # (bs, n_heads, gumbels_seq, 1)

    y_soft = gumbels_new.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class Q2V_DecoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False,
                 hard=True, gumbel=True, gumbel_tau=1.0, sum_assign=False, assign_eps=1e-9, q_length=32, **kwargs):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nheads, dropout=dropout)
        self.cross_attn = Multihead_CrossAttention_Q2V(d_model, nheads, dropout=dropout, hard=hard, gumbel=gumbel,
                                                   gumbel_tau=gumbel_tau, sum_assign=sum_assign, assign_eps=assign_eps,
                                                   q_length=q_length)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, c_att, Wword = self.cross_attn(query=self.with_pos_embed(tgt, query_pos),
                                      key=self.with_pos_embed(memory, pos),
                                      value=memory,
                                      attn_mask=memory_mask,
                                      key_padding_mask=memory_key_padding_mask,
                                      output_attentions=True)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, c_att, Wword


class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_lin = nn.Linear(in_features=self.dim, out_features=self.dim)

    def forward(self, query, key, value, key_padding_mask, attn_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        key_padding_mask: torch.tensor(bs, seq_length)
        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)   # [bs, nheads, seq_len, dim_head]

        def unshape(x):
            """ group heads """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        if key_padding_mask is not None:
            key_padding_mask = ((~key_padding_mask).view(mask_reshp).expand_as(scores))  # (bs, n_heads, q_length, k_length)
            scores = scores.masked_fill(key_padding_mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if attn_mask is not None:
            weights = weights * attn_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return context, weights.mean(1)
        else:
            return context

class Multihead_CrossAttention_Q2V(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1, hard=True, gumbel=True, gumbel_tau=1.0, sum_assign=False, assign_eps=1e-6, q_length=32):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_lin = nn.Linear(in_features=self.dim, out_features=self.dim)

        self.fc_k_dim = nn.Linear(in_features=self.dim, out_features=self.n_heads)

        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

        self.logit_gauss = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(q_length, 2),
            nn.Sigmoid(),
        )

    def get_attn(self, attn, attn_padding_mask=None, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, attn_padding_mask, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if attn_padding_mask is not None:
                attn = attn.masked_fill(attn_padding_mask, -float("inf"))  # (bs, n_heads, gumbels_seq, 1)

            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn

    def generate_gauss_weight(self, center, width, max_len):
        weight = torch.linspace(0, 1, max_len)           # shape:[32]
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)  # expand to [bs, 32]
        center = center.unsqueeze(-1)                               # shape: [bs, 1]
        width = width.unsqueeze(-1).clamp(1e-2) / 9        # shape: [bs, 1]

        w = 0.3989422804014327  # 1/(math.sqrt(2*math.pi))
        weight = w / width * torch.exp(-(weight - center) ** 2 / (2 * width ** 2))  # [bs, 32]

        # [bs, 32]
        return weight / weight.max(dim=-1, keepdim=True)[0]

    def forward(self, query, key, value, key_padding_mask, attn_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        key_padding_mask: torch.tensor(bs, seq_length)
        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)

        if key_padding_mask is not None:
            key_padding_mask2 = ((~key_padding_mask).view(mask_reshp).expand_as(scores))  # (bs, n_heads, q_length, k_length)
            scores = scores.masked_fill(key_padding_mask2, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)

        # ### compute each token weight of tensor k
        new_k = self.fc_k_dim(key).view(bs, k_length, self.n_heads, 1).transpose(1, 2)   # (bs, n_heads, k_length, 1)
        key_padding_mask3 = ((~key_padding_mask).unsqueeze(dim=-1).unsqueeze(dim=1).expand_as(new_k)) \
            if key_padding_mask is not None \
            else None
        weights_token_K = self.get_attn(new_k, key_padding_mask3)     # (bs, n_heads, k_length, 1)  {0, 1}

        weights_token_K = weights_token_K.sum(dim=1, keepdim=True) / (weights_token_K.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True) + 1e-6)
        weights_token_K = weights_token_K.repeat_interleave(self.n_heads, dim=1)

        new_weights = torch.matmul(weights, weights_token_K).view(bs*self.n_heads, q_length)         # (bs, n_heads, q_length, 1) -> (bs, n_heads, q_length)
        logits_cw = self.logit_gauss(new_weights).view(-1, 2)                   # [bs*n_heads, 2]
        gauss_c, gauss_w = logits_cw[:, 0], logits_cw[:, 1]                     # [bs*n_heads], [bs*n_heads]
        gauss_weight = self.generate_gauss_weight(gauss_c, gauss_w, q_length).view(bs, self.n_heads, q_length)
        gauss_weight = nn.Softmax(dim=-1)(gauss_weight)                         # [bs, n_heads, q_length]

        gauss_weight = gauss_weight.unsqueeze(-1).expand(-1, -1, -1, k_length).reshape(*weights.shape)
        weights = weights * (gauss_weight + 1e-10)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if attn_mask is not None:
            weights = weights * attn_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return context, weights.mean(1), weights_token_K.mean(dim=1).squeeze()
        else:
            return context


class V2Q_CAttLayer(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False, **kwargs):
        super().__init__()
        self.multihead_attn = MultiheadAttention(d_model, nheads, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        tgt2, att = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   output_attentions=True)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att

# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")