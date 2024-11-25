import torch
import torch.nn as nn
import numpy as np


def create_sinusoidal_embeddings(n_pos, dim, out):
    with torch.no_grad():
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class Embeddings(nn.Module):
    def __init__(
            self, d_model, language_len, vision_len, dropout, sinusoidal_pos_embds, d_pos=128
    ):
        super().__init__()
        max_position_embeddings = language_len + vision_len
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        # self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_model,
                # out=self.position_embeddings.weight,
                out=self.position_embeddings.weight,
            )
        # self.modality_embedding = nn.Embedding(2, d_model)
        self.language_len = language_len
        self.vision_len = vision_len
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        seq_length = embeddings.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=embeddings.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(embeddings[:, :, 0])  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        # if self.language_len != 0:
        #     modality_embeddings = self.modality_embedding(
        #         torch.tensor(
        #             [0] * self.language_len + [1] * self.vision_len, dtype=torch.long
        #         ).to(embeddings.device)
        #     )
        #     embeddings = (
        #         embeddings + position_embeddings + modality_embeddings
        #     )  # (bs, max_seq_length, dim)
        # else:
        embeddings = embeddings + position_embeddings  # (bs, max_seq_length, dim)

        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)

        return embeddings

