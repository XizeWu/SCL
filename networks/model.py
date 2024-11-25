from builtins import print, tuple
from signal import pause
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

import os
import sys

sys.path.append('../')
from einops import rearrange, repeat
from networks.multimodal_transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, \
    TransformerDecoder, CaAttLayer
from networks.AssignAttention import Q2V_DecoderLayer, V2Q_CAttLayer
from networks.position_encoding import PositionEmbeddingSine1D
from transformers import AutoModel, AutoTokenizer
from networks.embedding_layer import Embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)


class VideoQAmodel(nn.Module):
    def __init__(self, text_encoder_type="roberta-base", freeze_text_encoder=False, n_query=5, hard_eval=False, **kwargs):
        super(VideoQAmodel, self).__init__()
        self.d_model = kwargs['d_model']
        encoder_dropout = kwargs['encoder_dropout']
        self.num_encoder_layers = kwargs['num_encoder_layers']
        self.mc = n_query
        self.hard_eval = hard_eval
        self.max_word_qa = kwargs["max_word_qa"]
        self.hparam1D = 1
        self.hparam_X, self.hparam_Y, self.hparam_rho = 2, 2, 1
        self.prop_num = 1
        self.max_word_q = kwargs["max_word_q"]
        self.theta = kwargs["theta"]
        self.numFrames = kwargs['numF']

        self.G = kwargs["G"]

        # text encoder
        self.text_encoder = AutoModel.from_pretrained(text_encoder_type)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)

        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.FC_ans = nn.Linear(self.d_model, self.d_model)

        # position embedding
        self.pos_encoder_1d = PositionEmbeddingSine1D()
        self.pos_encoder_2d = self.build_2d_sincos_position_embedding()
        self.position_v = Embeddings(self.d_model, 0, self.numFrames, encoder_dropout, True)

        self.position_region = Embeddings(self.G*self.G, 0, self.numFrames, encoder_dropout, True)

        self.obj_resize = FeatureResizer(input_feat_size=768, output_feat_size=self.d_model, dropout=kwargs['dropout'])

        self.grid_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), self.num_encoder_layers, norm=nn.LayerNorm(self.d_model))
        self.frame_encoder = TransformerEncoder(TransformerEncoderLayer(**kwargs), self.num_encoder_layers, norm=nn.LayerNorm(self.d_model))

        self.uni_region = TransformerEncoderLayer(d_model=self.G*self.G, nheads=6, dim_feedforward=64, dropout=0.1, activation="relu")

        self.grid_encoder = TransformerEncoder(TransformerEncoderLayer(**kwargs), self.num_encoder_layers,
                                               norm=nn.LayerNorm(self.d_model))
        self.Spatial_Grid = TransformerEncoder(TransformerEncoderLayer(**kwargs), self.num_encoder_layers,
                                               norm=nn.LayerNorm(self.d_model))
        self.Temporal_Grid = TransformerEncoder(TransformerEncoderLayer(**kwargs), self.num_encoder_layers,
                                                norm=nn.LayerNorm(self.d_model))

        self.q2v_decoder = Q2V_DecoderLayer(d_model=self.d_model, nheads=kwargs["nheads"], hard=True, gumbel=True,
                                            gumbel_tau=1.0, sum_assign=False, assign_eps=1e-12, q_length=self.numFrames)
        self.v2q_catt = V2Q_CAttLayer(d_model=self.d_model, nheads=kwargs["nheads"])

        self.q2v_decoder2 = Q2V_DecoderLayer(d_model=self.d_model, nheads=kwargs["nheads"], hard=True, gumbel=True,
                                             gumbel_tau=1.0, sum_assign=False, assign_eps=1e-12, q_length=self.numFrames)
        self.v2q_catt2 = V2Q_CAttLayer(d_model=self.d_model, nheads=kwargs["nheads"])

        self.get_rho = nn.Sequential(
            nn.Linear(4, self.prop_num * 1),
            nn.Tanh(),
        )
        self.get_miuXY = nn.Sequential(
            nn.Dropout(encoder_dropout),
            nn.Linear(self.d_model, self.prop_num * 2),
            nn.Sigmoid(),
        )
        self.sigmaXY_FC_LN = nn.Sequential(
            nn.Linear(self.d_model, self.prop_num * 2),
            nn.Sigmoid(),
        )
        self.get_simgaXY = nn.Sequential(
            nn.Linear(self.prop_num * 4, self.prop_num * 2),
            nn.Sigmoid(),
        )

        self.vl_encoder = TransformerEncoder(TransformerEncoderLayer(**kwargs), self.num_encoder_layers,
                                             norm=nn.LayerNorm(self.d_model))

        self.satt_pool_frame_output = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Linear(self.d_model // 2, 1),
            nn.Softmax(dim=-2)
        )

        self.satt_pool_gird = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Linear(self.d_model // 2, 1),
            nn.Softmax(dim=-2)
        )

        self.vpos_proj = nn.Sequential(
            nn.Dropout(encoder_dropout),
            nn.Linear(self.d_model, self.d_model)
        )

        self.cat_vf = nn.Linear(self.d_model * 2, self.d_model)

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.G, self.G
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.d_model % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.d_model // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pos_embed = nn.Parameter(pos_emb)
        pos_embed.requires_grad = False

        return pos_embed

    def generate_gauss_weight(self, center, width, max_len):
        # code copied from https://github.com/minghangz/cpl
        weight = torch.linspace(0, 1, max_len)  # shape:[32]
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)  # expand to [bs, 32]
        center = center.unsqueeze(-1)  # shape: [bs, 1]
        width = width.unsqueeze(-1).clamp(1e-2) / 9  # shape: [bs, 1]

        w = 0.3989422804014327  # 1/(math.sqrt(2*math.pi))
        weight = w / width * torch.exp(-(weight - center) ** 2 / (2 * width ** 2))  # [bs, 32]

        # [bs, 32]
        return weight / weight.max(dim=-1, keepdim=True)[0]

    def generate_gauss_weight_2D(self, center_x, width_x, center_y, width_y, rho):
        weight_X = torch.linspace(0, 1, self.G)  # shape:[4]
        weight_X = weight_X.repeat_interleave(self.G, dim=0)  # [4*4]
        weight_X = weight_X.view(1, -1).expand(center_x.size(0), -1).to(center_x.device)  # expand to [bs, 16]

        weight_Y = torch.linspace(0, 1, self.G)  # shape:[4]
        weight_Y = weight_Y.unsqueeze(dim=0).repeat_interleave(self.G, dim=0).reshape(self.G*self.G)
        weight_Y = weight_Y.view(1, -1).expand(center_y.size(0), -1).to(center_y.device)  # expand to [bs, 16]

        center_x, center_y = center_x.unsqueeze(dim=-1), center_y.unsqueeze(dim=-1)  # shape: [bs, 1]
        width_x = width_x.unsqueeze(-1).clamp(1e-2) / self.hparam_X  # shape: [bs, 1]
        width_y = width_y.unsqueeze(-1).clamp(1e-2) / self.hparam_Y  # shape: [bs, 1]
        rho = rho.clamp(-0.99, 0.99) / self.hparam_rho

        w = 0.15915494309189535  # 1.0 / (2 * math.pi)
        numerator = (((weight_X - center_x) ** 2) / width_x ** 2
                     - 2 * rho * (weight_X - center_x) * (weight_Y - center_y) / (width_x * width_y)
                     + (weight_Y - center_y) ** 2 / width_y ** 2)
        weight = w / (width_x * width_y * torch.sqrt(1 - rho ** 2)) * torch.exp(-numerator / (2 * (1 - rho ** 2)))

        aaa = weight / weight.max(dim=-1, keepdim=True)[0]

        return aaa + 1e-8

    def module_1(self, q_local, q_mask, frame_feat, grid_feat, device):
        bs, numF, numG, hdim = grid_feat.size()

        grid_feat = self.pos_encoder_2d + grid_feat.view(bs * numF, numG, hdim)
        obj_local = self.obj_resize(grid_feat)
        obj_local, obj_att = self.grid_decoder(obj_local, q_local.repeat_interleave(numF, dim=0),
                                               memory_key_padding_mask=q_mask.repeat_interleave(numF, dim=0),
                                               output_attentions=True
                                               )  # [b*topFrame, 20, d]

        frame_mask = torch.ones(bs, numF).bool().to(device)
        grid_mask = torch.ones(bs * numF, numG).bool().to(device)
        obj_local = self.grid_encoder(obj_local, src_key_padding_mask=grid_mask, pos=self.pos_encoder_2d)

        logits_miuXY = self.get_miuXY(obj_local.max(dim=1)[0]).view(-1, 2)
        gauss_Xc, gauss_Yc = logits_miuXY[:, 0], logits_miuXY[:, 1]

        tmp_vg = torch.cat((self.sigmaXY_FC_LN(grid_feat.max(dim=1)[0]).view(-1, 2), logits_miuXY), dim=-1)
        logits_simgaXY = self.get_simgaXY(tmp_vg)
        gauss_Xw, gauss_Yw = logits_simgaXY[:, 0], logits_simgaXY[:, 1]

        gauss_rho = self.get_rho(torch.cat((logits_miuXY, logits_simgaXY), dim=-1)).view(bs, numF, 1).mean(dim=1)
        gauss_rho = gauss_rho.repeat_interleave(numF, dim=0).view(bs * numF, 1)
        gauss_weight2D = self.generate_gauss_weight_2D(gauss_Xc, gauss_Xw, gauss_Yc, gauss_Yw, gauss_rho)

        gauss_weight2D = self.position_region(gauss_weight2D.view(bs, numF, numG))
        gauss_weight2D = self.uni_region(gauss_weight2D, src_key_padding_mask=frame_mask)
        gauss_weight2D = torch.sigmoid(gauss_weight2D.view(bs * numF, numG))

        weight_2D = F.softmax(gauss_weight2D.unsqueeze(dim=1), dim=-1)  # [bs*numF, 1, numG]
        Vgrid_fhat = self.cat_vf(
            torch.cat((frame_feat, torch.bmm(weight_2D, grid_feat).squeeze().view(bs, numF, hdim)), dim=-1)
        )
        Vgrid_fhat = self.position_v(Vgrid_fhat)
        Vgrid_fhat = self.Temporal_Grid(Vgrid_fhat, src_key_padding_mask=frame_mask)  # [bs, numF, dim]

        Mask2D = (gauss_weight2D >= self.theta).bool()
        Vgrid_shat = self.Spatial_Grid(grid_feat, src_key_padding_mask=Mask2D, pos=self.pos_encoder_2d)

        Vgrid_ST = Vgrid_fhat.view(bs * numF, 1, hdim) + Vgrid_shat

        Gatt = self.satt_pool_gird(Vgrid_ST)
        globalF_feat = torch.sum(Vgrid_ST * Gatt, dim=1).view(bs, numF, hdim)

        return globalF_feat

    def forward(self, frame_feat, grid_feat, qns_word, ans_word, ans_id=None, stage="GQA"):
        bs, numF, numG, hdim = grid_feat.size()
        device = frame_feat.device
        q_local, q_mask = self.forward_text(list(qns_word), device, max_length=self.max_word_q)  # [batch, q_len, d_model]


        frame_mask = torch.ones(bs, numF).bool().to(device)
        frame_feat = self.frame_encoder(frame_feat, src_key_padding_mask=None, pos=self.pos_encoder_1d(frame_mask, self.d_model))

        # ########################### Dynamic ST Clue modeling ###################################
        globalF_feat = self.module_1(q_local, q_mask, frame_feat, grid_feat, device)
        # ########################### Dynamic ST Clue modeling ###################################

        # ################# Interactive Multi-modal Clue Reasoning (xN layer) ######################
        frame_critical1, _, _ = self.q2v_decoder(globalF_feat, q_local, memory_key_padding_mask=q_mask, query_pos=self.pos_encoder_1d(frame_mask, self.d_model))
        q_local_critical1, _ = self.v2q_catt(q_local, frame_critical1, memory_key_padding_mask=frame_mask, query_pos=self.pos_encoder_1d(q_mask, self.d_model),)

        frame_critical, _, _ = self.q2v_decoder2(frame_critical1, q_local_critical1, memory_key_padding_mask=q_mask, query_pos=self.pos_encoder_1d(frame_mask, self.d_model))
        q_local_critical, _ = self.v2q_catt2(q_local_critical1, frame_critical, memory_key_padding_mask=frame_mask, query_pos=self.pos_encoder_1d(q_mask, self.d_model),)
        # ################# Interactive Multi-modal Clue Reasoning (xN layer) ######################

        frame_qns_mask = torch.cat((frame_mask, q_mask), dim=-1)
        out_vq = self.vl_encoder(torch.cat((frame_critical, q_local_critical), dim=1),
                                 src_key_padding_mask=frame_qns_mask,
                                 pos=self.pos_encoder_1d(frame_qns_mask.bool(), self.d_model))

        out_att = self.satt_pool_frame_output(out_vq[:, :numF, :])
        out_feat = torch.sum(out_vq[:, :numF, :] * out_att, dim=1)
        fusion_proj = self.vpos_proj(out_feat)  # [bs, 768]

        a_seq, _ = self.forward_text(list(chain(*ans_word)), device, max_length=self.max_word_qa)
        a_seq = self.FC_ans(a_seq)
        a_seq = rearrange(a_seq, '(n b) t c -> b n t c', b=bs)
        answer_g = a_seq.mean(dim=2)

        return fusion_proj, answer_g

    def forward_text(self, text_queries, device, max_length):
        """
        text_queries : list of question str
        out: text_embedding: bs, len, dim
            mask: bs, len (bool) [1,1,1,1,0,0]
        """
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries,
                                                             add_special_tokens=True,
                                                             max_length=max_length,
                                                             padding='max_length',
                                                             truncation=True,
                                                             return_tensors='pt')
        tokenized_queries = tokenized_queries.to(device)
        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded_text = self.text_encoder(**tokenized_queries).last_hidden_state

        return encoded_text, tokenized_queries.attention_mask.bool()

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output