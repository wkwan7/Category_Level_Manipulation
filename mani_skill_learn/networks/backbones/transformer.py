import torch
import torch.nn as nn

from ..builder import BACKBONES, build_backbone
from ..modules import build_attention_layer


class TransformerBlock(nn.Module):
    def __init__(self, attention_cfg, mlp_cfg, dropout=None):
        super().__init__()
        self.attn = build_attention_layer(attention_cfg)
        self.mlp = build_backbone(mlp_cfg)
        assert mlp_cfg.mlp_spec[0] == mlp_cfg.mlp_spec[-1] == attention_cfg.embed_dim
        self.ln = nn.LayerNorm(attention_cfg.embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x, mask):
        """
        :param x: [B, N, C] [batch size, length, embed_dim]  the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, length, length] a mask for disallowing attention to padding tokens
        :return: [B, N, C] [batch size, length, length] a single tensor containing the output from the Transformer block
        """
        o = self.attn(x, mask)
        x = x + o
        x = self.ln(x)
        o = self.mlp(x)
        o = self.dropout(o)
        x = x + o
        x = self.ln(x)
        return x

@BACKBONES.register_module()
class TransformerEncoder(nn.Module):
    def __init__(self, block_cfg, pooling_cfg, mlp_cfg=None, num_blocks=6, TN_inc_steps = 10):
        super().__init__()
        embed_dim = block_cfg["attention_cfg"]["embed_dim"]
        self.task_embedding = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.xavier_normal_(self.task_embedding)
        self.attn_blocks = nn.ModuleList([TransformerBlock(**block_cfg) for i in range(num_blocks)])
        self.pooling = build_attention_layer(pooling_cfg, default_args=dict(type='AttentionPooling'))
        self.global_mlp = build_backbone(mlp_cfg) if mlp_cfg is not None else None
        self.TN_inc_steps = TN_inc_steps
        self.num_blocks = num_blocks

    def forward(self, x, mask, progressive_TN = False,
                enable_TN_progressive = False, TN_inc_iter = 0):
        """
        :param x: [B, N, C] [batch size, length, embed_dim] the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, len, len] a mask for disallowing attention to padding tokens.
        :return: [B, F] A single tensor containing the output from the Transformer
        """
        # print('1', x.shape, torch.isnan(x).any())
        one = torch.ones_like(mask[:,:,0])
        mask = torch.cat([one.unsqueeze(1), mask], dim=1) # (B, N+1, N)
        one = torch.ones_like(mask[:,:,0])
        mask = torch.cat([one.unsqueeze(2), mask], dim=2) # (B, N+1, N+1)
        x = torch.cat([torch.repeat_interleave(self.task_embedding, x.size(0), dim=0), x], dim=1)
        if not progressive_TN:
            for attn in self.attn_blocks:
                x = attn(x, mask)
        else:
            if not enable_TN_progressive:
                for attn in self.attn_blocks:
                    x = attn(x,mask)
                    break
            else:
                xlist = []
                itr = 0
                for attn in self.attn_blocks:
                    itr += 1
                    x = attn(x,mask)
                    xlist.append(x)
                    if TN_inc_iter < itr * self.TN_inc_steps: break
                
                cur_alpha = (TN_inc_iter % self.TN_inc_steps)/self.TN_inc_steps
                if TN_inc_iter < self.TN_inc_steps:
                    x = xlist[0]
                elif TN_inc_iter < self.num_blocks * self.TN_inc_steps:
                    x = xlist[-1] * cur_alpha + xlist[-2] * (1-cur_alpha)
                else:
                    x = xlist[-1]

        x = self.pooling(x, mask[:, -1:])
        if self.global_mlp is not None:
            x = self.global_mlp(x)
        return x
