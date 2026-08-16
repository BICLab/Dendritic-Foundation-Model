import torch
import math
from torch import nn 
import torch.nn.functional as F
from einops import rearrange
from fla.ops import fused_chunk_gla
from typing import Optional


class SparseLA(nn.Module):
    def __init__(self, d_model, n_head, layer_idx, use_gk, use_gv):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        
        self.gate_fn = nn.functional.silu
        assert use_gk and not use_gv, "Only use_gk is supported for simplicity."

        dk = self.embed_dim
        self.q_proj = nn.Linear(self.embed_dim, dk, bias=False)
        self.k_gate =  nn.Linear(self.embed_dim, dk, bias=False)

        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.g_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.head_dim = self.embed_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
        self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)

        self.d_conv = 2
        self.conv1d = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            bias=False,
            kernel_size=self.d_conv,
            groups=self.embed_dim,
            padding=self.d_conv - 1,
            # **factory_kwargs,
        )
        self.act = nn.SiLU()
        
        self.post_init()


    def post_init(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        if isinstance(self.k_gate, nn.Sequential):
            nn.init.xavier_uniform_(self.k_gate[0].weight, gain=2 ** -2.5)
            nn.init.xavier_uniform_(self.k_gate[1].weight, gain=2 ** -2.5)
        else:
            nn.init.xavier_uniform_(self.k_gate.weight, gain=2 ** -2.5)

    def forward(self, x, hidden_states=None):
        x = rearrange(x, 'b l d -> b d l').contiguous()
        seqlen = x.shape[-1]
        x = self.act(self.conv1d(x)[..., :seqlen])
        x = rearrange(x, 'b d l -> b l d').contiguous()
        q = self.q_proj(x)
        k_gate = self.k_gate(x)
        k = 1
        v = self.v_proj(x)
        g = self.g_proj(x)

        output, new_hidden_states = self.sparse_linear_attention(q, k, v, k_gate, hidden_states=hidden_states)
        output = self.gate_fn(g) * output
        output = self.out_proj(output)
        return output


    def sparse_linear_attention(self, q, k, v, gk, normalizer=16, hidden_states=None):
        
        k = 1 - gk
        k = torch.clamp(k, min=-7.0, max=7.0)

        k = rearrange(k, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        
        b,h,l,d = k.shape
        k = F.softmax(k, dim=-1)
        _, ind = torch.topk(k, 3*d//4, dim=-1, largest=False, sorted=False, out=None)
        k = k.scatter(-1, ind, torch.zeros_like(k, device=k.device))
        gk = (1 - k).log()
        
        o, new_hidden_states = fused_chunk_gla(q, k, v, gk, initial_state=hidden_states, output_final_state=True)
       
        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')
        return o, new_hidden_states

