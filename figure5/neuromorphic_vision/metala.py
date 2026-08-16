import torch.nn as nn
import torch
from einops import rearrange, repeat
from causal_conv1d import causal_conv1d_fn
import einops
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
import torch.nn.functional as F

class mela2d(nn.Module):
    def __init__(self, d_model,num_heads=4, lamda_ratio=1):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.gate_fn = nn.functional.silu

        self.q_dim = int(lamda_ratio*self.embed_dim)//num_heads * num_heads
        self.v_dim = int(lamda_ratio*self.embed_dim)//num_heads * num_heads

        self.q_proj = nn.Linear(self.embed_dim,self.q_dim,False)        
        self.k_gate = nn.Linear(self.embed_dim,self.q_dim,False)
        self.v_proj = nn.Linear(self.embed_dim,self.v_dim,False)
        self.g_proj = nn.Linear(self.embed_dim,self.v_dim,True)
        self.out_proj = nn.Linear(self.v_dim,self.embed_dim,False)
        self.head_dim = self.v_dim // self.num_heads
        self.key_dim = self.q_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
        # self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)
        self.group_norm = nn.LayerNorm(self.head_dim)
        self.aug_balance = nn.Parameter(0.0 * torch.zeros(self.q_dim))
        self.d_conv = 4
        self.conv1d = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            bias=False,
            kernel_size=self.d_conv,
            groups=self.embed_dim,
            padding=self.d_conv - 1,
        )
        self.act_q = nn.SiLU()

    def forward(self, x, state=None):
        x = rearrange(x, 'b l d -> b d l').contiguous()
        x = causal_conv1d_fn(
                x=x,
                weight=einops.rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias.to(self.precision)
                if self.conv1d.bias is not None
                else self.conv1d.bias,
                activation="silu",
            )
        x = rearrange(x, 'b d l -> b l d').contiguous()
        q = self.q_proj(x)
        q = self.act_q(q)
        k_gate = self.k_gate(x)
        k = 1
        v = self.v_proj(x)
        g = self.g_proj(x)
        output, new_hidden_states = self.gated_linear_attention(q, k, v, k_gate, state=state)
        output = self.gate_fn(g) * output
        output = self.out_proj(output)
        return output

    def gated_linear_attention(self, q, k, v, gk, normalizer=16, state=None):
        
        gk = F.logsigmoid(gk) / normalizer
        k = 1 - torch.exp(gk)
        
        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        k = rearrange(k, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        gk = rearrange(gk, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        aug_balance = rearrange(self.aug_balance, '(h d) -> h d', h = self.num_heads).contiguous()
        
        if self.training:
            o, new_hidden_states = fused_chunk_gla(q, k, v, gk, initial_state=state, output_final_state=True)
        else:
            o,_ = fused_recurrent_gla(q, k, v, gk)
            new_hidden_states = None
        
        augk = torch.einsum('bhld,hd->bhld', k, aug_balance)

        aug_w = torch.einsum('bhld,bhld->bhl', q, augk)
        o = o + F.sigmoid(aug_w.unsqueeze(-1) * v)
            
            
        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')
        return o, new_hidden_states


from hgru2_pytorch.hgru_real_cuda import HgruRealFunction

class mela1d(nn.Module):
    def __init__(self, d_model,num_heads=4, lamda_ratio=1):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.gate_fn = nn.functional.silu
        self.q_dim = int(lamda_ratio*self.embed_dim)//num_heads * num_heads
        self.v_dim = int(lamda_ratio*self.embed_dim)//num_heads * num_heads

        self.q_proj = nn.Linear(self.embed_dim,self.q_dim,True)        
        self.k_gate = nn.Linear(self.embed_dim,self.q_dim,True)
        self.v_proj = nn.Linear(self.embed_dim,self.v_dim,True)
        self.g_proj = nn.Linear(self.embed_dim,self.v_dim, True)
        self.out_proj = nn.Linear(self.v_dim,self.embed_dim,True)
        #self.head_dim = self.v_dim // self.num_heads
        #self.key_dim = self.q_dim // self.num_heads
        #self.scaling = self.key_dim ** -0.5
        #self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)
        self.aug_balance = nn.Parameter(0.0 * torch.zeros(self.q_dim))
        self.d_conv = 4
        self.conv1d = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            bias=False,
            kernel_size=self.d_conv,
            groups=self.embed_dim,
            padding=self.d_conv - 1,
        )
        self.act_q = nn.SiLU()
        self.act_v = nn.SiLU()
        self.act_w = nn.SiLU()
        self.scan = HgruRealFunction.apply
        # self.norm = nn.LayerNorm(self.q_dim, eps=1e-5, elementwise_affine=False)
        self.norm = nn.LayerNorm(self.q_dim)

    def forward(self, x, lower_bound = 0):
        x = rearrange(x, 'b l d -> b d l').contiguous()
        x = causal_conv1d_fn(
                x=x,
                weight=einops.rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias.to(self.precision)
                if self.conv1d.bias is not None
                else self.conv1d.bias,
                activation="silu",
            )
        x = rearrange(x, 'b d l -> b l d').contiguous()
        x = x.transpose(0,1)

        q = self.q_proj(x)
        q = self.act_q(q)
        k_gate = self.k_gate(x)
        k = 1
        v = self.act_v(self.v_proj(x))
        g = self.g_proj(x)
        # bound = F.softmax(self.lower_bounds, dim=0)[1].to(x)
        if type(lower_bound) == int:
            lower_bound = torch.zeros_like(x).to(x)
        output, new_hidden_states = self.gated_linear_attention(q, k, v, k_gate,lower_bound)
        output = self.gate_fn(g) * output
        output = self.out_proj(output).transpose(0,1)
        return output

    def gated_linear_attention(self, q, k, v, gk, lower_bound):
        gk = torch.exp(F.logsigmoid(gk)/16)
        gk = lower_bound + (1 - lower_bound) * gk
        k = 1 - gk 
        x = k*v # b l d 

        o = self.scan(x,gk)
        o = q*o

        augk  =   k* self.aug_balance
        aug_w =   q* augk
        o = o + self.act_w(aug_w * v)
        o = self.norm(o)
        return o, None
