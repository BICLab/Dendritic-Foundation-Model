import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.gla import fused_chunk_gla, chunk_gla, fused_recurrent_gla


class MinLSTM_matrix(nn.Module):
    """
        Only "parallel mode" is supported for conciseness.
        use log space.

        input shape: [batch, seq_len, in_chn]
        output shape: [batch,seq_len, out_chn]
    """
    def __init__(self, input_size: int, hidden_size: int, device=None, dtype=None):
        super().__init__()
        self.num_heads = 4
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size*3,bias=False,device=device, dtype=dtype)

    #log(1+exp(x)) softplus,用gla  qkv 不log,alpha log
    def forward(self, x_t):
        b,l,d = x_t.shape
        seq_len = x_t.shape[1]
        f,i,h = torch.chunk(self.linear(x_t),chunks=3,dim=-1)
        diff = F.softplus(-f) - F.softplus(-i)
        log_f = -F.softplus(diff) #need log
        in_i = torch.exp(-F.softplus(-diff))
        in_h = self.g(h)
        q = torch.ones((b,l,d), device=x_t.device, dtype=x_t.dtype)

        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        in_i = rearrange(in_i, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        in_h = rearrange(in_h, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        log_f = rearrange(log_f, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()

        # 头维度拆分成1
        # out,_ = fused_chunk_gla(q,in_i,in_h,log_f)
        out,_ = fused_chunk_gla(q,in_i,in_h,log_f)
        out = rearrange(out, 'b h l d -> b l (h d)')
        return out

    def g(self,x):
        return torch.where(x >= 0, x+0.5, torch.sigmoid(x))

    def log_g(self,x):
        return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))