from torch import nn
# from module.spikingjelly1.activation_based import neuron,surrogate
import torch
import torch.nn.functional as F
from einops import rearrange
from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla
from fla.modules import RotaryEmbedding
from causal_conv1d import causal_conv1d_fn
import einops
import numpy as np
import math
from module.pscan import pscan

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
# from hgru2_pytorch.modules import Hgru1_real_1d,Hgru2_1d

from hgru2_pytorch.gla.inter_chunk_contribution.fn import inter_chunk_onc
from hgru2_pytorch.gla.intra_chunk_contribution.fn import intra_chunk_onc
from hgru2_pytorch.helpers import get_activation_fn, get_norm_fn, print_module, print_params
from hgru2_pytorch.hgru_real_cuda import HgruRealFunction
from fla.ops.gla import fused_chunk_gla, chunk_gla, fused_recurrent_gla

#change somewhere to get comparable parameters
class Hgru1_real_1d(nn.Module):
    def __init__(
        self,
        embed_dim,
        act_fun="silu",
        expand_ratio=1,
        bias=True,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        ex_dim = int(expand_ratio * embed_dim)
        self.input_proj = nn.Linear(embed_dim, ex_dim, bias=bias)
        self.forget_gate = nn.Linear(embed_dim, ex_dim, bias=bias)
        self.output_gate = nn.Linear(embed_dim, ex_dim, bias=bias)
        self.out_proj = nn.Linear(ex_dim, embed_dim, bias=bias)
        self.norm = nn.LayerNorm(ex_dim)
        self.act = get_activation_fn(act_fun)

        self.scan = HgruRealFunction.apply

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        x = x.transpose(0,1)
        n, b, d = x.shape
        input = self.act(self.input_proj(x))
        output_gate = F.sigmoid(self.output_gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.forget_gate(x))
        input = (1 - lambda_) * input

        output_state = self.scan(input, lambda_)

        output_state = self.norm(output_state * output_gate)

        output = self.out_proj(output_state).transpose(0,1)
        return output

class Hgru2_1d(nn.Module):
    def __init__(
        self,
        embed_dim,
        expand_ratio=2,
        act_fun="silu",
        uv_act_fun="sigmoid",
        use_norm=True,
        bias=True,
        norm_type="layernorm",
        use_fla=False,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.expand_ratio = expand_ratio
        self.d_state = 2
        exdim = int(self.expand_ratio* embed_dim)//self.d_state*self.d_state
        self.in_proj = nn.Linear(embed_dim, 3 *exdim, bias=bias)
        self.out_proj = nn.Linear(exdim, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        self.out_act = get_activation_fn(uv_act_fun)
        self.use_norm = use_norm
        self.use_fla = use_fla
        if self.use_norm:
            self.norm = get_norm_fn(norm_type)(exdim)

        self.chunk_size = 128

        self.forward = self.forward_lesshead
        self.scan = HgruRealFunction.apply

    def forward(self, x, lower_bound=0):
        ## x: n b d #length batch dimension
        x = x.transpose(0,1)
        n, b, d = x.shape

        feature = self.in_proj(x)
        V, Q, F_ = feature.chunk(3, dim=-1)
        #V -input cell ,output_gate = q
        V = self.act(V)
        Q = self.out_act(Q)
        F_ = F.sigmoid(F_)# forget gate 
        if type(lower_bound) == int:
            lower_bound = torch.zeros_like(x).to(x)
                # reshape
        # h is num_head, d is head dimension
        V, Q, F_, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.d_state),
            [V, Q, F_, lower_bound],
        )
        # head 分配，在hgru1 无，除此以外确实是一样的
        lambda_ = lower_bound + (1 - lower_bound) * F_
        log_lambda_ = torch.log(lambda_)
        #print(log_lambda_)
        K = 1 - lambda_

        if self.use_fla:
            V, Q, G_K, K = map(
                lambda x: rearrange(x, "n b h d -> b h n d").contiguous(),
                [V, Q, log_lambda_, K],
            )

            o, _ = self.scan(Q, K, V, G_K, 1)
            o = rearrange(o, "b h n d -> n b (h d)")
        else:
            V, Q, G_K, K = map(
                lambda x: rearrange(
                    x, "(n c) b h d -> b h n c d", c=min(self.chunk_size, n)
                ).contiguous(),
                [V, Q, log_lambda_, K],
            )

            G_V = None
            G_K, G_V, o1 = inter_chunk_onc(Q, K.to(Q.dtype), V, G_K.to(Q.dtype), G_V)
            o2 = intra_chunk_onc(Q, K.to(Q.dtype), V, G_K.to(Q.dtype), G_V)
            o = o1 + o2
            o = rearrange(o, "b h n c d -> (n c) b (h d)")

        if self.use_norm:
            o = self.norm(o)

        # out proj                                                                                                                                                                                        
        output = self.out_proj(o).transpose(0,1)
        return output

    def forward_lesshead(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        # in bld
        x = x.transpose(0,1)
        n, b, d = x.shape
        feature = self.in_proj(x)
        input, output_gate, forget_gate = feature.chunk(3, dim=-1)
        input = self.act(input)
        output_gate = self.out_act(output_gate)
        forget_gate = F.sigmoid(forget_gate)
        if type(lower_bound) == int:
            lower_bound = torch.zeros_like(forget_gate).to(forget_gate)

        # reshape
        input, output_gate, forget_gate, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.d_state),#h太大，
            [input, output_gate, forget_gate, lower_bound],
        )
        # mix
        lambda_ = lower_bound + (1 - lower_bound) * forget_gate
        input = torch.einsum("... h d, ... h e -> ... h d e", 1 - lambda_, input)
        lambda_ = repeat(lambda_, "... h d -> ... h d e", e=self.d_state)

        # reshape
        input, lambda_ = map(
            lambda x: rearrange(x, "... h d e -> ... (h d e)"), [input, lambda_]
        )

        # mix
        output_state = self.scan(input, lambda_)

        # down
        output_state = rearrange(
            output_state,
            "... (h d e) -> ... h d e",
            d=self.d_state,
            e=self.d_state,
        )
        output_state = torch.einsum(
            "... h d e, ... h d -> ... h e", output_state, output_gate
        )
        output_state = rearrange(output_state, "... h e -> ... (h e)")

        # output gate
        if self.use_norm:
            output_state = self.norm(output_state)

        # out proj
        output = self.out_proj(output_state).transpose(0,1)

        return output

    def extra_repr(self):
        return print_module(self)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output

class LLaMAMLP(nn.Module):
    def __init__(
        self,
        d_model,
        intermediate_scale =4/3,
        act_func='gelu',
        dropout=0.0,
        transposed=False
    ):
        super().__init__()
        if act_func =='gelu':
            self.activation_func = F.gelu
        self.size = d_model
        ff_dim = int(intermediate_scale*d_model)
        self.w1 = nn.Linear(d_model,ff_dim,False)
        self.w3 = nn.Linear(d_model,ff_dim,False)
        self.w2 = nn.Linear(ff_dim,d_model,False)


    def forward(self,x,state = None):
        w1_out = self.w1(x)
        w3_out = self.w3(x)
        return self.w2(self.activation_func(w1_out) * w3_out)
      
    @property
    def d_output(self):
        return self.size

class self_atten_block(nn.Module):
    def __init__(self,d_model,
                num_heads=4,expand = 1,dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = d_model
        qk_dim = int(expand*self.embed_dim)//num_heads*num_heads
        self.head_dim = qk_dim//num_heads
        self.q_proj = nn.Linear(self.embed_dim,qk_dim,False)    
        self.k_proj = nn.Linear(self.embed_dim,qk_dim,False)
        self.v_proj = nn.Linear(self.embed_dim,qk_dim,False)
        self.out_proj = nn.Linear(qk_dim,self.embed_dim,False)
        
        self.rotary = RotaryEmbedding(self.head_dim)
        self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False) 
        self.apply(self._initialize_weights)
        self.drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def qkv(self,q,k,v):
        b,h,l,d = q.shape
        mask = (1 - torch.triu(torch.ones((1, l, l), device=q.device), diagonal=1)).bool()#仅保留下半
        qk = torch.matmul(q/self.scale,k.transpose(2,3))#
        qk = qk.masked_fill(mask==0,-1e9)
        qk = F.softmax(qk,dim=-1)
        qkv = torch.matmul(qk,v)
        return qkv
    
    def forward(self,x):
        b,l,d = x.shape
        q = rearrange(self.q_proj(x), '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(self.k_proj(x), '... (h d) -> ... h d', h=self.num_heads)
        v = rearrange(self.v_proj(x), '... (h d) -> ... h d', h=self.num_heads)
        q, k = self.rotary(q, k, 0, l)
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')
        o = self.qkv(q, k, v)
        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.out_proj(o)
        o = self.drop(o)
        return o

#copy from mamba
class Mamba_matrix(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # dt_rank delta d_state BC
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # 暂时没有修改串行计算的函数
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            # A = A.mean(dim=1,keepdim=True).contiguous()
            # B = B.mean(dim=1,keepdim=True).contiguous()
            # C = C.mean(dim=1,keepdim=True).contiguous()

            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # 状态扩大
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # Reused states must retain the original batch size unless reinitialized.
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class Mamba_vector(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # dt_rank delta d_state BC
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # 暂时没有修改串行计算的函数
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            A = A.mean(dim=1,keepdim=True).contiguous()
            B = B.mean(dim=1,keepdim=True).contiguous()
            C = C.mean(dim=1,keepdim=True).contiguous()

            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # 状态扩大
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # Reused states must retain the original batch size unless reinitialized.
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class MLPmix(nn.Module):
    def __init__(
        self,
        d_model,
        length,
        causal = False,
        act_func='gelu',
        scale = 2,
        dropout=0.0,
        transposed=False
    ):
        super().__init__()
        if act_func =='gelu':
            self.activation_func = F.gelu
        # self.multiple_of = multiple_of
        # 100 4000 128
        self.causal = causal
        self.size = d_model
        ff_dim = int(scale*self.size)
        self.bias = nn.Parameter(torch.zeros(length),requires_grad=True)
        self.w2 = nn.Linear(d_model,ff_dim,False)
        self.w3 = nn.Linear(ff_dim,d_model,False)

        weight = torch.zeros([length, length])
        #仅一半参数计算入参数量
        bias = torch.zeros([length, 1])
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, 0.)
        if causal == True:
            self.mask = (1 - torch.triu(torch.ones((length, length)), diagonal=1))#仅保留下半
        else: 
            self.mask = 1

    def forward(self,x,state = None):
        # in bld
        b,l,d = x.shape
        z = self.activation_func(self.w2(x))
        if self.causal == True:
            weight = self.mask.to(x)*self.weight
        else:
            weight = self.weight
        # weight = torch.zeros([l,l]).to(x)
        z_mix = torch.addmm(self.bias, weight, z.transpose(0,1).flatten(1)).view(l,b,-1)
        z_mix = z_mix.transpose(0,1)
        y = z*z_mix
        out = self.w3(y)
        return out      

#copy from Hyena
class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)

class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = (
            nn.Parameter(w * torch.ones(1, dim))
            if train_freq
            else w * torch.ones(1, dim)
        )

    def forward(self, x):
        return torch.sin(self.freq * x)

class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]

class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        decay = torch.exp(-t * self.deltas.abs())
        x = x * (decay + self.shift)
        return x

class Filter(OptimModule):
    def __init__(
        self,
        d_model,
        emb_dim=3,  # dim of input to MLP, augments with positional encoding
        order=16,  # width of the implicit MLP
        seq_len=1024,
        lr=1e-3,
        lr_pos_emb=1e-5,
        dropout=0.0,
        w=1,  # frequency of periodic activations
        wd=0,  # weight decay of kernel parameters
        bias=True,
        num_inner_mlps=2,
        linear_mixer=False,
        modulate: bool = True,
        normalized=False,
        num_heads: int = 1,
        **kwargs,
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()
        self.d_model = d_model 
        self.emb_dim = emb_dim 
        self.seq_len = seq_len 
        self.modulate = modulate
        self.num_heads = num_heads
        self.use_bias = bias
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        assert (
            emb_dim % 2 != 0 and emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        # uses a variable number of inner linear layers
        if linear_mixer is False:
            self.implicit_filter = [
                nn.Linear(emb_dim, order),
                act,
            ]
            for i in range(num_inner_mlps):
                self.implicit_filter.append(nn.Linear(order, order))
                self.implicit_filter.append(act)
            # final linear layer
            self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
            self.implicit_filter = nn.Sequential(*self.implicit_filter)
        else:
            self.implicit_filter = nn.Sequential(
                nn.Linear(emb_dim, d_model, bias=False),
            )

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)

        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)

        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None:
            # [MP] Currently does not work if k is None as the filter
            # comes in L, D instead of D, L
            k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k
        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias


        y = fftconv_ref(x, k, bias, dropout_mask=None, gelu=False)

        return y.to(dtype=x.dtype)

class Hyena(nn.Module):
    NUM_PROJECTIONS = 3 
    
    def __init__(
        self,
        d_model: int,
        l_max: int,
        filter_order: int=64,
        num_heads: int=1,
        num_blocks: int=1,
        outer_mixing: bool=False,
        dropout: float=0.0,
        filter_dropout: float=0.0,
        short_filter_order: int=3,
        return_state: bool=False,
        bidirectional: bool=False,
        layer_idx: int=None,
        **filter_args,
    ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            filter_order: (int): Width of the FFN parametrizing the implicit filter. Defaults to 64
            num_heads: (int): Number of heads. Defaults to 1
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
            return_state: (bool): whether to return a state
        """
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"Model dimension {d_model} must be divisible by num heads {num_heads}"
        assert (
            l_max % num_blocks == 0
        ), f"Maximum signal length {l_max} must be divisible by block dimension {num_blocks}"
        
        self.scale = 1.2
        block_dim = l_max // num_blocks
        head_dim = int(d_model*self.scale) // num_heads

        self.d_model = d_model
        self.l_max = l_max
        self.num_heads=num_heads
        self.block_dim = block_dim
        self.head_dim = head_dim
        self.filter_order=filter_order
        self.short_filter_order=short_filter_order
        self.num_blocks=num_blocks
        self.filter_dropout=filter_dropout
        self.outer_mixing=outer_mixing
        self.return_state=return_state
        
        self.dropout = nn.Dropout(dropout)
        # setup projections 
        self.in_proj = nn.Linear(self.d_model, self.NUM_PROJECTIONS * int(self.d_model*self.scale))
        self.out_proj = nn.Linear(int(self.scale*self.d_model), self.d_model)

        self.bidirectional = bidirectional

        total_width = int(self.scale*self.d_model) * self.NUM_PROJECTIONS

        self.short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=self.short_filter_order,
            groups=total_width,
            padding=self.short_filter_order - 1,
        )

        if "channels" not in filter_args:
            filter_args["channels"] = 1
        self.filter_fn = Filter(
            self.head_dim,
            order=self.filter_order,
            seq_len=self.l_max,
            dropout=self.filter_dropout,
            bidirectional=self.bidirectional,
            l_max=self.l_max,
            **filter_args,
        )


    def forward(self, u, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            u: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        l = u.size(1)
        assert l <= self.l_max, f"Input length {l} exceeds maximum length {self.max_l}"

        # in projection
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")

        # short filter
        uc = self.short_filter(u)[..., :l]

        uc = rearrange(
            uc,
            "b (ho v) (z l) -> b ho v z l",
            z=self.num_blocks,
            ho=self.num_heads,
            v=self.head_dim * self.NUM_PROJECTIONS,
        )

        x1, x2, v = uc.split(int(self.scale*self.d_model), dim=2)

        # pre-gating
        v = v * x1
        v = self.dropout(v) 

        # long convolution
        if self.bidirectional:
            # print(f"self.bidirectional: {self.bidirectional}")
            k_rev = self.filter_fn.filter_rev(l, device=u.device)
            k_rev = rearrange(k_rev, "c l d -> c d l")[0] # `c` is always 1 by default
        else:
            k_rev = None
        k = self.filter_fn.filter(l, device=u.device)
        k = rearrange(k, "c l d -> c d l")[0] # `c` is always 1 by default
        v = self.filter_fn(v, l, k=k, k_rev=k_rev, bias=self.filter_fn.bias[None, :, None])
        
        # post-gating
        v = v * x2

        y = rearrange(
            v,
            "b h v z l -> b (z l) (h v)",
            z=self.num_blocks,
            h=self.num_heads,
        )
        y = self.out_proj(y)

        if self.return_state:
            return y, None
        return y

    def state_size(self, sequence_length: int=2048) -> int:
        return self.d_model * sequence_length


class mela1d(nn.Module):
    def __init__(self, d_model,num_heads=4):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.gate_fn = nn.functional.silu
        self.q_dim = int(1.2*self.embed_dim)//num_heads * num_heads
        self.v_dim = int(1.2*self.embed_dim)//num_heads * num_heads

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
        self.norm = nn.LayerNorm(self.q_dim, eps=1e-5, elementwise_affine=False)

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

class mela2d(nn.Module):
    def __init__(self, d_model,num_heads=4):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.gate_fn = nn.functional.silu

        self.q_dim = int(1.2*self.embed_dim)//num_heads * num_heads
        self.v_dim = int(1.2*self.embed_dim)//num_heads * num_heads

        self.q_proj = nn.Linear(self.embed_dim,self.q_dim,False)        
        self.k_gate = nn.Linear(self.embed_dim,self.q_dim,False)
        self.v_proj = nn.Linear(self.embed_dim,self.v_dim,False)
        self.g_proj = nn.Linear(self.embed_dim,self.v_dim,True)
        self.out_proj = nn.Linear(self.v_dim,self.embed_dim,False)
        self.head_dim = self.v_dim // self.num_heads
        self.key_dim = self.q_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
        self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)
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

class SparseLAM(nn.Module):
    def __init__(self, d_model,num_heads=4):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.gate_fn = nn.functional.silu

        self.q_dim = int(1.2*self.embed_dim)//num_heads * num_heads
        self.v_dim = int(1.2*self.embed_dim)//num_heads * num_heads

        self.q_proj = nn.Linear(self.embed_dim,self.q_dim,False)        
        self.k_gate = nn.Linear(self.embed_dim,self.q_dim,False)
        self.v_proj = nn.Linear(self.embed_dim,self.v_dim,False)
        self.g_proj = nn.Linear(self.embed_dim,self.v_dim,True)
        self.out_proj = nn.Linear(self.v_dim,self.embed_dim,False)
        self.head_dim = self.v_dim // self.num_heads
        self.key_dim = self.q_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
        self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)
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
        self.gate_normalizer = 16 #min(2 * (self.layer_idx + 1), 16)

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
        gk = self.k_gate(x).float()
        v = self.v_proj(x)
        g = self.g_proj(x)

        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.num_heads)
        gk = rearrange(gk, 'b l (h d) -> b h l d', h = self.num_heads)
        meta_num, sparse_num = gk.shape[-1] // 2, 5

        gk_1 = F.logsigmoid(gk[..., :meta_num]) / self.gate_normalizer
        k_1 = 1 - torch.exp(gk_1)

        k_2 = 1 - gk[..., meta_num:]
        k_2 = torch.clamp(k_2, min=-7.0, max=7.0)
        k_2 = F.softmax(k_2, dim=-1)
        d = k_2.shape[-1]
        _, ind = torch.topk(k_2, d - sparse_num, dim=-1, largest=False, sorted=False, out=None)
        k_2 = k_2.scatter(-1, ind, torch.zeros_like(k_2, device=k_2.device))
        gk_2 = (1 - k_2).log()

        k = torch.cat((k_1, k_2), dim=-1)
        gk = torch.cat((gk_1, gk_2), dim=-1)
        o, new_hidden_states = fused_chunk_gla(q, k, v, gk, initial_state=state, output_final_state=True)
        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')
        output = self.gate_fn(g) * o
        output = self.out_proj(output)
        return output

class SparseLAV(nn.Module):
    def __init__(self, d_model,num_heads=4):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.gate_fn = nn.functional.silu

        self.q_dim = int(1.2*self.embed_dim)//num_heads * num_heads
        self.v_dim = int(1.2*self.embed_dim)//num_heads * num_heads

        self.q_proj = nn.Linear(self.embed_dim,self.q_dim,False)        
        self.k_gate = nn.Linear(self.embed_dim,self.q_dim,False)
        self.v_proj = nn.Linear(self.embed_dim,self.v_dim,False)
        self.g_proj = nn.Linear(self.embed_dim,self.v_dim,True)
        self.out_proj = nn.Linear(self.v_dim,self.embed_dim,False)
        self.head_dim = self.v_dim // self.num_heads
        self.key_dim = self.q_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
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
        self.gate_normalizer = 16 #min(2 * (self.layer_idx + 1), 16)
        self.scan = HgruRealFunction.apply
        self.norm = nn.LayerNorm(self.q_dim, eps=1e-5, elementwise_affine=False)



    def forward(self, x, state=None,lower_bound = 0):
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
        gk = self.k_gate(x).float()
        v = self.v_proj(x)
        g = self.g_proj(x)

        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.num_heads)
        gk = rearrange(gk, 'b l (h d) -> b h l d', h = self.num_heads)
        meta_num, sparse_num = gk.shape[-1] // 2, 5

        gk_1 = F.logsigmoid(gk[..., :meta_num]) / self.gate_normalizer
        k_1 = 1 - torch.exp(gk_1)
        gk_1 = torch.exp(gk_1)

        k_2 = 1 - gk[..., meta_num:]
        k_2 = torch.clamp(k_2, min=-7.0, max=7.0)
        k_2 = F.softmax(k_2, dim=-1)
        d = k_2.shape[-1]
        _, ind = torch.topk(k_2, d - sparse_num, dim=-1, largest=False, sorted=False, out=None)
        k_2 = k_2.scatter(-1, ind, torch.zeros_like(k_2, device=k_2.device))
        gk_2 = (1 - k_2)

        k = torch.cat((k_1, k_2), dim=-1)
        gk = torch.cat((gk_1, gk_2), dim=-1)
        #gk = torch.exp(gk)

        gk = rearrange(gk, 'b h l d -> l b (h d)')
        k = rearrange(k, 'b h l d -> l b (h d)')
        v = rearrange(v, 'b h l d -> l b (h d)')
        q = rearrange(q, 'b h l d -> l b (h d)')

        state = k*v
        o = self.scan(state,gk) #l b d
        o = q*o

        o = self.norm(o).transpose(0,1)

        output = self.gate_fn(g) * o
        output = self.out_proj(output) #l b d transpose
        return output
# Historical implementation names retained for compatibility.
spla2d = SparseLAM
spla1d = SparseLAV
sparsela_m = SparseLAM
sparsela_v = SparseLAV


class FC_att_layer4(nn.Module):
    def __init__(self,in_features ,channel_size, num_classes,block='self-att'):
        super().__init__()
        self.dp = 0.1
        self.drop = nn.Dropout(self.dp)
        self.encode = nn.Linear(in_features, channel_size)
        self.ln1 = nn.LayerNorm(channel_size)
        self.dropatt = nn.Dropout(self.dp)

        self.linear = nn.Linear(channel_size,channel_size)
        self.ln2 = nn.LayerNorm(channel_size)
        self.ln3 = nn.LayerNorm(channel_size)
        self.dropatt2 = nn.Dropout(self.dp)
        self.fc_out = nn.Linear(channel_size, num_classes)
        self.channal_size = channel_size
        block_aliases = {
            'spla1d': 'sparsela_v',
            'spla2d': 'sparsela_m',
        }
        self.block = block_aliases.get(block, block)
        block = self.block

        if block == 'self-att':
            self.att_1 = self_atten_block(channel_size,expand=1.5)
            self.att_2 = self_atten_block(channel_size,expand=1.5)
        elif block == 'mamba-vec':
            self.att_1 = Mamba_vector(channel_size,expand=1.8)
            self.att_2 = Mamba_vector(channel_size,expand=1.8)
        elif block == 'mamba-mat':
            self.att_1 = Mamba_matrix(channel_size,expand=1.8)
            self.att_2 = Mamba_matrix(channel_size,expand=1.8)
        elif block == 'hgrn1':
            self.expand_r = 1.5
            self.att_1 = Hgru1_real_1d(channel_size,expand_ratio=self.expand_r)
            self.att_2 = Hgru1_real_1d(channel_size,expand_ratio=self.expand_r)
            self.lower_bounds = nn.Parameter(torch.ones(2, int(channel_size*self.expand_r)), requires_grad=True)
        elif block == 'hgrn2':
            self.expand_r = 1.5
            self.att_1 = Hgru2_1d(channel_size,expand_ratio=self.expand_r)
            self.att_2 = Hgru2_1d(channel_size,expand_ratio=self.expand_r)
            self.lower_bounds = nn.Parameter(torch.ones(2, int(channel_size*self.expand_r)), requires_grad=True)
        elif block == 'Hyena':
            self.att_1 = Hyena(channel_size,l_max=250)
            self.att_2 = Hyena(channel_size,l_max=250)
        elif block == 'mlp':
            self.att_1 = MLPmix(channel_size,length=250,causal=True)
            self.att_2 = MLPmix(channel_size,length=250,causal=True)
        elif block == 'metala_m':
            self.att_1 = mela2d(channel_size)
            self.att_2 = mela2d(channel_size)
        elif block == 'metala_v':
            self.att_1 = mela1d(channel_size)
            self.att_2 = mela1d(channel_size)
            self.lower_bounds = nn.Parameter(torch.ones(2, int(channel_size*1.2//4*4)), requires_grad=True)
        elif block == 'sparsela_m':
            self.att_1 = sparsela_m(channel_size)
            self.att_2 = sparsela_m(channel_size)
        elif block == 'sparsela_v':
            self.att_1 = sparsela_v(channel_size)
            self.att_2 = sparsela_v(channel_size)
            self.lower_bounds = nn.Parameter(torch.ones(2, int(channel_size*1.2//4*4)), requires_grad=True)     
        else:
            raise ValueError(f"Unsupported block type: {block}")

    def forward(self,x):
        if self.block == 'hgrn1' or self.block == 'hgrn2' or self.block == 'metala_v' or self.block == 'sparsela_v':
            lower_bounds = F.softmax(self.lower_bounds, dim=0)
            lower_bound = torch.zeros_like(lower_bounds[0]).to(x).squeeze()
            x = F.gelu(self.drop(self.encode(x)))
            identity = x 
            x = self.ln1(x)
            x = self.dropatt(self.att_1(x,lower_bound = lower_bound))
            x = identity + x
            
            identity = x 
            x = self.ln2(x)
            x = self.dropatt(self.linear(x))
            x = identity + x
            
            lower_bound += lower_bounds[0]
            identity = x 
            x = self.ln3(x)
            x = self.dropatt2(self.att_2(x,lower_bound = lower_bound))
            x = identity + x

            x = self.fc_out(x)
            return x.mean(1)
        else:
            x = F.gelu(self.drop(self.encode(x)))
            identity = x 
            x = self.ln1(x)
            x = self.dropatt(self.att_1(x))
            x = identity + x
            
            identity = x 
            x = self.ln2(x)
            x = self.dropatt(self.linear(x))
            x = identity + x

            identity = x 
            x = self.ln3(x)
            x = self.dropatt2(self.att_2(x))
            x = identity + x

            x = self.fc_out(x)
            return x.mean(1)


