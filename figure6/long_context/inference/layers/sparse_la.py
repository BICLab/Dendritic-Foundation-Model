import torch
import triton
import triton.language as tl
from einops import rearrange

from typing import Iterable

from fla.ops.gla.fused_recurrent import fused_recurrent_gla
from fla.ops.gla.fused_chunk import fused_chunk_gla
from typing import Any, Callable, Dict, Literal, Optional, Tuple

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        q, 
        k, 
        start,
        end,
        v, 
        slope_rate, # [H, 1, 1]
        initial_state, # [B, H, K, V] #this initial_state
        is_sparsela: bool = True, 
        sparse_rate: float = 0.5, 
        scale: int = 1, 
        output_final_state: bool = True, 
        offsets: Iterable = None, 
        head_first: bool = True,
    ):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        slope_rate = slope_rate.contiguous()

        b, head, n, d = q.shape
        e = v.shape[-1]
        T = q.shape[2] 
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))

        # (1) expand the slope rate
        gk = -slope_rate.expand_as(q).to(torch.float32) # [H, 1, 1] -> [B, H, T, K]
        sparse_head_dim = int(sparse_rate * d)

        # (2) add sparse operations
        if is_sparsela:
            _, ind = torch.topk(torch.abs(k), sparse_head_dim, dim=-1, largest=False, sorted=False, out=None)
            k = k.scatter(-1, ind, torch.zeros_like(k))#ind 0
            gk = gk.scatter(-1, ind, torch.zeros_like(gk))
        initial = initial_state[:, :, start:end, :].clone()
        o,hidden_state = fused_chunk_gla(q,k,v,gk,scale=1,initial_state=initial,output_final_state=True,head_first=True)
        initial_state[:,:,start:end,:].copy_(hidden_state)

        ctx.save_for_backward(q, k, v, gk, initial_state)#get_final_state
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.head_first = head_first

        return o, initial_state



# Apply the modified sparse linear attention function
sparse_linear_attention_ = _attention.apply

def sparse_linear_attention(
    q, 
    k, 
    v, 
    ed, # [h, 1, 1] slope_rate
    is_sparsela: bool = True, 
    sparse_rate: float = 0.5, 
    block_size=256, 
    kv_history=None
):

    """
    Apply sparse linear attention algorithm 
    to compute attention efficiently.
    
    Args:
        q: Query tensor of shape [batch, heads, seq_len, dim]
        k: Key tensor of shape [batch, heads, seq_len, dim]
        v: Value tensor of shape [batch, heads, seq_len, dim_v]
        ed: Decay rate tensor of shape [heads]
        block_size: Size of blocks for block-sparse attention
        kv_history: Optional key-value history from previous computations
        
    Returns:
        output: Attention output
        kv: Updated key-value history
    """
    d = q.shape[-1]
    e = v.shape[-1] # 1 h n d
    assert (d == 128 or d == 64)
    assert (e == 128 or e == 64)

    if ed.dim() == 1:
        ed = ed.view(1, -1, 1, 1) # 1 h 1 1

    # Split the computation into chunks for better parallelism
    #
    m = 128 if d >= 128 else 64
    assert d % m == 0, f"Dimension d ({d}) must be divisible by m ({m})"
    arr = [m * i for i in range(d // m + 1)]#0,d//m  d//m,2d//m ... 
    if arr[-1] != d:
        arr.append(d)
    n = len(arr)
    output = 0

    # Initialize or clone key-value history
    if kv_history is None:
        kv_history = torch.zeros((q.shape[0], q.shape[1], d, e),
                                 dtype=torch.float32,
                                 device=q.device)
    else:
        kv_history = kv_history.clone().contiguous()

    # Process each chunk and accumulate results
    for i in range(n - 1):
        s = arr[i]
        e = arr[i + 1]
        q1 = q[..., s:e]
        k1 = k[..., s:e]
        o, kv = sparse_linear_attention_(q1, k1,s,e, v, ed,kv_history,is_sparsela, sparse_rate)
        output = output + o
    return output, kv 



def linear_decode_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_caches: torch.Tensor, # [B, H, K, V]   #hidden_state
    slope_rate: torch.Tensor,
    slot_idx: torch.Tensor, # used for padding in BATCH dimension
    is_sparsela: bool = True, 
    sparse_rate: float = 0.5, 
    BLOCK_SIZE: int = 32,
) -> torch.Tensor:
    B, H, _, D = q.shape
    assert k.shape == (B, H, 1, D)
    assert v.shape == (B, H, 1, D)
    assert slope_rate.shape == (H, 1, 1)
    # print(kv_caches.shape)
    #256 8 128 128

    kv_history = kv_caches[slot_idx, ...].clone().contiguous()
    # (1) expand the slope rate
    gk = -(slope_rate.expand_as(q).to(torch.float32))  # [H, 1, 1] -> [B, H, T, K]
    sparse_head_dim = int(sparse_rate * q.shape[-1])

    # (2) add sparse operations
    if is_sparsela:
        _, ind = torch.topk(torch.abs(k), sparse_head_dim, dim=-1, largest=False, sorted=False, out=None)
        k = k.scatter(-1, ind, torch.zeros_like(k))
        gk = gk.scatter(-1, ind, torch.zeros_like(gk))
    q,k,v,gk = map(lambda x : rearrange(x,'b h t d -> b t h d'),(q,k,v,gk))#head——first false

    output,out_state = fused_recurrent_gla(q,k,v,gk,scale = 1,initial_state=kv_history,output_final_state=True)
    kv_caches[slot_idx, ...] = out_state.to(kv_caches.dtype)
    output = rearrange(output,"b t h d -> b t (h d)")
    return output.squeeze(1).contiguous()