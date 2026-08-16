import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ..gla.inter_chunk_contribution.fn import inter_chunk_onc
from ..gla.intra_chunk_contribution.fn import intra_chunk_onc
from ..helpers import get_activation_fn, get_norm_fn, print_module, print_params
from ..hgru_real_cuda import HgruRealFunction


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
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        self.out_act = get_activation_fn(uv_act_fun)
        self.use_norm = use_norm
        self.use_fla = use_fla
        if self.use_norm:
            self.norm = get_norm_fn(norm_type)(embed_dim)

        self.chunk_size = 128

        if self.expand_ratio < 16:
            self.forward = self.forward_lesshead
            self.scan = HgruRealFunction.apply

        if self.use_fla:
            from fla.ops import fused_chunk_gla

            self.scan = fused_chunk_gla
    #lower_bound 需要在整个模型上额外设计一些参数
    '''
    self.lower_bounds = nn.Parameter(torch.ones(n_layers, d_model), requires_grad=True)
    if "hgru" in self.layer_name:
            lower_bounds = F.softmax(self.lower_bounds, dim=0)
            lower_bound = torch.zeros(self.d_output).to(outputs)
        
        cnt = 0
        for layer, prev_state in zip(self.layers, prev_states):
            if "hgru" in self.layer_name:
                kwargs["lower_bound"] = lower_bound
            outputs, state = layer(
                outputs, *args, state=prev_state, **kwargs
            )
            next_states.append(state)
            if self.track_norms:
                output_norms.append(torch.mean(outputs.detach() ** 2))
            if "hgru" in self.layer_name:
                # skip glu
                if cnt % 2 == 0:
                    lower_bound += lower_bounds[cnt // 2]
            cnt += 1
        outputs = self.norm(outputs)
    意思是设计如果2层,d通道,先生成一个2*d的额外参数,过softmax后,随后按照层数进行加和,第一层为0,
    '''


    def forward(self, x, lower_bound=0):
        ## x: n b d #length batch dimension
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
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [V, Q, F_, lower_bound],
        )
        # head 分配，在hgru1 无，除此以外确实是一样的
        lambda_ = lower_bound + (1 - lower_bound) * F_

        log_lambda_ = torch.log(lambda_)

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
        output = self.out_proj(o)
        return output

    def forward_lesshead(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        feature = self.in_proj(x)
        input, output_gate, forget_gate = feature.chunk(3, dim=-1)
        input = self.act(input)
        output_gate = self.out_act(output_gate)
        forget_gate = F.sigmoid(forget_gate)
        if type(lower_bound) == int:
            lower_bound = torch.zeros_like(x).to(x)

        # reshape
        input, output_gate, forget_gate, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [input, output_gate, forget_gate, lower_bound],
        )

        # mix
        lambda_ = lower_bound + (1 - lower_bound) * forget_gate
        input = torch.einsum("... h d, ... h e -> ... h d e", 1 - lambda_, input)
        lambda_ = repeat(lambda_, "... h d -> ... h d e", e=self.expand_ratio)

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
            d=self.expand_ratio,
            e=self.expand_ratio,
        )
        output_state = torch.einsum(
            "... h d e, ... h d -> ... h e", output_state, output_gate
        )
        output_state = rearrange(output_state, "... h e -> ... (h e)")

        # output gate
        if self.use_norm:
            output_state = self.norm(output_state)

        # out proj
        output = self.out_proj(output_state)

        return output

    def extra_repr(self):
        return print_module(self)
