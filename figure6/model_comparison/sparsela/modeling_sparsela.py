# coding=utf-8
""" PyTorch SparseLA model."""
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from einops import rearrange

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from fla.ops.gla import fused_chunk_gla, chunk_gla, fused_recurrent_gla
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

from .configuration_sparsela import SparseLAConfig
from .norm import SimpleRMSNorm
from .utils import (
    get_activation_fn,
    get_norm_fn,
    logging_info,
    print_module,
    print_params,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "SparseLAConfig"


class SparseLACache(Cache):
    """
    A cache used for storing past states produced by SparseLA. Refer to fla.
    [conv states, hidden states], where
    conv_states: [batch_size, embed_dim, conv_size],
    hidden states: [batch_size, key_dim, value_dim].
    """

    def __init__(
        self,
        seen_tokens: int = 0
    ):

        self.states: List[torch.Tensor] = []
        self._seen_tokens = seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        state: Tuple[torch.Tensor],
        layer_idx: int,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Updates the cache with the new `state` for the layer `layer_idx`.

        Parameters:
            state (`Tuple[torch.Tensor]`):
                The new state to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            offset (`int`):
                The offset of current fed tokens.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            The updated state.
        """

        if isinstance(state, torch.Tensor):
            state = [state,]
        if len(self.states) <= layer_idx:
            self.states.append(list(state))
        else:
            for i, s in enumerate(state):
                self.states[layer_idx][i] = s
            # update the number of seen tokens once we achieve the last layer
            if layer_idx == len(self) - 1:
                self._seen_tokens += offset

        return state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.states)):
            device = self.states[layer_idx].device
            self.states[layer_idx] = self.states[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[torch.Tensor]:
        return tuple(self.states)

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        seen_tokens: int = 0
    ):
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(seen_tokens)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                cache.update(past_key_values[layer_idx], layer_idx)
        return cache


class SparseLAConv(nn.Conv1d):
    """
    Short convolution of SparseLA, used for generation.
    """
        
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        padding: int = 0,
        bias: bool = False,
        precision: torch.dtype = torch.bfloat16,
        **factory_kwargs
    ):
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding=padding,
            bias=bias,
            **factory_kwargs
        )

        self.kernel_size = kernel_size
        self.precision = precision

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):

        if mask is not None:
            x = x.mul_(mask.unsqueeze(-1))
    
        if cache is not None and x.shape[1] == 1:
            x = causal_conv1d_update(
                x.squeeze(1),
                cache,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias.to(self.precision) if self.bias is not None else self.bias,
                activation="silu",
            )
            x = x.unsqueeze(1)
            return x, cache
        
        x = rearrange(x, 'b l d -> b d l').contiguous()
        if use_cache:
            cache = F.pad(x, (self.kernel_size - x.shape[-1], 0))
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(self.weight, "d 1 w -> d w"),
            bias=self.bias.to(self.precision) if self.bias is not None else self.bias,
            activation="silu",
        )
        x = rearrange(x, 'b d l -> b l d').contiguous()
        return x, cache


class SparseLinearAttention(nn.Module):

    def __init__(self, mode, embed_dim, num_heads, layer_idx):
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.gate_normalizer = min(2 * (self.layer_idx + 1), 16)

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not suppoerted mode `{mode}`."

        precision = "bf16"
        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[precision]
        self.precision = dtype
        # Initialize on the caller's default device. This keeps random
        # initialization compatible with both CPU and GPU workflows; callers
        # can move the complete model with ``model.to(device)`` afterwards.
        factory_kwargs = {"dtype": dtype}
        
        self.gate_fn = nn.functional.silu

        dk = self.embed_dim // 2
        dv = self.embed_dim

        self.q_proj = nn.Linear(self.embed_dim, dk, bias=False)
        self.gk_proj =  nn.Linear(self.embed_dim, dk, bias=False)

        self.v_proj = nn.Linear(self.embed_dim, dv, bias=False)
        self.g_proj = nn.Linear(self.embed_dim, dv, bias=True)
        self.out_proj = nn.Linear(dv, self.embed_dim, bias=False)

        self.head_dim = dv // self.num_heads
        self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)

        self.d_conv = 4  
        self.conv1d = SparseLAConv(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            bias=True,
            kernel_size=self.d_conv,
            groups=self.embed_dim,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d.to(self.precision)
        
        self.post_init()

    def post_init(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.gk_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -2.5)
        nn.init.zeros_(self.g_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs
    ):
        mode = 'fused_recurrent' if x.shape[1] == 1 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        conv_states = last_state[0] if last_state is not None else None
        x, conv_states = self.conv1d(x=x, mask=padding_mask, cache=conv_states, use_cache=use_cache)

        q = self.q_proj(x)
        v = self.v_proj(x)
        gk = self.gk_proj(x)
        gk = gk.float()

        if padding_mask is not None:
            v = v.mul_(padding_mask.unsqueeze(-1))

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
        
        hidden_states = last_state[1] if last_state is not None else None
        if mode == 'fused_recurrent':
            o, hidden_states = fused_recurrent_gla(q, k, v, gk, initial_state=hidden_states, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            o, hidden_states = fused_chunk_gla(q, k, v, gk, initial_state=hidden_states, output_final_state=use_cache)
        elif mode == 'chunk':
            o, hidden_states = chunk_gla(q, k, v, gk, initial_state=hidden_states, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
        if use_cache and past_key_values is not None:
            last_state = (conv_states, hidden_states)
            past_key_values.update(last_state, self.layer_idx, q.shape[2])
       
        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')

        g = self.g_proj(x)
        o = self.gate_fn(g) * o.to(x.dtype)
        o = self.out_proj(o)

        return o, past_key_values

    def extra_repr(self):
        return print_module(self)


class GLU(nn.Module):

    def __init__(self, d1, d2, act_fun, bias=False):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.l1 = nn.Linear(d1, d2, bias=bias)
        self.l2 = nn.Linear(d1, d2, bias=bias)
        self.l3 = nn.Linear(d2, d1, bias=bias)
        self.act_fun = get_activation_fn(act_fun)

    def forward(self, x):
        o1 = self.act_fun(self.l1(x))
        o2 = self.l2(x)
        output = o1 * o2
        output = self.l3(output)

        return output


class SparseLADecoderLayer(nn.Module):

    def __init__(self, config: SparseLAConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.decoder_embed_dim
        ## token mixer
        self.token_mixer = SparseLinearAttention(
            mode=config.attn_mode,
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            layer_idx=layer_idx
        )
        self.token_norm = get_norm_fn(config.norm_type)(self.embed_dim)

        ## channel mixer
        self.glu_act = config.glu_act
        self.glu_dim = config.glu_dim
        if self.glu_dim == -1:
            self.glu_dim = self.embed_dim
        bias = config.bias
        self.channel_mixer = GLU(self.embed_dim, self.glu_dim, self.glu_act, bias=bias)
        self.channel_norm = get_norm_fn(config.norm_type)(self.embed_dim)

    def forward(self,
                x,
                padding_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
                use_cache: Optional[bool] = False,):
        residual = x

        x = self.token_norm(x)

        x, past_key_values = self.token_mixer(
            x=x, 
            padding_mask=padding_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        x = x + residual
        x = self.channel_mixer(self.channel_norm(x)) + x

        outputs = x
        return outputs, past_key_values


SPARSELA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SparseLAConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(SPARSELA_START_DOCSTRING, )
class SparseLAPreTrainedModel(PreTrainedModel):
    config_class = SparseLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SparseLADecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, SparseLAModel):
            module.gradient_checkpointing = value


SPARSELA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(SPARSELA_START_DOCSTRING, )
class SparseLAModel(SparseLAPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`SparseLADecoderLayer`]

    Args:
        config: SparseLAConfig
    """

    def __init__(self, config: SparseLAConfig):
        super().__init__(config)
        # hf origin
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        # params
        self.embed_tokens = nn.Embedding(config.vocab_size, config.decoder_embed_dim, self.padding_idx)
        self.layers = nn.ModuleList([SparseLADecoderLayer(config, layer_idx) for layer_idx in range(config.decoder_layers)])
        self.final_norm = get_norm_fn(config.norm_type)(config.decoder_embed_dim)
        self.embed_dim = config.decoder_embed_dim
        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(self.embed_dim)
        self.num_layers = config.decoder_layers

        # Initialize weights and apply final processing
        self.post_init()

    def extra_repr(self):
        return print_module(self)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(SPARSELA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if not self.training and padding_mask != None and padding_mask.eq(self.padding_idx).any():
            raise ValueError(
                "During the inference stage, attn_padding_mask should be either None or should not include the pad token."
            )

        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            # !!! use embed_scale
            inputs_embeds = self.embed_scale * self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if use_cache and not isinstance(past_key_values, SparseLACache):
            past_key_values = SparseLACache.from_legacy_cache(past_key_values)

        all_hidden_states = () if output_hidden_states else None
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, None)

                    return custom_forward

                hidden_states, past_key_values = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    padding_mask,
                    past_key_values,
                    use_cache,
                )
            else:
                hidden_states, past_key_values = layer(hidden_states,
                                      padding_mask=padding_mask,
                                      past_key_values=past_key_values,
                                      use_cache=use_cache,)

        hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class SparseLAForCausalLM(SparseLAPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = SparseLAModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.decoder_embed_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(SPARSELA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, SparseLAForCausalLM

        >>> model = SparseLAForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            padding_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      use_cache=True,
                                      **kwargs):
        
        if past_key_values:
            if not isinstance(past_key_values, SparseLACache):
                past_key_values = SparseLACache.from_legacy_cache(past_key_values, input_ids.shape[1] - 1)
            input_ids = input_ids[:, -1:]
            attention_mask = attention_mask[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past), )
        return reordered_past
