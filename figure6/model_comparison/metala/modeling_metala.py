# coding=utf-8
""" PyTorch MetaLA model."""
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
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from fla.ops.gla import fused_chunk_gla, chunk_gla, fused_recurrent_gla
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

from .configuration_metala import MetaLAConfig
from .norm import SimpleRMSNorm
from .utils import (
    get_activation_fn,
    get_norm_fn,
    logging_info,
    print_module,
    print_params,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MetaLAConfig"


class QVMetaLA_self_aug(nn.Module):

    def __init__(self, embed_dim, num_heads, layer_idx):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx

        self.gate_fn = nn.functional.silu

        dk = self.embed_dim // 2

        self.q_proj = nn.Linear(self.embed_dim, dk, bias=False)
        self.k_gate = nn.Linear(self.embed_dim, dk, bias=False)

        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.g_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.aug_balance = nn.Parameter(0.0 * torch.zeros(dk))

        self.head_dim = self.embed_dim // self.num_heads
        self.key_dim = dk // self.num_heads
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
        self.act = nn.SiLU()

        self.post_init()

    def post_init(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.k_gate.weight, gain=2**-2.5)

    def forward(self, x, attention_mask=None, hidden_state=None, conv_state=None, use_cache=False):
        if attention_mask is not None:
            x = x.mul_(attention_mask.unsqueeze(-1))
    
        # short convolution
        if conv_state is not None and x.shape[1] == 1:
            x = causal_conv1d_update(
                x.squeeze(1),
                conv_state,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation="silu",
            )
            x = x.unsqueeze(1)
        else:
            x = rearrange(x, 'b l d -> b d l').contiguous()
            if use_cache:
                conv_state = F.pad(x, (self.d_conv - x.shape[-1], 0))
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation="silu",
            )
            x = rearrange(x, 'b d l -> b l d').contiguous()

        q = self.q_proj(x)
        k_gate = self.k_gate(x)
        k = 1
        v = self.v_proj(x)
        g = self.g_proj(x)
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))

        output, new_hidden_state = self.MetaLA(q, k, v, k_gate, state=hidden_state)
        output = self.gate_fn(g) * output.to(x.dtype)
        output = self.out_proj(output)

        return output, new_hidden_state, conv_state

    def MetaLA(self, q, k, v, gk, normalizer=16, state=None):

        gk = F.logsigmoid(gk) / normalizer
        k = 1 - torch.exp(gk)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads).to(torch.float32).contiguous()
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads).to(torch.float32).contiguous()
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads).to(torch.float32).contiguous()
        gk = rearrange(gk, 'b l (h d) -> b h l d', h=self.num_heads).to(torch.float32).contiguous()
        aug_balance = rearrange(self.aug_balance, '(h d) -> h d', h=self.num_heads).to(torch.float32).contiguous()

        if self.training or q.shape[2] != 1:
            o, state = fused_chunk_gla(q, k, v, gk, initial_state=state, output_final_state=True)
        else:
            o, state = fused_recurrent_gla(q, k, v, gk, initial_state=state, output_final_state=True)

        # self-augmentation
        augk = torch.einsum('bhld,hd->bhld', k, aug_balance)
        aug_w = torch.einsum('bhld,bhld->bhl', q, augk)
        o = o + F.sigmoid(aug_w.unsqueeze(-1) * v)

        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')

        return o, state

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


class MetaLADecoderLayer(nn.Module):

    def __init__(self, config: MetaLAConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.decoder_embed_dim
        ## token mixer
        self.token_mixer = QVMetaLA_self_aug(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            layer_idx=layer_idx,
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
                state: Optional[Tuple[torch.Tensor]] = None,
                conv_state: Optional[torch.Tensor] = None,
                use_cache: Optional[bool] = False):
        # current does not support padding_mask!
        residual = x

        x = self.token_norm(x)

        x, state, conv_state = self.token_mixer(x=x.transpose(0, 1), attention_mask=padding_mask, hidden_state=state, conv_state=conv_state, use_cache=use_cache)

        x = x.transpose(0, 1) + residual
        x = self.channel_mixer(self.channel_norm(x)) + x

        outputs = x
        return outputs, state, conv_state


METALA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MetaLAConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(METALA_START_DOCSTRING, )
class MetaLAPreTrainedModel(PreTrainedModel):
    config_class = MetaLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MetaLADecoderLayer"]
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
        if isinstance(module, MetaLAModel):
            module.gradient_checkpointing = value


@dataclass
class MetaLAModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    conv_caches: Optional[List[torch.FloatTensor]] = None


METALA_INPUTS_DOCSTRING = r"""
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


@add_start_docstrings(METALA_START_DOCSTRING, )
class MetaLAModel(MetaLAPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MetaLADecoderLayer`]

    Args:
        config: MetaLAConfig
    """

    def __init__(self, config: MetaLAConfig):
        super().__init__(config)
        # hf origin
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        # params
        self.embed_tokens = nn.Embedding(config.vocab_size, config.decoder_embed_dim, self.padding_idx)
        self.layers = nn.ModuleList([MetaLADecoderLayer(config, i) for i in range(config.decoder_layers)])
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

    @add_start_docstrings_to_model_forward(METALA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        conv_caches: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if not self.training and padding_mask != None and padding_mask.eq(self.padding_idx).any():
            raise ValueError(
                "During the inference stage, attn_padding_mask should be either None or should not include the pad token."
            )

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

        # new states of each layer
        # Cache recurrent and convolution states as per-layer tuples.
        state_cache_values = () if use_cache else None
        conv_cache_values = () if use_cache else None

        # b, n, d -> n, b, d
        hidden_states = hidden_states.transpose(1, 0)

        for idx, layer in enumerate(self.layers):
            past_key_value = (past_key_values[idx] if past_key_values is not None else None)
            conv_cache = (conv_caches[idx] if conv_caches is not None else None)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    padding_mask,
                    past_key_value,
                )
            else:
                layer_outputs = layer(hidden_states,
                                      padding_mask=padding_mask,
                                      state=past_key_value,
                                      conv_state=conv_cache,
                                      use_cache=use_cache)

            hidden_states = layer_outputs[0]

            if use_cache:
                state_cache_values += (layer_outputs[1], )
                conv_cache_values += (layer_outputs[-1], )

        hidden_states = self.final_norm(hidden_states)

        # n, b, d -> b, n, d
        hidden_states = hidden_states.transpose(1, 0)

        if not return_dict:
            return tuple(v for v in [hidden_states, state_cache_values, conv_cache_values] if v is not None)
        return MetaLAModelOutputWithPast(last_hidden_state=hidden_states,
                                         past_key_values=state_cache_values,
                                         conv_caches=conv_cache_values)


@dataclass
class MetaLACausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Tuple[Tuple[torch.FloatTensor]]` of shape `(batch_size, num_heads, key_dim, value_dim)`):
            The Linear Attention states of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        conv_caches (`List[torch.FloatTensor]` of shape `(batch_size, hidden_size, kernel_size)`):
            The Convolutional states of the model.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    conv_caches: Optional[List[torch.FloatTensor]] = None


class MetaLAForCausalLM(MetaLAPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = MetaLAModel(config)

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

    @add_start_docstrings_to_model_forward(METALA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MetaLACausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        conv_caches: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MetaLACausalLMOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MetaLAForCausalLM

        >>> model = MetaLAForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
            conv_caches=conv_caches,
            use_cache=use_cache,
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

        return MetaLACausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            conv_caches=outputs.conv_caches,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      conv_caches=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      use_cache=True,
                                      **kwargs):

        if past_key_values:
            input_ids = input_ids[:, -1:]
            attention_mask = attention_mask[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "conv_caches": conv_caches,
            'attention_mask': attention_mask,
            'use_cache': use_cache,
        })
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs: ModelOutput, model_kwargs: Dict[str, Any],
                                            **kwargs) -> Dict[str, Any]:

        super()._update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs)
        model_kwargs["conv_caches"] = outputs.get("conv_caches", None)

        return model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past), )
        return reordered_past
