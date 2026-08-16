#    Copyright 2024 OpenNLPLab
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# coding=utf-8
""" Transnormer configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class TransnormerConfig(PretrainedConfig):
    model_type = "transnormer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=64000,
        use_cache=True,
        init_std=0.02,
        # model config
        decoder_embed_dim=1024,
        decoder_layers=24,
        decoder_attention_heads=8,
        no_scale_embedding=False,
        add_bos_token=False,
        norm_type="simplermsnorm",
        linear_use_lrpe_list=[],
        hidden_dim=1024,
        linear_act_fun="silu",
        glu_dim=2816,
        bias=False,
        gate_dim=16,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        # hf origin
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.init_std = init_std
        # add
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.no_scale_embedding = no_scale_embedding
        self.add_bos_token = add_bos_token
        self.norm_type = norm_type
        self.linear_use_lrpe_list = linear_use_lrpe_list
        self.hidden_dim = hidden_dim
        self.linear_act_fun = linear_act_fun
        self.glu_dim = glu_dim
        self.bias = bias
        self.gate_dim = gate_dim
