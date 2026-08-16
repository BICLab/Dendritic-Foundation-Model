# coding=utf-8
""" MetaLA configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class MetaLAConfig(PretrainedConfig):
    model_type = "metala"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        pad_token_id=2,
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=32000,
        use_cache=True,
        init_std=0.02,
        # model config
        decoder_embed_dim=1024,
        decoder_layers=24,
        decoder_attention_heads=8,
        add_bos_token=False,
        causal=True,
        glu_act="silu",
        glu_dim=5632,
        bias=False,
        norm_type="simplermsnorm",
        no_scale_embedding=True,
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
        self.add_bos_token = add_bos_token
        self.causal = causal
        self.glu_act = glu_act
        self.glu_dim = glu_dim
        self.bias = bias
        self.norm_type = norm_type
        self.no_scale_embedding = no_scale_embedding