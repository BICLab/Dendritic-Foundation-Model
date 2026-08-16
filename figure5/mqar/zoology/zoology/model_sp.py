import os
import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, DataSegmentConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "figure2" + sweep_id


VOCAB_SIZE = 8_192
MIXER_NAMES = tuple(
    name.strip()
    for name in os.getenv("MQAR_MIXERS", "sparsela").split(",")
    if name.strip()
)


configs = []
for input_seq_len, num_kv_pairs in [
    (512, 64),
]:
    if input_seq_len == 1024:
        batch_size = 64
    elif input_seq_len == 512:
        batch_size = 128
    elif input_seq_len == 256:
        batch_size = 256
    else:
        batch_size = 512


    factory_kwargs = {
        "num_kv_pairs": num_kv_pairs,
        "train_power_a": 0.01,
        "test_power_a": 0.01,
        "random_non_queries": False
    }

    data = DataConfig(
        train_configs=[MQARConfig(num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        test_configs=[MQARConfig(num_examples=3_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        batch_size=batch_size,
        cache_dir=os.getenv("ZOOLOGY_CACHE_DIR", ".cache/zoology"),
    )

    for d_model in [
        64, 
        # 128,
        # 256,
        # 512
    ]:
        cnt = 0
        for lr in  np.logspace(-4, -2, 4):            
            cnt += 1
            if cnt!=2:
                continue
            MIXERS = {
                "attention": dict(
                    name="zoology.mixers.attention.MHA",
                    kwargs={
                        "dropout": 0.1,
                        "num_heads": 2
                    },
                ),
                "mlpattn": dict(
                    name="zoology.mixers.mlpattn.gMLPBlock",
                    kwargs={
                        "seq_len": input_seq_len
                    },
                ),
                "hgrn1": dict(
                    name="zoology.mixers.hgru1_real_1d.Hgru1_real_1d",
                    kwargs={
                    },
                ),
                "hgrn2": dict(
                    name="zoology.mixers.hgru2_1d.Hgru2_1d",
                    kwargs={
                        "expand_ratio":d_model
                    },
                ),
                "m_minilstm": dict(
                    name="zoology.mixers.matrix_minilstm.MinLSTM_matrix",
                    kwargs={
                        "hidden_size": d_model
                    },
                ),
                "minilstm": dict(
                    name="zoology.mixers.minLSTMNet.MinLSTM",
                    kwargs={
                        "hidden_size": d_model
                    },
                ),
                "hyena": dict(
                    name="zoology.mixers.hyena.Hyena",
                    kwargs={
                        "l_max": input_seq_len
                    },
                ),
                "rwkv": dict(
                    name="zoology.mixers.rwkv.RWKVTimeMixer",
                    kwargs={
                        "l_max": input_seq_len,
                    },
                ),
                "base_conv": dict(
                    name="zoology.mixers.base_conv.BaseConv",
                    kwargs={
                        "l_max": input_seq_len,
                        # pass a list of kernel sizes for each of four layers
                        "kernel_size": [3, -1, 3, -1]
                    }
                ),
                "h3": dict(
                    name="zoology.mixers.h3.H3",
                    kwargs={
                        "l_max": input_seq_len,
                        "d_state": input_seq_len,  # makes it mathematically equivalent to Hyena
                        "head_dim": 2
                    }
                ),
                "based": dict(
                    name="zoology.mixers.hybrid.Hybrid",
                    kwargs={
                        "configs": [
                            dict(
                                name="zoology.mixers.base_conv.BaseConv",
                                kwargs={
                                    "l_max": input_seq_len,
                                    # pass a list of kernel sizes for each of four layers
                                    "kernel_size": 3,
                                    "implicit_long_conv": True,
                                }
                            ),
                            dict(
                                name="zoology.mixers.based.Based",
                                kwargs={
                                    "l_max": input_seq_len,
                                    "feature_dim": 8,
                                    "num_key_value_heads": 1,
                                    "num_heads": 1,
                                    "feature_name": "taylor_exp",
                                    "train_view": "quadratic"
                                }
                            )
                        ]
                    }
                ),
                "mamba": dict(
                    name="zoology.mixers.mamba.Mamba",
                    kwargs={}
                ),
                "mamba_v": dict(
                    name="zoology.mixers.mamba_vector.Mamba",
                    kwargs={}
                ),
                "mamba2": dict(
                    name="zoology.mixers.mamba2.Mamba2",
                    kwargs={"d_state": d_model//4, "headdim": 256, "use_mem_eff_path": False}
                ),
                "gla": dict(
                    name="zoology.mixers.gla.GatedLinearAttention",
                    kwargs={"n_head": 2, "use_gk": True, "use_gv": False}
                ),
                "sparsela": dict(
                    name="zoology.mixers.sparsela.SparseLA",
                    kwargs={"n_head": 2, "use_gk": True, "use_gv": False}
                ),
            }
            

            unknown_mixers = sorted(set(MIXER_NAMES) - set(MIXERS))
            if unknown_mixers:
                raise ValueError(
                    "Unknown MQAR mixer(s): " + ", ".join(unknown_mixers)
                )

            for sequence_mixer in MIXER_NAMES:

                if 'mamba' == sequence_mixer or 'mamba_v' == sequence_mixer:
                    block_type = "MambaBlock"
                elif 'mamba2' == sequence_mixer:
                    block_type = "Mamba2Block"
                elif 'hgrn1' == sequence_mixer or 'hgrn2' == sequence_mixer:
                    block_type = "TransformerBlock_hgrn"
                else:
                    block_type = "TransformerBlock"

                pe_flag = sequence_mixer in {"attention","mlpattn", "slideattn"}
                model = ModelConfig(
                    d_model=d_model,
                    n_layers=2,
                    block_type=block_type,
                    max_position_embeddings=input_seq_len if pe_flag else 0,
                    vocab_size=VOCAB_SIZE,
                    sequence_mixer=MIXERS[sequence_mixer],
                    state_mixer=dict(name="zoology.mixers.mlp.LLaMAMLP", kwargs={})
                )
                config = TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    max_epochs=64,
                    run_id=f"{sequence_mixer}-seqlen{input_seq_len}-dmodel{d_model}-lr{lr}-kv{num_kv_pairs}-half",
                    logger=LoggerConfig(
                        project_name=os.getenv("WANDB_PROJECT", "mqar"),
                        entity=os.getenv("WANDB_ENTITY"),
                    )

                )
                configs.append(config)