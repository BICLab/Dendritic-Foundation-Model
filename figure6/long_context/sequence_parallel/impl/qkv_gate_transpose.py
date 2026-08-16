import torch

if torch.cuda.is_available():
    from meepo.cuda.ops import qkv_gate_trans_backward, qkv_gate_trans_forward


def naive_qkv_preprocess(
    qkv_layer, gate_layer, num_heads, num_groups, head_dim
):
    """
    WARNING: the qkv output of this function is in [s, b, nh, d] format
    """
    qkv_layer = torch.nn.functional.silu(qkv_layer)
    gate_layer = torch.nn.functional.sigmoid(gate_layer)

    # [seqlen, batch, hidden_per_partition] -->
    # [seqlen, batch, num_groups_per_partition, (num_heads_per_partition // num_groups_per_partition + 2) * head_dim]
    new_tensor_shape = qkv_layer.size()[:-1] + (
        num_groups,
        -1,
    )
    qkv_layer = qkv_layer.view(*new_tensor_shape)

    # [seqlen, batch, num_groups_per_partition, (num_heads_per_partition // num_groups_per_partition + 2) * head_dim]
    # --> [seqlen, batch, num_groups_per_partition, num_heads_per_partition // num_groups_per_partition * head_dim],
    #     [seqlen, batch, num_groups_per_partition, head_dim],
    #     [seqlen, batch, num_groups_per_partition, head_dim],
    q_layer, k_layer, v_layer = torch.split(
        qkv_layer,
        [
            num_heads // num_groups * head_dim,
            head_dim,
            head_dim,
        ],
        dim=3,
    )
    # [seqlen, batch, num_groups_per_partition, num_heads_per_partition // num_groups_per_partition * head_dim]
    #   --> [seqlen, batch, num_heads_per_partition, head_dim]
    q_layer = q_layer.reshape(q_layer.size(0), q_layer.size(1), -1, head_dim)

    return q_layer, k_layer, v_layer, gate_layer


class FusedQKVGateTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv_gate, num_heads, num_groups):
        if num_heads != num_groups:
            raise ValueError("num_heads must be equal to num_groups")

        q_out, k_out, v_out, gate_out = qkv_gate_trans_forward(
            qkv_gate, num_heads
        )

        ctx.save_for_backward(qkv_gate)

        return q_out, k_out, v_out, gate_out

    @staticmethod
    def backward(ctx, grad_q_out, grad_k_out, grad_v_out, grad_gate_out):
        (qkv_gate,) = ctx.saved_tensors

        grad_qkv_gate = qkv_gate_trans_backward(
            grad_q_out,
            grad_k_out,
            grad_v_out,
            grad_gate_out,
            qkv_gate,
            grad_q_out.shape[1],  # num_heads
        )

        return grad_qkv_gate, None, None


def fused_qkv_gate_transpose(qkv_gate, num_heads, num_groups):
    return FusedQKVGateTranspose.apply(qkv_gate, num_heads, num_groups)
