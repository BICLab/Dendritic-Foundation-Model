# from visualizer import get_local
import torch
import torchinfo
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

from hgru2_pytorch import Hgru2_1d
from hgru2_pytorch import Hgru1_real_1d

# from mamba_ssm import Mamba, Mamba_vector
from mamba_pytorch import Mamba, Mamba_vector

from sparsela_hf.modeling_sparsela import sparsela_m, sparsela_v

class SepConv_Spike(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
        
    ):
        super().__init__()
        self.spike1 = nn.ReLU()
        self.spike2 = nn.ReLU()
        self.spike3 = nn.ReLU()
        
        med_channels = int(expansion_ratio * dim)
        
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
            )
        
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        
        x = self.spike1(x)
            
        x = self.pwconv1(x)
        
        x = self.spike2(x)
            
        x = self.dwconv(x)

        x = self.spike3(x)

        x = self.pwconv2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim)

        self.spike1 = nn.ReLU()
        self.spike2 = nn.ReLU()
        
        self.mlp_ratio = mlp_ratio
        hidden_features = int(dim*mlp_ratio)
        
        self.conv1 = nn.Conv2d(
            dim, hidden_features, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(hidden_features)  # 这里可以进行改进
        
        self.conv2 = nn.Conv2d(
            hidden_features, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x))
        x = self.spike2(x)
        x = self.bn2(self.conv2(x))
        x = x_feat + x

        return x


class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0, D_Norm=8, neuron_type='SFA',
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_spike = nn.ReLU()
        self.fc2_spike = nn.ReLU()
        
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        BT, C, H, W = x.shape
        x = x.flatten(2)
        x = self.fc1_spike(x)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x)
        x = self.fc2_spike(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(BT, C, H, W).contiguous()

        return x

class SparseLA2d_Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, lamda_ratio=2, timesteps=16, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.timesteps = timesteps

        self.attn = sparsela_m(embed_dim=dim, num_heads=4, lamda_ratio=lamda_ratio)

        
    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.timesteps
        B = BT // T
        N = H * W
        x = x.reshape(B, T, C, N).permute(0,3,1,2).flatten(0, 1) # [B*N, T, C]
        x, _ = self.attn(x) # [B*N, T, C]
        x = x.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2) # [B, T, C, H, W]
        x = x.flatten(0, 1) # [BT, C, H, W]
        return x

class SparseLA1d_Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, lamda_ratio=2, timesteps=16, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.timesteps = timesteps

        self.attn = sparsela_v(d_model=dim, num_heads=4, lamda_ratio=lamda_ratio)

        
    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.timesteps
        B = BT // T
        N = H * W
        x = x.reshape(B, T, C, N).permute(0,3,1,2).flatten(0, 1) # [B*N, T, C]
        x = self.attn(x) # [B*N, T, C]
        x = x.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2) # [B, T, C, H, W]
        x = x.flatten(0, 1) # [BT, C, H, W]
        return x

class Mamba2d_Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, lamda_ratio=2, timesteps=16, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.timesteps = timesteps

        self.attn = Mamba_vector(d_model=dim, d_state=16, d_conv=4, expand=lamda_ratio)

        
    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.timesteps
        B = BT // T
        N = H * W
        x = x.reshape(B, T, C, N).permute(0,3,1,2).flatten(0, 1) # [B*N, T, C]
        x = self.attn(x) # [B*N, T, C]
        x = x.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2) # [B, T, C, H, W]
        x = x.flatten(0, 1) # [BT, C, H, W]
        return x

class Mamba1d_Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, lamda_ratio=2, timesteps=16, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.timesteps = timesteps

        self.attn = Mamba(d_model=dim, d_state=16, d_conv=4, expand=lamda_ratio)

        
    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.timesteps
        B = BT // T
        N = H * W
        x = x.reshape(B, T, C, N).permute(0,3,1,2).flatten(0, 1) # [B*N, T, C]
        x = self.attn(x) # [B*N, T, C]
        x = x.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2) # [B, T, C, H, W]
        x = x.flatten(0, 1) # [BT, C, H, W]
        return x


class HGRU2_Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, lamda_ratio=2, timesteps=16, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.timesteps = timesteps

        self.attn = Hgru2_1d(embed_dim=dim, expand_ratio=lamda_ratio)

        
    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.timesteps
        B = BT // T
        N = H * W
        x = x.reshape(B, T, C, N).permute(0,3,1,2).flatten(0, 1) # [B*N, T, C]
        x = self.attn(x.transpose(0, 1)).transpose(0, 1) # [B*N, T, C]
        x = x.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2) # [B, T, C, H, W]
        x = x.flatten(0, 1) # [BT, C, H, W]
        return x

class HGRU1_Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, lamda_ratio=2, timesteps=16, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.timesteps = timesteps

        self.attn = Hgru1_real_1d(embed_dim=dim, expand_ratio=lamda_ratio)

        
    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.timesteps
        B = BT // T
        N = H * W
        x = x.reshape(B, T, C, N).permute(0,3,1,2).flatten(0, 1) # [B*N, T, C]
        x = self.attn(x.transpose(0, 1)).transpose(0, 1) # [B*N, T, C]
        x = x.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2) # [B, T, C, H, W]
        x = x.flatten(0, 1) # [BT, C, H, W]
        return x

class SoftmaxAttention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, head_dim=32, num_heads=8, timesteps=16, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.timesteps = timesteps

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.timesteps
        B = BT // T
        N = H * W
        x = x.reshape(B, T, C, N).permute(0,3,1,2).flatten(0, 1) # [B*N, T, C]
        qkv = self.qkv(x).reshape(B*N, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        causal_mask = torch.zeros(T, T)
        causal_mask = causal_mask.masked_fill(torch.triu(torch.ones(T, T), diagonal=1).bool(), float('-inf')).to(x.device)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn += causal_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, T, C).transpose(0, 1).reshape(BT, N, C)
        x = self.proj(x).transpose(1, 2).reshape(BT, C, H, W)
        x = self.proj_drop(x)
        return x

class ConvAttention(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, lamda_ratio=2,
        act1_layer=nn.SiLU, act2_layer=nn.Identity, 
        bias=False, kernel_size=7, padding=3, timesteps=16,
        **kwargs):
        super().__init__()
        med_channels = int(lamda_ratio * dim)
        self.input_resolution = timesteps
        self.timesteps = timesteps

        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = nn.ReLU()
        self.dwconv = nn.Conv1d(med_channels, med_channels, kernel_size, stride=1, padding=3)
        self.act2 = nn.Identity()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.timesteps
        B = BT // T
        N = H * W
        x = x.reshape(B, T, C, N).permute(0,3,1,2).flatten(0, 1) # [B*N, T, C]

        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.dwconv(x.transpose(1,2)).transpose(1,2)
        x = self.act2(x)
        x = self.pwconv2(x)
        x = x.reshape(B, N, T, C).permute(0, 2, 3, 1).reshape(BT, C, H, W).contiguous()
        return x
    
class MLPAttention(nn.Module):
    def __init__(self, dim, timesteps=16, lamda_ratio=8, act_layer=nn.ReLU, **kwargs):
        super().__init__()
        in_features = timesteps
        self.timesteps = timesteps

        hidden_features = int(in_features * lamda_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, in_features)
    
    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.timesteps
        B = BT // T
        N = H * W
        x = x.reshape(B, T, C, N).permute(0,3,1,2).flatten(0, 1) # [B*N, T, C]
        
        x = x.transpose(-1, -2).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.reshape(B, N, C, T).permute(0, 3, 2, 1).reshape(BT, C, H, W).contiguous()
        return x


class Block_tem_attn(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        init_values=1e-6,
        tem_attention=SoftmaxAttention,
        lamda_ratio=4,
        T=16,
    ):
        super().__init__()

        self.tem_attn = tem_attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            lamda_ratio=lamda_ratio,
            timesteps=T,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.tem_attn(x)
        x = x + self.mlp(x)

        return x


class DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        neuron_type = 'SFA',
        D_Norm=8
        
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike=nn.ReLU()
        
    def forward(self, x):

        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_bn(self.encode_conv(x))

        return x


class V3_tem_attn(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dim=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        tem_attention=SoftmaxAttention,
        lamda_ratio=4,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        T=36,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        # embed_dim = [64, 128, 256, 512]
        self.T = T

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1_1 = DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios,)]
        )

        self.downsample2 = DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios,)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios,)]
        )

        self.downsample3 = DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            
        )

        self.block3 = nn.ModuleList(
            [
                Block_tem_attn(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    tem_attention=tem_attention,
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    lamda_ratio=lamda_ratio,
                    T=self.T,
                    
                )
                for j in range(int(depths*0.5))
            ]
        )

        self.downsample4 = DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
            
        )

        self.block4 = nn.ModuleList(
            [
                Block_tem_attn(
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    tem_attention=tem_attention,
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    lamda_ratio=lamda_ratio,
                    T=self.T,
                )
                for j in range(int(depths*0.5))
            ]
        )
        
        self.head = (
            nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        )
        self.spike = nn.ReLU()
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) and m.elementwise_affine == True:
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)

        return x

    def forward(self, x): # [B, T, C, H, W]
        B, T, _, _, _ = x.shape
        x = x.flatten(0, 1).contiguous()
        x = self.forward_features(x) # [BT,C,H,W]
        x = x.flatten(2).mean(2)
        x = self.spike(x)
        x = self.head(x).reshape(B, T, -1).mean(1)
        return x


def V3_tem_attn_sparsela_m_tiny(**kwargs):
    model = V3_tem_attn(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[16, 32, 64, 128],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tem_attention=SparseLA2d_Attention,
        lamda_ratio=1,
        depths=2,
        sr_ratios=1,
        **kwargs,
    )
    return model

def V3_tem_attn_sparsela_v_tiny(**kwargs):
    model = V3_tem_attn(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[16, 32, 64, 128],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tem_attention=SparseLA1d_Attention,
        lamda_ratio=1,
        depths=2,
        sr_ratios=1,
        **kwargs,
    )
    return model


# Backward-compatible model names used by earlier experiment scripts.
V3_tem_attn_spla2d_tiny = V3_tem_attn_sparsela_m_tiny
V3_tem_attn_spla1d_tiny = V3_tem_attn_sparsela_v_tiny

def V3_tem_attn_mamba1d_tiny(**kwargs):
    model = V3_tem_attn(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[16, 32, 64, 128],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tem_attention=Mamba1d_Attention,
        lamda_ratio=1,
        depths=2,
        sr_ratios=1,
        **kwargs,
    )
    return model

def V3_tem_attn_mamba2d_tiny(**kwargs):
    model = V3_tem_attn(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[16, 32, 64, 128],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tem_attention=Mamba2d_Attention,
        lamda_ratio=1,
        depths=2,
        sr_ratios=1,
        **kwargs,
    )
    return model

def V3_tem_attn_hgru2d_tiny(**kwargs):
    model = V3_tem_attn(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[16, 32, 64, 128],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tem_attention=HGRU2_Attention,
        lamda_ratio=1,
        depths=2,
        sr_ratios=1,
        **kwargs,
    )
    return model

def V3_tem_attn_hgru1d_tiny(**kwargs):
    model = V3_tem_attn(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[16, 32, 64, 128],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tem_attention=HGRU1_Attention,
        lamda_ratio=1,
        depths=2,
        sr_ratios=1,
        **kwargs,
    )
    return model

def V3_tem_attn_softmax_tiny(**kwargs):
    model = V3_tem_attn(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[16, 32, 64, 128],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tem_attention=SoftmaxAttention,
        depths=2,
        sr_ratios=1,
        **kwargs,
    )
    return model

def V3_tem_attn_mlp_tiny(**kwargs):
    model = V3_tem_attn(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[16, 32, 64, 128],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tem_attention=MLPAttention,
        lamda_ratio=16,
        depths=2,
        sr_ratios=1,
        T=36,
        **kwargs,
    )
    return model

def V3_tem_attn_conv_tiny(**kwargs):
    model = V3_tem_attn(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[16, 32, 64, 128],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tem_attention=ConvAttention,
        lamda_ratio=0.64,
        depths=2,
        sr_ratios=1,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    model = V3_tem_attn_sparsela_v_tiny().cuda()
    x = torch.randn(2, 36, 2, 128, 128).cuda()
    print(model)
    y = model(x)
    torchinfo.summary(model, (2, 36, 2, 128, 128), device='cuda')