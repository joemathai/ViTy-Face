# single file & easily hackable dinov2 ViT code
# OG version can be found here : https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py

from typing import Sequence, Tuple, Union, Callable
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from xformers.ops import SwiGLU, memory_efficient_attention, unbind


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224, 
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
    ):
        super().__init__()
        patch_grid_size = (img_size // patch_size, img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=(patch_size, patch_size), 
                              stride=(patch_size, patch_size))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, h, w = x.shape
        assert h % self.patch_size == 0
        assert w % self.patch_size == 0
        x = self.proj(x)
        h, w = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        return x


class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=None,
        drop=0.0,
        bias=True
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                bias=bias,
        )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim,
        init_values=1e-5,
        inplace=False,
    ):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias=True,
        ffn_bias=True,
        drop=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=MemEffAttention,
        ffn_layer=SwiGLUFFNFused,
    ):
        super().__init__()
        self.ls1 = LayerScale(dim, init_values=1.0)
        self.ls2 = LayerScale(dim, init_values=1.0)
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                bias=ffn_bias,
        )

    def forward(self, x):
        def attn_residual_func(x):
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self, 
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        num_register_tokens=0,
    ):
        super().__init__()
        self.n_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.n_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                      in_chans=in_chans, embed_dim=embed_dim, norm_layer=None)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 1370 is picked up from the Dinov2 large model, this is so that the weights can be loaded
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, self.num_register_tokens, embed_dim)) if num_register_tokens else None
        )
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))  # not really used here
        
        blocks_list = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
            )
            for _ in range(depth)
        ]
        
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

        def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
            if not depth_first and include_root:
                fn(module=module, name=name)
            for child_name, child_module in module.named_children():
                child_name = ".".join((name, child_name)) if name else child_name
                named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
            if depth_first and include_root:
                fn(module=module, name=name)
            return module

        def init_weights_vit_timm(module: nn.Module, name: str = ""):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_embedding(self, x, w, h):
        prev_type = x.dtype
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        interpolate_offset = 0.1  # avoid floating point error in interpolation
        N = self.pos_embed.shape[1] - 1
        M = int(math.sqrt(N))
        assert N == M * M
        sx = float(w0 + interpolate_offset) / M
        sy = float(h0 + interpolate_offset) / M
        kwargs = {'scale_factor': (sx, sy)}
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode='bicubic',
            antialias=False,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(prev_type)


    def forward(self, x):
        B, c, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_embedding(x, w, h) 
        if self.register_tokens is not None:
            x = torch.cat(
                    (
                        x[:, :1],
                        self.register_tokens.expand(x.shape[0], -1, -1),
                        x[:, 1:],
                    ),
                    dim=1,
            )
        
        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1: self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
        }


if __name__ == "__main__":
    print('testing ViT model from dinov2')
    from torch.profiler import profile, record_function, ProfilerActivity
    import requests
    from io import BytesIO
    
    # Download the model weights
    url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth'
    response = requests.get(url)
    response.raise_for_status()
    weights = BytesIO(response.content)
    state_dict = torch.load(weights)

    x = torch.randn(4, 3, 224, 224).cuda()
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        num_register_tokens=4,
    )
    model = model.cuda()
    model.load_state_dict(state_dict)
    with torch.no_grad():
        model.eval()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                     profile_memory=True, record_shapes=True) as prof:
            for _ in range(100):
                model(x)
    print(prof.key_averages().table(sort_by='cuda_time_total'))


