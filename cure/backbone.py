# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted for fixed-resolution CuRe inference. See LICENSE and NOTICE.

"""PE Core L/14 patch features with Q/V LoRA at 336 x 336 resolution."""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class Rotary2D(nn.Module):
    def __init__(self):
        super().__init__()
        frequencies = 1.0 / (10000 ** (torch.arange(0, 32, 2).float() / 32))
        positions = torch.arange(24).to(frequencies.dtype) + 1
        axis = torch.einsum("..., f -> ... f", positions, frequencies)
        axis = axis.repeat_interleave(2, dim=-1)
        vertical = axis[:, None].expand(24, 24, -1)
        horizontal = axis[None, :].expand(24, 24, -1)
        angles = torch.cat([horizontal, vertical], dim=-1).reshape(576, 64)
        angles = torch.cat([torch.zeros(1, 64), angles], dim=0)
        self.register_buffer("angles", angles[None, None], persistent=False)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, values):
        first, second = values.reshape(*values.shape[:-1], -1, 2).unbind(-1)
        rotated = torch.stack([-second, first], dim=-1).flatten(-2)
        result = values * self.angles.cos() + rotated * self.angles.sin()
        return result.contiguous().to(values.dtype)


class Attention(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty(3072, 1024))
        self.in_proj_bias = nn.Parameter(torch.empty(3072))
        self.out_proj = nn.Linear(1024, 1024)
        self.lora_A_q = nn.Parameter(torch.empty(64, 1024))
        self.lora_B_q = nn.Parameter(torch.empty(1024, 64))
        self.lora_A_v = nn.Parameter(torch.empty(64, 1024))
        self.lora_B_v = nn.Parameter(torch.empty(1024, 64))
        self.rope = rope

    def forward(self, x):
        batch, sequence, width = x.shape
        projection = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        projection = (
            projection.unflatten(-1, (3, width))
            .unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        )
        query, key, value = projection[0], projection[1], projection[2]
        query = query + 2.0 * F.linear(F.linear(x, self.lora_A_q), self.lora_B_q)
        value = value + 2.0 * F.linear(F.linear(x, self.lora_A_v), self.lora_B_v)
        query = query.reshape(batch, sequence, 16, 64).transpose(1, 2)
        key = key.reshape(batch, sequence, 16, 64).transpose(1, 2)
        value = value.reshape(batch, sequence, 16, 64).transpose(1, 2)
        query, key = self.rope(query), self.rope(key)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=64 ** -0.5,
        )
        attended = attended.transpose(1, 2).reshape(batch, sequence, width)
        return F.linear(attended, self.out_proj.weight, self.out_proj.bias)


class ResidualBlock(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.attn = Attention(rope)
        self.ln_1 = nn.LayerNorm(1024, eps=1e-5)
        self.ln_2 = nn.LayerNorm(1024, eps=1e-5)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(1024, 4096)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(4096, 1024)),
        ]))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.resblocks = nn.ModuleList([ResidualBlock(rope) for _ in range(24)])

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


class PECore(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1024, kernel_size=14, stride=14, bias=False)
        self.class_embedding = nn.Parameter(torch.empty(1024))
        self.positional_embedding = nn.Parameter(torch.empty(577, 1024))
        self.ln_pre = nn.LayerNorm(1024, eps=1e-5)
        self.ln_post = nn.LayerNorm(1024, eps=1e-5)
        self.transformer = Transformer(Rotary2D())

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 336, 336):
            raise ValueError("Expected image tensors with shape [N, 3, 336, 336]")
        batch = images.shape[0]
        x = self.conv1(images).permute(0, 2, 3, 1).reshape(batch, 576, 1024)
        cls = self.class_embedding.view(1, 1, -1).expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding[None]
        x = self.transformer(self.ln_pre(x))
        return self.ln_post(x)[:, 1:].mean(dim=1).float()
