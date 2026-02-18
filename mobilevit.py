"""
MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer

Independent re-implementation based on Mehta & Rastegari (ICLR 2022).
Paper: https://arxiv.org/abs/2110.02178

Architecture reference: Table 4 (Appendix A)
Block design: Figure 1b, Section 3.1
Key design choices:
    - FFN hidden dim = 2d (not 4d); see Appendix A
    - Patch h = w = 2 at all spatial levels; see Section 3.1, Table 6
    - Expansion ratio = 4 (XS/S) or 2 (XXS); see Appendix A
    - Activation = SiLU (Swish); see Section 3.1
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

__all__ = [
    "MobileViT", "mobilevit_xxs", "mobilevit_xs", "mobilevit_s",
]


# ------------------------------------------------------------------ #
#  Primitive layers                                                    #
# ------------------------------------------------------------------ #

class ConvBnAct(nn.Module):
    """Conv2d → BN → SiLU.  The paper uses Swish throughout."""

    def __init__(self, inp, oup, ks, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, ks, stride, ks // 2,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """MobileNetV2 block.  Shortcut when stride=1 and channels match."""

    def __init__(self, inp, oup, stride=1, expand=4):
        super().__init__()
        mid = int(inp * expand)
        self.shortcut = (stride == 1 and inp == oup)

        layers = []
        if expand != 1:                             # pointwise expansion
            layers.append(ConvBnAct(inp, mid, 1))
        layers += [
            ConvBnAct(mid, mid, 3, stride, groups=mid),  # depthwise
            nn.Conv2d(mid, oup, 1, bias=False),           # projection
            nn.BatchNorm2d(oup),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        out = self.body(x)
        return out + x if self.shortcut else out


# ------------------------------------------------------------------ #
#  Transformer                                                         #
# ------------------------------------------------------------------ #

class MHSA(nn.Module):
    """Multi-head self-attention with fused QKV projection."""

    def __init__(self, dim, heads=1, attn_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        h = self.heads
        qkv = self.qkv(x).reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                    # each (B, h, N, d_k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer.  FFN ratio = 2 (paper Appendix A)."""

    def __init__(self, dim, heads=1, ffn_mult=2.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, heads, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hid), nn.SiLU(inplace=True),
            nn.Linear(hid, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ------------------------------------------------------------------ #
#  MobileViT block  (Figure 1b, Section 3.1)                          #
# ------------------------------------------------------------------ #

class MobileViTBlock(nn.Module):
    """
    Local rep → Unfold → Transformer (global) → Fold → Project → Fuse

    Key insight from the paper (Eq. 1):
        For each pixel position p within a patch, the transformer
        attends across *all* patches.  This gives every pixel an
        effective receptive field of H × W without losing spatial order.
    """

    def __init__(self, in_ch, d_model, depth, heads=1,
                 patch_h=2, patch_w=2, attn_drop=0.0):
        super().__init__()
        self.ph, self.pw = patch_h, patch_w

        # local representation  (conv-3x3 → conv-1x1)
        self.local_rep = nn.Sequential(
            ConvBnAct(in_ch, in_ch, 3),
            nn.Conv2d(in_ch, d_model, 1, bias=False),
        )

        # global: L transformer layers operating on inter-patch tokens
        self.global_rep = nn.Sequential(
            *[TransformerBlock(d_model, heads, attn_drop=attn_drop)
              for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

        # back-projection + fusion with skip
        self.proj = ConvBnAct(d_model, in_ch, 1)
        self.fuse = ConvBnAct(2 * in_ch, in_ch, 3)

    # -- unfold / fold ------------------------------------------------
    # Notation:  B=batch, d=channels, H/W=spatial, ph/pw=patch dims
    #            P = ph*pw  (pixels per patch)
    #            N = (H/ph)*(W/pw)  (number of patches)

    def _unfold(self, x: torch.Tensor):
        """(B, d, H, W) → (B·P, N, d)  ready for transformer."""
        B, d, H, W = x.shape
        ph, pw = self.ph, self.pw
        assert H % ph == 0 and W % pw == 0, \
            f"Spatial dims ({H}×{W}) not divisible by patch ({ph}×{pw})"
        nh, nw = H // ph, W // pw

        # (B, d, nh, ph, nw, pw) → (B, ph, pw, nh, nw, d)
        x = x.reshape(B, d, nh, ph, nw, pw)
        x = x.permute(0, 3, 5, 2, 4, 1)           # pixel-pos first
        x = x.reshape(B * ph * pw, nh * nw, d)     # (B·P, N, d)
        return x, (B, d, nh, nw)

    def _fold(self, x: torch.Tensor, info: Tuple):
        """(B·P, N, d) → (B, d, H, W)."""
        B, d, nh, nw = info
        ph, pw = self.ph, self.pw
        x = x.reshape(B, ph, pw, nh, nw, d)
        x = x.permute(0, 5, 3, 1, 4, 2)           # (B, d, nh, ph, nw, pw)
        return x.reshape(B, d, nh * ph, nw * pw)

    def forward(self, x):
        skip = x

        x = self.local_rep(x)
        x, info = self._unfold(x)
        x = self.global_rep(x)
        x = self.norm(x)
        x = self._fold(x, info)

        x = self.proj(x)
        return self.fuse(torch.cat([skip, x], dim=1))


# ------------------------------------------------------------------ #
#  Network configurations  (Table 4)                                   #
# ------------------------------------------------------------------ #

_CONFIGS: Dict[str, dict] = {
    # ch: [stem, mv2_0, mv2_1, block3, block4, block5, head_conv1x1]
    "xxs": dict(                                    # 1.3 M  (ImageNet)
        ch=[16, 16, 24, 48, 64, 80, 320],
        dims=[64, 80, 96],
        depths=[2, 4, 3],
        heads=[1, 1, 1],                            # paper doesn't specify
        expand=2,                                   # Appendix A exception
    ),
    "xs": dict(                                     # 2.3 M
        ch=[16, 32, 48, 64, 80, 96, 384],
        dims=[96, 120, 144],
        depths=[2, 4, 3],
        heads=[1, 2, 3],                            # following ml-cvnets defaults
        expand=4,
    ),
    "s": dict(                                      # 5.6 M
        ch=[16, 32, 64, 96, 128, 160, 640],
        dims=[144, 192, 240],
        depths=[2, 4, 3],
        heads=[1, 2, 3],
        expand=4,
    ),
}

# After stem(↓2) + 4 stride-2 stages the input is downsampled 32×.
# Also, patch_size=2 at every MobileViT block requires even spatial dims.
_MIN_DIVISOR = 32


# ------------------------------------------------------------------ #
#  Full network                                                        #
# ------------------------------------------------------------------ #

class MobileViT(nn.Module):
    """
    Architecture (Figure 1b):
        Conv-3×3 ↓2  →  MV2  →  MV2↓2 + 2×MV2
        →  [MV2↓2 + MViT-block]×3  →  Conv-1×1 → pool → linear
    """

    def __init__(self, config: str = "xxs", num_classes: int = 1000,
                 attn_drop: float = 0.0):
        super().__init__()
        c = _CONFIGS[config]
        ch = c["ch"]
        dims, depths = c["dims"], c["depths"]
        heads_list = c["heads"]
        exp = c["expand"]

        self.stem = ConvBnAct(3, ch[0], 3, stride=2)

        # MV2-only stages
        self.mv2_0 = InvertedResidual(ch[0], ch[1], 1, exp)
        self.mv2_1 = nn.Sequential(
            InvertedResidual(ch[1], ch[2], 2, exp),
            InvertedResidual(ch[2], ch[2], 1, exp),
            InvertedResidual(ch[2], ch[2], 1, exp),
        )

        # stages with MobileViT blocks
        self.block3 = nn.Sequential(
            InvertedResidual(ch[2], ch[3], 2, exp),
            MobileViTBlock(ch[3], dims[0], depths[0],
                           heads=heads_list[0], attn_drop=attn_drop),
        )
        self.block4 = nn.Sequential(
            InvertedResidual(ch[3], ch[4], 2, exp),
            MobileViTBlock(ch[4], dims[1], depths[1],
                           heads=heads_list[1], attn_drop=attn_drop),
        )
        self.block5 = nn.Sequential(
            InvertedResidual(ch[4], ch[5], 2, exp),
            MobileViTBlock(ch[5], dims[2], depths[2],
                           heads=heads_list[2], attn_drop=attn_drop),
        )

        self.head = nn.Sequential(
            ConvBnAct(ch[5], ch[6], 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(ch[6], num_classes)

        self._init_weights()

    def _check_input(self, x: torch.Tensor):
        _, _, H, W = x.shape
        if H % _MIN_DIVISOR != 0 or W % _MIN_DIVISOR != 0:
            raise ValueError(
                f"Input size {H}×{W} must be divisible by {_MIN_DIVISOR}. "
                f"Try 128, 256, or 384.")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        self._check_input(x)
        x = self.stem(x)
        x = self.mv2_0(x)
        x = self.mv2_1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.head(x)
        return self.fc(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---- factory helpers ------------------------------------------------

def mobilevit_xxs(**kw) -> MobileViT:  return MobileViT("xxs", **kw)
def mobilevit_xs(**kw) -> MobileViT:   return MobileViT("xs", **kw)
def mobilevit_s(**kw) -> MobileViT:    return MobileViT("s", **kw)


if __name__ == "__main__":
    for tag in ("xxs", "xs", "s"):
        m = MobileViT(tag, num_classes=10)
        y = m(torch.randn(1, 3, 256, 256))
        print(f"MobileViT-{tag.upper():<3s}  "
              f"params={m.n_params/1e6:.2f}M  out={y.shape}")
