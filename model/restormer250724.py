import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# LayerNorm2d: 對每個通道做 LayerNorm（模仿官方實作）
# --------------------------------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1)) # 每個channel獨特的weight, bias(比起單純用1, 0更有表現力)
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean([2, 3], keepdim=True) # (B, C, H, W) -> (B, C, 1, 1)
        var = x.var([2, 3], keepdim=True, unbiased=False) # (B, C, H, W) -> (B, C, 1, 1)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias

# --------------------------------------------------
# MDTA: Multi-Dconv Head Transposed Attention
# --------------------------------------------------
class MDTA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可學習溫度參數

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)  # 1x1 conv → q, k, v
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False)  # depth-wise conv (每個通道只跟自己conv)
        self.project = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)             # [B, 3*C, H, W]
        qkv = self.dwconv(qkv)        # [B, 3*C, H, W]
        q, k, v = qkv.chunk(3, dim=1) # 各為 [B, C, H, W]

        # reshape: [B, heads, C//heads, H*W]
        q = q.view(B, self.num_heads, C // self.num_heads, -1)  # [B, heads, C_head, HW]
        k = k.view(B, self.num_heads, C // self.num_heads, -1)  # [B, heads, C_head, HW]
        v = v.view(B, self.num_heads, C // self.num_heads, -1)  # [B, heads, C_head, HW]

        # Transposed attention: q @ k^T → [B, heads, C_head, C_head]
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.temperature  # [B, heads, C_head, HW] @ [B, heads, HW, C_head] -> [B, heads, C_head, C_head]
        attn = F.softmax(attn, dim=-1)

        # apply attention: attn @ v → [B, heads, C_head, HW]
        out = torch.matmul(attn, v)  # [B, heads, C_head, HW]

        # reshape back to [B, C, H, W]
        out = out.view(B, C, H, W)
        return self.project(out)

# --------------------------------------------------
# Gated-DFFN: Feed-forward Network with Gating
# --------------------------------------------------
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False) # 升維
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

# --------------------------------------------------
# ★ CHANGED: Restormer Block 加入 LayerScale（更穩）
# --------------------------------------------------
class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads, layerscale_init=1e-2, ffn_expansion_factor=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn  = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = GDFN(dim, ffn_expansion_factor)   # ✅ 把 config 的值傳進來
        self.gamma_attn = nn.Parameter(layerscale_init * torch.ones(1, dim, 1, 1))
        self.gamma_ffn  = nn.Parameter(layerscale_init * torch.ones(1, dim, 1, 1))
    def forward(self, x):
        x = x + self.gamma_attn * self.attn(self.norm1(x))
        x = x + self.gamma_ffn  * self.ffn(self.norm2(x))
        return x

# --------------------------------------------------
# ★ CHANGED: Downsample 修正為 PixelUnshuffle → 1×1 Conv
#   輸入 C → 輸出 2C（空間 /2）
# --------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),                                    # (B, C, H, W) → (B, 4C, H/2, W/2)
            nn.Conv2d(in_channels * 4, in_channels * 2, 1, bias=False),  # 4C → 2C
            LayerNorm2d(in_channels * 2)   # ★ 建議加上 更穩定
        )
    def forward(self, x):
        return self.body(x)

# --------------------------------------------------
# ★ CHANGED: Upsample 對稱於 Downsample：1×1 Conv → PixelShuffle
#   輸入 C → 輸出 C/2（空間 *2）
# --------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 1, bias=False),  # 先把通道數調整為 PixelShuffle 的輸入要求
            nn.PixelShuffle(2),                                       # 空間尺寸 *2, 通道減半
            LayerNorm2d(in_channels // 2)                             # ★ 建議加 LayerNorm
        )
    def forward(self, x):
        return self.body(x)

# --------------------------------------------------
# ★ CHANGED: Skip 融合（cat 後用 1×1 Conv 壓回原通道）
# --------------------------------------------------
class SkipFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
    def forward(self, up_feat, enc_feat):
        return self.fuse(torch.cat([up_feat, enc_feat], dim=1))

# --------------------------------------------------
# Restormer 主結構（含改良）
# --------------------------------------------------
class Restormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inp_channels  = config["inp_channels"]
        self.out_channels  = config["out_channels"]
        dim                = config["embed_dim"]
        num_blocks         = config["num_blocks"]
        num_heads          = config["num_heads"]
        layerscale_init    = config.get("layerscale_init", 1e-2)
        self.with_residual = config.get("with_global_residual", True)
        ffn_expansion_factor = config.get("ffn_expansion_factor", 2.66)

        self.patch_embed = nn.Conv2d(self.inp_channels, dim, kernel_size=3, padding=1)

        # Encoder
        self.encoder1 = nn.Sequential(*[RestormerBlock(dim, num_heads[0], layerscale_init, ffn_expansion_factor) for _ in range(num_blocks[0])])
        self.down1    = Downsample(dim)          # dim → 2*dim

        self.encoder2 = nn.Sequential(*[RestormerBlock(dim * 2, num_heads[1], layerscale_init, ffn_expansion_factor) for _ in range(num_blocks[1])])
        self.down2    = Downsample(dim * 2)      # 2*dim → 4*dim

        self.encoder3 = nn.Sequential(*[RestormerBlock(dim * 4, num_heads[2], layerscale_init, ffn_expansion_factor) for _ in range(num_blocks[2])])
        self.down3    = Downsample(dim * 4)      # 4*dim → 8*dim

        # Bottleneck
        self.bottleneck = nn.Sequential(*[RestormerBlock(dim * 8, num_heads[3], layerscale_init, ffn_expansion_factor) for _ in range(num_blocks[3])])

        # Decoder
        self.up3      = Upsample(dim * 8)        # 8*dim → 4*dim
        self.fuse3    = SkipFusion(dim * 4)      # concat(4*dim, 4*dim) → 4*dim
        self.decoder3 = nn.Sequential(*[RestormerBlock(dim * 4, num_heads[2], layerscale_init, ffn_expansion_factor) for _ in range(num_blocks[2])])

        self.up2      = Upsample(dim * 4)        # 4*dim → 2*dim
        self.fuse2    = SkipFusion(dim * 2)
        self.decoder2 = nn.Sequential(*[RestormerBlock(dim * 2, num_heads[1], layerscale_init, ffn_expansion_factor) for _ in range(num_blocks[1])])

        self.up1      = Upsample(dim * 2)        # 2*dim → dim
        self.fuse1    = SkipFusion(dim)
        self.decoder1 = nn.Sequential(*[RestormerBlock(dim, num_heads[0], layerscale_init, ffn_expansion_factor) for _ in range(num_blocks[0])])

        self.output   = nn.Conv2d(dim, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        inp = x  # ★ 用於全域殘差
        x = self.patch_embed(x)

        enc1 = self.encoder1(x)                 # dim
        enc2 = self.encoder2(self.down1(enc1))  # 2*dim
        enc3 = self.encoder3(self.down2(enc2))  # 4*dim
        bot  = self.bottleneck(self.down3(enc3))# 8*dim

        up3  = self.up3(bot)                    # 4*dim
        dec3 = self.decoder3(self.fuse3(up3, enc3))

        up2  = self.up2(dec3)                   # 2*dim
        dec2 = self.decoder2(self.fuse2(up2, enc2))

        up1  = self.up1(dec2)                   # dim
        dec1 = self.decoder1(self.fuse1(up1, enc1))

        out  = self.output(dec1)
        if self.with_residual and (self.inp_channels == self.out_channels):
            # 全域殘差：預測殘差 + 輸入
            return out + inp
        return out