import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# ✅ 可自定義參數（建議調整這裡）
# --------------------------------------------------
RESTORMER_CONFIG = {
    "inp_channels": 1,        # 輸入影像通道數（灰階 = 1）
    "out_channels": 1,       # 輸出影像通道數（灰階 = 1）
    "embed_dim": 48,         # 初始嵌入通道數
    "num_blocks": [4, 6, 6, 8],  # 每個 Encoder/Decoder stage 的 block 數量
    "num_heads": [1, 2, 4, 8]    # 對應每層的 attention head 數量
}

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
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False) # 生成query key value
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3) # Depth-wise 3*3 conv
        self.project = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x) # [B, 3*C, H, W]
        qkv = self.dwconv(qkv) # 同上，但融合局部空間資訊
        q, k, v = qkv.chunk(3, dim=1) # 分割成 Query, Key, Value，各 [B, C, H, W]

        # reshape for attention: [B, heads, C//heads, H*W]
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)

        q = F.normalize(q, dim=2) # 對 dimension 2 做 L2 normalization
        k = F.normalize(k, dim=2)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.temperature  # 對調最後兩個維度, [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v.transpose(-2, -1))  # [B, heads, HW, C//heads]
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        return self.project(out)

# --------------------------------------------------
# Gated-DFFN: Feed-forward Network with Gating
# --------------------------------------------------
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

# --------------------------------------------------
# Restormer Block: MDTA + GDFN with residual & norm
# --------------------------------------------------
class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# --------------------------------------------------
# Downsample: Conv + PixelUnshuffle to reduce H, W
# --------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, bias=False),
            nn.PixelUnshuffle(2)  # spatial size /2, channels *4
        )

    def forward(self, x):
        return self.body(x)

# --------------------------------------------------
# Upsample: PixelShuffle + Conv to recover size
# --------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.PixelShuffle(2),  # spatial size *2, channels /4
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.body(x)

# --------------------------------------------------
# Restormer 主結構
# --------------------------------------------------
class Restormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        inp_ch = config["inp_channels"]
        out_ch = config["out_channels"]
        dim = config["embed_dim"]
        num_blocks = config["num_blocks"]
        num_heads = config["num_heads"]

        self.patch_embed = nn.Conv2d(inp_ch, dim, kernel_size=3, padding=1)

        # Encoder
        self.encoder1 = nn.Sequential(*[RestormerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])
        self.down1 = Downsample(dim)

        self.encoder2 = nn.Sequential(*[RestormerBlock(dim * 2, num_heads[1]) for _ in range(num_blocks[1])])
        self.down2 = Downsample(dim * 2)

        self.encoder3 = nn.Sequential(*[RestormerBlock(dim * 4, num_heads[2]) for _ in range(num_blocks[2])])
        self.down3 = Downsample(dim * 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[RestormerBlock(dim * 8, num_heads[3]) for _ in range(num_blocks[3])])

        # Decoder
        self.up3 = Upsample(dim * 8)
        self.decoder3 = nn.Sequential(*[RestormerBlock(dim * 4, num_heads[2]) for _ in range(num_blocks[2])])

        self.up2 = Upsample(dim * 4)
        self.decoder2 = nn.Sequential(*[RestormerBlock(dim * 2, num_heads[1]) for _ in range(num_blocks[1])])

        self.up1 = Upsample(dim * 2)
        self.decoder1 = nn.Sequential(*[RestormerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])

        self.output = nn.Conv2d(dim, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.patch_embed(x)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.down1(enc1))
        enc3 = self.encoder3(self.down2(enc2))
        bottleneck = self.bottleneck(self.down3(enc3))

        dec3 = self.decoder3(self.up3(bottleneck) + enc3)
        dec2 = self.decoder2(self.up2(dec3) + enc2)
        dec1 = self.decoder1(self.up1(dec2) + enc1)

        return self.output(dec1)