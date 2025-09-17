import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
註解符號說明:
    ###說明概念
    #說明程式碼
"""
# ==========================================
"""
    The code below is for DNN-end
"""
# ==========================================

# ====== Sinusoidal Time Embedding ======
class TimeEmbedding(nn.Module):  # 用sinusoidal embedding方法把t接入
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        inv_freq = torch.exp(torch.arange(half_dim) * -emb_scale)
        self.register_buffer('inv_freq', inv_freq)  # register_buffer()用來註冊「模型中要跟著儲存，但不訓練的張量」

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        t = t.to(self.inv_freq.device).unsqueeze(1)  # t shape: (B,) → (B, 1), to: move to same device
        freqs = t * self.inv_freq.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=1)  # (B, dim)
        return self.mlp(emb)  # (B, dim)

# ====== Residual Block ======
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim, bottleneck_ratio=0.5):
        super().__init__()
        mid_channels = int(out_channels * bottleneck_ratio)

        # 時間嵌入線性層
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, mid_channels)
        )

        # Bottleneck 結構
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(8, mid_channels)  # 根據channels分組進行normalization

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.norm3 = nn.GroupNorm(8, out_channels)

        # 殘差分支
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.activation = nn.SiLU()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        # 多數時間感知模型都選擇在第一層 activation 之後加上時間 embedding。
        # 如果太早加（如還沒卷積時），t_feat 的特徵尚無法與空間資訊對齊
        # 如果太晚加（如輸出前），時間資訊就來不及參與中間層的建構
        time_emb = self.time_mlp(t).view(t.size(0), -1, 1, 1)
        h = h + time_emb

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        h = self.conv3(h)
        h = self.norm3(h)

        return self.activation(h + self.skip(x))

# ====== Conditioned UNet ======
class ConditionedUNet(nn.Module):
    def __init__(self, img_channels=1, t_dim=64, latent_channels=1, base_channels=64):
        super().__init__()
        self.time_mlp = TimeEmbedding(t_dim)

        in_ch = img_channels + latent_channels  # img_channel: noise channel

        # Encoder: 128 → 64 → 32 → 16 → 8
        self.down1 = ResidualBlock(in_ch, base_channels, t_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ResidualBlock(base_channels, base_channels * 2, t_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, t_dim)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = ResidualBlock(base_channels * 4, base_channels * 8, t_dim)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(base_channels * 8, base_channels * 8, t_dim)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec4 = ResidualBlock(base_channels * 8 + base_channels * 8, base_channels * 4, t_dim)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = ResidualBlock(base_channels * 4 + base_channels * 4, base_channels * 2, t_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, t_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ResidualBlock(base_channels + base_channels, base_channels, t_dim)

        self.out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        t_embed = self.time_mlp(t)

        x = torch.cat([x, cond], dim=1)  # noise + latent, (B, 2, 128, 128)
        # Encoder
        x1 = self.down1(x, t_embed)  # (B, C, 64, 64)
        x2 = self.down2(self.pool1(x1), t_embed)  # (B, C, 32, 32)
        x3 = self.down3(self.pool2(x2), t_embed)  # (B, C, 16, 16)
        x4 = self.down4(self.pool3(x3), t_embed)  # (B, C, 8, 8)

        bottleneck = self.bottleneck(self.pool4(x4), t_embed)  # (B, C, 8, 8)

        # Decoder
        y = self.dec4(torch.cat([self.up4(bottleneck), x4], dim=1), t_embed)  # (B, C, 16, 16)
        y = self.dec3(torch.cat([self.up3(y), x3], dim=1), t_embed)  # (B, C, 32, 32)
        y = self.dec2(torch.cat([self.up2(y), x2], dim=1), t_embed)  # (B, C, 64, 64)
        y = self.dec1(torch.cat([self.up1(y), x1], dim=1), t_embed)  # (B, 2, 128, 128)

        return self.out(y) # (B, 1, 128, 128)


# ====== Diffusion Decoder ======
class DiffusionDecoder(nn.Module):
    def __init__(self, model, timesteps=1000, image_shape=(1, 128, 128), device='cuda'):
        super().__init__()
        self.model = model.to(device)
        self.T = timesteps
        self.image_shape = image_shape
        self.device = device

        self.betas = torch.linspace(1e-4, 0.02, self.T).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)  # cumulative product

    def forward_diffusion(self, x0, t, latent):
        noise = torch.randn_like(latent)
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1 - alpha_hat_t) * noise
        return x_t, noise

    def reverse_sample(self, cond):
        B = cond.size(0)
        x = torch.randn((B, *self.image_shape), device=self.device)
        for t in reversed(range(self.T)):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x, t_tensor, cond)
            alpha = self.alphas[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.betas[t]

            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
        return x



# ====== Autoencoder Integration ======
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, sensor=None, sensor_noise=None):
        super().__init__()
        self.encoder = encoder  # ONN and lens
        self.sensor = sensor  # terahertz sensor
        self.sensor_noise = sensor_noise
        self.decoder = decoder  # diffusion model

    def forward(self, x, t=None, mode='train'):
        latent = self.encoder(x)

        # plug-and-play sensor module
        if self.sensor is not None:
            latent = self.sensor(latent)
        
        # plug-and-play sensor noise module
        if self.sensor_noise is not None:
            latent = self.sensor_noise(latent)

        if mode == 'train':
            x_t, noise = self.decoder.forward_diffusion(x, t, latent)
            noise_pred = self.decoder.model(x_t, t, latent)
            return noise_pred, noise
        elif mode == 'sample':
            return self.decoder.reverse_sample(latent)

# ====== Latent examination ======
class LatentExamination(nn.Module):
    def __init__(self, encoder, sensor=None, sensor_noise=None):
        super().__init__()
        self.encoder = encoder  # ONN and lens
        self.sensor = sensor  # terahertz sensor
        self.sensor_noise = sensor_noise

    def forward(self, x, mode):
        latent = self.encoder(x)

        # plug-and-play sensor module
        if self.sensor is not None:
            latent = self.sensor(latent)
        
        # plug-and-play sensor noise module
        if self.sensor_noise is not None:
            latent = self.sensor_noise(latent)
        
        return latent
