import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====== ONN to Sensor Calculation ======
class Sensor(nn.Module):
    def __init__(self, config):
        """
        config:
            - output_dim: 輸出維度 (選擇性，用於裁切)
        """
        super().__init__()
        self.output_dim = config.get("output_dim", None)

    def forward(self, x):
        latent = torch.abs(x)  # 轉成 real

        """
        # 如果要裁切到 output_dim
        if self.output_dim is not None:
            assert int(math.sqrt(self.output_dim)) ** 2 == self.output_dim, \
                "output_dim must be perfect square"
            width = int(self.output_dim ** 0.5)
            latent = latent[:, :, 
                (latent.size(2)-width)//2:(latent.size(2)+width)//2,
                (latent.size(3)-width)//2:(latent.size(3)+width)//2
            ]
        """
            
        if latent.dtype != torch.float32:  
            latent = latent.to(torch.float32)
        return latent.float()   # (B, C, H, W)


# ====== Sensor Noise Simulation ======
class SensorNoise(nn.Module):
    def __init__(self, config):
        """
        config:
            - blur_kernel_size
            - blur_sigma
            - gray_mean
            - gray_sigma
            - gray_ratio
            - noise_std
        """
        super().__init__()
        self.blur_kernel_size = config.get("blur_kernel_size", 15)
        self.blur_sigma = config.get("blur_sigma", 5)
        self.gray_mean = config.get("gray_mean", 0.6)
        self.gray_sigma = config.get("gray_sigma", 0.02)
        self.gray_ratio = config.get("gray_ratio", 0.55)
        self.noise_std = config.get("noise_std", 10/255.)

        # 建立 differentiable Gaussian kernel
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())

    def forward(self, x):
        # Gaussian blur
        x = self._gaussian_blur(x)

        # Gray background
        gray_bg = torch.randn_like(x) * self.gray_sigma + self.gray_mean  
        x = (1 - self.gray_ratio) * x + self.gray_ratio * gray_bg

        # Add Gaussian noise
        noise = torch.randn_like(x) * self.noise_std
        x = torch.clamp(x + noise, 0.0, 1.0)  

        return x

    def _create_gaussian_kernel(self):
        k = self.blur_kernel_size
        sigma = self.blur_sigma
        coords = torch.arange(k) - k // 2
        grid = coords.repeat(k).view(k, k)
        x = grid
        y = grid.t()
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k, k)  
        kernel = kernel.repeat(3, 1, 1, 1)  # RGB 每個 channel 卷積
        return kernel

    def _gaussian_blur(self, x):
        return F.conv2d(x, self.gaussian_kernel, 
                        padding=self.blur_kernel_size // 2, groups=3)
