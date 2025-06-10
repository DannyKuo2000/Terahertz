import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Experiments Relative parameters:
    Refractive index: 1.7
    Absorption coefficient: 1e-5
    Sub THz: 0.2e12

註解符號說明:
    ###說明概念
    #說明程式碼
"""

# ====== Air Diffraction Calculation ======
"""
這段程式碼模擬的是：給定一個以 dx 為取樣解析度的波前（E），這個波前在空氣中傳播距離 z 後，到達前方某一平面時的波場分布。
重點觀念：
無限長的平面波之所以看起來沒有繞射，是因為都會有其他部分進行相消。如果我們只注意有限區域，其他部分視作被遮擋，繞射的情況就會出現
| 可能修正方法               | 效果                                 |
| -----------------------   | ------------------------------------ |
| 降低 `dx`                 | 增加 Nyquist frequency，降低 aliasing |
| 增加 `num_size`（區域大小）| 降低邊界效應與頻率截斷誤差              |
| 初始波前 band-limiting     | 確保不超過模擬頻率範圍                 |
| 使用 zero-padding         | 緩解邊界效應，讓 FFT 更精確            |
| 使用 spectral method 判斷誤差 | 頻譜分析可以幫你預估保留了多少能量   |

"""
class DiffractiveLayer(nn.Module):
    def __init__(self, dx=0.00075, num_size=128, frequency=0.2e12, z=0.1, refractive_index=1):
        super().__init__()
        self.dx = dx  # resolution (m)
        self.size = num_size  # number of optical neurons in one dimension
        # self.ll = ll        # layer length (m)
        self.wl = 2.998e8 / frequency  # wavelength = light speed / frequency (m)
        self.z = z  # distance between two layers (m)
        self.n = refractive_index

        ### 計算一個node傳播在空氣中的convolution 
        # 用angular spectrum技巧，預先計算 kz(因為與輸入無關)
        # 計算一維頻率軸fx, 範圍為[-1/(2*dx), 1/(2*dx)], 分成size份 (以0為中心，因為有用np.fft.fftshift)
        fx = np.fft.fftshift(np.fft.fftfreq(self.size, d=self.dx))
        fxx, fyy = np.meshgrid(fx, fx)  # 弄出網格

        ### 傅立葉轉換的頻率成分正好就是k: k_x=2*pi*f_x, k_y=2*pi*f_y (angular spectrum重要觀念)
        # 計算從一個node跑出的kz，kz**2 = k**2 - kx**2 - ky**2
        argument = (2 * np.pi * self.n)**2 * ((1. / self.wl)**2 - fxx**2 - fyy**2)
        tmp = np.sqrt(np.abs(argument))

        # >=0: propagating, <0: evanescent(光學細節)
        kz = np.where(argument >= 0, tmp, 1j * tmp)

        # 考慮兩層ONN之間的距離
        self.jkz = torch.from_numpy(np.exp(1j * kz * self.z)).to(device)

    def forward(self, E):
        c_fft = torch.fft.fft2(E)  # 2D FFT
        c = torch.fft.fftshift(c_fft)  # 移動低頻分量到中心(常見的十字狀圖)

        # 在frequency domain相乘, 等於在space domain做convolution (驗證?)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * self.jkz))
        return angular_spectrum

# ======= Interface Interaction Calculation ======
class FresnelInterface(nn.Module):
    def __init__(self, dx=0.00075, num_size=128, frequency=0.2e12, keep_reflection=False, complex_index=False, n1=1, n2=1.7):
        """
        擴充版 FresnelInterface 支援：
        - 偏振分離計算（TE/TM） : 同時考慮兩種偏振態的 Fresnel 係數。
        - 全反射處理（虛數透射角）: 若入射角超過臨界角，自動產生虛數的折射角，保留反射波。
        - 複數折射率（模擬吸收介質）: 模擬吸收介質或金屬等材料（e.g. 𝑛=1.5+0.2𝑖）。
        - 選擇性保留反射波 : 你可選擇是否返回反射波（如干涉模擬時很有用）。 
        
        參數說明：
        dx                : 空間解析度（每點距離，m）
        num_size          : 點陣大小（如128表示128x128）
        n1, n2            : 折射率（可為複數）
        frequency         : 波頻率（Hz）
        keep_reflection   : 是否保留反射波
        complex_index     : 是否使用複數折射率
        """
        super().__init__()
        self.dx = dx
        self.size = num_size
        self.n1 = n1 if complex_index else complex(n1, 0.0)
        self.n2 = n2 if complex_index else complex(n2, 0.0)
        self.keep_reflection = keep_reflection
        self.wl = 2.998e8 / frequency  # 真空波長
        self.k0 = 2 * np.pi / self.wl  # 真空波數

        # 建立頻率網格
        fx = np.fft.fftshift(np.fft.fftfreq(self.size, d=self.dx))
        fxx, fyy = np.meshgrid(fx, fx)
        kx = 2 * np.pi * fxx
        ky = 2 * np.pi * fyy
        k_perp = np.sqrt(kx**2 + ky**2)

        # 入射角的 sin(theta_i)
        sin_theta_i = k_perp / (self.k0 * abs(self.n1))
        sin_theta_i = np.clip(sin_theta_i, 0, 1)

        # cos(theta_i), sin(theta_t), cos(theta_t)
        cos_theta_i = np.sqrt(1 - sin_theta_i**2 + 0j)
        sin_theta_t = (self.n1 / self.n2) * sin_theta_i
        cos_theta_t = np.sqrt(1 - sin_theta_t**2 + 0j)  # 虛數表示全反射

        # Fresnel TE (s) 和 TM (p) 偏振反射與透射係數
        rs = (self.n1 * cos_theta_i - self.n2 * cos_theta_t) / (self.n1 * cos_theta_i + self.n2 * cos_theta_t)
        ts = (2 * self.n1 * cos_theta_i) / (self.n1 * cos_theta_i + self.n2 * cos_theta_t)

        rp = (self.n2 * cos_theta_i - self.n1 * cos_theta_t) / (self.n2 * cos_theta_i + self.n1 * cos_theta_t)
        tp = (2 * self.n1 * cos_theta_i) / (self.n2 * cos_theta_i + self.n1 * cos_theta_t)

        # 將 rs, rp, ts, tp 組成平均強度反射率與透射率
        R = 0.5 * (np.abs(rs)**2 + np.abs(rp)**2)
        T = 0.5 * (np.abs(ts)**2 + np.abs(tp)**2)

        self.R = torch.from_numpy(R).to(torch.float32)  # 強度反射率
        self.T = torch.from_numpy(T).to(torch.float32)  # 強度透射率

        # 若保留複數振幅的反射波與透射波
        self.rs = torch.from_numpy(rs).to(torch.complex64)
        self.rp = torch.from_numpy(rp).to(torch.complex64)
        self.ts = torch.from_numpy(ts).to(torch.complex64)
        self.tp = torch.from_numpy(tp).to(torch.complex64)

    def forward(self, E):
        """
        輸入 E 是一個空間波前（複數值的張量），尺寸為 (B, H, W) 或 (H, W)
        根據設定回傳透射波，必要時也可同時回傳反射波
        """
        E_f = torch.fft.fftshift(torch.fft.fft2(E))

        # 計算複數振幅平均的透射分量（可拓展為偏振分離）
        t_avg = 0.5 * (self.ts + self.tp).to(E.device)
        r_avg = 0.5 * (self.rs + self.rp).to(E.device)

        E_f_transmitted = E_f * t_avg
        E_f_reflected = E_f * r_avg

        E_out = torch.fft.ifft2(torch.fft.ifftshift(E_f_transmitted))

        if self.keep_reflection:
            E_ref = torch.fft.ifft2(torch.fft.ifftshift(E_f_reflected))
            return E_out, E_ref
        else:
            return E_out

# ====== Material Phase Control ======
class MaterialLayer(nn.Module):
    def __init__(self, num_size=128):
        super().__init__()
        init_phase = 2 * np.pi * np.random.rand(num_size, num_size)

        # 這裡才是實際印製產生的phase變化
        self.phase = nn.Parameter(torch.from_numpy(init_phase))

    def forward(self, x):
        # 加入印製的相位調整
        phase_mask = torch.exp(1j * self.phase)
        return x * phase_mask

# ====== ONN ensemblance ======
class ONN(nn.Module):
    def __init__(self, num_layers=3, num_size=128):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DiffractiveLayer(num_size=num_size))  # 空氣層
            self.layers.append(MaterialLayer(num_size=num_size))  # 材料層
        # 到sensor的空氣層在那邊計算

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
# ====== ONN to Sensor Calculation ====== (計算ONN出來到camera後經過air or lens路徑)
class Sensor(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.last_diffraction_layer = DiffractiveLayer(dx=0.00075, num_size=128, frequency=0.2e12, z=0.1)  # 理論上是一層空氣 但可能有鏡片
        # self.sensor_kernel_size = 14  # average pooling kernel size 
        # self.sensor_stride_size = 14

    def forward(self, x):
        latent = self.last_diffraction_layer(x)
        latent = torch.abs(latent)  # 轉成 real

        # print(f"latent size: {latent.size()}")
        # latent = F.avg_pool2d(latent, kernel_size=self.sensor_kernel_size, stride=self.sensor_stride_size)
        
        ### 如果sensor只取中間某些部分
        """
        assert int(math.sqrt(self.output_dim)) ** 2 == self.output_dim, "output_dim must be a perfect square"
        width = int(self.output_dim ** 0.5)
        latent = latent[:, :, (latent.size(2)-width)//2:(latent.size(2)+width)//2, (latent.size(3)-width)//2:(latent.size(3)+width)//2]  # 裁切中間 28x28
        print(f"latent size: {latent.size()}")
        """
        if latent.dtype != torch.float32:  # change dtype to consist with encoder
            latent = latent.to(torch.float32)
        return latent.float()   # 2D output

# ====== Sensor Noise Simulation ======
class SensorNoise(nn.Module):
    def __init__(self, blur_kernel_size=15, blur_sigma=5, gray_mean=0.6, gray_sigma=0.02, gray_ratio=0.55, noise_std=10/255.):
        super().__init__()
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.gray_mean = gray_mean  # gray background mean
        self.gray_sigma = gray_sigma  # gray background sigma
        self.gray_ratio = gray_ratio  # gray background ratio
        self.noise_std = noise_std

        # 預建立differentiable Gaussian kernel（固定參數部會進行訓練）
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())

    def forward(self, x):
        """
        x: shape (B, C, H, W), dtype=torch.float32, range=[0,1]
        模擬實測感測器影像效果：模糊、加灰背景、加雜訊
        """
        ### Gaussian blur
        x = self._gaussian_blur(x)

        ### Gray background: simulate sensor back light, making back light value around 0.55~0.65
        # ~ N(0.6, 0.02), 0.6 ~= 155/255, 0.02 ~= 5/255
        gray_bg = torch.randn_like(x) * self.gray_sigma + self.gray_mean  
        # mix with ratio
        x = (1 - self.gray_ratio) * x + self.gray_ratio * gray_bg

        ### add Gaussian noise
        # ~ N(0, 10/255)
        noise = torch.randn_like(x) * self.noise_std
        # constrain value between 0.0~1.0
        x = torch.clamp(x + noise, 0.0, 1.0)  

        return x

    def _create_gaussian_kernel(self):
        """建立一個可用於 conv2d 的 Gaussian kernel"""
        k = self.blur_kernel_size
        sigma = self.blur_sigma
        coords = torch.arange(k) - k // 2
        grid = coords.repeat(k).view(k, k)
        x = grid
        y = grid.t()
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k, k)  # shape = (1, 1, k, k)
        kernel = kernel.repeat(3, 1, 1, 1)  # 對 RGB 每個 channel 分別卷積
        return kernel

    def _gaussian_blur(self, x):
        """以 depthwise conv2d 實作高斯模糊"""
        return F.conv2d(x, self.gaussian_kernel, padding=self.blur_kernel_size // 2, groups=3)

# ==========================================
    '''
    The code below is for DNN-end
    '''
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
