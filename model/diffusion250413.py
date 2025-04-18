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
# ==== Diffraction Calculation ====
class DiffractiveLayer(nn.Module):
    def __init__(self, dx=0.00075, num_size=28*4, ll=0.01, frequency=0.2e12, z=0.01): # original: size = 36
        super().__init__()
        self.dx = dx       # resolution (m)
        self.size = num_size       # number of optical neurons in one dimension
        # self.ll = ll        # layer length (m)
        self.wl = 2.998e8 / frequency    # wavelength = light speed / frequency (m)
        self.z = z         # distance between two layers (m)

    def forward(self, E):
        """
        這段forward主要考慮的是在空氣中傳播的疊加情形, ONN的影響會在Net()再額外加入。
        先將每個node傳播到下一層的kz大小寫出(前半部分), 再把ONN之間的距離考慮進去, 用以算出相位變化的convolution(jkz = torch.from_numpy...)
        最後乘上經過FFT過的input訊號(angular_spectrum = ...)
        Question: 把全反射在這裡考慮似乎有些奇怪
        """
        ### 從frequency domain進行計算
        # 2D FFT
        c_fft = torch.fft.fft2(E) 
        # 移動低頻分量到中心(常見的十字狀圖)
        c = torch.fft.fftshift(c_fft) 

        # 計算一維頻率軸fx, 範圍為[-1/(2*dx), 1/(2*dx)], 分成size份 (以0為中心，因為有用np.fft.fftshift)
        fx = np.fft.fftshift(np.fft.fftfreq(self.size, d = self.dx)) 
        # 弄出網格
        fxx, fyy = np.meshgrid(fx, fx)

        ### 算kz**2 = k**2 - kx**2 - ky**2
        # 計算從一個node跑出的kz
        argument = (2 * np.pi)**2 *((1. / self.wl)**2 - fxx**2 - fyy**2)
        # 算kz
        tmp = np.sqrt(np.abs(argument))

        # >=0: propagating, <0: evanescent(光學細節)
        # 角度過大產生全反射, 剩下會exponential decay波跟平行於介面的evanescent波. 乘上i, 使其有exponential decay
        kz = np.where(argument >= 0, tmp, 1j*tmp)

        ### 加入通過ONN後，在空氣中傳播的疊加：
        # 考慮兩層ONN之間的距離
        jkz = torch.from_numpy(np.exp(1j * kz * self.z)).to(device)
        # 在frequency domain相乘, 等於在space domain做convolution (公式需驗證)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * jkz))

        return angular_spectrum

# ==== ONN Model ====
class Net(nn.Module):
    def __init__(self, num_layers=3, num_size=28*4):
        super().__init__()
        # random initialized [0, 2*pi] 每層長寬各為size, 共num_layers, call: self.phase1[i]
        # torch.from_numpy(): 將NumPy轉換成PyTorch
        # torch.nn.Parameter: 允許在反向傳播中更新
        self.phase1 = [torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size = (num_size, num_size))))for _ in range(num_layers)] # original: size = (36, 36), .astype("float32")

        ### 將 self.phase1[i] 的每個張量註冊到模型中，使它們可以被 PyTorch 的自動微分系統追蹤。
        for i in range(num_layers):
            self.register_parameter("phase1" + "_" + str(i), self.phase1[i])

        ### 用一個layers組(block)裝入所有layers
        # torch.nn.ModuleList：用list存許多層DiffractiveLayer(), PyTorch的特殊list
        self.diffractive_layers_block1 = torch.nn.ModuleList([DiffractiveLayer() for _ in range(num_layers)])

        # 單獨定義最後一層(沒用到)
        # self.last_diffractive_layer1 = DiffractiveLayer()

        # 定義一個 Softmax 函數(沒用到)
        # self.softmax = torch.nn.Softmax(dim = -1)

    def forward(self, x):
        for index, layer in enumerate(self.diffractive_layers_block1):
            # 自動調用forward() (PyTorch特性)
            # 把layers組的每層layer輪流抓出來算
            #print(f"x shape in ONN: {x.shape}")
            temp = layer(x)

            # 這裡才是實際印製產生的phase變化
            exp_j_phase = self.phase1[index]

            # 加入印製的相位調整
            x = temp * torch.exp(1j * exp_j_phase)

        output = x #output = torch.abs(x)
        return output
    
# ==== ONN to Sensor Calculation ==== (計算ONN出來到camera的路徑)
class Sensor(nn.Module):
    def __init__(self, output_dim=(8*8)):
        super().__init__()
        self.output_dim = output_dim
        self.air_or_lens_layers = DiffractiveLayer(dx=0.00075, num_size=28*4, ll=0.01, frequency=0.2e12, z=0.01)  # 理論上是一層空氣 但可能有鏡片
        # self.sensor_kernel_size = 14  # average pooling kernel size 
        # self.sensor_stride_size = 14

    def forward(self, x):
        latent = self.air_or_lens_layers(x)
        latent = torch.abs(latent)  # 轉成 real

        # print(f"latent size: {latent.size()}")
        # latent = F.avg_pool2d(latent, kernel_size=self.sensor_kernel_size, stride=self.sensor_stride_size)
        
        assert int(math.sqrt(self.output_dim)) ** 2 == self.output_dim, "output_dim must be a perfect square"
        width = int(self.output_dim ** 0.5)
        latent = latent[:, :, (latent.size(2)-width)//2:(latent.size(2)+width)//2, (latent.size(3)-width)//2:(latent.size(3)+width)//2]  # 裁切中間 28x28
        print(f"latent size: {latent.size()}")
        return latent  # 2D output

# ==== Sinusoidal Time Embedding ====
def sinusoidal_embedding(timesteps, dim): # 用sinusoidal embedding方法把t接入
    """
    將整數 timestep 轉成 sinusoidal embedding 向量。
    timesteps: shape [B]
    return: shape [B, dim]
    """
    device = timesteps.device
    half_dim = dim // 2  # 這裡是 32
    emb = math.log(10000) / (half_dim - 1)  # 計算頻率比例
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # shape: [32]
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)  # shape: [B, 32]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # shape: [B, 64]
    return emb  # shape: (B, dim)


# ========= ResidualBlock =========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),  # 分組正規化，這是對卷積層的輸出進行正規化處理，可以加速訓練並提高模型性能。
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()  # 如果channel數一樣就identity

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))


# ========= Conditioned UNet =========
class ConditionedUNet(nn.Module):
    def __init__(self, noise_channel=1, t_dim=64, latent_side_length=8, inner_channels=32):
        super().__init__()
        self.latent_side_length = latent_side_length
        self.t_dim = t_dim

        # cond 與 t 各自轉成 latent feature map (B, C, 8, 8)
        self.fc_t = nn.Linear(t_dim, latent_side_length**2)

        # 處理步驟一
        self.res1 = ResidualBlock(noise_channel + 2, inner_channels)

        # 上採樣：8×8 → 14×14
        self.up1 = nn.Upsample(scale_factor=1.75, mode='bilinear', align_corners=False),  # 8 → 14

        # 處理步驟二
        self.res2 = ResidualBlock(inner_channels, inner_channels // 2)

        # 上採樣：14×14 → 28×28
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 14 → 28

        # 最終輸出層
        self.out = nn.Conv2d(inner_channels // 2, 1, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        t_embed = sinusoidal_embedding(t, self.t_dim)
        t_feat = self.fc_t(t_embed).view(-1, 1, self.latent_side_length, self.latent_side_length)

        x = torch.cat([x, cond, t_feat], dim=1)         # (B, C+2, 8, 8)
        x = self.res1(x)                                # (B, inner_C, 8, 8)
        x = self.up1(x)                                 # (B, inner_C, 14, 14)
        x = self.res2(x)                                # (B, inner_C//2, 14, 14)
        x = self.up2(x)                                 # (B, inner_c//2, 28, 28)
        return self.out(x)                              # (B, 1, 28, 28)


# ========= Diffusion Decoder =========
class DiffusionDecoder(nn.Module):
    def __init__(self, model, timesteps=1000, image_shape=(1, 28, 28), device='cpu'):
        super().__init__()
        self.model = model.to(device)
        self.T = timesteps
        self.image_shape = image_shape
        self.device = device

        self.betas = torch.linspace(1e-4, 0.02, self.T).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)  # cumulative product

    def forward_diffusion(self, x0, t, latent_shape):
        noise = torch.randn_like(latent_shape)
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



# ========= Autoencoder 整合 =========
class Autoencoder(nn.Module):
    def __init__(self, encoder, sensor, decoder):
        super().__init__()
        self.encoder = encoder  # ONN and lens
        self.sensor = sensor  # terahertz sensor
        self.decoder = decoder  # diffusion model

    def forward(self, x, t=None, mode='train'):
        latent = self.encoder(x)
        latent = self.sensor(latent)
        
        if mode == 'train':
            x_t, noise = self.decoder.forward_diffusion(x, t)
            noise_pred = self.decoder.model(x_t, t, latent)
            return noise_pred, noise
        elif mode == 'sample':
            return self.decoder.reverse_sample(latent)

