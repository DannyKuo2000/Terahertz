import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from config import ENCODER_CONFIG

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
    def __init__(self, config=ENCODER_CONFIG):
        super().__init__()
        self.layers = nn.ModuleList()
        num_layers = config["num_layers"]
        num_size = config["num_size"]
        dx = config["dx"]
        z = config["z"]
        n = config["refractive_index"]
        frequency = config["frequency"]

        for _ in range(num_layers):
            self.layers.append(DiffractiveLayer(dx=dx, num_size=num_size, frequency=frequency, z=z, refractive_index=n))
            self.layers.append(MaterialLayer(num_size=num_size))
        # 最後一層 DiffractiveLayer 可做成單點
        self.layers.append(DiffractiveLayer(dx=dx, num_size=1, frequency=frequency, z=z, refractive_index=n))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x