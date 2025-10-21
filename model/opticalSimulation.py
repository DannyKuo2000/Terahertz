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
    Sub THz: 0.2004e12

註解符號說明:
    ###說明概念
    #說明程式碼

器材訊息:
透鏡架2: Newport M-LH-2A: https://www.newport.com/p/M-LH-2A
透鏡2: 1.55 µm BCX Lens: 針對 1.55 µm 波長最佳化設計的雙凸透鏡

器材位置:
sample: (22.1+19.9)/2+0.2 = 21.2
len2: (36.5+34.3)/2 = 35.4 
camera: 39.5
"""
class SourceLayer(nn.Module):
    def __init__(self, use_input=True, input=None, mode="white", size_source=(128, 128),
                 sigma=0.3, amplitude=1.0, center=(0.0, 0.0),
                 rotation=0.0, aspect_ratio=1.0,
                 resize_size_source=None, new_size_source=None):
        super().__init__()
        self.use_input = use_input
        self.input = input
        self.mode = mode
        self.size_source = size_source

        # Gaussian beam 參數
        self.sigma = sigma
        self.amplitude = amplitude
        self.center = center
        self.rotation = rotation
        self.aspect_ratio = aspect_ratio

        # Resize/pad layer
        self.resize_pad = ResizePadLayer(resize_size_source, new_size_source)

    def forward(self, x):
        device = x.device  # ✅ 確保所有 tensor 都跟 x 在同一裝置上

        if self.use_input:
            if self.input is None:
                raise ValueError("use_input=True 時必須在 init 傳入 condition")

            # ✅ 保證 self.input 在同一 device
            input_resized = self.resize_pad(self.input.to(device=device, dtype=x.dtype))
            return x * input_resized

        else:
            # -------------------
            # 產生自訂光源
            # -------------------
            H, W = self.size_source

            if self.mode == "white":
                src = torch.ones((1, 1, H, W), dtype=x.dtype, device=device)

            elif self.mode == "gaussian":
                yy, xx = torch.meshgrid(
                    torch.linspace(-1, 1, H, device=device, dtype=x.dtype),
                    torch.linspace(-1, 1, W, device=device, dtype=x.dtype),
                    indexing="ij"
                )

                # 平移中心
                xx = xx - self.center[0]
                yy = yy - self.center[1]

                # 旋轉
                if self.rotation != 0.0:
                    cos_t = math.cos(self.rotation)
                    sin_t = math.sin(self.rotation)
                    x_rot = cos_t * xx - sin_t * yy
                    y_rot = sin_t * xx + cos_t * yy
                    xx, yy = x_rot, y_rot

                # 橢圓比例與 Gaussian 分布
                xx = xx / self.aspect_ratio
                r2 = xx**2 + yy**2
                src = self.amplitude * torch.exp(-r2 / (2 * self.sigma**2))
                src = src.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            else:
                raise ValueError(f"Unknown source mode: {self.mode}")

            src_resized = self.resize_pad(src)
            return x * src_resized.to(device=device, dtype=x.dtype)

class ResizePadLayer(nn.Module):
    def __init__(self, resize_size=None, pad_size=None, mode='bilinear'):
        """
        resize_size: tuple (H_resize, W_resize) or None，先 resize 到此大小
        new_size: tuple (H_new, W_new) or None，最後強制調整成這個大小（padding 或裁切）
        mode: resize interpolation mode ('bilinear', 'nearest', etc.)
        """
        super().__init__()
        self.resize_size = resize_size
        self.pad_size = pad_size
        self.mode = mode

    def forward(self, x):
        """
        x: tensor (B, C, H, W)，可以是實數或複數
        """
        B, C, H, W = x.shape

        # -------------------------
        # Step 1: Resize
        # -------------------------
        if self.resize_size is not None and (H != self.resize_size[0] or W != self.resize_size[1]):
            if torch.is_complex(x):
                real = F.interpolate(x.real, size=self.resize_size, mode=self.mode, align_corners=False)
                imag = F.interpolate(x.imag, size=self.resize_size, mode=self.mode, align_corners=False)
                x = torch.complex(real, imag)
            else:
                x = F.interpolate(x, size=self.resize_size, mode=self.mode, align_corners=False)

        # -------------------------
        # Step 2: Padding / Cropping 到 pad_size
        # -------------------------
        if self.pad_size is not None:
            H_cur, W_cur = x.shape[-2:]
            target_h, target_w = self.pad_size

            pad_h = target_h - H_cur
            pad_w = target_w - W_cur

            if pad_h >= 0 and pad_w >= 0:
                # zero padding
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                if torch.is_complex(x):
                    real = F.pad(x.real, (pad_left, pad_right, pad_top, pad_bottom))
                    imag = F.pad(x.imag, (pad_left, pad_right, pad_top, pad_bottom))
                    x = torch.complex(real, imag)
                else:
                    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
            else:
                # 中心裁切
                start_h = (H_cur - target_h) // 2 if H_cur > target_h else 0
                start_w = (W_cur - target_w) // 2 if W_cur > target_w else 0
                x = x[..., start_h:start_h + target_h, start_w:start_w + target_w]

        return x

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
    def __init__(self, dx=0.00075, num_size=128, frequency=0.2004e12, z=0.1, refractive_index=1, 
                 pad_factor=2, mask_evanescent=False, reverse_z=False):
        super().__init__()
        self.dx = dx  # resolution (m)
        self.size = num_size  # number of optical neurons in one dimension
        self.wl = 2.998e8 / frequency  # wavelength = light speed / frequency (m)
        self.z = z  # distance between two layers (m)
        self.n = refractive_index
        self.pad_factor = pad_factor  # zero-padding 倍數
        self.mask_evanescent = mask_evanescent  # 是否遮掉 evanescent
        self.reverse_z = reverse_z  # 是否反向傳播 (-z)

        # ==============================================
        # Step 1. 空間頻率軸
        # ==============================================
        fx = np.fft.fftshift(np.fft.fftfreq(self.size * self.pad_factor, d=self.dx))
        fxx, fyy = np.meshgrid(fx, fx)

        # ==============================================
        # Step 2. 波數分量
        # ==============================================
        kx = 2 * np.pi * fxx
        ky = 2 * np.pi * fyy
        k = 2 * np.pi * self.n / self.wl  # 總波數

        # ==============================================
        # Step 3. 縱向分量 kz
        # ==============================================
        argument = k**2 - kx**2 - ky**2
        tmp = np.sqrt(np.abs(argument))
        kz = np.where(argument >= 0, tmp, 1j * tmp)

        # ==============================================
        # Step 4. 傳播相因子
        # ==============================================
        if self.reverse_z:
            H = np.exp(-1j * kz * self.z)
            H[argument < 0] = 0.0  # 避免 evanescent 爆炸
        else:
            H = np.exp(1j * kz * self.z)
            if self.mask_evanescent:
                H[argument < 0] = 0.0

        self.H = torch.from_numpy(H.astype(np.complex64)).to(device)

        # ==============================================
        # Step 5. aliasing 檢查
        # ==============================================
        if self.dx > self.wl / 2:
            print(f"⚠️ Warning: dx={self.dx*1e3:.3f} mm > λ/2={self.wl/2*1e3:.3f} mm, 可能 aliasing")

    def forward(self, E):
        # 確保 tensor
        if isinstance(E, np.ndarray):
            E = torch.from_numpy(E).to(device)
        E = E.to(torch.complex64)

        # Step A. Padding
        if self.pad_factor > 1:
            pad = (self.size * (self.pad_factor - 1)) // 2
            if torch.is_complex(E):
                real = torch.nn.functional.pad(E.real, (pad, pad, pad, pad))
                imag = torch.nn.functional.pad(E.imag, (pad, pad, pad, pad))
                E = torch.complex(real, imag)
            else:
                E = torch.nn.functional.pad(E, (pad, pad, pad, pad), value=0.0)

        # Step B. 傅立葉域傳播
        F = torch.fft.fftshift(torch.fft.fft2(E))
        propagated = torch.fft.ifft2(torch.fft.ifftshift(F * self.H))

        # Step C. 還原原始大小
        if self.pad_factor > 1:
            start = pad
            end = start + self.size
            propagated = propagated[..., start:end, start:end]

        return propagated

'''class DiffractiveLayer(nn.Module):
    def __init__(self, dx=0.00075, num_size=128, frequency=0.2004e12, z=0.1, refractive_index=1, 
                 pad_factor=4, keep_pad=False, mask_evanescent=False, reverse_z=False, multi_step=1, eps=1e-3,
                 alpha_global=0.0, beta_freq=0.0, use_geom_atten=False):
        """
        衰減項說明:
        1. alpha_global: 全域衰減係數 (m^-1)，傳播距離 z 時因子為 exp(-alpha_global * z)
            與空氣吸收有關(太赫茲波段嚴重且水蒸氣影響大): 常用 1~10(m**-1)
        2. beta_freq: 高頻衰減係數 (m^-1)，對應 exp(-beta_freq * (kx^2+ky^2) * z)
            與儀器建模有關(可能原因: 1.光學系統有限 NA 2.偏離軸的波能量效率下降 3.散射或鏡頭成像限制): 常用 1e-7
        3. use_geom_atten: 是否開啟幾何 1/z 衰減
            Angular spectrum 本質上是平面波分解，所以沒有自動包含這個 1/z 幾何衰減。
        """
        super().__init__()
        self.dx = dx
        self.size = num_size
        self.wl = 2.998e8 / frequency
        self.z = z
        self.n = refractive_index
        self.pad_factor = pad_factor
        self.keep_pad = keep_pad
        self.mask_evanescent = mask_evanescent
        self.reverse_z = reverse_z
        self.multi_step = multi_step   # 拆成幾步傳播
        self.eps = eps  # evanescent 判斷閾值

        # 衰減參數
        self.alpha_global = alpha_global
        self.beta_freq = beta_freq
        self.use_geom_atten = use_geom_atten

        # ======================================================
        # Step 1. 建立頻率軸
        # ======================================================
        fx = np.fft.fftshift(np.fft.fftfreq(self.size * self.pad_factor, d=self.dx))  
        fxx, fyy = np.meshgrid(fx, fx)

        # Step 2. kx, ky
        self.kx = 2 * np.pi * fxx
        self.ky = 2 * np.pi * fyy

        # Step 3. 總波數
        self.k = 2 * np.pi * self.n / self.wl

        # Step 4. 縱向分量
        argument = self.k**2 - self.kx**2 - self.ky**2
        # ---------- 在 __init__ 中（建立 numpy 資料後，轉成 torch 前） ----------
        # 建議保留 numpy 的高精度運算，但在丟入 torch 前轉 dtype

        # 先把 tmp, kx, ky, argument 等 cast 到 float32 / complex64 最後形式
        # 1) 確保 tmp 是 float32（或在最後再轉）
        tmp = np.sqrt(np.abs(argument)).astype(np.float32)

        # 2) 建立 kz：對於 argument>=0 用實數 tmp，否則用 1j*tmp
        #    先用 float32 tmp，再製作 complex64 結果
        kz = np.where(argument >= 0, tmp, 1j * tmp).astype(np.complex64)

        # 3) is_evan, alpha, freq_decay 轉成 float32
        is_evan = (argument < 0)
        alpha = np.where(is_evan, np.real(kz), 0.0).astype(np.float32)
        freq_decay = (self.kx**2 + self.ky**2).astype(np.float32)

        # 4) 最後把 numpy -> torch，並指定 dtype/device 正確
        self.kz_torch = torch.from_numpy(kz).to(device=device, dtype=torch.complex64)
        self.alpha_torch = torch.from_numpy(alpha).to(device=device, dtype=torch.float32)
        self.is_evan_torch = torch.from_numpy(is_evan.astype(np.float32)).to(device=device, dtype=torch.float32)
        self.freq_decay_torch = torch.from_numpy(freq_decay).to(device=device, dtype=torch.float32)

        # dx check
        if self.dx > self.wl / 2:
            print(f"⚠️ Warning: dx={self.dx*1e3:.3f} mm > λ/2={self.wl/2*1e3:.3f} mm, 可能 aliasing")

    def forward(self, E):
        if isinstance(E, np.ndarray):
            E = torch.from_numpy(E).to(device)

        # padding: 注意 F.pad 不支援 complex，若是 complex 要分開 pad real/imag（或先作 real/imag pad）
        if self.pad_factor > 1:
            pad = (self.size * (self.pad_factor - 1)) // 2
            if torch.is_complex(E):
                real = torch.nn.functional.pad(E.real, (pad, pad, pad, pad))
                imag = torch.nn.functional.pad(E.imag, (pad, pad, pad, pad))
                E = torch.complex(real, imag)
            else:
                E = torch.nn.functional.pad(E, (pad, pad, pad, pad), mode='constant', value=0)

        dz = self.z / self.multi_step
        z_done = 0.0
        for step in range(self.multi_step):
            z_rem = self.z - z_done - dz

            # ------------------------------
            # Step A. 建立傳播因子 (含衰減)
            # ------------------------------
            if self.reverse_z:
                H = torch.exp(-1j * self.kz_torch * dz)
                H = H * (1 - self.is_evan_torch)
            else:
                H = torch.exp(1j * self.kz_torch * dz)

                if self.mask_evanescent:
                    atten = torch.exp(-self.alpha_torch * z_rem)
                    keep = ((1 - self.is_evan_torch) > 0) | (atten >= self.eps)
                    H = H * keep
                
                """# 全域衰減（你原本的做法保留，但確保用 tensor）
                if self.alpha_global > 0:
                    H = H * torch.exp(torch.tensor(-self.alpha_global * dz, dtype=H.dtype, device=H.device))

                # 高頻衰減
                if self.beta_freq > 0:
                    H = H * torch.exp(-self.beta_freq * self.freq_decay_torch * dz)"""

            # ------------------------------
            # Step B. 頻域傳播
            # ------------------------------
            c_fft = torch.fft.fftshift(torch.fft.fft2(E, dim=(-2, -1)))  # 改用norm="ortho"幾乎沒有影響
            angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c_fft * H))

            E = angular_spectrum
            z_done += dz

            # 幾何擴散 (controlled, avoid division-by-small)
            if self.use_geom_atten and z_done > 0: # 盡量不要用幾何衰減或只在最後一步用(容易讓數值爆炸或不穩定)
                # MODIFIED: use 1/sqrt(z) (less aggressive than 1/z), prevent tiny z causing huge factors
                #           and ensure tensor dtype/device matching E, clamp maximum factor
                eps_z = 1e-9  # floor to avoid division by zero
                z_safe = max(z_done, eps_z)  # python float safe lower bound
                z_tensor = torch.tensor(z_safe, dtype=E.dtype, device=E.device)
                geom_factor = 1.0 / torch.sqrt(z_tensor)  # amplitude ~ 1/sqrt(z)
                # limit the maximum amplification to avoid pathological case
                geom_factor = torch.clamp(geom_factor, max=1e3)  # MODIFIED: guard against huge factor
                E = E * geom_factor  # MODIFIED

            # ------------------------------
            # Step C. 動態裁切 (保留 ≥99% 能量)
            # ------------------------------
            E = self._crop_and_pad(E, energy_frac=0.99)

        # 最後回傳裁切後大小
        if self.pad_factor > 1 and self.keep_pad == False:
            start = pad
            end = start + self.size
            E = E[..., start:end, start:end]
        return E

    
    def _crop_and_pad(self, E, energy_frac=0.99, target_shape=None, margin_pix=10):
        """
        裁切到 >= energy_frac 的能量再 pad 回 target_shape（確保回填後尺寸**精確**等於 target_shape）。
        - E: torch tensor (complex or real) shape (H,W)
        - target_shape: tuple (H_target, W_target). 若 None，則用當前 E 的 shape（不改變）
        - 回傳值 dtype/device 與輸入 E 一致。
        """
        # 目標尺寸
        H_target, W_target = (E.shape[-2], E.shape[-1]) if target_shape is None else tuple(target_shape)

        # 強度（以 numpy 做排序以節省實作量；若需效率可改成 torch 實作）
        I_np = (E.abs()**2).detach().cpu().numpy()
        B, C, H, W = I_np.shape
        cy, cx = H // 2, W // 2

        # 半徑格
        y = np.arange(H) - cy # y = [-cy, -cy+1, -cy+2, ...]
        x = np.arange(W) - cx
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2

        # 由內到外累積能量
        idx = np.argsort(R2.ravel()) # 把所有 pixel 按由小到大的半徑排序，idx 為排序後的一維索引陣列
        cumulative = np.cumsum(I_np.ravel()[idx]) # 由半徑小到大加總
        total = cumulative[-1] # 總能量
        target = energy_frac * total
        cut_idx = np.searchsorted(cumulative, target)
        cut_idx = min(cut_idx, len(idx)-1)

        r2_target = R2.ravel()[idx[cut_idx]]
        r = int(np.ceil(np.sqrt(r2_target))) + margin_pix # 因為切成正方形，預留 10 pixels，以免少於99%

        # clamp r 以避免超出邊界（保證取 slice 時不會用到負索引）
        r = min(r, cy, cx, H - cy - 1, W - cx - 1)

        y0, y1 = cy - r, cy + r
        x0, x1 = cx - r, cx + r

        # 若意外 y1<=y0 或 x1<=x0（極端情況），直接回傳原圖
        if y1 <= y0 or x1 <= x0:
            return E
        
        E_crop = E[..., y0:y1, x0:x1]
        newH, newW = E_crop.shape[-2], E_crop.shape[-1]

        # 計算非對稱 padding，保證最終尺寸 EXACT 等於 target
        pad_top = (H_target - newH) // 2
        pad_bottom = H_target - newH - pad_top
        pad_left = (W_target - newW) // 2
        pad_right = W_target - newW - pad_left

        # 若需要填回的量為負 (代表 crop 後比 target 大)，就不做 crop，直接回傳原始 E
        if pad_top < 0 or pad_bottom < 0 or pad_left < 0 or pad_right < 0:
            return E

        # pad 支援 complex：對 real/imag 分別 pad
        if torch.is_complex(E_crop):
            real = torch.nn.functional.pad(E_crop.real, (pad_left, pad_right, pad_top, pad_bottom))
            imag = torch.nn.functional.pad(E_crop.imag, (pad_left, pad_right, pad_top, pad_bottom))
            E_new = torch.complex(real, imag)
        else:
            E_new = torch.nn.functional.pad(E_crop, (pad_left, pad_right, pad_top, pad_bottom))

        # 確保尺寸正確
        assert E_new.shape[-2] == H_target and E_new.shape[-1] == W_target, \
            f"pad failed: got {E_new.shape[-2:]} expected {(H_target, W_target)}"

        # 保持 device 與原始 E 一致
        return E_new.to(E.device)'''

class LensLayer(nn.Module):
    def __init__(self, focal_length, dx, num_size, wavelength,
                 pupil_type=None, pupil_radius=None, pupil_width=None,
                 phase_model="exact", mode="forward", outside="one",
                 frame=False, frame_inner=0.02375, frame_outer=0.0254):
        super().__init__()
        self.f = float(focal_length)
        self.dx = float(dx)
        self.N  = int(num_size)
        self.wl = float(wavelength)
        self.frame_inner = frame_inner
        self.frame_outer = frame_outer

        k = 2*np.pi / self.wl
        # 以中心為0的座標（公尺）
        x = (np.arange(self.N) - self.N/2) * self.dx
        X, Y = np.meshgrid(x, x)

        if phase_model == "exact":
            # 精確球面相位
            if mode == "backward":
                H_lens = np.exp(1j * k * (np.sqrt(X**2 + Y**2 + self.f**2) - self.f))
            else:
                H_lens = np.exp(-1j * k * (np.sqrt(X**2 + Y**2 + self.f**2) - self.f))
        else:
            # 傳統拋物近似
            if mode == "backward":
                H_lens = np.exp(1j * k * (X**2 + Y**2) / (2*self.f))
            else: 
                H_lens = np.exp(-1j * k * (X**2 + Y**2) / (2*self.f))

        # pupil（可關閉、圓形或方形）
        if pupil_type == "circular":
            assert pupil_radius is not None, "請設定 pupil_radius（m）"
            P = ((X**2 + Y**2) <= (pupil_radius**2)).astype(np.float32)
        elif pupil_type == "square":
            assert pupil_width is not None, "請設定 pupil_width（m）"
            half = pupil_width/2
            P = ((np.abs(X) <= half) & (np.abs(Y) <= half)).astype(np.float32)
        else:
            P = np.ones_like(X, dtype=np.float32)

        # outside 選項
        if outside == "one":
            self.H = H_lens * P + (1 - P)   # 外面=1
        else:
            self.H = H_lens * P             # 外面=0（預設）

        if frame == True:
            frame_mask = (((X**2 + Y**2) >= (frame_inner**2)) & ((X**2 + Y**2) <= (frame_outer**2))).astype(np.float32)
            self.H = self.H * (1 - frame_mask)
    def forward(self, E):
        return E * self.H

# ======= Interface Interaction Calculation ====== 
class FresnelInterface(nn.Module):
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
    def __init__(self, dx=0.00075, num_size=128, frequency=0.2e12, keep_reflection=False, complex_index=False, n1=1, n2=1.7):
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

class RadialAttenuationLayer(nn.Module):
    """
    對複數場 E 做徑向衰減，避免邊緣數值過大。
    
    參數:
    ----------
    E : np.ndarray 或 torch.Tensor, complex
        傳播後的場
    R0 : float, optional
        衰減開始的參考半徑 (pixel)。若 None，預設為圖像一半寬度。
    exponent : float
        控制衰減曲線陡峭度，越大越陡
    min_factor : float
        最遠處的最小強度因子，避免完全為 0

    回傳:
    ----------
    E_out : same type as E
        徑向衰減後的場
    """
    def __init__(self, R0_ratio=0.8, exponent=2, min_factor=0):
        super().__init__()
        self.R0_ratio = R0_ratio
        self.exponent = exponent # 衰減速度
        self.min_factor = min_factor # 邊界保留的最低強度
        
    def forward(self, E):
        H, W = E.shape[-2:]
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=E.device) - cy
        x = torch.arange(W, device=E.device) - cx
        X, Y = torch.meshgrid(x, y, indexing='xy')
        R = torch.sqrt(X**2 + Y**2) # 每個pixel到中心的距離
        R0 = self.R0_ratio * R.max()

        attenuation = torch.ones_like(R)
        mask = R >= R0
        attenuation[mask] = torch.exp(- ((R[mask]-R0)/(max(H,W)-R0))**self.exponent)
        attenuation = torch.clamp(attenuation, self.min_factor, 1.0)

        return E * attenuation

class SensorLayer(nn.Module):
    def __init__(self, crop_size=128, bin_size=1, flip=False):
        """
        crop_size: 裁切大小 (pixels)
        bin_size: 像素合併 (模擬binning)
        flip: 是否模擬相機倒像 (上下+左右翻轉)
        """
        super().__init__()
        self.crop_size = crop_size
        self.bin_size = bin_size
        self.flip = flip

    def forward(self, E):
        """
        E: 輸入場 (complex tensor, shape = [H, W])
        回傳裁切/合併後的場 (crop_size x crop_size)
        """
        B, C, H, W = E.shape
        ch = self.crop_size // 2
        center_h, center_w = H // 2, W // 2

        # --- Step 1: 裁切中央區域 ---
        E_crop = E[..., center_h - ch:center_h + ch, center_w - ch:center_w + ch]

        # --- Step 2: 像素 binning (平均合併區塊) ---
        if self.bin_size > 1:
            new_size = self.crop_size // self.bin_size
            # E_crop shape = [..., crop_size, crop_size]
            E_crop = E_crop.unfold(-2, self.bin_size, self.bin_size).unfold(-1, self.bin_size, self.bin_size)
            # shape [..., new_size, bin_size, new_size, bin_size]
            E_crop = E_crop.mean(dim=(-3, -1))  # 對 bin_size 做平均


        # --- Step 3: 是否翻轉 (模擬相機倒像) ---
        if self.flip:
            E_crop = torch.flip(E_crop, dims=[-2, -1])

        # --- Step 4: 轉換到Intensity ---
        I_crop = torch.abs(E_crop)
        I_crop = I_crop.to(torch.float32)

        # 回傳 intensity 而不是 complex
        return I_crop


# ====== Sensor Noise Simulation ======
class SensorNoiseLayer(nn.Module):
    def __init__(self, blur_kernel_size=15, blur_sigma=5, gray_mean=0.6, gray_sigma=0.02, gray_ratio=0.55, noise_std=10/255.):
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
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.gray_mean = gray_mean
        self.gray_sigma = gray_sigma
        self.gray_ratio = gray_ratio
        self.noise_std = noise_std

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


# ====== Material Phase Control ======
class MaterialLayer(nn.Module):
    def __init__(self, num_size=128, block_size=(1, 1)):
        super().__init__()
        self.block_size = block_size
        h_small = math.ceil(num_size / block_size[0])
        w_small = math.ceil(num_size / block_size[1])

        # 初始化相位 (小網格)
        init_phase = 2 * np.pi * np.random.rand(h_small, w_small).astype(np.float32)
        self.phase = nn.Parameter(torch.from_numpy(init_phase))

    def forward(self, x):
        """
        x: (B, C, H, W) tensor，可以是 real 或 complex
        block_size: (block_h, block_w)，每個 phase 對應多少輸入 pixel
        """
        B, C, H, W = x.shape
        block_h, block_w = self.block_size

        # 放大相位網格成 full size，對應 floor(i/block_h), floor(j/block_w)
        phase_full = self.phase.repeat_interleave(block_h, dim=0).repeat_interleave(block_w, dim=1) # 複製
        phase_full = phase_full[:H, :W]  # 裁切到輸入大小

        # 建立複數相位遮罩
        phase_mask = torch.exp(1j * phase_full).to(x.device)

        # 相乘（自動 broadcast 到 B, C）
        return x * phase_mask

# ====== ONN ensemblance ======
class ONN(nn.Module):
    def __init__(self, config=ENCODER_CONFIG):
        super().__init__()
        self.layers = nn.ModuleList()  # 用 ModuleList 代替普通 list
        self.layer_names = []  # 存每一層的「語意名字」
        
        # SourceLayer
        use_input           = config["use_input"]
        input               = config["input"]
        mode_source         = config["mode_source"]
        size_source         = config["size_source"]
        sigma               = config["sigma"]
        amplitude           = config["amplitude"]
        center              = config["center"]
        rotaion             = config["rotation"]
        aspect_ratio        = config["aspect_ratio"]
        resize_size_source  = config["resize_size_source"]
        new_size_source     = config["new_size_source"]

        # ResizePadLayer
        resize_size = config["resize_size"]
        pad_size    = config["pad_size"]

        # DiffractiveLayer 
        num_layers      = config["num_layers"]
        dx              = config["dx"]
        num_size        = config["num_size"]
        frequency       = config["frequency"]
        z_values        = config["z"]  # 可能是 float 或 list
        n               = config["refractive_index"]
        pad_factor      = config["pad_factor"]
        #keep_pad        = config["keep_pad"]
        mask_evanescent = config["mask_evanescent"]
        reverse_z       = config["reverse_z"]
        #multi_step      = config["multi_step"]
        #eps             = config["eps"]
        #alpha_global    = config["alpha_global"]
        #beta_freq       = config["beta_freq"]
        #use_geom_atten  = config["use_geom_atten"]

        # LensLayer 
        focal_length = config["focal_length"]
        dx           = config["dx"]
        num_size     = config["num_size"]
        wavelength   = config["wavelength"]
        pupil_type   = config["pupil_type"]
        pupil_radius = config["pupil_radius"]
        pupil_width  = config["pupil_width"]
        phase_model  = config["phase_model"]
        mode_lens    = config["mode_lens"]
        outside      = config["outside"]
        frame        = config["frame"]
        frame_inner  = config["frame_inner"]
        frame_outer  = config["frame_outer"]

        # SensorLayer
        active_sensor   = config["active_sensor"]
        crop_size       = config["crop_size"]
        bin_size        = config["bin_size"]
        flip            = config["flip"]

        # SensorNoiseLayer
        active_sensor_noise = config["active_sensor_noise"]
        blur_kernel_size    = config["blur_kernel_size"]
        blur_sigma          = config["blur_sigma"]
        gray_mean           = config["gray_mean"]
        gray_sigma          = config["gray_sigma"]
        gray_ratio          = config["gray_ratio"]
        noise_std           = config["noise_std"]

        # MaterialLayer
        num_size_material   = config["num_size"]
        block_size          = config["block_size"]

        # -------------------------------
        # 建立 layers
        # -------------------------------
        resize_pad_layer_index = 0
        diffractive_layer_index = 0
        material_layer_index = 0
        
        self.layers.append(ResizePadLayer(resize_size=(160, 160), pad_size=(160, 160)))
        self.layer_names.append(f"ResizePadLayer{resize_pad_layer_index + 1}")
        resize_pad_layer_index += 1

        self.layers.append(SourceLayer(use_input=use_input, input=input, mode=mode_source, size_source=size_source, sigma=sigma, amplitude=amplitude, 
                                       center=center, rotation=rotaion, aspect_ratio=aspect_ratio, resize_size_source=resize_size_source, new_size_source=new_size_source))
        self.layer_names.append(f"SourceLayer")

        self.layers.append(ResizePadLayer(resize_size=resize_size, pad_size=pad_size))
        self.layer_names.append(f"ResizePadLayer{resize_pad_layer_index + 1}")
        resize_pad_layer_index += 1

        # 每一層使用不同的 z (如果超出長度，就循環使用)
        z_values_index = 0
        for z_values_index in range(num_layers):
            self.layers.append(
                DiffractiveLayer(dx=dx, num_size=num_size, frequency=frequency, z=z_values[z_values_index], refractive_index=n,
                                 pad_factor=pad_factor, mask_evanescent=mask_evanescent, reverse_z=reverse_z)
            )
            self.layer_names.append(f"DiffractiveLayer{diffractive_layer_index + 1}")
            diffractive_layer_index += 1
            """DiffractiveLayer(dx=dx, num_size=num_size, frequency=frequency, z=z_values[z_values_index], refractive_index=n,
                                 pad_factor=pad_factor, keep_pad=keep_pad, mask_evanescent=mask_evanescent,
                                 reverse_z=reverse_z, multi_step=multi_step, eps=eps,
                                 alpha_global=alpha_global, beta_freq=beta_freq, use_geom_atten=use_geom_atten)"""
            self.layers.append(MaterialLayer(num_size=num_size_material, block_size=block_size))
            self.layer_names.append(f"MaterialLayer{material_layer_index + 1}")
            material_layer_index += 1

        self.layers.append(DiffractiveLayer(dx=dx, num_size=num_size, frequency=frequency, z=z_values[z_values_index], refractive_index=n,
                                            pad_factor=pad_factor, mask_evanescent=mask_evanescent,
                                            reverse_z=reverse_z))
        self.layer_names.append(f"DiffractiveLayer{diffractive_layer_index + 1}")
        diffractive_layer_index += 1
        """self.layers.append(LensLayer(focal_length=focal_length, dx=dx, num_size=num_size, wavelength=wavelength, pupil_type=pupil_type,
                                    pupil_radius=pupil_radius, pupil_width=pupil_width, phase_model=phase_model, mode=mode_lens, outside=outside, frame=frame,
                                    frame_inner=frame_inner, frame_outer=frame_outer))"""
        # Sensor / Noise
        if active_sensor:
            self.layers.append(SensorLayer(crop_size=crop_size, bin_size=bin_size, flip=flip))
            self.layer_names.append("SensorLayer")
        if active_sensor_noise:
            self.layers.append(SensorNoiseLayer(blur_kernel_size=blur_kernel_size, blur_sigma=blur_sigma,
                                                gray_mean=gray_mean, gray_sigma=gray_sigma,
                                                gray_ratio=gray_ratio, noise_std=noise_std))
            self.layer_names.append("SensorNoiseLayer")
        
        self.layers.append(ResizePadLayer(resize_size=(128, 128), pad_size=(128, 128)))
        self.layer_names.append(f"ResizePadLayer{resize_pad_layer_index + 1}")
        resize_pad_layer_index += 1

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x