"""
This file is a examination about changing the structure of ONN
to three class: DiffractiveLayer(air), MaterialLayer(material), 
ONN(assemblance)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== Air Diffraction Calculation ====
"""
這段forward主要考慮的是在空氣中傳播的疊加情形, ONN的影響會在Net()再額外加入。
先將每個node傳播到下一層的kz大小寫出(前半部分), 再把ONN之間的距離考慮進去, 用以算出相位變化的convolution(jkz = torch.from_numpy...)
最後乘上經過FFT過的input訊號(angular_spectrum = ...)
Question: 把全反射在這裡考慮似乎有些奇怪
"""
class DiffractiveLayer(nn.Module):
    def __init__(self, dx=0.00075, num_size=128, frequency=0.2e12, z=0.06):
        super().__init__()
        self.dx = dx  # resolution (m)
        self.size = num_size  # number of optical neurons in one dimension
        # self.ll = ll        # layer length (m)
        self.wl = 2.998e8 / frequency  # wavelength = light speed / frequency (m)
        self.z = z  # distance between two layers (m)

        ### 預先計算 kz，因為與輸入無關，只取決於尺寸與參數
        # 計算一維頻率軸fx, 範圍為[-1/(2*dx), 1/(2*dx)], 分成size份 (以0為中心，因為有用np.fft.fftshift)
        fx = np.fft.fftshift(np.fft.fftfreq(self.size, d=self.dx))
        fxx, fyy = np.meshgrid(fx, fx)  # 弄出網格

        ### 算kz**2 = k**2 - kx**2 - ky**2
        # 計算從一個node跑出的kz
        argument = (2 * np.pi)**2 * ((1. / self.wl)**2 - fxx**2 - fyy**2)
        tmp = np.sqrt(np.abs(argument))  # 算kz

        # >=0: propagating, <0: evanescent(光學細節)
        # 角度過大產生全反射, 剩下會exponential decay波跟平行於介面的evanescent波. 乘上i, 使其有exponential decay
        kz = np.where(argument >= 0, tmp, 1j * tmp)

        ### 加入通過ONN後，在空氣中傳播的疊加：
        # 考慮兩層ONN之間的距離
        self.jkz = torch.from_numpy(np.exp(1j * kz * self.z)).to(device)

    def forward(self, E):
        c_fft = torch.fft.fft2(E)  # 2D FFT
        c = torch.fft.fftshift(c_fft)  # 移動低頻分量到中心(常見的十字狀圖)

        # 在frequency domain相乘, 等於在space domain做convolution (驗證?)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * self.jkz))
        return angular_spectrum


# ==== Material Phase Control ====
class MaterialLayer(nn.Module):
    def __init__(self, num_size=128):
        super().__init__()
        init_phase = 2 * np.pi * np.random.rand(num_size, num_size)
        #init_phase = np.zeros((num_size, num_size), dtype=np.float32)

        # 這裡才是實際印製產生的phase變化
        self.phase = nn.Parameter(torch.from_numpy(init_phase).float())

    def forward(self, x):
        # 加入印製的相位調整
        phase_mask = torch.exp(1j * self.phase)
        return x * phase_mask


# ==== ONN 組合架構 ====
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


# ==== 視覺化工具 ====
def plot_field(field, title_prefix="", save_path=None):
    magnitude = torch.abs(field).detach().cpu().numpy()
    phase = torch.angle(field).detach().cpu().numpy()

    plt.figure(figsize=(10, 4))

    # Magnitude
    plt.subplot(1, 2, 1)
    plt.imshow(magnitude, cmap='gray')
    plt.title(f"{title_prefix} Magnitude")
    plt.colorbar()

    # Phase
    plt.subplot(1, 2, 2)
    plt.imshow(phase, cmap='twilight')
    plt.title(f"{title_prefix} Phase")
    plt.colorbar()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"已儲存圖片至 {save_path}")

    plt.show()


# ==== 測試 ====
if __name__ == "__main__":
    num_size = 1024
    num_layers = 3
    input_field = torch.zeros((num_size, num_size), dtype=torch.cfloat).to(device)
    input_field[num_size//2, num_size//2] = 1.0 + 0j  # 中央點光源

    model = ONN(num_layers=num_layers, num_size=num_size).to(device)
    output = model(input_field)

    print("輸出張量大小:", output.shape)
    os.makedirs("./ONN_modelVerification2_result", exist_ok=True)
    plot_field(input_field, title_prefix="Input", save_path="./ONN_modelVerification2_result/Input.png")
    plot_field(output, title_prefix="Output", save_path="./ONN_modelVerification2_result/Output_3layers_random.png")