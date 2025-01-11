import numpy as np
import torch
import math
from torch.nn import functional as F
import torch.optim as optim
import torch.nn as nn

class DiffractiveLayer(torch.nn.Module):
    def __init__(self, frequency = 0.4e12, size = 32): # original: size = 36
        super(DiffractiveLayer, self).__init__()
        self.dx = 0.005       # resolution (m)
        self.size = size       # number of optical neurons in one dimension
        #self.ll = 0.01        # layer length (m)
        self.wl = 3e8 / frequency    # wavelength = light speed / frequency (m)
        self.z = 0.01         # distance between two layers (m)

    def forward(self, E):
        # 從frequency domain進行計算
        c_fft = torch.fft.fft2(E) # 2D FFT
        c = torch.fft.fftshift(c_fft) # 移動低頻分量到中心(常見的十字狀圖)

        fx = np.fft.fftshift(np.fft.fftfreq(self.size, d = self.dx)) # 計算一維頻率軸fx, 範圍為[-1/(2*dx), 1/(2*dx)], 分成size份
        fxx, fyy = np.meshgrid(fx, fx)
        # 算kz**2 = k**2 - kx**2 - ky**2
        # 計算從一個node跑出的kz
        argument = (2 * np.pi)**2 *((1. / self.wl)**2 - fxx**2 - fyy**2)
        # 算kz
        tmp = np.sqrt(np.abs(argument))

        # >=0: propagating, <0: evanescent(光學細節)
        # 角度過大產生全反射, 剩下會exponential decay波跟平行於介面的evanescent波. 乘上i, 使其有exponential decay
        kz = np.where(argument >= 0, tmp, 1j*tmp)

        # 考慮兩層ONN之間的距離
        jkz = torch.from_numpy(np.exp(1j * kz * self.z)).to(device)
        # 在frequency domain相乘, 等於在space domain做convolution (公式需驗證)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * jkz))

        return angular_spectrum

class Net(torch.nn.Module):
    def __init__(self, num_layers=3):
        super(Net, self).__init__()
        # torch.nn.Parameter: 允許在反向傳播中更新
        # torch.from_numpy(): 將NumPy轉換成PyTorch
        # random initialized [0, 2*pi] 每層大小為size, 共num_layers, call: self.phase1[i]
        self.phase1 = [torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size = (32, 32)).astype("float32")))for _ in range(num_layers)] # original: size = (36, 36)
        # 將 self.phase1[i] 的每個張量註冊到模型中，使它們可以被 PyTorch 的自動微分系統追蹤。
        for i in range(num_layers):
            self.register_parameter("phase1" + "_" + str(i), self.phase1[i])
        # torch.nn.ModuleList：用list存許多層DiffractiveLayer(), PyTorch的特殊list
        self.diffractive_layers1 = torch.nn.ModuleList([DiffractiveLayer() for _ in range(num_layers)])

        # 單獨定義最後一層(沒用到)
        # self.last_diffractive_layer1 = DiffractiveLayer()

        # 定義一個 Softmax 函數(沒用到)
        # self.softmax = torch.nn.Softmax(dim = -1)

    def forward(self, x):
        for index, layer in enumerate(self.diffractive_layers1):
            # 自動調用forward() (PyTorch特性)
            temp = layer(x)
            # 這裡才是實際印製產生的phase變化
            exp_j_phase = self.phase1[index]
            # 加入印製的相位調整
            x = temp * torch.exp(1j * exp_j_phase)

        output = torch.abs(x)
        print(f"Encoded output size: {output.shape}")  # print output size
        return output

class Decoder_1(nn.Module):
    def __init__(self, input_size = 32*32, output_size = 32*32):
        super(Decoder_1, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_size, 512, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(512, output_size, dtype=torch.float64),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x.view(-1, 32, 32)

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.flatten = nn.Flatten()  # flatten the output
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_flat = self.flatten(encoded)  # flatten the output
        encoded_flat_real = torch.abs(encoded_flat).float()
        decoded = self.decoder(encoded_flat)
        return decoded




