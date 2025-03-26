import numpy as np
import torch
import math
from torch.nn import functional as F
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Relative parameters:
    Refractive index: 1.7
    Absorption coefficient: 1e-5
    Sub THz: 0.2e12
"""

"""
註解符號說明:
    ###說明概念
    #說明程式碼
"""


class DiffractiveLayer(torch.nn.Module):
    def __init__(self, frequency = 0.2e12, num_size = 28*4): # original: size = 36
        super(DiffractiveLayer, self).__init__()
        self.dx = 0.00075       # resolution (m)
        self.size = num_size       # number of optical neurons in one dimension
        #self.ll = 0.01        # layer length (m)
        self.wl = 2.998e8 / frequency    # wavelength = light speed / frequency (m)
        self.z = 0.01         # distance between two layers (m)

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

class Net(torch.nn.Module):
    def __init__(self, num_layers=3, num_size=28*4):
        super(Net, self).__init__()
        # random initialized [0, 2*pi] 每層大小為size, 共num_layers, call: self.phase1[i]
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

        output = torch.abs(x)
        #print(f"Encoded output size: {output.shape}")  # print output size
        return output



class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim, 128)   # 從潛在向量到128維
        self.fc2 = nn.Linear(128, 256)          # 從128維到256維
        self.fc3 = nn.Linear(256, output_dim)   # 從256維到輸出尺寸
        
        self.relu = nn.ReLU()                   # 激活函數
        self.sigmoid = nn.Sigmoid()             # 用於將輸出壓縮到[0,1]範圍

    def forward(self, x):
        x = x.to(torch.float32)
        #print(f"fc1 shape: {self.fc1.weight.shape}")
        #print(f"x shape: {x.shape}")
        #print(f"fc1 weight data type: {self.fc1.weight.dtype}")
        #print(f"x data type: {x.dtype}")
        x = self.relu(self.fc1(x))  # 第一層經過ReLU激活
        x = self.relu(self.fc2(x))  # 第二層經過ReLU激活
        x = self.sigmoid(self.fc3(x))  # 最後一層經過Sigmoid激活，將值壓縮到[0,1]
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(Autoencoder, self).__init__()
        
        # Net()的dim寫在前面了
        self.encoder = Net()  # 初始化Encoder

        ### 模擬攝影機將他拍成8*8大小，這裡用average壓縮 
        # Sensor Layer: 平均池化將 (112, 112) 壓縮為 (8, 8)
        self.sensor_size = 14  # 112 / 8 = 14

        self.decoder = Decoder(latent_dim, output_dim)  # 初始化Decoder

    def forward(self, x):
        latent = self.encoder(x)  # Encoder輸出潛在向量

        # Sensor Layer: 空間平均池化
        latent = F.avg_pool2d(latent, kernel_size=self.sensor_size)
        latent = latent.view(latent.size(0), -1)
        reconstructed = self.decoder(latent)  # Decoder重建輸出
        return reconstructed




