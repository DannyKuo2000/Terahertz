import numpy as np
import torch
import math
from torch.nn import functional as F
import torch.nn as nn

def detector_region(x):
    return torch.cat((
        x[:, 46 : 66, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46 : 66, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46 : 66, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 78 : 98].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 109 : 129].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)

# Create CNN Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(1,28,28)
        
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2 
        out = self.cnn2(out)
        # Max pool 2 
        out = self.maxpool2(out)
        return out

class Diffractivelayers_(torch.nn.Module):
    def __init__(self, lambda_wavelength = 0.0015, k = 0.047429572, na = 1, z = 0.06):
        super(Diffractivelayers_, self).__init__()
        self.lambda_wavelength = lambda_wavelength
        self.k = k
        self.na = na
        self.z = z
        self.n = 1.75 
        # Parameters for transmittance and thickness
        
    def compute_transmittance(self, tau, na, thickness):
        exponent = (2j * np.pi / self.lambda_wavelength) * (tau - na) * thickness
        return torch.exp(exponent).cuda()

    def propagate(self, U):
        x = torch.linspace(-64, 63, 128).cuda()
        y = torch.linspace(-64, 63, 128).cuda()
        x, y = torch.meshgrid(x, y, indexing='ij')
        z = self.z
        r = torch.sqrt(x**2 + y**2 + z**2).cuda()
        w = (z / (r**2)) * (1/(2*np.pi) + 1j/self.lambda_wavelength) * torch.exp(1j * 2 * np.pi * r / self.lambda_wavelength)
        return torch.fft.ifft2(torch.fft.fft2(U) * torch.fft.fft2(w)).cuda()

    def forward(self, input_wavefield, thickness):
        U = self.propagate(input_wavefield)
        tau = self.n + 1j * self.k
        T = self.compute_transmittance( tau, self.na, thickness)
        U *= T
        return U

class angular_spectrum_method():
    def __init__(self, z, freq = 0.2e12, size = 128):
        self.z = z
        self.dx = 0.00075
        self.freq = freq
        self.wl = 3e8/self.freq
        self.size = size

    def forward(self, E):
        c_fft = torch.fft.fft2(E)
        c = torch.fft.fftshift(c_fft)

        fx = np.fft.fftshift(np.fft.fftfreq(self.size, d = self.dx))
        fxx, fyy = np.meshgrid(fx, fx)
        argument = (2 * np.pi)**2 *((1. / self.wl)**2 - fxx**2 - fyy**2)

         #Calculate the propagating and the evanescent modes
        tmp = np.sqrt(np.abs(argument))
        kz = np.where(argument >= 0, tmp, 1j*tmp)

        jkz = torch.from_numpy(np.exp(1j * kz * self.z)).cuda()

         #Propagate the angular spectrum a distance z
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * jkz))

        return angular_spectrum

class DiffractiveLayer(torch.nn.Module):
    def __init__(self, size = 128, freq = 0.5e12):  ###freq=0.2e12
        super(DiffractiveLayer, self).__init__()
        self.dx = 0.00075
        self.freq = freq
        self.size = size                        # 200 * 200 neurons in one layer
        self.wl = 3e8/self.freq                   # wavelength

    def forward(self, E, z):
        #Compute angular spectrum
         c_fft = torch.fft.fft2(E)
         c = torch.fft.fftshift(c_fft)

         fx = np.fft.fftshift(np.fft.fftfreq(self.size, d = self.dx))
         fxx, fyy = np.meshgrid(fx, fx)
         argument = (2 * np.pi)**2 *((1. / self.wl)**2 - fxx**2 - fyy**2)

         #Calculate the propagating and the evanescent modes
         tmp = np.sqrt(np.abs(argument))
         kz = np.where(argument >= 0, tmp, 1j*tmp)
         jkz = torch.from_numpy(np.exp(1j * kz * z)).cuda()

         #Propagate the angular spectrum a distance z
         angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * jkz))

         return angular_spectrum

class Net(torch.nn.Module):
    ## phase only modulation ##
    def __init__(self, num_layers=3):
        super(Net, self).__init__()

        self.vol_size = 0.00005
        self.PI = torch.tensor(np.pi)
        self.n = 1.75
        self.dn = self.n -1
        c = 3e8
        self.lam = c/0.2e12
        self.k = 2*self.PI/self.lam

        self.phase1 = [torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size = (128, 128)).astype("float32")))for _ in range(num_layers)]
        for i in range(num_layers):
           self.register_parameter("phase1" + "_" + str(i), self.phase1[i])
        self.diffractive_layers1 = torch.nn.ModuleList([DiffractiveLayer() for _ in range(num_layers)])
        #self.last_diffractive_layer1 = DiffractiveLayer()

        self.sofmax = torch.nn.Softmax(dim = -1)
        self.relu = torch.nn.ReLU()

        nn.init.constant_(self.phase1_0, 0)
        nn.init.constant_(self.phase1_1, 0)
        nn.init.constant_(self.phase1_2, 0)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.up1 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.up2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.up3 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        Attenuation_coefficient = 0.03 # !!!

        for index, layer in enumerate(self.diffractive_layers1):
            if index == 0:
                z = 0.02
            else:
                z = 0.06
            temp = layer(x, z)
            exp_j_phase = self.phase1[index]
            exp_j_phase = torch.sin(exp_j_phase) * self.PI
            exp_j_phase = exp_j_phase + self.PI
            L = exp_j_phase / self.k / self.dn

            x = temp * torch.exp(1j * exp_j_phase) * torch.exp(-Attenuation_coefficient * L)

        x = angular_spectrum_method(0.02).forward(x)

        x = torch.abs(x)

        # 获取张量的高度和宽度
        height, width = x.shape[1], x.shape[2]

        # 检查高度是否至少为2以进行切片
        if height > 1 and width > 72:
            slice_h_start = min(56, height)  # 确保起点不超过张量高度
            slice_h_end = min(72, height)    # 确保终点不超过张量高度
            slice_w_start = min(56, width)   # 确保起点不超过张量宽度
            slice_w_end = min(72, width)     # 确保终点不超过张量宽度

            # 如果切片结果为空，则抛出错误
            if slice_h_end - slice_h_start <= 0 or slice_w_end - slice_w_start <= 0:
                raise ValueError(f"Invalid slice operation: height={height}, width={width}")

            # 对张量 x 进行切片操作
            cs = x[:, slice_h_start:slice_h_end, slice_w_start:slice_w_end]

        else:
            # 如果高度小于2，直接用整个张量作为输入
            cs = x

        # 增加 channel 维度
        cs = torch.unsqueeze(cs, 1)  # 增加 channel 维度
        # 移除多余的批次维度
        cs = cs.squeeze(0)  # 移除批次维度，将张量形状变为 [channels, height, width]

        # 打印张量形状，检查是否符合卷积层要求
        print(f"Shape of cs before conv layers: {cs.shape}")

        cs = cs.float()
        cs = self.conv1(cs)
        cs = self.conv2(cs)
        cs = self.up1(cs)
        cs = self.conv3(cs)
        cs = self.conv4(cs)
        cs = self.up2(cs)
        cs = self.conv5(cs)
        cs = self.conv6(cs)
        cs = self.up3(cs)
        cs = self.conv7(cs)
        cs = self.conv8(cs)
        cs = cs.double()

        output = cs.squeeze()

        return output, cs

if __name__ == '__main__':
    print(Net())