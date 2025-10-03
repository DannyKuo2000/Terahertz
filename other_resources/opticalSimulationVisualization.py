import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
# 取得這個檔案的資料夾
#current_dir = os.path.dirname(os.path.abspath(__file__))

# 把 model 資料夾加入 sys.path
#sys.path.append(os.path.join(current_dir, "../model"))

# 現在可以 import opticalSimulation 了
# 將 script 的上一層 Terahertz 加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SourceLayer, ResizePadLayer, DiffractiveLayer, LensLayer, SensorLayer, SensorNoiseLayer, MaterialLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_CONFIG = {
    # SourceLayer: length: 0.03m, size: 160, dx: 0.0001875
    "use_input": "white",  # 是否使用自訂source
    "input": None,  # source
    "mode_source": "gaussian",  # 不使用自訂source的話，要使用"white" or "gaussian"
    "size_source": (160, 160),  # 想要製作的gaussian beam大小
    "sigma": 1.0,  # sigma of gaussian
    "amplitude": 1.0,  # amplitude of gaussian
    "center": (0.0, 0.0),  # center of gaussian
    "rotation": 0.0,  # rotation of gaussian
    "aspect_ratio": 1.0,  # 橢圓比例
    "resize_size_source": (160, 160),  # resize size, e.g., (H, W)
    "new_size_source": (160, 160),  # final size, e.g., (H, W)

    # ResizePadLayer
    "resize_size": (160, 160),  # resize size of input, e.g., (H, W)
    "pad_size": (512, 512),  # final size of input, e.g., (H, W)

    # Number of MaterialLayer
    "num_layers": 0,          # ONN layer數量

    # DiffractiveLayer
    "dx": 0.00075/4,            # 空間解析度 (m)
    "num_size": 512,          # 每層大小
    "frequency": 0.2e12,      # THz頻率
    "z": [0.142, 0.041],        # 層間距離 (m)
    "refractive_index": 1,  # 空氣折射率或介質折射率
    "pad_factor": 1,
    "keep_pad": False,
    "mask_evanescent": False,
    "reverse_z": False,
    "multi_step": 2,
    "eps": 1e-3,
    "alpha_global": 0.0,
    "beta_freq": 0.0,
    "use_geom_atten": False,
    
    # MaterialLayer
    "num_size_material": 128,
    "block_size": (4, 4),

    # LensLayer
    "focal_length": 0.029,
    "dx": 0.00075/4,
    "num_size": 512,
    "wavelength": 2.998e8 / 0.2004e12,
    "pupil_type": "circular",
    "pupil_radius": 0.02375,
    "pupil_width": None,
    "phase_model": "exact",
    "mode_lens": "forward",
    "outside": "one",
    "frame": True,
    "frame_inner": 0.02375,
    "frame_outer": 0.0254,


    # SensorLayer
    "active_sensor": True,
    "crop_size": 160,
    "bin_size": 1,
    "flip": True,

    # SensorNoiseLayer
    "active_sensor_noise": False,
    "blur_kernel_size": 15,
    "blur_sigma": 5,
    "gray_mean": 0.6,     # 背景灰階均值
    "gray_sigma": 0.02,   # 背景灰階標準差
    "gray_ratio": 0.55,   # 背景混合比例
    "noise_std": 10/255,  # 高斯雜訊標準差
}

class ONN(nn.Module):
    def __init__(self, config=ENCODER_CONFIG):
        super().__init__()
        self.layers = nn.ModuleList()
        
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
        keep_pad        = config["keep_pad"]
        mask_evanescent = config["mask_evanescent"]
        reverse_z       = config["reverse_z"]
        multi_step      = config["multi_step"]
        eps             = config["eps"]
        alpha_global    = config["alpha_global"]
        beta_freq       = config["beta_freq"]
        use_geom_atten  = config["use_geom_atten"]

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
        self.layers.append(ResizePadLayer(resize_size=(160, 160), pad_size=(160, 160)))
        self.layers.append(SourceLayer(use_input=use_input, input=input, mode=mode_source, size_source=size_source, sigma=sigma, amplitude=amplitude, 
                                       center=center, rotation=rotaion, aspect_ratio=aspect_ratio, resize_size_source=resize_size_source, new_size_source=new_size_source))
        self.layers.append(ResizePadLayer(resize_size=resize_size, pad_size=pad_size))

        # 每一層使用不同的 z (如果超出長度，就循環使用)
        z_values_index = 0
        for z_values_index in range(num_layers):
            self.layers.append(
                DiffractiveLayer(dx=dx, num_size=num_size, frequency=frequency, z=z_values[z_values_index], refractive_index=n,
                                 pad_factor=pad_factor, keep_pad=keep_pad, mask_evanescent=mask_evanescent,
                                 reverse_z=reverse_z, multi_step=multi_step, eps=eps,
                                 alpha_global=alpha_global, beta_freq=beta_freq, use_geom_atten=use_geom_atten)
            )
            self.layers.append(MaterialLayer(num_size=num_size_material, block_size=block_size))


        self.layers.append(DiffractiveLayer(dx=dx, num_size=num_size, frequency=frequency, z=z_values[z_values_index], refractive_index=n,
                                            pad_factor=pad_factor, keep_pad=keep_pad, mask_evanescent=mask_evanescent,
                                            reverse_z=reverse_z, multi_step=multi_step, eps=eps,
                                            alpha_global=alpha_global, beta_freq=beta_freq, use_geom_atten=use_geom_atten))
        z_values_index += 1
        self.layers.append(LensLayer(focal_length=focal_length, dx=dx, num_size=num_size, wavelength=wavelength, pupil_type=pupil_type,
                                    pupil_radius=pupil_radius, pupil_width=pupil_width, phase_model=phase_model, mode=mode_lens, outside=outside, frame=frame,
                                    frame_inner=frame_inner, frame_outer=frame_outer))
        self.layers.append(DiffractiveLayer(dx=dx, num_size=num_size, frequency=frequency, z=z_values[z_values_index], refractive_index=n,
                                            pad_factor=pad_factor, keep_pad=keep_pad, mask_evanescent=mask_evanescent,
                                            reverse_z=reverse_z, multi_step=multi_step, eps=eps,
                                            alpha_global=alpha_global, beta_freq=beta_freq, use_geom_atten=use_geom_atten))
        # Sensor / Noise
        if active_sensor:
            self.layers.append(SensorLayer(crop_size=crop_size, bin_size=bin_size, flip=flip))
        if active_sensor_noise:
            self.layers.append(SensorNoiseLayer(blur_kernel_size=blur_kernel_size, blur_sigma=blur_sigma,
                                                gray_mean=gray_mean, gray_sigma=gray_sigma,
                                                gray_ratio=gray_ratio, noise_std=noise_std))

    def forward(self, x, return_intermediate=False):
        outputs = []
        layer_names = []
        for layer in self.layers:
            # 收集 layer 名稱
            layer_names.append(layer.__class__.__name__)
            
            x = layer(x)
            if return_intermediate:
                output = torch.abs(x) ** 2
                outputs.append(output.cpu().numpy())
                
        return (layer_names, outputs) if return_intermediate else x


    def visualize(self, x):
        # forward 並拿到所有中間層
        layer_names, outputs = self.forward(x, return_intermediate=True)

        n = len(outputs)
        fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
        if n == 1:
            axes = [axes]

        for i, out in enumerate(outputs):
            # 假設輸入是 [B, C, H, W]，這裡取 batch=0, channel=0 來畫
            img = out[0, 0]
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"{layer_names[i]}")  # 顯示 layer 名稱
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
    
def load_image(path):
    img = Image.open(path).convert("L")

    return img

def main():
    img_path = "data/GroundTruth-800-v1/003.png"
    x = load_image(img_path)
    x = np.sqrt(x)   # 🔹 取平方根得到電場幅值
    x = torch.from_numpy(x).to(device).type(torch.complex64)

    # 將 [H, W] 轉成 [1, 1, H, W]
    x = x.unsqueeze(0).unsqueeze(0)

    model = ONN(ENCODER_CONFIG)
    model.visualize(x)

if __name__ == "__main__":
    main()
