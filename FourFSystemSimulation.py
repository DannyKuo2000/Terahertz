# ======
# This file is for real NMLab251205_measurement check 
# ======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
from PIL import Image
from model.opticalSimulation import CropResizeDisplacePadLayer, DiffractiveLayer, LensLayer, RadialAttenuationLayer, SensorLayer, SensorNoiseLayer, SourceLayer, MaterialLayer
from FourFSystemSimulation_config import ENCODER_CONFIG

# ====== Image Loader ======
def load_image(path, cut=None, size=None):
    img = Image.open(path).convert("L")
    print(f"Original size {img.size}")
    
    if cut is not None:
        img = img.crop([img.size[0]//2-cut[0]//2, img.size[1]//2-cut[1]//2, img.size[0]//2+(cut[0]-cut[0]//2), img.size[1]//2+(cut[1]-cut[1]//2)])
        print(f"Cutted size {img.size}")
    
    if size is not None:
        img = img.resize((size[0], size[1]), Image.BICUBIC)
        print(f"Resized size {img.size}")
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array

# ====== Image Moving ======
def shift_image(img_array, shift):
    """
    對影像做水平 & 垂直位移
    img_array: numpy 2D (灰階) or 3D (彩色) array, 值域 [0,1]
    shift: (shift_h, shift_w)
        shift_h > 0 向下移, < 0 向上移
        shift_w > 0 向右移, < 0 向左移
    """
    h, w = img_array.shape[:2]
    shift_h, shift_w = shift

    # 建立一張全黑影像
    shifted = np.zeros_like(img_array)

    # 計算有效範圍
    src_y_start = max(0, -shift_h)
    src_y_end   = min(h, h - shift_h)   # 原圖範圍
    dst_y_start = max(0, shift_h)
    dst_y_end   = min(h, h + shift_h)   # 新圖範圍

    src_x_start = max(0, -shift_w)
    src_x_end   = min(w, w - shift_w)
    dst_x_start = max(0, shift_w)
    dst_x_end   = min(w, w + shift_w)

    # 複製有效範圍
    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        img_array[src_y_start:src_y_end, src_x_start:src_x_end]

    return shifted

# ====== ONN ensemblance ======
class ONN(nn.Module):
    def __init__(self, config=ENCODER_CONFIG):
        super().__init__()
        self.layers = nn.ModuleList()  # 用 ModuleList 代替普通 list
        self.layer_names = []  # 存每一層的「語意名字」
        
        self.return_phases = False

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
        crop_size_source    = config["crop_size_source"]
        resize_size_source  = config["resize_size_source"]
        pad_size_source     = config["pad_size_source"]
        source_is_intensity = config["source_is_intensity"]

        # CropResizeDisplacePadLayer
        crop_size   = config["crop_size"]
        resize_size = config["resize_size"]
        pad_size    = config["pad_size"]

        # DiffractiveLayer 
        num_layers      = config["num_layers"]
        dx              = config["dx"]
        num_size_diffractive        = config["num_size_diffractive"]
        frequency       = config["frequency"]
        z_values        = config["z"]  # 可能是 float 或 list
        n               = config["refractive_index"]
        pad_factor      = config["pad_factor"]
        window          = config["window"]
        mask_evanescent = config["mask_evanescent"]
        reverse_z       = config["reverse_z"]

        # MaterialLayer
        num_size_material = config["num_size_material"]
        block_size = config["block_size"]
        return_phases = config["return_phases"]

        # LensLayer
        wavelength = config["wavelength"]
        lens_configs = config["lens_configs"]
        if len(z_values) != len(lens_configs) + 1:
            raise ValueError(
                "4F config expects len(z) == len(lens_configs) + 1 "
                "for object->lens1, lens-to-lens, and lens2->image propagation."
            )

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
        
        # -------------------------------
        # 建立 layers
        # -------------------------------
        total_index = 1
        resize_pad_layer_index = 1
        diffractive_layer_index = 1
        material_layer_index = 0
        len_layer_index = 1
        z_values_index = 0
        
        # ==========================
        self.layers.append(CropResizeDisplacePadLayer(resize_size=resize_size, pad_size=pad_size))
        self.layer_names.append(f"{total_index}_ResizePadLayer{resize_pad_layer_index}")
        resize_pad_layer_index += 1
        total_index += 1

        self.layers.append(SourceLayer(use_input=use_input, input=input, mode=mode_source, size_source=size_source, sigma=sigma, amplitude=amplitude, 
                            center=center, rotation=rotaion, aspect_ratio=aspect_ratio, crop_size_source=crop_size_source,
                            resize_size_source=resize_size_source, pad_size_source=pad_size_source, source_is_intensity=source_is_intensity))
        self.layer_names.append(f"{total_index}_SourceLayer")
        total_index += 1

        for lens_config in lens_configs:
            self.layers.append(
                DiffractiveLayer(dx=dx, num_size=num_size_diffractive, frequency=frequency, z=z_values[z_values_index], refractive_index=n,
                                pad_factor=pad_factor, window=window, mask_evanescent=mask_evanescent, reverse_z=reverse_z)
            )
            self.layer_names.append(f"{total_index}_DiffractiveLayer{diffractive_layer_index}")
            z_values_index += 1
            diffractive_layer_index += 1
            total_index += 1

            self.layers.append(
                LensLayer(
                    focal_length=lens_config["focal_length"],
                    dx=lens_config.get("dx", dx),
                    num_size=lens_config.get("num_size", num_size_diffractive),
                    wavelength=lens_config.get("wavelength", wavelength),
                    pupil_type=lens_config.get("pupil_type"),
                    pupil_radius=lens_config.get("pupil_radius"),
                    pupil_width=lens_config.get("pupil_width"),
                    phase_model=lens_config.get("phase_model", "exact"),
                    mode=lens_config.get("mode", "forward"),
                    outside=lens_config.get("outside", "one"),
                    frame=lens_config.get("frame", False),
                    frame_inner=lens_config.get("frame_inner", 0.02375),
                    frame_outer=lens_config.get("frame_outer", 0.0254),
                )
            )
            lens_name = lens_config.get("name", f"LensLayer{len_layer_index}")
            self.layer_names.append(f"{total_index}_{lens_name}")
            len_layer_index += 1
            total_index += 1

        self.layers.append(
            DiffractiveLayer(dx=dx, num_size=num_size_diffractive, frequency=frequency, z=z_values[z_values_index], refractive_index=n,
                            pad_factor=pad_factor, window=window, mask_evanescent=mask_evanescent, reverse_z=reverse_z)
        )
        self.layer_names.append(f"{total_index}_DiffractiveLayer{diffractive_layer_index}")
        total_index += 1

        # Sensor / Noise
        if active_sensor:
            self.layers.append(SensorLayer(crop_size=crop_size, bin_size=bin_size, flip=flip))
            self.layer_names.append(f"{total_index}_SensorLayer")
            total_index += 1
        if active_sensor_noise:
            self.layers.append(SensorNoiseLayer(blur_kernel_size=blur_kernel_size, blur_sigma=blur_sigma,
                                                gray_mean=gray_mean, gray_sigma=gray_sigma,
                                                gray_ratio=gray_ratio, noise_std=noise_std))
            self.layer_names.append(f"{total_index}_SensorNoiseLayer")
            total_index += 1

    def forward(self, x, return_intermediate=True):
        # ======
        # 若 return_phases=True，則除了輸出結果外，也會回傳所有 MaterialLayer 的相位參數。
        # return_intermediate=True → 會同時回傳每層 output
        # ======
        phase_list = []
        outputs = []  # <--- 新增：存每層輸出

        for name, layer in zip(self.layer_names, self.layers):

            # MaterialLayer 另外處理 phase
            if self.return_phases and isinstance(layer, MaterialLayer):
                x, phase = layer(x)
                phase_list.append(phase)
            else:
                x = layer(x)

            # 每層輸出都保存
            outputs.append((name, x.detach().clone()))

        # 回傳三種形式
        if return_intermediate and self.return_phases:
            return x, phase_list, outputs

        if return_intermediate:
            return x, outputs

        if self.return_phases:
            return x, phase_list

        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(ENCODER_CONFIG["save_path"], exist_ok=True)

    path = ENCODER_CONFIG["image_path"]
    I0 = load_image(path)
    print(I0.size)
    E0 = np.sqrt(I0)
    E0 = torch.from_numpy(E0).to(device).type(torch.complex64)
    E0 = E0.unsqueeze(0).unsqueeze(0)
    
    model = ONN()

    # forward，要求所有中間 layer output
    final_output, all_outputs = model(E0, return_intermediate=True)

    gain = ENCODER_CONFIG["gain"]
    noise_level = ENCODER_CONFIG["noise_level"]

    # 印出每層的 output (shape)
    for name, out in all_outputs: # 測試
        print(name, out.shape)    

        # 如果要處理成 intensity
        if torch.is_complex(out):
            img = (out.abs() ** 2)
        else:
            img = out.squeeze()
        
        img = img * gain
        noise = torch.randn_like(img) * noise_level
        img = img + noise
        img = torch.clamp(img, 0, 1)

        vutils.save_image(img, os.path.join(ENCODER_CONFIG["save_path"], f"{name}_abs.png"), normalize=False)
        print(f"[ONN DEBUG] Saved layer '{name}' intensity output")
