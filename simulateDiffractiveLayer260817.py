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
from model.opticalSimulation import DiffractiveLayer, LensLayer, CropResizeDisplacePad, RadialAttenuationLayer, SensorLayer, SensorNoiseLayer, SourceLayer, MaterialLayer
from simulateDiffractiveLayer260817_config import ENCODER_CONFIG

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

        self.layers = nn.ModuleList()
        self.layer_names = []

        self.return_phases = config.get("return_phases", False)

        # =========================
        # Config blocks
        # =========================
        transform_configs = config["transform_configs"]
        diffractive_configs = config["diffractive_configs"]
        lens_configs = config["lens_configs"]
        material_configs = config.get("material_configs", [])

        # Source
        use_input = config["use_input"]
        input_path = config["input"]
        mode_source = config["mode_source"]
        created_size = config["created_size"]
        source_is_intensity = config["source_is_intensity"]

        sigma = config["sigma"]
        amplitude = config["amplitude"]
        center = config["center"]
        rotation = config["rotation"]
        aspect_ratio = config["aspect_ratio"]

        crop_size_source = config["crop_size_source"]
        resize_size_source = config["resize_size_source"]
        displace_size_source = config["displace_size_source"]
        pad_size_source = config["pad_size_source"]

        # Sensor
        active_sensor = config["active_sensor"]
        crop_size = config["crop_size"]
        sensor_displacement = config["sensor_displacement"]
        sensor_psf_enabled = config["sensor_psf_enabled"]
        sensor_psf_sigma = config["sensor_psf_sigma"]
        sensor_psf_kernel_size = config["sensor_psf_kernel_size"]
        simulation_pitch = config["simulation_pitch"]
        target_pitch = config["target_pitch"]
        bin_size = config["bin_size"]
        flip = config["flip"]

        # Noise
        active_sensor_noise = config["active_sensor_noise"]
        blur_kernel_size = config["blur_kernel_size"]
        blur_sigma = config["blur_sigma"]
        gray_mean = config["gray_mean"]
        gray_sigma = config["gray_sigma"]
        gray_ratio = config["gray_ratio"]
        noise_std = config["noise_std"]

        # Final process
        # self.gain = config["gain"]
        # self.bias = config["bias"]
        # self.noise_level = config["noise_level"]


        # wavelength
        wavelength = config["wavelength"]

        optical_chain = config.get("OPTICAL_CHAIN")
        if optical_chain is not None:
            if not isinstance(optical_chain, (list, tuple)) or len(optical_chain) == 0:
                raise ValueError("OPTICAL_CHAIN must be a non-empty list")

            source_base = {
                "use_input": use_input,
                "input": input_path,
                "mode": mode_source,
                "created_size": created_size,
                "source_is_intensity": source_is_intensity,
                "sigma": sigma,
                "amplitude": amplitude,
                "center": center,
                "rotation": rotation,
                "aspect_ratio": aspect_ratio,
                "crop_size_source": crop_size_source,
                "resize_size_source": resize_size_source,
                "displace_size_source": displace_size_source,
                "pad_size_source": pad_size_source,
            }

            sensor_base = {
                "crop_size": crop_size,
                "sensor_displacement": sensor_displacement,
                "sensor_psf_enabled": sensor_psf_enabled,
                "sensor_psf_sigma": sensor_psf_sigma,
                "sensor_psf_kernel_size": sensor_psf_kernel_size,
                "simulation_pitch": simulation_pitch,
                "target_pitch": target_pitch,
                "bin_size": bin_size,
                "flip": flip,
            }

            noise_base = {
                "blur_kernel_size": blur_kernel_size,
                "blur_sigma": blur_sigma,
                "gray_mean": gray_mean,
                "gray_sigma": gray_sigma,
                "gray_ratio": gray_ratio,
                "noise_std": noise_std,
            }

            total_index = 1

            def append_named_layer(layer, layer_name):
                nonlocal total_index
                self.layers.append(layer)
                self.layer_names.append(f"{total_index}_{layer_name}")
                total_index += 1

            for spec in optical_chain:
                if not isinstance(spec, dict) or "type" not in spec:
                    raise ValueError("Each OPTICAL_CHAIN item must be a dict with a 'type' key")

                layer_type = spec["type"]
                params_override = spec.get("params", {})
                layer_name = spec.get("name")

                if layer_type == "source":
                    params = dict(source_base)
                    params.update(params_override)
                    append_named_layer(SourceLayer(**params), layer_name or "SourceLayer")

                elif layer_type == "diffractive":
                    idx = spec["index"]
                    diff_cfg = dict(diffractive_configs[idx])
                    diff_cfg.update(params_override)
                    append_named_layer(
                        DiffractiveLayer(
                            dx=diff_cfg["dx"],
                            num_size=diff_cfg["num_size"],
                            frequency=diff_cfg["frequency"],
                            z=diff_cfg["z"],
                            refractive_index=diff_cfg["refractive_index"],
                            pad_factor=diff_cfg["pad_factor"],
                            window=diff_cfg["window"],
                            mask_evanescent=diff_cfg["mask_evanescent"],
                            reverse_z=diff_cfg["reverse_z"],
                        ),
                        layer_name or diff_cfg.get("name", f"Diffractive{idx}")
                    )

                elif layer_type == "material":
                    idx = spec["index"]
                    material_cfg = dict(material_configs[idx])
                    material_cfg.update(params_override)
                    #material_return_phases = material_cfg.get("return_phases", self.return_phases)
                    append_named_layer(
                        MaterialLayer(
                            num_size=material_cfg["num_size"],
                            block_size=material_cfg["block_size"],
                            mode=material_cfg["mode"],
                            border_width=material_cfg["border_width"],
                            return_phases=material_cfg["return_phases"],
                        ),
                        layer_name or material_cfg.get("name", f"Material{idx}")
                    )

                elif layer_type == "lens":
                    idx = spec["index"]
                    lens_cfg = dict(lens_configs[idx])
                    lens_cfg.update(params_override)
                    append_named_layer(
                        LensLayer(
                            focal_length=lens_cfg["focal_length"],
                            dx=lens_cfg["dx"],
                            num_size=lens_cfg["num_size"],
                            wavelength=wavelength,
                            pupil_type=lens_cfg["pupil_type"],
                            pupil_radius=lens_cfg["pupil_radius"],
                            pupil_width=lens_cfg["pupil_width"],
                            phase_model=lens_cfg["phase_model"],
                            mode=lens_cfg["mode"],
                            outside=lens_cfg["outside"],
                            frame=lens_cfg["frame"],
                            frame_inner=lens_cfg["frame_inner"],
                            frame_outer=lens_cfg["frame_outer"],
                        ),
                        layer_name or lens_cfg.get("name", f"Lens{idx}")
                    )

                elif layer_type == "sensor":
                    if not active_sensor:
                        continue
                    params = dict(sensor_base)
                    params.update(params_override)
                    append_named_layer(
                        SensorLayer(**params),
                        layer_name or "SensorLayer"
                    )

                elif layer_type == "noise":
                    if not active_sensor_noise:
                        continue
                    params = dict(noise_base)
                    params.update(params_override)
                    append_named_layer(
                        SensorNoiseLayer(**params),
                        layer_name or "SensorNoiseLayer"
                    )

                else:
                    raise ValueError(f"Unknown optical chain layer type: {layer_type}")

            print(f"[ONN] Built from OPTICAL_CHAIN with {len(self.layers)} layers")
            return

        # information reminder
        print(f"Number of diffractive layers: {len(diffractive_configs)}")
        print(f"Number of lens layers: {len(lens_configs)}")
        print(f"Number of material layers: {len(material_configs)}")

        material_map = {}
        for idx, material_cfg in enumerate(material_configs):
            attach_after = material_cfg.get("attach_after_diffractive_index", idx)
            material_map.setdefault(attach_after, []).append(material_cfg)

        # =========================
        # Layer indexing
        # =========================
        total_index = 1

        # =========================
        # Input transform
        # =========================
        t0 = transform_configs[0]
        self.layers.append(
            CropResizeDisplacePad(
                crop_size=t0["crop_size"],
                resize_size=t0["resize_size"],
                displace=t0["displace_size"],
                pad_size=t0["pad_size"],
            )
        )
        self.layer_names.append(f"{total_index}_{t0['name']}")
        total_index += 1

        # =========================
        # Source
        # =========================
        self.layers.append(
            SourceLayer(
                use_input=use_input,
                input=input_path,
                mode=mode_source,
                created_size=created_size,
                source_is_intensity=source_is_intensity,
                sigma=sigma,
                amplitude=amplitude,
                center=center,
                rotation=rotation,
                aspect_ratio=aspect_ratio,
                crop_size_source=crop_size_source,
                resize_size_source=resize_size_source,
                displace_size_source=displace_size_source,
                pad_size_source=pad_size_source,
            )
        )
        self.layer_names.append(f"{total_index}_SourceLayer")
        total_index += 1

        def append_diffractive_layer(diff_cfg):
            nonlocal total_index
            self.layers.append(
                DiffractiveLayer(
                    dx=diff_cfg["dx"],
                    num_size=diff_cfg["num_size"],
                    frequency=diff_cfg["frequency"],
                    z=diff_cfg["z"],
                    refractive_index=diff_cfg["refractive_index"],
                    pad_factor=diff_cfg["pad_factor"],
                    window=diff_cfg["window"],
                    mask_evanescent=diff_cfg["mask_evanescent"],
                    reverse_z=diff_cfg["reverse_z"],
                )
            )
            self.layer_names.append(f"{total_index}_{diff_cfg['name']}")
            total_index += 1

        def append_lens_layer(lens_cfg):
            nonlocal total_index
            self.layers.append(
                LensLayer(
                    focal_length=lens_cfg["focal_length"],
                    dx=lens_cfg["dx"],
                    num_size=lens_cfg["num_size"],
                    wavelength=wavelength,
                    pupil_type=lens_cfg["pupil_type"],
                    pupil_radius=lens_cfg["pupil_radius"],
                    pupil_width=lens_cfg["pupil_width"],
                    phase_model=lens_cfg["phase_model"],
                    mode=lens_cfg["mode"],
                    outside=lens_cfg["outside"],
                    frame=lens_cfg["frame"],
                    frame_inner=lens_cfg["frame_inner"],
                    frame_outer=lens_cfg["frame_outer"],
                )
            )
            self.layer_names.append(f"{total_index}_{lens_cfg['name']}")
            total_index += 1

        def append_material_layers(after_diff_idx):
            nonlocal total_index
            for material_cfg in material_map.get(after_diff_idx, []):
                self.layers.append(
                    MaterialLayer(
                        num_size=material_cfg["num_size"],
                        block_size=material_cfg["block_size"],
                        return_phases=material_cfg.get("return_phases", True),
                    )
                )
                self.layer_names.append(f"{total_index}_{material_cfg['name']}")
                total_index += 1

        # =========================
        # Optical chain
        # =========================
        for i, lens_cfg in enumerate(lens_configs):
            append_diffractive_layer(diffractive_configs[i])
            append_material_layers(i)
            append_lens_layer(lens_cfg)

        append_diffractive_layer(diffractive_configs[-1])
        append_material_layers(len(lens_configs))

        # =========================
        # Sensor
        # =========================
        if active_sensor:
            self.layers.append(
                SensorLayer(
                    crop_size=crop_size,
                    sensor_displacement=sensor_displacement,
                    sensor_psf_enabled=sensor_psf_enabled,
                    sensor_psf_sigma=sensor_psf_sigma,
                    sensor_psf_kernel_size=sensor_psf_kernel_size,
                    simulation_pitch=simulation_pitch,
                    target_pitch=target_pitch,
                    bin_size=bin_size,
                    flip=flip,
                )
            )
            self.layer_names.append(f"{total_index}_SensorLayer")
            total_index += 1

        # =========================
        # Noise
        # =========================
        if active_sensor_noise:
            self.layers.append(
                SensorNoiseLayer(
                    blur_kernel_size=blur_kernel_size,
                    blur_sigma=blur_sigma,
                    gray_mean=gray_mean,
                    gray_sigma=gray_sigma,
                    gray_ratio=gray_ratio,
                    noise_std=noise_std,
                )
            )
            self.layer_names.append(f"{total_index}_SensorNoiseLayer")
            total_index += 1

    def forward(self, x, return_intermediate=False):
        phase_list = []
        outputs = []

        for name, layer in zip(self.layer_names, self.layers):
            out = layer(x)
            if isinstance(out, tuple):
                x = out[0]
                if len(out) > 1 and isinstance(layer, MaterialLayer):
                    phase_list.append(out[1])
            else:
                x = out
            # x = x * self.gain + self.bias #! temporary
            outputs.append((name, x.detach().clone()))

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
    bias = ENCODER_CONFIG["bias"]
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
        img = img + noise + bias
        img = torch.clamp(img, 0, 1)

        vutils.save_image(img, os.path.join(ENCODER_CONFIG["save_path"], f"{name}_abs.png"), normalize=False)
        print(f"[ONN DEBUG] Saved layer '{name}' intensity output")