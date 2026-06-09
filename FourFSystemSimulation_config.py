# --------------------------------------------------
# Optical Encoder Configuration
# --------------------------------------------------
# Basic calculation: 
# ONN size: 0.00075 (m)
# simulation pixel size: 0.00075/4 (m) = 0.0001875 (m) (每個ONN element 用4*4的模擬去跑)
# image length = 0.03 (m) => 0.03/0.0001875 = 160 pixels


ENCODER_CONFIG = {
    # "image_path": "data/GroundTruth-800-v1/083.png",
    # "save_path": "results/terahertz_group_260106",
    # "image_path": "other_data/NVLab251224_fixed/camera_34.0.bmp",
    # "save_path": "results/NVLab251224_simulation",
    "image_path": "data/GroundTruth-800-v1/003.png",
    "save_path": "other_data/FourFSystemSimulationResult",

    #====== SourceLayer ======: length: 0.03m, size: 160, dx: 0.0001875
    "use_input": True,  # 是否使用自訂source
    "input": "other_data/NVLab260608/MultiSnap_2026-06-08_10-59-23_0265_0000.bmp",  # source
    "mode_source": "white",  # 不使用自訂source的話，要使用"white" or "gaussian"
    "size_source": (512, 512),  # 想要製作的gaussian beam大小
    "sigma": 0.3,  # sigma of gaussian
    "amplitude": 1.0,  # amplitude of gaussian, range: [0, 1]
    "center": (0.0, 0.0),  # center of gaussian(pixel)
    "rotation": 0.0,  # rotation of gaussian(angle)
    "aspect_ratio": 1.0,  # 橢圓比例

    # 最後的size處理(重複使用以下的ResizePadLayer)
    "crop_size_source": None,
    "resize_size_source": (2286, 2286),  # resize size, e.g., (H, W)
    "pad_size_source": (8192, 8192),  # final size, e.g., (H, W)
    "source_is_intensity": True,

    #====== ResizePadLayer ======
    "crop_size": None,
    "resize_size": (2286, 2286),  # resize size of input, e.g., (H, W)
    # "pad_size": (512, 512),  # final size of input, e.g., (H, W)
    # "resize_size": (256, 256),  # resize size, e.g., (H, W)
    "pad_size": (8192, 8192),  # final size, e.g., (H, W)
    #35um * 384 = 0.01344m,
    #35um * 288 = 0.01008m,
    #pitch: 35um
    
    #====== Number of MaterialLayer ======
    "num_layers": 0,          # ONN layer數量

    #====== DiffractiveLayer ======
    # "dx": 0.00075/2,            # 空間解析度 (m)
    # "dx": 0.000234375,
    "dx": 0.000035/2,
    "num_size_diffractive": 8192,          # 每層大小
    "frequency": 0.2004e12,      # THz頻率
    #"z": [0.06, 0.06, 0.06, 0.06],        # 層間距離 (m)
    #"z": [0.142, 0.041],        # 層間距離 (m)
    # 4F layout: object -> f1 -> lens1 -> f1 + f2 -> lens2 -> f2 -> image plane
    "z": [0.26, 0.303, 0.043],
    "refractive_index": 1,  # 空氣折射率或介質折射率
    "pad_factor": 1,
    "window": "hann",
    #"keep_pad": False,
    "mask_evanescent": False,
    "reverse_z": False,
    
    #====== MaterialLayer ======
    "num_size_material": 128,
    "block_size": (2, 2),
    "return_phases": False,  # 開關: return phases for manufacture loss calculation


    #====== LensLayer ======
    # Each lens needs its own config because a 4F system can use different focal lengths/apertures.
    # The lens grid must match the propagated field grid: num_size_lens == num_size_diffractive.
    "wavelength": 2.998e8 / 0.2004e12,
    "lens_configs": [
        {
            "name": "Lens1",
            "focal_length": 0.26,
            "dx": 0.000035/2,
            "num_size": 8192,
            "pupil_type": "circular",
            "pupil_radius": 0.05, # 0.05
            "pupil_width": None,
            "phase_model": "exact",
            "mode": "forward",
            "outside": "zero",
            "frame": False,
            "frame_inner": None,
            "frame_outer": None,
        },
        {
            "name": "Lens2",
            "focal_length": 0.043,
            "dx": 0.000035/2,
            "num_size": 8192, 
            "pupil_type": "circular",
            "pupil_radius": 0.06, # 0.025
            "pupil_width": None,
            "phase_model": "exact",
            "mode": "forward",
            "outside": "zero",
            "frame": False,
            "frame_inner": None,
            "frame_outer": None,
        },
    ],


    #====== SensorLayer ======
    "active_sensor": True, # 開關
    "crop_size": (288*2, 384*2),
    "bin_size": 2,
    "flip": True,

    #====== SensorNoiseLayer ======
    "active_sensor_noise": False, # 開關
    "blur_kernel_size": 15,
    "blur_sigma": 5,
    "gray_mean": 0.6,     # 背景灰階均值
    "gray_sigma": 0.02,   # 背景灰階標準差
    "gray_ratio": 0.55,   # 背景混合比例
    "noise_std": 10/255,  # 高斯雜訊標準差


    #====== Final Process ======
    "gain": 1,
    "noise_level": 0,
}
