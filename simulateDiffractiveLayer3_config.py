# --------------------------------------------------
# Optical Encoder Configuration
# --------------------------------------------------
# Basic calculation: 
# ONN size: 0.00075 (m)
# simulation pixel size: 0.00075/4 (m) = 0.0001875 (m) (每個ONN element 用4*4的模擬去跑)
# image length = 0.03 (m) => 0.03/0.0001875 = 160 pixels
ENCODER_CONFIG = {
    "image_path": "data/GroundTruth-800-v1/043.png",
    "save_path": "results/simulateDiffractiveLayer3_GT043",

    #====== SourceLayer ======: length: 0.03m, size: 160, dx: 0.0001875
    "use_input": False,  # 是否使用自訂source
    "input": None,  # source
    "mode_source": "white",  # 不使用自訂source的話，要使用"white" or "gaussian"
    "size_source": (160, 160),  # 想要製作的gaussian beam大小
    "sigma": 0.3,  # sigma of gaussian
    "amplitude": 1.0,  # amplitude of gaussian, range: [0, 1]
    "center": (0.0, 0.0),  # center of gaussian(pixel)
    "rotation": 0.0,  # rotation of gaussian(angle)
    "aspect_ratio": 1.0,  # 橢圓比例

    # 最後的size處理(重複使用以下的ResizePadLayer)
    "resize_size_source": (160, 160),  # resize size, e.g., (H, W)
    "new_size_source": (160, 160),  # final size, e.g., (H, W)

    #====== ResizePadLayer ======
    "resize_size": (160, 160),  # resize size of input, e.g., (H, W)
    "pad_size": (512, 512),  # final size of input, e.g., (H, W)

    #====== Number of MaterialLayer ======
    "num_layers": 0,          # ONN layer數量

    #====== DiffractiveLayer ======
    "dx": 0.00075/4,            # 空間解析度 (m)
    "num_size": 128*4,          # 每層大小
    "frequency": 0.2004e12,      # THz頻率
    #"z": [0.06, 0.06, 0.06, 0.06],        # 層間距離 (m)
    #"z": [0.142, 0.041],        # 層間距離 (m)
    "z": [0.15, 0.045],
    "refractive_index": 1,  # 空氣折射率或介質折射率
    "pad_factor": 1,
    "window": "hann",
    #"keep_pad": False,
    "mask_evanescent": False,
    "reverse_z": False,
    
    #====== MaterialLayer ======
    "num_size_material": 128,
    "block_size": (4, 4),
    "return_phases": False,  # 開關: return phases for manufacture loss calculation


    #====== LensLayer ======
    "focal_length": 0.045, #0.029,
    "dx": 0.00075/4,
    "num_size": 128*4,
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


    #====== SensorLayer ======
    "active_sensor": True, # 開關
    "crop_size": 92,
    "bin_size": 1,
    "flip": False,

    #====== SensorNoiseLayer ======
    "active_sensor_noise": False, # 開關
    "blur_kernel_size": 15,
    "blur_sigma": 5,
    "gray_mean": 0.6,     # 背景灰階均值
    "gray_sigma": 0.02,   # 背景灰階標準差
    "gray_ratio": 0.55,   # 背景混合比例
    "noise_std": 10/255,  # 高斯雜訊標準差
}