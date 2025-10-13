# --------------------------------------------------
# Real Dataset Configuration
# --------------------------------------------------
DATASET_CONFIG = {
    "dataset_name": "EMNIST",   # 可選: "MNIST" | "FashionMNIST" | "EMNIST" | "Custom"
    "emnist_split": "byclass",  # 只有 EMNIST 用
    "root": "./data/RealDataset-800-v1",  # Custom dataset 的資料夾 (Custom dataset專用)
    "batch_size": 64,
    "num_workers": 0,
    "valid_ratio": 0.1,   # 10% 驗證
    "test_ratio": 0.1,    # 10% 測試
    "resize": 128, # 把最短邊resize到128
    "center_crop": (128, 128),
    "augmentation": {
        "use_random_rotation": False,
        "rotation_degrees": 0,
        "use_random_affine": False,
        "translate_ratio": (0, 0)
    }
}
    
# --------------------------------------------------
# Optical Encoder Configuration
# --------------------------------------------------
ENCODER_CONFIG = {
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
    "num_layers": 3,          # ONN layer數量

    #====== DiffractiveLayer ======
    "dx": 0.00075/4,            # 空間解析度 (m)
    "num_size": 128*4,          # 每層大小
    "frequency": 0.2004e12,      # THz頻率
    "z": [0.06, 0.06, 0.06, 0.06],        # 層間距離 (m)
    #"z": [0.142, 0.041],        # 層間距離 (m)
    "refractive_index": 1,  # 空氣折射率或介質折射率
    "pad_factor": 1,
    #"keep_pad": False,
    "mask_evanescent": False,
    "reverse_z": False,
    #"multi_step": 1,
    #"eps": 1e-3,
    #"alpha_global": 0.0,
    #"beta_freq": 0.0,
    #"use_geom_atten": False,
    
    #====== MaterialLayer ======
    "num_size_material": 128,
    "block_size": (4, 4),

    #====== LensLayer ======
    "focal_length": 0.029,
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
    "active_sensor": True,
    "crop_size": 40,
    "bin_size": 1,
    "flip": True,

    #====== SensorNoiseLayer ======
    "active_sensor_noise": False,
    "blur_kernel_size": 15,
    "blur_sigma": 5,
    "gray_mean": 0.6,     # 背景灰階均值
    "gray_sigma": 0.02,   # 背景灰階標準差
    "gray_ratio": 0.55,   # 背景混合比例
    "noise_std": 10/255,  # 高斯雜訊標準差
}

# --------------------------------------------------
# Restormer Configuration
# --------------------------------------------------
RESTORMER_CONFIG = {
    # I/O
    "inp_channels": 1,               # 輸入通道數（灰階=1，RGB=3）
    "out_channels": 1,               # 輸出通道數

    # Embedding & Blocks
    #"embed_dim": 48,                 # 初始通道數
    "embed_dim": 16,
    #"num_blocks": [4, 6, 6, 8],      # 每層 RestormerBlock 數量
    "num_blocks": [2, 3, 3, 4],
    "num_heads":  [1, 2, 4, 8],      # Multi-head Attention 每層 head 數量

    # Training Stability
    "layerscale_init": 1e-2,         # LayerScale 初始化值（小一點比較穩）
    "with_global_residual": True,    # 是否啟用全域殘差（輸入+輸出）

    # Feed-forward 設定
    "ffn_expansion_factor": 2.66,    # GDFN 隱層放大倍率

    # Normalization
    #"eps": 1e-6                      # LayerNorm2d epsilon
}

# --------------------------------------------------
# Autoencoder Configuration
# --------------------------------------------------
AUTOENCODER_CONFIG = {
    "use_encoder": True,
    "use_decoder": True,
}

# --------------------------------------------------
# Training Configuration
# --------------------------------------------------
TRAINING_CONFIG = {    
    # Set up
    "writer_save_path": "runs/baseline_restormer_ONN",
    "weight_save_dir": './checkpoints',
    "weight_save_name": 'baseline_restormer_ONN.pth',
    
    # hyperparameters
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 1e-3,
    "patience": 5,

}

# --------------------------------------------------
# Testing Configuration
# --------------------------------------------------
TESTING_CONFIG = {    
    # load config
    "weight_save_dir": './checkpoints',
    "weight_save_name": '20250821_031543_restormer_baseline.pth',
    
    # save config
    "results_save_dir": './results',
    "results_save_name_suffix": '_metrics.json'

}

