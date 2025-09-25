# --------------------------------------------------
# Real Dataset Configuration
# --------------------------------------------------
DATASET_CONFIG = {
    "dataset_name": "Custom",   # 可選: "MNIST" | "FashionMNIST" | "EMNIST" | "Custom"
    "emnist_split": None,  # 只有 EMNIST 用
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
    # ResizePadLayer
    "resize_size": (128, 128),
    "pad_size": (128*4, 128*4),

    # number of ONN
    "num_layers": 0,          # ONN layer數量

    # DiffractiveLayer
    "dx": 0.00075,            # 空間解析度 (m)
    "num_size": 128 * 4,          # 每層大小，128x128
    "frequency": 0.2e12,      # THz頻率
    "z": 0.06,                 # 層間距離 (m)
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
    
    # SensorLayer
    "active_sensor": True,
    "crop_size": 128,
    "bin_size": 1,
    "flip": False,

    # SensorNoiseLayer
    "active_sensor_noise": True,
    "blur_kernel_size": 15,
    "blur_sigma": 5,
    "gray_mean": 0.6,     # 背景灰階均值
    "gray_sigma": 0.02,   # 背景灰階標準差
    "gray_ratio": 0.55,   # 背景混合比例
    "noise_std": 10/255.  # 高斯雜訊標準差
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
    "use_sensor": False,        # 是否啟用 sensor
    "use_sensor_noise": False,  # 是否啟用 sensor noise
    "use_decoder": True,
}

# --------------------------------------------------
# Training Configuration
# --------------------------------------------------
TRAINING_CONFIG = {    
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 1e-3,
    "patience": 5,
}

