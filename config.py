
# ========= Dataset =========
DATASET_CONFIG = {
    "dataset_name": "MNIST",       # 可選 "MNIST" / "EMNIST" / "FashionMNIST"
    "emnist_split": "byclass",      # EMNIST 專用，其他 dataset 可忽略
    "batch_size": 64,
    "num_workers": 0,
    "valid_ratio": 0.1,
    "resize": (128, 128),
    "augmentation": {
        "use_random_rotation": True,
        "rotation_degrees": 10,
        "use_random_affine": False,
        "translate_ratio": (0.1, 0.1)
    }
}
    

# ===== ONN / Optical Encoder Config =====
ENCODER_CONFIG = {
    "num_layers": 3,          # ONN layer數量
    "num_size": 128,          # 每層大小，128x128
    "dx": 0.00075,            # 空間解析度 (m)
    "frequency": 0.2e12,      # THz頻率
    "refractive_index": 1.7,  # 空氣折射率或介質折射率
    "z": 0.1,                 # 層間距離 (m)
}
    

# --------------------------------------------------
# Sensor Configuration
# --------------------------------------------------
SENSOR_CONFIG = {
    "output_dim": 128,   # 輸出 latent 尺寸 (若有裁切)

    # sensor noise
    "blur_kernel_size": 15,
    "blur_sigma": 5,
    "gray_mean": 0.6,     # 背景灰階均值
    "gray_sigma": 0.02,   # 背景灰階標準差
    "gray_ratio": 0.55,   # 背景混合比例
    "noise_std": 10/255.  # 高斯雜訊標準差
}

# --------------------------------------------------
# Restormer Config
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
    "use_sensor": True,        # 是否啟用 sensor
    "use_sensor_noise": False,  # 是否啟用 sensor noise
}

# --------------------------------------------------
# ========= Training =========
# --------------------------------------------------
TRAINING_CONFIG = {    
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 1e-3,
    "patience": 5,
}

