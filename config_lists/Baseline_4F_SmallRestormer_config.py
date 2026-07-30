# --------------------------------------------------
# Real Dataset Configuration
# --------------------------------------------------
DATASET_CONFIG = {
    "dataset_name": "MNIST+EMNIST",   # Options: "MNIST" | "FashionMNIST" | "EMNIST" | "Custom" | "MNIST+EMNIST"
    
    "emnist_split": "byclass",  # dataset type of EMNIST (only available in EMNIST or MNIST+EMNIST)
    "mnist_size": 60000,  # size of mnist for training (only available in MNIST+EMNIST)
    "emnist_size": 15000,  # size for emnist training (only available in MNIST+EMNIST)
    "emnist_test_ratio": 0.25,  # propotion of EMNIST dataset joined in test (only available in MNIST+EMNIST)
    "mnist_test_size": None,  # spare parameter
    "emnist_test_size": None,  # spare parameter
    "seed": 42,  # seed (only available in MNIST+EMNIST), using 42 as default
    "root": "./data/RealDataset-800-v1",  # Path of custom dataset (only available in custom dataset)
    
    "valid_ratio": 0.05,   # propotion of validation
    "test_ratio": None,    # propotion of testing (if needed)
    "resize": 267, # resize shortest side to 128. e.g. 267 or None
    "center_crop": None, #(400, 400), # e.g. (400, 400) or None
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
# Basic calculation: 
# ONN size: 0.00075 (m)
# simulation pixel size: 0.00075/4 (m) = 0.0001875 (m) (each ONN neuron used 4*4 simulation pixels)
# ONN size: 0.00075m = 0.75mm = 750um
# Camera pixel size: 0.000035 = 35um
# 35um * 384 = 0.01344m
# 35um * 288 = 0.01008m
ENCODER_CONFIG = {
    #====== Transformation for input image ======
    "transform_configs": [
        {
            "name": "InputTransform",
            "crop_size": None,  # crop size, e.g., (H, W)
            "resize_size": (267, 267),  # resize size, e.g., (H, W)
            "displace_size": (0, 0),#(int(30*3483/400), int(100*3483/400)),  # displace size, e.g., (H, W), down & right are positive
            "pad_size": (640, 640),  # pad size, e.g., (H, W)
        },
    ],


    #====== SourceLayer ======: length: 0.03m, size: 160, dx: 0.0001875
    "use_input": True,  # use input source
    "input": "other_data/NVLab260608/MultiSnap_2026-06-08_10-59-23_0265_0000.bmp",  # input source path
    "mode_source": "white",  # default source mode, e.g., "white" or "gaussian"
    "created_size": (512, 512),  # size of gaussian beam
    "source_is_intensity": True,

    "sigma": 0.3,  # sigma of gaussian
    "amplitude": 1.0,  # amplitude of gaussian, range: [0, 1]
    "center": (0.0, 0.0),  # center of gaussian(pixel)
    "rotation": 0.0,  # rotation of gaussian(angle)
    "aspect_ratio": 1.0,  # oval ratio

    "crop_size_source": None,
    "resize_size_source": (267, int(267*384/288)),
    "displace_size_source": None,
    "pad_size_source": (640, 640),


    #====== DiffractiveLayer ======
    # 4F layout: object -> f1 -> lens1 -> f1 + f2 -> lens2 -> f2 -> image plane
    "diffractive_configs": [
        {
            "name": "ObjectToLens1",
            "z": 0.255,                  # distance (m)
            "dx": 0.00015,           # spatial resolution (m)
            "num_size": 640,           # size of each layer
            "frequency": 0.2004e12,     
            "refractive_index": 1,      # refractive index
            "pad_factor": 1,
            "window": "hann",
            "mask_evanescent": False,
            "reverse_z": False,
        },
        {
            "name": "Lens1ToLens2",
            "z": 0.357,
            "dx": 0.00015,
            "num_size": 640,
            "frequency": 0.2004e12,
            "refractive_index": 1,
            "pad_factor": 1,
            "window": "hann",
            "mask_evanescent": False,
            "reverse_z": False,
        },
        {
            "name": "Lens2ToCamera",
            "z": 0.102,
            "dx": 0.00015,
            "num_size": 640,
            "frequency": 0.2004e12,
            "refractive_index": 1,
            "pad_factor": 1,
            "window": "hann",
            "mask_evanescent": False,
            "reverse_z": False,
        },
    ],
    
    #====== MaterialLayer ======
    "num_layers": 0,          # Number of ONN layers
    "material_configs": [
        {
            "name": "Material1",
            "num_size": 128,
            "block_size": (2,2),
            "return_phases": False,  # Switch: return phases for manufacture loss calculation
        },
        {
            "name": "Material2",
            "num_size": 128,
            "block_size": (2,2),
            "return_phase": False,
        },
    ],


    #====== LensLayer ======
    # Each lens needs its own config because a 4F system can use different focal lengths/apertures.
    # The lens grid must match the propagated field grid: num_size_lens == num_size_diffractive.
    "wavelength": 2.998e8 / 0.2004e12,
    "lens_configs": [
        {   # simulation range: dx * num_size
            "name": "Lens1",
            "focal_length": 0.255,
            "dx": 0.00015,
            "num_size": 640,
            "pupil_type": "circular",
            "pupil_radius": 0.0508, # 0.05 # radius usually in inches
            "pupil_width": None,
            "phase_model": "exact", # high NA: exact, NA ~ sin(arctan(r/f))
            "mode": "forward",
            "outside": "zero",
            "frame": False,
            "frame_inner": None,
            "frame_outer": None,
        },
        {
            "name": "Lens2",
            "focal_length": 0.102,
            "dx": 0.00015,
            "num_size": 640, 
            "pupil_type": "circular",
            "pupil_radius": 0.0508, # 0.025 # radius usually in inches
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
    "active_sensor": True, # switch

    "crop_size": (62.3, 62.3), #(67.2, 67.2), #(288*2, 384*2),
    "sensor_displacement": (0, 0),

    "sensor_psf_enabled": False, # switch, PSF (thermal diffusion)
    "sensor_psf_sigma": 1.0,
    "sensor_psf_kernel_size": 9, 

    "simulation_pitch": 150,  # in um (micrometer)
    "target_pitch": 35,  # in um (micrometer)

    "bin_size": 1,
    "flip": True,

    #====== SensorNoiseLayer ======
    "active_sensor_noise": False, # Switch
    "blur_kernel_size": 15,
    "blur_sigma": 5,
    "gray_mean": 0.6,     # background gray mean
    "gray_sigma": 0.02,   # background gray std
    "gray_ratio": 0.55,   # background mixed ratio
    "noise_std": 10/255,  # Gaussian noise std


    #====== Final Process ====== simulate as Brightness and Contrast
    "gain": 1, 
    "bias": 0/255, # should between 0 ~ 1, e.g. 0.001, 1/255
    "noise_level": 0,
}


# --------------------------------------------------
# Restormer Configuration
# --------------------------------------------------
RESTORMER_CONFIG = {
    # alignment and downsampling padding
    "padding_factor": 8,
    
    # I/O
    "inp_channels": 1,               # input channel number（gray=1，RGB=3）
    "out_channels": 1,               # output channel number

    # Embedding & Blocks
    #"embed_dim": 48,                 # initial dim
    "embed_dim": 16,
    #"num_blocks": [4, 6, 6, 8],      # number of each RestormerBlock
    "num_blocks": [2, 3, 3, 4],
    "num_heads":  [1, 2, 4, 8],      # number of Multi-head Attention of each RestormerBlock

    # Training Stability
    "layerscale_init": 1e-2,         # LayerScale initail value（stable in small value）
    "with_global_residual": True,    # switch of global residual (input+output)

    # Feed-forward setup
    "ffn_expansion_factor": 2.66,    # GDFN expansion factor

    # Normalization
    #"eps": 1e-6                      # LayerNorm2d epsilon
}

# --------------------------------------------------
# Autoencoder Configuration
# --------------------------------------------------
AUTOENCODER_CONFIG = {
    "use_encoder": True, # switch
    "use_decoder": True, # switch
    "return_phases": True, # switch，for calculation of Phase local contrast loss
}

# --------------------------------------------------
# Training Configuration
# --------------------------------------------------
TRAINING_CONFIG = {    
    # ====== Save path & model setting ======
    "checkpoints_weights_save_dir": "./checkpoints_weights/Baseline_4F_SmallRestormer",  # ./checkpoints_weights/{run_file_name}
    "writer_save_path": "runs/Baseline_4F_SmallRestormer",  # TensorBoard save path, e.g. runs/{run_file_name}

    "csv_log_enabled": True,  # Enable CSV logging for per-epoch metrics
    "csv_log_path": "./checkpoints_weights/Baseline_4F_SmallRestormer/training_log.csv",  # CSV log file path
    "best_model_name": "best_model.pth",  # Filename for the best model weights
    "last_model_name": "last_model.pth",  # Filename for the latest model weights
    "best_checkpoint_name": "best_checkpoint.pth",  # Filename for the best full checkpoint
    "last_checkpoint_name": "last_checkpoint.pth",  # Filename for the latest full checkpoint
    
    # ====== TensorBoard setting ======
    "tb_log_lr": True,  # Log learning rate to TensorBoard
    "tb_log_epoch_time": True,  # Log epoch duration to TensorBoard
    "tb_log_gpu_memory": True,  # Log GPU memory usage to TensorBoard
    "tb_log_recon_every_n_epochs": 5,  # Log reconstruction images every N epochs
    "tb_log_recon_num_images": 20,  # Number of images to log for reconstruction visualization


    # ====== Resume training ======
    "resume_training": False,  # switch, if want to start trainging from checkpoint
    "resume_checkpoint_path": "./checkpoints_weights/Baseline_4F_SmallRestormer/checkpoints/epoch30_valLoss0.0123_20251026_154501.pth",  # ./checkpoints_weights/{run_file_name}/checkpoints/...
    
    # ====== Training hyperparameters ======
    # === Parallel ===
    "distributed": False,
    "num_workers": 0,  # using 0 in single GPU
    # === Memory and Time Optimization ===
    "use_amp": False,  # Automatic Mixed Precision: for reduction calculation cost and memory
    "grad_accum_steps": 2,  # number of mini batches
    # === Others ===
    "batch_size": 8,
    "epochs": 100,
    "learning_rate": 1e-3,
    "patience": 10,
    "use_scheduler": True,                     # switch, using scheduler
    "scheduler_type": "ReduceLROnPlateau",     # scheduler type: "StepLR", "CosineAnnealingLR"
    "scheduler_params": {                      # scheduler parameters
        "mode": "min",  # minimization val loss
        "factor": 0.5,  # LR = LR * 0.5
        "min_lr": 1e-6,  # minimum of LR
        "patience": 4,  # number of epochs waiting for reducing LR
        #"verbose": True,  # print LR changing infor
    },

    # ====== Phase Local Contrast loss =======
    "return_phases": False,  # add Phase local contrast loss. Switch
    "plc_loss_weight": 1e-4,  # loss weight of phase local contrast loss
    "plc_sigma": 100,  # std
    "use_weight": True,
    "loss_mode": "margin"  # "margin": Margin-based Gradient Loss, "mean"
    #"margin":  # for margin mode only
}

# --------------------------------------------------
# Testing Configuration
# --------------------------------------------------
TESTING_CONFIG = {    
    # === Parallel ===
    "distributed": False,
    "num_workers": 0,  # using 0 in single GPU
    "batch_size": 8,

    # load config
    "weight_save_dir": './checkpoints_weights/Baseline_4F_SmallRestormer/weights',  # e.g.: ./checkpoints_weights/{run_name}/weights
    "weight_save_name": 'epoch98_Loss0.0013_20260721_111517.pth',
    # "weight_save_dir": './checkpoints_weights/baseline_restormer_ONN_PLC/weights',  # e.g.: ./checkpoints_weights/{run_name}/weights
    # "weight_save_name": 'epoch56_valLoss0.0025_20251101_071249.pth',
    # "weight_save_dir": './checkpoints_weights/baseline_restormer_ONN/weights',  # e.g.: ./checkpoints_weights/{run_name}/weights
    # "weight_save_name": 'epoch60_valLoss0.0021_20251030_224229.pth',


    # save config
    "results_save_dir": './results/Baseline_4F_SmallRestormer_test',
    # "results_save_dir": './results/baseline_restormer_ONN_pad2',
    "results_save_name_suffix": '_metrics.json',

    # ONN debug
    "onn_debug": True, # print each layer of encoder (including ONN). Switch
    "ONN_input_select": "fix",  # fix or random
    "ONN_input_idx": 0,  # if fixxed select, input the image number
    "seed": None,  # if randomly select, choose a seed
}
