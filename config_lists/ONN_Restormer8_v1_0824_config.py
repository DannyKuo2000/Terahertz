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
    "resize": 160, # resize shortest side to 128. e.g. 267 or None
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
            "resize_size": (160, 160),  # resize size, e.g., (H, W)
            "displace_size": (0, 0),#(int(30*3483/400), int(100*3483/400)),  # displace size, e.g., (H, W), down & right are positive
            "pad_size": (384, 384),  # pad size, e.g., (H, W)
        },
    ],


    #====== SourceLayer ======: length: 0.03m, size: 160, dx: 0.0001875
    "use_input": False,  # use input source
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
    "pad_size_source": (384, 384),


    #====== DiffractiveLayer ======
    # 4F layout: object -> f1 -> lens1 -> f1 + f2 -> lens2 -> f2 -> image plane
    "diffractive_configs": [
        {
            "name": "ObjectToMaterial1",
            "z": 0.06,                  # distance (m)
            "dx": 0.00025,           # spatial resolution (m)
            "num_size": 384,           # size of each layer
            "frequency": 0.2004e12,     
            "refractive_index": 1,      # refractive index
            "pad_factor": 1,
            "window": "hann",
            "mask_evanescent": False,
            "reverse_z": False,
        },
        {
            "name": "Material1ToMaterial2",
            "z": 0.06,                  # distance (m)
            "dx": 0.00025,           # spatial resolution (m)
            "num_size": 384,           # size of each layer
            "frequency": 0.2004e12,     
            "refractive_index": 1,      # refractive index
            "pad_factor": 1,
            "window": "hann",
            "mask_evanescent": False,
            "reverse_z": False,
        },
        {
            "name": "Material2ToMaterial3",
            "z": 0.06,                  # distance (m)
            "dx": 0.00025,           # spatial resolution (m)
            "num_size": 384,           # size of each layer
            "frequency": 0.2004e12,     
            "refractive_index": 1,      # refractive index
            "pad_factor": 1,
            "window": "hann",
            "mask_evanescent": False,
            "reverse_z": False,
        },
        {
            "name": "Material3ToLens1",
            "z": 0.306,                  # distance (m)
            "dx": 0.00025,           # spatial resolution (m)
            "num_size": 384,           # size of each layer
            "frequency": 0.2004e12,     
            "refractive_index": 1,      # refractive index
            "pad_factor": 1,
            "window": "hann",
            "mask_evanescent": False,
            "reverse_z": False,
        },
        {
            "name": "Lens1ToLens2",
            "z": 0.408,
            "dx": 0.00025,
            "num_size": 384,
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
            "dx": 0.00025,
            "num_size": 384,
            "frequency": 0.2004e12,
            "refractive_index": 1,
            "pad_factor": 1,
            "window": "hann",
            "mask_evanescent": False,
            "reverse_z": False,
        },
    ],

    #====== MaterialLayer ======
    "num_layers": 3,  # Number of material layers
    "material_configs": [
        {
            "name": "Material1",
            "num_size": 128*3,  # number of (ONN neurons * block_size)
            "block_size": (3, 3),  # simulation range of one ONN neuron
            "return_phases": False,  # Switch: return phases for manufacture loss calculation
            "attach_after_diffractive_index": 0,  # Insert after diffractive layer 0 (object -> material1)
        },
        {
            "name": "Material2",
            "num_size": 128*3,
            "block_size": (3, 3),
            "return_phases": False,
            "attach_after_diffractive_index": 1,  # Insert after diffractive layer 1 (material1 -> material2)
        },
        {
            "name": "Material3",
            "num_size": 128*3,
            "block_size": (3, 3),
            "return_phases": False,
            "attach_after_diffractive_index": 2,  # Insert after diffractive layer 2 (material2 -> material3)
        },
    ],
    #====== LensLayer ======
    # Each lens needs its own config because a 4F system can use different focal lengths/apertures.
    # The lens grid must match the propagated field grid: num_size_lens == num_size_diffractive.
    "wavelength": 2.998e8 / 0.2004e12,
    "lens_configs": [
        {   # simulation range: dx * num_size
            "name": "Lens1",
            "focal_length": 0.306,
            "dx": 0.00025,
            "num_size": 384,
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
            "dx": 0.00025,
            "num_size": 384, 
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

    "crop_size": (53.3333, 53.3333), #(67.2, 67.2), #(288*2, 384*2),
    "sensor_displacement": (0, 0),

    "sensor_psf_enabled": False, # switch, PSF (thermal diffusion)
    "sensor_psf_sigma": 1.0,
    "sensor_psf_kernel_size": 9, 

    "use_target_resize": 160,  # force resize to target size after cropping, if None, would use the pitch ratio below
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
# Optical Chain Configuration
# --------------------------------------------------
# This chain controls the execution order of the optical simulation.
# You can reorder, insert, or remove layers here without changing model code.
OPTICAL_CHAIN = [
    {"type": "source", "name": "SourceLayer"},
    {"type": "diffractive", "name": "ObjectToMaterial1", "index": 0},
    {"type": "material", "name": "Material1", "index": 0},
    {"type": "diffractive", "name": "Material1ToMaterial2", "index": 1},
    {"type": "material", "name": "Material2", "index": 1},
    {"type": "diffractive", "name": "Material2ToMaterial3", "index": 2},
    {"type": "material", "name": "Material3", "index": 2},
    {"type": "diffractive", "name": "Material3ToLens1", "index": 3},
    {"type": "lens", "name": "Lens1", "index": 0},
    {"type": "diffractive", "name": "Lens1ToLens2", "index": 4},
    {"type": "lens", "name": "Lens2", "index": 1},
    {"type": "diffractive", "name": "Lens2ToCamera", "index": 5},
    {"type": "sensor", "name": "SensorLayer"},
]


# --------------------------------------------------
# Restormer Configuration
# --------------------------------------------------
RESTORMER_CONFIG = {
    # alignment and downsampling padding
    "padding_factor": 8,
    "use_input_padding": True,   # Pad input only when the spatial size is not divisible by padding_factor
    "output_crop_size": 160,     # Center-crop the final output to this size; set to None to disable
    
    # I/O
    "inp_channels": 1,               # input channel number（gray=1，RGB=3）
    "out_channels": 1,               # output channel number

    # Embedding & Blocks
    "dim": 8,                 # initial dim
    # "dim": 48,
    "num_blocks": [4, 6, 6, 8],      # number of each RestormerBlock
    # "num_blocks": [2, 3, 3, 4],
    "num_refinement_blocks": 2,
    "heads":  [1, 2, 4, 8],      # number of Multi-head Attention of each RestormerBlock

    # Feed-forward setup
    "ffn_expansion_factor": 2.66,    # GDFN expansion factor
    "bias": False,
    "LayerNorm_type": "WithBias",  # "WithBias" or "BiasFree"
    "dual_pixel_task": False,

    # Training Stability
    # "layerscale_init": 1e-2,         # LayerScale initail value（stable in small value）
    # "with_global_residual": True,    # switch of global residual (input+output)
    # Normalization
    # "eps": 1e-6                      # LayerNorm2d epsilon
}

# --------------------------------------------------
# Autoencoder Configuration
# --------------------------------------------------
AUTOENCODER_CONFIG = {
    "use_encoder": True, # switch
    "use_decoder": True, # switch
    "return_phases": False, # switch，for calculation of Phase local contrast loss
}

# --------------------------------------------------
# Training Configuration
# --------------------------------------------------
TRAINING_CONFIG = {    
    # ====== Save path & model setting ======
    "checkpoints_weights_save_dir": "./checkpoints_weights/ONN_Restormer8_v1_0824",  #! check before training. e.g. ./checkpoints_weights/{run_file_name}
    "writer_save_path": "runs/ONN_Restormer8_v1_0824",  #! check before training. TensorBoard save path, e.g. runs/{run_file_name}
    "csv_log_enabled": False,  # Enable CSV logging for per-epoch metrics  #! close to improve training speed
    "csv_log_path": "./checkpoints_weights/ONN_Restormer8_v1_0824/training_log.csv",  #! check before trainging. CSV log file path
    "best_model_name": "best_model.pth",  # Filename for the best model weights
    "last_model_name": "last_model.pth",  # Filename for the latest model weights
    "best_checkpoint_name": "best_checkpoint.pth",  # Filename for the best full checkpoint
    "last_checkpoint_name": "last_checkpoint.pth",  # Filename for the latest full checkpoint
    
    # ====== TensorBoard setting ======  #! close or reduce to improve training speed
    "tb_log_lr": False,  # Log learning rate to TensorBoard  
    "tb_log_epoch_time": False,  # Log epoch duration to TensorBoard  
    "tb_log_gpu_memory": False,  # Log GPU memory usage to TensorBoard  
    "tb_log_recon_every_n_epochs": 5,  # Log reconstruction images every N epochs
    "tb_log_recon_num_images": 8,  # Number of images to log for reconstruction visualization


    # ====== Resume training ======
    "resume_training": False,  # switch, if want to start trainging from checkpoint
    "resume_checkpoint_path": "./checkpoints_weights/ONN_Restormer8_v1_0824/checkpoints/epoch30_valLoss0.0123_20251026_154501.pth",  #! check before resume. ./checkpoints_weights/{run_file_name}/checkpoints/...
    
    # ====== Experiments hyperparameters ======
    # === Debug ===
    "enable_profiling": False,  # extra info: para number, runtime  #! This would consume a lot of computation
    "profile_steps": 0,  # how many iteration to print out extra info

    # === Parallel ===
    "distributed": True,
    "num_workers": 4,  # using 0 in single GPU
    
    # === Memory and Time Optimization ===
    "use_amp": False,  #! Automatic Mixed Precision (AMP): for reduction calculation cost and memory. This may cause training error in optical simulation
    "grad_accum_steps": 1,  # number of mini batches 
    
    # === Training hyperparameters ===
    "global_batch_size": 40,
    "epochs": 500,
    "max_iterations": 300_000,

    # loss
    "loss": "L1",

    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "optimizer_params": {  #! params for AdamW
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
    },

    # Scheduler
    "use_scheduler": True,
    "scheduler_type": "CosineAnnealingLR",
    "scheduler_params": {
        "T_max": 300_000,  # how many scheduler.step() to reach "eta_min"
        "eta_min": 1e-6,
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
    "distributed": True,
    "num_workers": 4,  # using 0 in single GPU
    "batch_size": 40,

    # # load config
    # "weight_save_dir": './checkpoints_weights/ONN_Restormer8_v1_0824/weights',  #! check before testing. e.g.: ./checkpoints_weights/{run_name}/weights
    # "weight_save_name": 'epoch144_loss0.0101_20260810_123923.pth',
    "weight_save_dir": './checkpoints_weights/ONN_Restormer8_v1_0824',  #! check before testing. e.g.: ./checkpoints_weights/{run_name}/weights
    "weight_save_name": 'last_model.pth',

    # save config
    "results_save_dir": './results/ONN_Restormer8_v1_0824',  #! check before testing.
    "results_save_name_suffix": '_metrics.json',

    # ONN debug
    "onn_debug": True, # print each layer of encoder (including ONN). Switch
    "ONN_input_select": "fix",  # fix or random
    "ONN_input_idx": 0,  # if fixxed select, input the image number
    "seed": None,  # if randomly select, choose a seed
}



# --------------------------------------------------
# Latent Analysis Configuration
# --------------------------------------------------
LATENT_ANALYSIS_CONFIG = {    
    # === Parallel ===
    "distributed": False,
    "num_workers": 0,  # using 0 in single GPU
    "batch_size": 40,

    # load config
    # "weight_save_dir": './checkpoints_weights/ONN_Restormer8_v1_0824/weights',  #! check before analyzing. e.g.: ./checkpoints_weights/{run_name}/weights
    # "weight_save_name": 'epoch99_Loss0.0005_20260731_203404.pth',
    "weight_save_dir": './checkpoints_weights/ONN_Restormer8_v1_0824',  #! check before analyzing. e.g.: ./checkpoints_weights/{run_name}/weights
    "weight_save_name": 'best_model.pth',

    # save config
    "results_save_dir": './latent_analysis/ONN_Restormer8_v1_0824_analysis5',  #! check before analyzing.
    "results_save_name_suffix": '_metrics.json',

    # latent analysis modes
    "analysis_modes": [
        # "mask_outer",
        # "mask_center",
        "ring_by_ring",
        # "random_mask",
    ],

    # shared mask ratios for mask_outer / mask_center / random_mask
    "mask_ratios": [
        # 0.01,
        # 0.02,
        # 0.03,
        # 0.04,
        # 0.05,
        # 0.06,
        # 0.07,
        # 0.08,
        # 0.09,
        # 0.10,
        # 0.111,
        # 0.112,
        # 0.113,
        # 0.114,
        # 0.115,
        # 0.116,
        # 0.117,
        0.1180,
        0.1185,
        0.1190,
        0.1195,
        0.1200,
        0.1205,  
        0.1210,
        0.1215,
        0.1220,
        0.1225,
        0.1230,
        0.1235,
        0.1240,
        0.1245,
        0.1250,
        0.1255,
        0.126,
        # 0.127,
        # 0.128,
        # 0.129,
        # 0.130,
        # 0.131,
        # 0.132,
        # 0.133,
        # 0.134,
        # 0.135,
        # 0.136,
        # 0.137,
        # 0.138,
        # 0.139,
        # 0.140,
        # 0.141,
        # 0.142,
        # 0.143,
        # 0.144,
        # 0.145,
        # 0.146,
        # 0.147,
        # 0.148,
        # 0.149, 
        # 0.150,
        # 0.151,
        # 0.152,
        # 0.153,
        # 0.154,
        # 0.155,
        # 0.156,
        # 0.157,
        # 0.158,
        # 0.159,
        # 0.160,
    ],

    # ring-by-ring settings
    # larger ring number means outer ring, smaller ring number means inner ring.
    "ring_num": 60,
    "ring_mask_steps": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30], # ring numbers to be masked in each step
    
    # Optional explicit ring combinations, e.g. [[7], [6, 7], [5, 6, 7]]
    # If None, the code uses cumulative outer-ring masking from ring_mask_steps.
    "ring_mask_patterns": None, #[[15, 16, 17], [16, 17], [17]],  # None,

    # random mask control
    "mask_seed": 1234,

    # visualization control
    "visualize_samples": 20,
    "save_sample_panels": True,
    "save_mask_previews": True,
    "save_metric_curves": True,
}


# --------------------------------------------------
# Latent Analysis V2 Configuration
# --------------------------------------------------
LATENT_ANALYSIS_V2_CONFIG = {
    **LATENT_ANALYSIS_CONFIG,

    # analysis name
    "analysis_name": "region_heatmap",

    # latent space partition
    "region_rows": 8,
    "region_cols": 8,

    # If None, analyze all regions.
    # If provided, only these region ids will be analyzed.
    # Region ids are assigned in row-major order.
    "region_ids_to_analyze": None,

    # heatmap metrics to save
    # supported: delta_psnr, delta_mse, relative_mse, delta_ssim, masked_psnr,
    # masked_mse, masked_ssim, recon_diff_l1, recon_diff_mse
    "heatmap_metrics": [
        "delta_psnr",
        "delta_mse",
        "relative_mse",
        "recon_diff_l1",
    ],

    # visualization control
    "annotate_heatmap": False,
    "save_baseline_panel": True,
    "save_region_previews": True,
    "preview_region_limit": 6,
    "preview_region_ids": None,
    "save_output_diff_maps": True,
    "save_output_diff_maps_limit": 6,

    # whether to save per-region JSON files
    "save_region_json": True,
}
