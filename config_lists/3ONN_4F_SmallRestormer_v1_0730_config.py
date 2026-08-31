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
    "resize": 167, # resize shortest side to 128. e.g. 267 or None
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
            "resize_size": (167, 167),  # resize size, e.g., (H, W)
            "displace_size": (0, 0),#(int(30*3483/400), int(100*3483/400)),  # displace size, e.g., (H, W), down & right are positive
            "pad_size": (640, 640),  # pad size, e.g., (H, W)
        },
    ],


    #====== SourceLayer ======: length: 0.03m, size: 160, dx: 0.0001875
    "use_input": False,  # use input source
    "input": "other_data/NVLab260608/MultiSnap_2026-06-08_10-59-23_0265_0000.bmp",  # input source path
    "mode_source": "white",  # default source mode, e.g., "white" or "gaussian"
    "created_size": (512, 512),  # size of gaussian beam
    "source_is_intensity": True,
    "return_phases": False,  # Return phase tensors from material layers for phase-based losses

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
    # Ordered propagation segments between optical elements
    "diffractive_configs": [
        {
            "name": "ObjectToMaterial1",
            "z": 0.06,                  # distance (m)
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
            "name": "Material1ToMaterial2",
            "z": 0.06,                  # distance (m)
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
            "name": "Material2ToMaterial3",
            "z": 0.06,                  # distance (m)
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
            "name": "Material3ToLens1",
            "z": 0.306,                  # distance (m)
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
            "z": 0.408,
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
    "num_layers": 3,  # Number of material layers
    "material_configs": [
        {
            "name": "Material1",
            "num_size": 128*5,  # number of (ONN neurons * block_size)
            "block_size": (5, 5),  # simulation range of one ONN neuron
            "return_phases": False,  # Switch: return phases for manufacture loss calculation
            "attach_after_diffractive_index": 0,  # Insert after diffractive layer 0 (object -> material1)
        },
        {
            "name": "Material2",
            "num_size": 128*5,
            "block_size": (5, 5),
            "return_phases": False,
            "attach_after_diffractive_index": 1,  # Insert after diffractive layer 1 (material1 -> material2)
        },
        {
            "name": "Material3",
            "num_size": 128*5,
            "block_size": (5, 5),
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

    "crop_size": (55.5556, 55.5556), #(67.2, 67.2), #(288*2, 384*2),
    "sensor_displacement": (0, 0),

    "sensor_psf_enabled": False, # switch, PSF (thermal diffusion)
    "sensor_psf_sigma": 1.0,
    "sensor_psf_kernel_size": 9, 

    "use_target_resize": 168,  # force resize to target size after cropping, if None, would use the pitch ratio below
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
    "output_crop_size": 167,     # Center-crop the final output to this size; set to None to disable
    
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
    "eps": 1e-6                      # LayerNorm2d epsilon
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
    "checkpoints_weights_save_dir": "./checkpoints_weights/3ONN_4F_SmallRestormer_v1_0730",  #! check before training. e.g. ./checkpoints_weights/{run_file_name}
    "writer_save_path": "runs/3ONN_4F_SmallRestormer_v1_0730",  #! check before training. TensorBoard save path, e.g. runs/{run_file_name}

    "csv_log_enabled": True,  # Enable CSV logging for per-epoch metrics
    "csv_log_path": "./checkpoints_weights/3ONN_4F_SmallRestormer_v1_0730/training_log.csv",  #! check before trainging. CSV log file path
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
    "resume_checkpoint_path": "./checkpoints_weights/3ONN_4F_SmallRestormer_v1_0730/checkpoints/epoch99_Loss0.0005_20260731_203404.pth",  #! check before resume. ./checkpoints_weights/{run_file_name}/checkpoints/...
    
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
    # "weight_save_dir": './checkpoints_weights/3ONN_4F_SmallRestormer_v1_0730/weights',  #! check before testing. e.g.: ./checkpoints_weights/{run_name}/weights
    # "weight_save_name": 'epoch99_Loss0.0005_20260731_203404.pth',
    "weight_save_dir": './checkpoints_weights/3ONN_4F_SmallRestormer_v1_0730',  #! check before testing. e.g.: ./checkpoints_weights/{run_name}/weights
    "weight_save_name": 'best_model.pth',

    # save config
    "results_save_dir": './results/3ONN_4F_SmallRestormer_v1_0730',  #! check before testing
    "results_save_name_suffix": '_metrics.json',

    # ONN debug
    "onn_debug": True, # print each layer of encoder (including ONN). Switch
    "ONN_input_select": "fix",  # fix or random
    "ONN_input_idx": 0,  # if fixxed select, input the image number
    "seed": None,  # if randomly select, choose a seed
}
