# --------------------------------------------------
# Optical Encoder Configuration
# --------------------------------------------------
# Basic calculation: 
# ONN size: 0.00075 (m)
# simulation pixel size: 0.00075/4 (m) = 0.0001875 (m) (每個ONN element 用4*4的模擬去跑)
# image length = 0.03 (m) => 0.03/0.0001875 = 160 pixels


ENCODER_CONFIG = {
    "image_path": "data/GroundTruth-800-v1/001.png",
    "save_path": "results/Presentation260818/Equivalent4FDemonstration",

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
    "created_size": (384, 384),  # size of gaussian beam
    "source_is_intensity": True,

    "sigma": 0.2,  # sigma of gaussian
    "amplitude": 1.0,  # amplitude of gaussian, range: [0, 1]
    "center": (0.0, 0.0),  # center of gaussian(pixel)
    "rotation": 0.0,  # rotation of gaussian(angle)
    "aspect_ratio": 1.0,  # oval ratio

    "crop_size_source": None,
    "resize_size_source": (384, 384),
    "displace_size_source": None,
    "pad_size_source": (384, 384),


    #====== DiffractiveLayer ======
    # 4F layout: object -> f1 -> lens1 -> f1 + f2 -> lens2 -> f2 -> image plane
    "diffractive_configs": [
        {
            "name": "ObjectToLens",
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
            "name": "LensToSensor",
            "z": 0.102,                  # distance (m)
            "dx": 0.00025,           # spatial resolution (m)
            "num_size": 384,           # size of each layer
            "frequency": 0.2004e12,     
            "refractive_index": 1,      # refractive index
            "pad_factor": 1,
            "window": "hann",
            "mask_evanescent": False,
            "reverse_z": False,
        },
        
    ],
    
    #====== MaterialLayer ======
    "num_layers": 0,          # Number of ONN layers
    "material_configs": [
        # {
        #     "name": "Material1",
        #     "num_size": 128,
        #     "block_size": (2, 2),
        #     "mode": "default",  # e.g. "default", "border"
        #     "border_width": None,  # only valid when mode is "border"
        #     "return_phases": False,  # Switch: return phases for manufacture loss calculation
        # },
        # {
        #     "name": "Material2",
        #     "num_size": 128,
        #     "block_size": (2, 2),
        #     "mode": "default",  # e.g. "default", "border"
        #     "border_width": None,  # only valid when mode is "border"
        #     "return_phase": False,
        # },
    ],


    #====== LensLayer ======
    # Each lens needs its own config because a 4F system can use different focal lengths/apertures.
    # The lens grid must match the propagated field grid: num_size_lens == num_size_diffractive.
    "wavelength": 2.998e8 / 0.2004e12,
    "lens_configs": [
        {   # simulation range: dx * num_size
            "name": "Lens",
            "focal_length": 0.0765,
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
    ],


    #====== SensorLayer ======
    "active_sensor": True, # switch

    "crop_size": (128, 128),#(128, 128), #(67.2, 67.2), #(288*2, 384*2),
    "sensor_displacement": (0, 0),

    "sensor_psf_enabled": False, # switch, PSF (thermal diffusion)
    "sensor_psf_sigma": 1.0,
    "sensor_psf_kernel_size": 9, 

    "use_target_resize": 128,  # force resize to target size after cropping, if None, would use the pitch ratio below
    "simulation_pitch": 250,  # in um (micrometer)
    "target_pitch": 250,  # in um (micrometer)

    "bin_size": 1,
    "flip": True,

    #====== SensorNoiseLayer ======
    "active_sensor_noise": True, # Switch
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
    {"type": "diffractive", "name": "ObjectToLens", "index": 0},
    {"type": "lens", "name": "Lens", "index": 0},
    {"type": "diffractive", "name": "LensToSensor", "index": 1},
    {"type": "sensor", "name": "SensorLayer"},
]