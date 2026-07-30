# --------------------------------------------------
# Optical Encoder Configuration
# --------------------------------------------------
# Basic calculation: 
# ONN size: 0.00075m = 0.75mm = 750um
# Camera pixel size: 0.000035 = 35um
# 35um * 384 = 0.01344m
# 35um * 288 = 0.01008m


ENCODER_CONFIG = {
    #====== Input image ======
    "image_path": "data/GroundTruth-800-v1/001.png",
    # "image_path": "other_data/FourFSystemSimulationResult/8_SensorLayer_abs.png",
    "save_path": "other_data/FourFSystemSimulationResult",

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
            "z": 0.26,                  # distance (m)
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
            "z": 0.346,
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
            "z": 0.086,
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
            "focal_length": 0.26,
            "dx": 0.00015,
            "num_size": 640,
            "pupil_type": "circular",
            "pupil_radius": 0.0508, # 0.05
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
            "focal_length": 0.086,
            "dx": 0.00015,
            "num_size": 640, 
            "pupil_type": "circular",
            "pupil_radius": 0.0508, # 0.025
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

    "crop_size": (67.2, 67.2), #(288*2, 384*2),
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
    "gain": 0.85, 
    "bias": 0/255, # should between 0 ~ 1, e.g. 0.001, 1/255
    "noise_level": 0,
}
