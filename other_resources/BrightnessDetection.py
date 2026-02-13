import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from imageio import imread

# ==================================================
# Configuration
# ==================================================

ROI_SIZE = 288
TOP_PERCENT = 1.0
OUTPUT_DIR = "results/DecayAnalysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# ROI utilities
# ==================================================

def extract_center_roi(img, roi_size=288):
    if img.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {img.shape}")

    H, W = img.shape
    if roi_size > min(H, W):
        raise ValueError("ROI larger than image")

    y0 = (H - roi_size) // 2
    x0 = (W - roi_size) // 2
    return img[y0:y0+roi_size, x0:x0+roi_size]

# ==================================================
# Image loading (SINGLE FILE)
# ==================================================

def load_single_image(image_path):
    """
    Load a single image file.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    img = imread(image_path).astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)  # convert to grayscale if needed
    return img

# ==================================================
# Intensity metrics
# ==================================================

def compute_roi_metrics(img, roi_size=288):
    roi = extract_center_roi(img, roi_size)
    return np.mean(roi), np.sum(roi)

def compute_peak_metric(img, roi_size=288, top_percent=1.0):
    roi = extract_center_roi(img, roi_size)
    k = max(1, int(roi.size * top_percent / 100))
    return np.mean(np.sort(roi.ravel())[-k:])

# ==================================================
# Physical attenuation model
# ==================================================

def attenuation_model(z, C, alpha):
    return C * z**(-2) * np.exp(-alpha * z)

# ==================================================
# Main analysis
# ==================================================

def analyze_distance_attenuation(image_files, z_positions):

    if len(image_files) != len(z_positions):
        raise ValueError("image_files and z_positions must have the same length")

    I_mean = []
    I_peak = []

    for path in image_files:
        img = load_single_image(path)

        mean_I, _ = compute_roi_metrics(img, ROI_SIZE)
        peak_I = compute_peak_metric(img, ROI_SIZE, TOP_PERCENT)

        I_mean.append(mean_I)
        I_peak.append(peak_I)

    I_mean = np.array(I_mean)
    I_peak = np.array(I_peak)
    z_positions = np.array(z_positions)

    popt, pcov = curve_fit(
        attenuation_model,
        z_positions,
        I_mean,
        p0=(np.max(I_mean), 0.1),
        maxfev=10000
    )

    C_fit, alpha_fit = popt
    alpha_std = np.sqrt(np.diag(pcov))[1]

    print("\n========== Fitting Results ==========")
    print(f"C       = {C_fit:.3e}")
    print(f"alpha   = {alpha_fit:.3e} 1/m")
    print(f"alpha σ = {alpha_std:.3e} 1/m")
    print("=====================================\n")

    z_fit = np.linspace(z_positions.min(), z_positions.max(), 300)
    I_fit = attenuation_model(z_fit, C_fit, alpha_fit)

    # ---------- Plot: intensity vs distance ----------
    plt.figure()
    plt.loglog(z_positions, I_mean, "o", label="ROI mean")
    plt.loglog(z_positions, I_peak, "x", label="Peak (top 1%)")
    plt.loglog(z_fit, I_fit, "-", label="Fit")
    plt.xlabel("z (m)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "intensity_vs_distance.png"), dpi=300)
    plt.show()

    # ---------- Plot: residual ----------
    plt.figure()
    plt.plot(
        z_positions,
        I_mean - attenuation_model(z_positions, C_fit, alpha_fit),
        "o-"
    )
    plt.xlabel("z (m)")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fit_residual.png"), dpi=300)
    plt.show()

    # ---------- Save numeric results ----------
    np.savetxt(
        os.path.join(OUTPUT_DIR, "attenuation_results.txt"),
        np.column_stack([z_positions, I_mean, I_peak]),
        header="z(m)  I_mean  I_peak"
    )

# ==================================================
# Entry point
# ==================================================

if __name__ == "__main__":

    image_files = [
        "other_data/NVLab260116_processed/34cm_AGCoff.bmp",
        "other_data/NVLab260116_processed/35cm_AGCoff.bmp",
        "other_data/NVLab260116_processed/36cm_AGCoff.bmp",
        "other_data/NVLab260116_processed/37cm_AGCoff.bmp",
        "other_data/NVLab260116_processed/38cm_AGCoff.bmp",
    ]

    z_positions = [0.08, 0.09, 0.10, 0.11, 0.12]

    analyze_distance_attenuation(image_files, z_positions)
