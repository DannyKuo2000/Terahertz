import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# SETTINGS
# ============================================================
input_dir = "other_data/NVLab260612_fixed/Type2"
output_dir = "other_data/NVLab260612_results/Type2"

mode = "beam_fit"
# "profile"   -> 1D Gaussian fit
# "moment"    -> second moment method
# "beam_fit"  -> propagation fitting

# replace with real z if available
z_values = [60.0, 60.5, 61.0, 61.5, 62.0, 62.5, 63.0, 63.5, 64.0, 64.5, 65.0]
# z_values = [10.7, 10.8, 10.9]  

exts = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")

# ============================================================
# ROI CROP SETTING
# ============================================================
# Format: [x_min, x_max, y_min, y_max]
# Set to None -> use full image
H = 288
W = 384
h = 150  # height
w = 150  # width
cx = W // 2
cy = H // 2
x_min = cx - w // 2
x_max = x_min + w
y_min = cy - h // 2
y_max = y_min + h
roi = [x_min, x_max, y_min, y_max]
# roi = None  # Example: roi = [80, 240, 60, 200]

def apply_roi(img, roi):
    """
    Crop image to region of interest.
    roi format: [x_min, x_max, y_min, y_max]
    """
    if roi is None:
        return img
    x_min, x_max, y_min, y_max = roi
    return img[y_min:y_max, x_min:x_max]


# ============================================================
# Gaussian model
# ============================================================
def gaussian(x, A, x0, sigma, B):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + B


# ============================================================
# Beam center
# ============================================================
def find_beam_center(img):
    y, x = np.unravel_index(np.argmax(img), img.shape)
    return x, y


# ============================================================
# Fit Gaussian
# ============================================================
def fit_profile(profile):
    profile = profile.astype(np.float64)
    profile = profile - np.min(profile)  # remove background

    if np.max(profile) <= 0:
        return None

    profile = profile / np.max(profile)  # max normalization, make calculation stable

    x = np.arange(len(profile))

    try:
        popt, _ = curve_fit(
            gaussian,
            x,
            profile,
            p0=[1.0, np.argmax(profile), len(profile)/10, 0.0],
            maxfev=10000
        )
        return abs(popt[2])

    except:
        return None


# ============================================================
# Moment method
# ============================================================
def moment_sigma(profile):
    profile = profile.astype(np.float64)
    profile = profile - np.min(profile)  # remove background

    if profile.sum() <= 0:
        return None

    x = np.arange(len(profile))
    x0 = np.sum(x * profile) / np.sum(profile)

    sigma = np.sqrt(np.sum(profile * (x - x0)**2) / np.sum(profile))
    return sigma


# ============================================================
# Beam model (for fitting)
# ============================================================
def beam_model(z, sigma0, z0, zR):
    return sigma0 * np.sqrt(1 + ((z - z0) / zR)**2)


# ============================================================
# Load images
# ============================================================
image_files = sorted([
    f for f in os.listdir(input_dir)
    if f.lower().endswith(exts)
])

N = len(image_files)

if z_values is None:
    print("[WARNING] Using index as pseudo-z (not physical)")
    z_values = np.arange(N)

assert len(z_values) == N

results = []


# ============================================================
# MAIN LOOP
# ============================================================
for idx, fname in enumerate(image_files):

    img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_UNCHANGED)

    if img is None:
        continue

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = apply_roi(img, roi)
    img = img.astype(np.float64)

    hx = np.sum(img, axis=0)
    vy = np.sum(img, axis=1)

    # --------------------------------------------------------
    # MODE SWITCH
    # --------------------------------------------------------
    if mode == "profile":
        sx = fit_profile(hx)
        sy = fit_profile(vy)
    elif mode == "moment":
        sx = moment_sigma(hx)
        sy = moment_sigma(vy)
    elif mode == "beam_fit":
        sx = fit_profile(hx)
        sy = fit_profile(vy)
    else:
        raise ValueError("Invalid mode")

    if sx is None or sy is None:
        print(f"[FAIL] {fname}")
        continue

    sigma_avg = (sx + sy) / 2

    results.append({
        "z": z_values[idx],
        "sigma_x": sx,
        "sigma_y": sy,
        "sigma_avg": sigma_avg
    })

    # ========================================================
    # PRINT EACH RESULT
    # ========================================================
    print(
        f"{fname} | "
        f"z={z_values[idx]:.3f} | "
        f"sigma_x={sx:.3f} | "
        f"sigma_y={sy:.3f} | "
        f"sigma_avg={sigma_avg:.3f}"
    )


# ============================================================
# SAVE CSV
# ============================================================
csv_name = f"{output_dir}/beam_{mode}.csv"
os.makedirs(os.path.dirname(csv_name), exist_ok=True)

with open(csv_name, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["z", "sigma_x", "sigma_y", "sigma_avg"]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved CSV -> {csv_name}")


# ============================================================
# PLOT sigma(z)
# ============================================================
z = [r["z"] for r in results]
sx = [r["sigma_x"] for r in results]
sy = [r["sigma_y"] for r in results]
sa = [r["sigma_avg"] for r in results]

plt.figure(figsize=(10, 5))
plt.plot(z, sx, label="sigma_x")
plt.plot(z, sy, label="sigma_y")
plt.plot(z, sa, label="sigma_avg")

plt.xlabel("z")
plt.ylabel("Beam width σ")
plt.title(f"Beam evolution ({mode})")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# OPTIONAL: Beam propagation fitting
# ============================================================
if mode == "beam_fit" and len(results) > 3:
    try:
        popt, _ = curve_fit(
            beam_model,
            z,
            sa,
            p0=[min(sa), np.mean(z), (max(z)-min(z))/2],
            maxfev=20000
        )

        sigma0, z0, zR = popt

        print("\n=== Beam propagation fit ===")
        print(f"sigma0 = {sigma0:.4f}")
        print(f"z0     = {z0:.4f}")
        print(f"zR     = {zR:.4f}")

        z_fit = np.linspace(min(z), max(z), 200)
        y_fit = beam_model(z_fit, *popt)

        plt.figure()
        plt.plot(z, sa, "o", label="data")
        plt.plot(z_fit, y_fit, "-", label="fit")

        plt.xlabel("z")
        plt.ylabel("sigma")
        plt.title("Gaussian beam propagation fit")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("Fit failed:", e)