import math
import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import zoom
import cv2
from scipy.ndimage import shift
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== load image ======
def load_intensity_image(path):
    img = imageio.imread(path)
    if img.ndim == 3:  # RGB
        img = img.mean(axis=2)
    img = img.astype(np.float64)
    # print(np.max(img))
    return img / 255

def resolve_dark_current_eff_drift(dark_current_eff_drift, target_shape=None):
    if isinstance(dark_current_eff_drift, str):
        drift = np.loadtxt(dark_current_eff_drift, dtype=np.float64) / 255.0
        if target_shape is not None and drift.shape != target_shape:
            raise ValueError(
                f"dark_current_eff_drift shape mismatch: got {drift.shape}, expected {target_shape}"
            )
        return drift

    if dark_current_eff_drift is None:
        return 0.0

    if np.isscalar(dark_current_eff_drift):
        return float(dark_current_eff_drift)

    drift = np.asarray(dark_current_eff_drift, dtype=np.float64)
    if target_shape is not None and drift.shape != target_shape:
        raise ValueError(
            f"dark_current_eff_drift shape mismatch: got {drift.shape}, expected {target_shape}"
        )
    return drift

def has_dark_current_eff_drift(dark_current_eff_drift, eps=1e-12):
    if dark_current_eff_drift is None:
        return False

    if isinstance(dark_current_eff_drift, str):
        return True

    if np.isscalar(dark_current_eff_drift):
        return abs(float(dark_current_eff_drift)) > eps

    drift = np.asarray(dark_current_eff_drift, dtype=np.float64)
    return bool(np.any(np.abs(drift) > eps))

def subtract_dark_current_eff_drift(img, dark_current_eff_drift=0.0, clip_min=0.0):
    drift = resolve_dark_current_eff_drift(
        dark_current_eff_drift,
        target_shape=np.asarray(img).shape
    )
    img = np.asarray(img, dtype=np.float64) - drift
    if clip_min is not None:
        img = np.clip(img, a_min=clip_min, a_max=None)
    return img

# ====== crop image center ======
def center_crop(img, crop_size):
    if crop_size is None:
        return img

    h, w = img.shape
    if isinstance(crop_size, int):
        crop_h = crop_size
        crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)

    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    return img[y0:y0+crop_h, x0:x0+crop_w]

# ====== find true image center ======
def find_center_robust(img, blur_ksize=11, threshold_ratio=0.3):
    """
    img: 2D numpy array (intensity)
    return: (cy, cx)
    """
    img = img.astype(np.float64)

    # Gaussian blur 
    img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    fig, axs = plt.subplots(1,2, figsize=(10,5))

    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(img_blur, cmap="gray")
    axs[1].set_title("Gaussian Blur")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

    # 2 threshold 
    threshold = threshold_ratio * np.max(img_blur)
    mask = img_blur > threshold

    if np.sum(mask) == 0:
        raise ValueError(f"Threshold ...")

    # 3 find center
    y, x = np.indices(img.shape)

    weighted = img_blur * mask

    cx = np.sum(x * weighted) / np.sum(weighted)
    cy = np.sum(y * weighted) / np.sum(weighted)
    print(f"Y replacement:{cy -  img.shape[0]//2:.1f}, X replacement:{cx - img.shape[1]//2:.1f}")
    return cy, cx

# ====== move image to center ======
def center_image(img):
    ny, nx = img.shape

    cy, cx = find_center_robust(img)

    # ????豲??
    target_y = ny / 2
    target_x = nx / 2

    shift_y = target_y - cy
    shift_x = target_x - cx

    img_shifted = shift(img, shift=(shift_y, shift_x), order=3)

    return img_shifted, (shift_y, shift_x)

# ====== ??? ======
def align_img_list(Img_list):
    """
    Img_list: list of 2D numpy arrays
    return: aligned list
    """

    aligned_list = []
    shifts = []

    for img in Img_list:
        img_aligned, s = center_image(img)
        aligned_list.append(img_aligned)
        shifts.append(s)

    return aligned_list, shifts

# ====== upsample intensity image ======
def upscale_intensity(img, scale_factor=2, order=3):
    """
    img: 2D numpy array
    scale_factor: upscale factor
    order: interpolation order
            1 = bilinear
            3 = bicubic 
    """
    return zoom(img, zoom=scale_factor, order=order)

def spatial_bin_intensity(img, binning_factor=1):
    if binning_factor is None:
        return np.asarray(img, dtype=np.float64)

    binning_factor = int(binning_factor)
    if binning_factor <= 1:
        return np.asarray(img, dtype=np.float64)

    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    binned_h = h // binning_factor
    binned_w = w // binning_factor

    if binned_h <= 0 or binned_w <= 0:
        raise ValueError(
            f"binning_factor={binning_factor} is too large for image shape {img.shape}"
        )

    trimmed = img[:binned_h * binning_factor, :binned_w * binning_factor]
    return trimmed.reshape(
        binned_h, binning_factor, binned_w, binning_factor
    ).mean(axis=(1, 3))

# ====== angular spectrum propagation simulation ======
def angular_spectrum_propagate(U, wavelength, dx, z, include_evanescent=False):
    ny, nx = U.shape
    fx = torch.fft.fftfreq(nx, d=dx, device=U.device)
    fy = torch.fft.fftfreq(ny, d=dx, device=U.device)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    k = 2 * np.pi / wavelength
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY

    argument = k**2 - kx**2 - ky**2
    kz_real = torch.sqrt(torch.clamp(argument, min=0.0))

    if include_evanescent:
        kz_imag = torch.sqrt(torch.clamp(-argument, min=0.0))
        H = torch.exp(1j * kz_real * z) * torch.exp(-kz_imag * abs(z))
    else:
        H = torch.exp(1j * kz_real * z)

    F = torch.fft.fft2(U)
    U_prop = torch.fft.ifft2(F * H)

    return U_prop

# ====== angular spectrum propagation simulation numpy version ======
def angular_spectrum_propagate_numpy(U, wavelength, dx, z, include_evanescent=False):
    ny, nx = U.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")

    k = 2 * np.pi / wavelength
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY

    argument = k**2 - kx**2 - ky**2
    kz_real = np.sqrt(np.maximum(argument, 0.0))

    if include_evanescent:
        kz_imag = np.sqrt(np.maximum(-argument, 0.0))
        H = np.exp(1j * kz_real * z) * np.exp(-kz_imag * abs(z))
    else:
        H = np.exp(1j * kz_real * z)

    F = np.fft.fft2(U)
    U_prop = np.fft.ifft2(F * H)

    return U_prop

# ====== phase smoothness loss calculation ======
def phase_smoothness_loss(U, grad_weight=1.0, curv_weight=0.2):
    # gradient term
    dx = torch.angle(U[:, 1:] * torch.conj(U[:, :-1]))
    dy = torch.angle(U[1:, :] * torch.conj(U[:-1, :]))
    loss_grad = torch.mean(torch.abs(dx**2)) + torch.mean(torch.abs(dy**2))

    # curvature term
    dxx = dx[:, 1:] - dx[:, :-1]
    dyy = dy[1:, :] - dy[:-1, :]

    loss_curv = torch.mean(dxx**2) + torch.mean(dyy**2)
    return grad_weight * loss_grad + curv_weight * loss_curv

    # return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))  # L1 loss
    # return torch.mean(dx**2) + torch.mean(dy**2)  # L2 loss

    # phase = torch.angle(U)
    # dx = phase[:, 1:] - phase[:, :-1]
    # dy = phase[1:, :] - phase[:-1, :]
    # return torch.mean(dx**2) + torch.mean(dy**2)

# ====== phase initialization ======
def build_initial_phase(ny, nx, device, phase_init_mode="random", radial_phase_positive=True,
        stripe_phase_num_y=6, stripe_phase_num_x=7,
        phase_init_center=None, phase_init_scale=math.pi):

    if phase_init_mode == "zero":
        return torch.zeros((ny, nx), dtype=torch.float32, device=device)

    if phase_init_mode == "stripe":
        y = torch.arange(ny, dtype=torch.float32, device=device)
        x = torch.arange(nx, dtype=torch.float32, device=device)
        Y, X = torch.meshgrid(y, x, indexing="ij")

        # Use two evenly spaced smooth stripe fields and multiply them so the maxima
        # appear as a grid at the stripe intersections instead of adding on top of each other.
        stripe_y = 0.5 * (1.0 + torch.cos(2 * math.pi * stripe_phase_num_y * Y / max(ny, 1)))
        stripe_x = 0.5 * (1.0 + torch.cos(2 * math.pi * stripe_phase_num_x * X / max(nx, 1)))

        # Grid-like phase peaks at the crossings, then smoothly decays away in both directions.
        phase = phase_init_scale * (stripe_y * stripe_x)
        return phase

    if phase_init_mode == "radial":
        if phase_init_center is None:
            cy = 0.5 * (ny - 1)
            cx = 0.5 * (nx - 1)
        else:
            cy, cx = phase_init_center

        y = torch.arange(ny, dtype=torch.float32, device=device)
        x = torch.arange(nx, dtype=torch.float32, device=device)
        Y, X = torch.meshgrid(y, x, indexing="ij")

        radius = torch.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        radius_max = torch.max(radius).clamp_min(1e-12)
        phase = phase_init_scale * (radius / radius_max)

        if not radial_phase_positive:
            phase = -phase

        return phase

    if phase_init_mode != "random":
        raise ValueError(f"Unsupported phase_init_mode: {phase_init_mode}")

    return math.pi * torch.rand((ny, nx), device=device) - 0.5 * math.pi

# ====== training loss history plot ======
def plot_loss_history(loss_history, keys_to_plot=None, save_path=None, show_plot=True):
    iterations = np.asarray(loss_history["iter"])
    eps = 1e-12

    if keys_to_plot is None:
        keys_to_plot = ["total", "data", "corr", "ssim", "phase", "z", "monotonic"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for key in keys_to_plot:
        if key not in loss_history:
            continue
        ax.semilogy(iterations, np.clip(loss_history[key], eps, None), label=key)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss History")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)

def compute_pearson_corr_np(img_a, img_b, eps=1e-12):
    a = np.asarray(img_a, dtype=np.float64).reshape(-1)
    b = np.asarray(img_b, dtype=np.float64).reshape(-1)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2)) + eps
    return float(np.sum(a * b) / denom)

def compute_masked_ssim(img_a, img_b, mask=None, data_range=1.0):
    img_a = np.asarray(img_a, dtype=np.float64)
    img_b = np.asarray(img_b, dtype=np.float64)
    if mask is None:
        return float(ssim(img_a, img_b, data_range=data_range))

    if not np.any(mask):
        return np.nan

    a = np.where(mask, img_a, 0.0)
    b = np.where(mask, img_b, 0.0)
    return float(ssim(a, b, data_range=data_range))

def evaluate_basic_metrics(pred, target, mask_threshold_ratio=0.1, eps=1e-12):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    mask = target > (mask_threshold_ratio * np.max(target) + eps)

    metrics = {
        "ssim": float(ssim(pred, target, data_range=1.0)),
        "corr": compute_pearson_corr_np(pred, target),
        "mae": float(np.mean(np.abs(pred - target))),
        "energy_err": float(np.abs(np.sum(pred) - np.sum(target)) / (np.sum(target) + eps)),
        "mask_ratio": float(np.mean(mask)),
        "ssim_masked": compute_masked_ssim(pred, target, mask=mask, data_range=1.0),
    }

    if np.any(mask):
        metrics["mae_masked"] = float(np.mean(np.abs(pred[mask] - target[mask])))
        metrics["corr_masked"] = compute_pearson_corr_np(pred[mask], target[mask])
    else:
        metrics["mae_masked"] = np.nan
        metrics["corr_masked"] = np.nan

    return metrics

def evaluate_intensity_metrics(pred, target, mask_threshold_ratio=0.1, crop_size=(256, 256), eps=1e-12):
    metrics = evaluate_basic_metrics(pred, target, mask_threshold_ratio=mask_threshold_ratio, eps=eps)

    pred_crop = center_crop(np.asarray(pred, dtype=np.float64), crop_size)
    target_crop = center_crop(np.asarray(target, dtype=np.float64), crop_size)
    crop_metrics = evaluate_basic_metrics(pred_crop, target_crop, mask_threshold_ratio=mask_threshold_ratio, eps=eps)

    metrics.update({
        "ssim_center256": crop_metrics["ssim"],
        "corr_center256": crop_metrics["corr"],
        "mae_center256": crop_metrics["mae"],
        "energy_err_center256": crop_metrics["energy_err"],
        "ssim_masked_center256": crop_metrics["ssim_masked"],
        "corr_masked_center256": crop_metrics["corr_masked"],
        "mae_masked_center256": crop_metrics["mae_masked"],
        "mask_ratio_center256": crop_metrics["mask_ratio"],
    })

    return metrics

# ====== data loss calculation ======
def compute_intensity_loss(I_pred, I_meas, mode="l1", weight_power=1.0, eps=1e-12):
    if mode == "l1":
        return torch.mean(torch.abs(I_pred - I_meas))

    if mode == "weighted_l1":
        I_pred_w = I_pred
        I_meas_w = I_meas
        if weight_power != 1.0:
            I_pred_w = I_pred ** weight_power
            I_meas_w = I_meas ** weight_power
        return torch.mean(torch.abs(I_pred_w - I_meas_w))
    
    if mode == "weighted_l2":
        I_pred_w = I_pred
        I_meas_w = I_meas
        if weight_power != 1.0:
            I_pred_w = I_pred ** weight_power
            I_meas_w = I_meas ** weight_power
        return torch.mean(torch.abs(I_pred_w - I_meas_w) ** 2)

    if mode == "normalized_l1":
        I_pred_n = I_pred / (torch.sum(I_pred) + eps)
        I_meas_n = I_meas / (torch.sum(I_meas) + eps)
        return torch.mean(torch.abs(I_pred_n - I_meas_n))

    if mode == "weighted_normalized_l1":
        I_pred_n = I_pred / (torch.sum(I_pred) + eps)
        I_meas_n = I_meas / (torch.sum(I_meas) + eps)

        # Put more emphasis on informative bright regions instead of large dark background.
        weight = I_meas_n / (torch.mean(I_meas_n) + eps)
        weight = torch.clamp(weight, min=1e-3)
        if weight_power != 1.0:
            weight = weight ** weight_power
        weight = weight / (torch.mean(weight) + eps)
        return torch.mean(weight * torch.abs(I_pred_n - I_meas_n))

    raise ValueError(f"Unsupported data_loss_mode: {mode}")

def compute_correlation_loss(I_pred, I_meas, eps=1e-12):
    # Use normalized intensity before correlation so the term focuses more on pattern similarity.
    I_pred_n = I_pred / (torch.sum(I_pred) + eps)
    I_meas_n = I_meas / (torch.sum(I_meas) + eps)

    pred_flat = I_pred_n.reshape(-1)
    meas_flat = I_meas_n.reshape(-1)

    pred_centered = pred_flat - torch.mean(pred_flat)
    meas_centered = meas_flat - torch.mean(meas_flat)

    numerator = torch.sum(pred_centered * meas_centered)
    denominator = torch.sqrt(
        torch.sum(pred_centered ** 2) * torch.sum(meas_centered ** 2) + eps
    )
    corr = numerator / denominator

    # Correlation close to 1 is good, so convert it to a minimization loss.
    return 1.0 - corr

def compute_ssim_loss(I_pred, I_meas, window_size=7, c1=1e-4, c2=9e-4, eps=1e-12):
    # Normalize first so SSIM focuses more on structure than absolute power scale.
    I_pred_n = I_pred / (torch.sum(I_pred) + eps)
    I_meas_n = I_meas / (torch.sum(I_meas) + eps)

    pred = I_pred_n[None, None, :, :]
    meas = I_meas_n[None, None, :, :]

    mu_pred = F.avg_pool2d(pred, kernel_size=window_size, stride=1, padding=window_size // 2)
    mu_meas = F.avg_pool2d(meas, kernel_size=window_size, stride=1, padding=window_size // 2)

    mu_pred_sq = mu_pred ** 2
    mu_meas_sq = mu_meas ** 2
    mu_pred_meas = mu_pred * mu_meas

    sigma_pred_sq = F.avg_pool2d(pred ** 2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu_pred_sq
    sigma_meas_sq = F.avg_pool2d(meas ** 2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu_meas_sq
    sigma_pred_meas = F.avg_pool2d(pred * meas, kernel_size=window_size, stride=1, padding=window_size // 2) - mu_pred_meas

    ssim_map = ((2 * mu_pred_meas + c1) * (2 * sigma_pred_meas + c2)) / (
        (mu_pred_sq + mu_meas_sq + c1) * (sigma_pred_sq + sigma_meas_sq + c2) + eps
    )

    return 1.0 - torch.mean(ssim_map)

# ====== amplitude field helper ======
def build_amplitude_field(amplitude_param, target_shape):
    target_h, target_w = target_shape
    if amplitude_param.shape == (target_h, target_w):
        return amplitude_param

    return F.interpolate(
        amplitude_param[None, None, :, :],
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False
    )[0, 0]

# ====== grid shape to target shape
def build_phase_field(phase_param, target_shape):
    target_h, target_w = target_shape
    if phase_param.shape == (target_h, target_w):
        return phase_param

    return F.interpolate(
        phase_param[None, None, :, :],
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False
    )[0, 0]

def resolve_weight_schedule(weight_spec, progress):
    if isinstance(weight_spec, (tuple, list)) and len(weight_spec) == 2:
        start, end = weight_spec
        return start + (end - start) * progress
    return weight_spec

def resolve_curriculum_planes(progress, n_planes, device, curriculum_plan=None):
    if curriculum_plan is None:
        if progress < 1/3:
            planes = list(range(max(1, n_planes - 3), n_planes + 1))
            stage_label = f"last_{len(planes)}"
        elif progress < 2/3:
            planes = list(range(max(1, n_planes - 6), n_planes + 1))
            stage_label = f"last_{len(planes)}"
        else:
            planes = list(range(1, n_planes + 1))
            stage_label = "all"
    else:
        selected_stage = curriculum_plan[-1]
        for stage in curriculum_plan:
            if progress < stage["until"]:
                selected_stage = stage
                break

        if selected_stage["planes"] is None:
            planes = list(range(1, n_planes + 1))
            stage_label = selected_stage.get("label", "all")
        else:
            planes = [int(p) for p in selected_stage["planes"]]
            stage_label = selected_stage.get("label", str(planes))

    return torch.tensor(planes, dtype=torch.long, device=device), stage_label

# ====== save results as png ======
def save_png(img, title):
        save_path = f"{output_dir}/{title}.png"
        if "phase" in title.lower():
            phase_display = np.angle(np.exp(1j * img))
            fig, ax = plt.subplots(figsize=(5.8, 5))
            im = ax.imshow(phase_display, cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            return

        img = img * 255
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        Image.fromarray(img).save(save_path)

def make_offset_corrected_title(base_title, dark_current_eff_drift):
    if not has_dark_current_eff_drift(dark_current_eff_drift):
        return base_title
    return f"{base_title}_offset_subtracted"

def pooled_adjacent_violators(y, weights=None):
    y = np.asarray(y, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(y, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    block_values = []
    block_weights = []
    block_starts = []
    block_ends = []

    for idx, (value, weight) in enumerate(zip(y, weights)):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_starts.append(idx)
        block_ends.append(idx + 1)

        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]
            ) / max(merged_weight, 1e-12)
            merged_start = block_starts[-2]
            merged_end = block_ends[-1]

            block_values = block_values[:-2] + [merged_value]
            block_weights = block_weights[:-2] + [merged_weight]
            block_starts = block_starts[:-2] + [merged_start]
            block_ends = block_ends[:-2] + [merged_end]

    fitted = np.empty_like(y, dtype=np.float64)
    for value, start, end in zip(block_values, block_starts, block_ends):
        fitted[start:end] = value
    return fitted

def estimate_monotone_gain_curve_from_image_pairs(predicted_imgs, measured_imgs, num_bins=512):
    pred_flat = np.concatenate([
        np.asarray(img, dtype=np.float64).reshape(-1) for img in predicted_imgs
    ])
    meas_flat = np.concatenate([
        np.asarray(img, dtype=np.float64).reshape(-1) for img in measured_imgs
    ])

    valid = np.isfinite(pred_flat) & np.isfinite(meas_flat)
    pred_flat = pred_flat[valid]
    meas_flat = meas_flat[valid]
    if pred_flat.size == 0:
        raise ValueError("No valid pixels available for monotone gain curve estimation.")

    order = np.argsort(pred_flat, kind="mergesort")
    pred_sorted = pred_flat[order]
    meas_sorted = meas_flat[order]

    num_bins = max(8, min(int(num_bins), pred_sorted.size))
    bin_edges = np.linspace(0, pred_sorted.size, num_bins + 1, dtype=int)

    x_bins = []
    y_bins = []
    w_bins = []
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if end <= start:
            continue
        pred_chunk = pred_sorted[start:end]
        meas_chunk = meas_sorted[start:end]
        x_bins.append(float(np.mean(pred_chunk)))
        y_bins.append(float(np.mean(meas_chunk)))
        w_bins.append(float(end - start))

    x_bins = np.asarray(x_bins, dtype=np.float64)
    y_bins = np.asarray(y_bins, dtype=np.float64)
    w_bins = np.asarray(w_bins, dtype=np.float64)
    y_iso = pooled_adjacent_violators(y_bins, weights=w_bins)

    mapped_pred = np.interp(pred_flat, x_bins, y_iso, left=y_iso[0], right=y_iso[-1])
    mae_before = float(np.mean(np.abs(pred_flat - meas_flat)))
    mae_after = float(np.mean(np.abs(mapped_pred - meas_flat)))

    return {
        "x_knots": x_bins,
        "y_knots_raw": y_bins,
        "y_knots_iso": y_iso,
        "num_pixels": int(pred_flat.size),
        "num_bins": int(len(x_bins)),
        "mae_before": mae_before,
        "mae_after": mae_after,
    }

def save_monotone_gain_curve(curve_result, output_dir):
    curve_csv_path = os.path.join(output_dir, "monotone_gain_curve.csv")
    with open(curve_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted_linear_intensity", "measured_mean_raw", "measured_monotone_fit"])
        for x, y_raw, y_fit in zip(
            curve_result["x_knots"],
            curve_result["y_knots_raw"],
            curve_result["y_knots_iso"]
        ):
            writer.writerow([x, y_raw, y_fit])

    summary_csv_path = os.path.join(output_dir, "monotone_gain_curve_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["num_pixels", curve_result["num_pixels"]])
        writer.writerow(["num_bins", curve_result["num_bins"]])
        writer.writerow(["mae_before", curve_result["mae_before"]])
        writer.writerow(["mae_after", curve_result["mae_after"]])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve_result["x_knots"], curve_result["y_knots_raw"], color="0.75", linewidth=1.2, label="Binned mean")
    ax.plot(curve_result["x_knots"], curve_result["y_knots_iso"], color="tab:red", linewidth=2.0, label="Monotone fit")
    ax.set_xlabel("Predicted linear intensity")
    ax.set_ylabel("Measured pixel value")
    ax.set_title("Estimated Monotone Gain Curve")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "monotone_gain_curve.png"), dpi=200)
    plt.close(fig)

def inverse_softplus(value, eps=1e-12):
    value = max(float(value), eps)
    return math.log(math.exp(value) - 1.0)

def build_learned_monotone_gain_curve(num_knots, initial_input_scale, device):
    if num_knots < 2:
        raise ValueError("num_knots must be at least 2 for monotone gain learning.")

    init_increment = inverse_softplus(1.0)
    gain_increment_param = torch.full(
        (num_knots - 1,),
        init_increment,
        dtype=torch.float32,
        device=device
    )
    gain_increment_param.requires_grad_(True)

    gain_log_input_scale = torch.tensor(
        math.log(max(float(initial_input_scale), 1e-6)),
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )
    return gain_increment_param, gain_log_input_scale

def get_learned_monotone_gain_knots_torch(gain_increment_param, gain_log_input_scale, dtype, device, eps=1e-12):
    increments = F.softplus(gain_increment_param.to(dtype=dtype)) + 1e-6
    knots_y = torch.cat([
        torch.zeros(1, dtype=dtype, device=device),
        torch.cumsum(increments, dim=0)
    ])
    knots_y = knots_y / knots_y[-1].clamp_min(eps)
    input_scale = torch.exp(gain_log_input_scale).to(dtype=dtype, device=device)
    knots_x = torch.linspace(0.0, 1.0, steps=knots_y.numel(), dtype=dtype, device=device) * input_scale
    return knots_x, knots_y, input_scale

def export_learned_monotone_gain_curve(gain_increment_param, gain_log_input_scale):
    with torch.no_grad():
        knots_x, knots_y, input_scale = get_learned_monotone_gain_knots_torch(
            gain_increment_param,
            gain_log_input_scale,
            dtype=gain_increment_param.dtype,
            device=gain_increment_param.device
        )

    return {
        "x_knots": knots_x.detach().cpu().numpy(),
        "y_knots": knots_y.detach().cpu().numpy(),
        "input_scale": float(input_scale.item()),
        "num_knots": int(knots_y.numel()),
    }

def apply_learned_monotone_gain_torch(I_linear, gain_increment_param, gain_log_input_scale, eps=1e-12):
    _, knots_y, input_scale = get_learned_monotone_gain_knots_torch(
        gain_increment_param,
        gain_log_input_scale,
        dtype=I_linear.dtype,
        device=I_linear.device,
        eps=eps
    )
    x_norm = torch.clamp(I_linear / input_scale.clamp_min(eps), 0.0, 1.0)
    knots_x = torch.linspace(0.0, 1.0, steps=knots_y.numel(), dtype=I_linear.dtype, device=I_linear.device)

    flat_x = x_norm.reshape(-1)
    segment_idx = torch.bucketize(flat_x, knots_x[1:-1])
    segment_idx = torch.clamp(segment_idx, 0, knots_y.numel() - 2)

    x0 = knots_x[segment_idx]
    x1 = knots_x[segment_idx + 1]
    y0 = knots_y[segment_idx]
    y1 = knots_y[segment_idx + 1]
    t = (flat_x - x0) / (x1 - x0).clamp_min(eps)
    flat_y = y0 + t * (y1 - y0)
    return flat_y.reshape_as(I_linear)

def apply_learned_monotone_gain_numpy(I_linear, gain_curve_result):
    I_linear = np.asarray(I_linear, dtype=np.float64)
    return np.interp(
        I_linear,
        gain_curve_result["x_knots"],
        gain_curve_result["y_knots"],
        left=gain_curve_result["y_knots"][0],
        right=gain_curve_result["y_knots"][-1]
    )

def apply_inverse_learned_monotone_gain_numpy(I_measured, gain_curve_result):
    I_measured = np.asarray(I_measured, dtype=np.float64)
    return np.interp(
        I_measured,
        gain_curve_result["y_knots"],
        gain_curve_result["x_knots"],
        left=gain_curve_result["x_knots"][0],
        right=gain_curve_result["x_knots"][-1]
    )

def make_gain_corrected_title(base_title, dark_current_eff_drift, learned_gain_curve=None):
    has_drift = has_dark_current_eff_drift(dark_current_eff_drift)
    has_gain = learned_gain_curve is not None
    if has_drift and has_gain:
        return f"{base_title}_drift_gain_corrected"
    if has_drift:
        return f"{base_title}_offset_subtracted"
    if has_gain:
        return f"{base_title}_gain_corrected"
    return base_title

def save_learned_monotone_gain_curve(gain_curve_result, output_dir):
    curve_csv_path = os.path.join(output_dir, "learned_monotone_gain_curve.csv")
    with open(curve_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted_linear_intensity", "measured_domain_value"])
        for x, y in zip(gain_curve_result["x_knots"], gain_curve_result["y_knots"]):
            writer.writerow([x, y])

    summary_csv_path = os.path.join(output_dir, "learned_monotone_gain_curve_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["input_scale", gain_curve_result["input_scale"]])
        writer.writerow(["num_knots", gain_curve_result["num_knots"]])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gain_curve_result["x_knots"], gain_curve_result["y_knots"], color="tab:red", linewidth=2.0)
    ax.set_xlabel("Predicted linear intensity")
    ax.set_ylabel("Measured-domain pixel value")
    ax.set_title("Learned Monotone Gain Curve")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "learned_monotone_gain_curve.png"), dpi=200)
    plt.close(fig)

def load_preprocessed_images(img_paths, crop_size=None, scale=1, dark_current_eff_drift=0.0,
        spatial_binning_factor=1):
    img_list = []
    for path in img_paths:
        img = load_intensity_image(path)
        img = subtract_dark_current_eff_drift(img, dark_current_eff_drift=dark_current_eff_drift)
        img = center_crop(img, crop_size)
        img = spatial_bin_intensity(img, binning_factor=spatial_binning_factor)
        if scale is not None:
            img = upscale_intensity(img, scale_factor=scale)
        img_list.append(img)
    return img_list

def build_plane_evaluation_entries(U_recon, reference_img, reference_z, measured_imgs, measured_z_list,
        wavelength, dx, include_evanescent=False, split_label="train", learned_gain_curve=None):
    entries = []
    U_zero = np.sqrt(reference_img) * np.exp(1j * 0)

    for plane_idx, (z_value, measured_img) in enumerate(zip(measured_z_list, measured_imgs), start=1):
        U_generated = angular_spectrum_propagate_numpy(
            U_recon, wavelength, dx, z_value - reference_z,
            include_evanescent=include_evanescent
        )
        I_generated_linear = np.abs(U_generated) ** 2

        U_zero_plane = angular_spectrum_propagate_numpy(
            U_zero, wavelength, dx, z_value - reference_z,
            include_evanescent=include_evanescent
        )
        I_zero_linear = np.abs(U_zero_plane) ** 2

        if learned_gain_curve is not None:
            I_generated = apply_learned_monotone_gain_numpy(I_generated_linear, learned_gain_curve)
            I_zero = apply_learned_monotone_gain_numpy(I_zero_linear, learned_gain_curve)
            I_measured_linear = apply_inverse_learned_monotone_gain_numpy(measured_img, learned_gain_curve)
        else:
            I_generated = I_generated_linear
            I_zero = I_zero_linear
            I_measured_linear = measured_img

        entries.append({
            "split": split_label,
            "plane": plane_idx,
            "z_m": z_value,
            "i_gen": I_generated,
            "i_zero": I_zero,
            "i_gen_linear": I_generated_linear,
            "i_zero_linear": I_zero_linear,
            "i_meas": measured_img,
            "i_meas_linear": I_measured_linear,
        })

    return entries

# ==========================
# Multi-plane optimization version 2: 
# ==========================
def multi_plane_gradient(Img_list, z_list, wavelength, dx, n_iter=500, lr=5e-3,
        lr_phase=None,
        lr_z=None,
        lr_amp=None,
        lr_gain=None,
        use_lr_scheduler=False,
        lr_decay_gamma=0.5,
        lr_plateau_patience=1000,
        lr_plateau_threshold=1e-4,
        lr_min=1e-7,
        phase_grid_shape=None,
        amplitude_grid_shape=None,
        train_amplitude=False,
        amplitude_update_limit=0.2,
        amplitude_weight=1e-3,
        learn_monotone_gain=False,
        monotone_gain_num_knots=16,
        monotone_gain_weight=0.0,
        monotone_gain_smoothness_weight=0.0,
        include_reference_plane_in_loss=True,
        reference_plane_weight=1.0,
        train_delta_z=True,
        z_param_mode="plane_shift",
        data_weight=1.0,
        data_loss_mode="l1",
        data_loss_weight_power=1.0,
        correlation_loss_weight=0.0,
        ssim_loss_weight=0.0,
        phase_smoothness_weight=1e-3,
        phase_grad_weight=1.0,
        phase_curv_weight=0.2,
        z_weight=1e3,
        monotonic_weight=1e5,
        max_z_update=2e-4,
        include_evanescent=False,
        phase_init_mode="random",
        stripe_phase_num_y=8,
        stripe_phase_num_x=8,
        radial_phase_positive=True,
        phase_init_center=None,
        phase_init_scale=math.pi,
        curriculum=False,
        curriculum_plan=None,
        stochastic=False,
        stoch_k=0.75,
        device='cpu'
):

    ny, nx = Img_list[0].shape

    # amplitude initialization
    A0_nominal = torch.sqrt(Img_list[0])

    # phase initialization
    phase_init = build_initial_phase(
        ny,
        nx,
        device=device,
        phase_init_mode=phase_init_mode,
        stripe_phase_num_y=stripe_phase_num_y,
        stripe_phase_num_x=stripe_phase_num_x,
        radial_phase_positive=radial_phase_positive,
        phase_init_center=phase_init_center,
        phase_init_scale=phase_init_scale
    )

    if phase_grid_shape is None:
        phase_param = phase_init.detach().clone()
    else:
        grid_h, grid_w = phase_grid_shape
        phase_param = F.interpolate(
            phase_init[None, None, :, :],
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False
        )[0, 0]
    phase_param.requires_grad_(True)

    if lr_phase is None:
        lr_phase = lr
    if lr_z is None:
        lr_z = lr
    if lr_amp is None:
        lr_amp = lr
    if lr_gain is None:
        lr_gain = lr

    if amplitude_grid_shape is None:
        amplitude_param = torch.zeros((ny, nx), dtype=torch.float32, device=device)
    else:
        amp_h, amp_w = amplitude_grid_shape
        amplitude_param = torch.zeros((amp_h, amp_w), dtype=torch.float32, device=device)
    amplitude_param.requires_grad_(train_amplitude)

    z_nominal = torch.tensor(z_list, dtype=torch.float32, device=device)
    nominal_spacing = z_nominal[1:] - z_nominal[:-1]
    min_gap = 0.5 * torch.min(nominal_spacing).item()
    initial_gain_input_scale = max(float(torch.max(Img_list[0]).item()), 1e-3)

    if learn_monotone_gain:
        gain_increment_param, gain_log_input_scale = build_learned_monotone_gain_curve(
            monotone_gain_num_knots,
            initial_input_scale=initial_gain_input_scale,
            device=device
        )
    else:
        gain_increment_param = None
        gain_log_input_scale = None

    if train_delta_z:
        if z_param_mode not in ["plane_shift", "spacing"]:
            raise ValueError(f"Unsupported z_param_mode: {z_param_mode}")
        z_param = torch.zeros(len(z_list) - 1, dtype=torch.float32, device=device, requires_grad=True)
        param_groups = [
            {"params": [phase_param], "lr": lr_phase},
            {"params": [z_param], "lr": lr_z},
        ]
    else:
        z_param = None
        param_groups = [
            {"params": [phase_param], "lr": lr_phase},
        ]

    if train_amplitude:
        param_groups.append({"params": [amplitude_param], "lr": lr_amp})
    if learn_monotone_gain:
        param_groups.append({
            "params": [gain_increment_param, gain_log_input_scale],
            "lr": lr_gain
        })

    optimizer = torch.optim.Adam(param_groups)

    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_decay_gamma,
            patience=lr_plateau_patience,
            threshold=lr_plateau_threshold,
            threshold_mode="rel",
            min_lr=lr_min
        )
    else:
        scheduler = None

    n_planes = len(z_list) - 1
    count = torch.zeros(n_planes + 1)
    loss_history = {
        "iter": [],
        "total": [],
        "data": [],
        "data_raw": [],
        "corr": [],
        "corr_raw": [],
        "ssim": [],
        "ssim_raw": [],
        "phase": [],
        "amplitude": [],
        "gain_reg": [],
        "gain_smooth": [],
        "z": [],
        "monotonic": [],
    }

    for it in range(n_iter):
        optimizer.zero_grad()

        if train_delta_z:
            z_update_limited = max_z_update * torch.tanh(z_param / max_z_update)
            if z_param_mode == "plane_shift":
                delta_z_limited = z_update_limited
                delta_z_full = torch.cat([
                    torch.zeros(1, dtype=torch.float32, device=device),
                    delta_z_limited
                ])
                z_current = z_nominal + delta_z_full
                spacing_update_limited = delta_z_limited[1:] - delta_z_limited[:-1]
            else:
                spacing_update_limited = z_update_limited
                spacing_current = nominal_spacing + spacing_update_limited
                z_current = torch.cat([
                    z_nominal[:1],
                    z_nominal[:1] + torch.cumsum(spacing_current, dim=0)
                ])
                delta_z_full = z_current - z_nominal
                delta_z_limited = delta_z_full[1:]
        else:
            delta_z_limited = torch.zeros(len(z_list) - 1, dtype=torch.float32, device=device)
            spacing_update_limited = torch.zeros(len(z_list) - 1, dtype=torch.float32, device=device)
            delta_z_full = torch.zeros(len(z_list), dtype=torch.float32, device=device)
            z_current = z_nominal

        phase = build_phase_field(phase_param, (ny, nx))
        if train_amplitude:
            amplitude_update = amplitude_update_limit * torch.tanh(
                build_amplitude_field(amplitude_param, (ny, nx)) / amplitude_update_limit
            )
            A0 = A0_nominal * (1.0 + amplitude_update)
            loss_amplitude = amplitude_weight * torch.mean(amplitude_update ** 2)
        else:
            amplitude_update = torch.zeros((ny, nx), dtype=torch.float32, device=device)
            A0 = A0_nominal
            loss_amplitude = torch.tensor(0.0, dtype=torch.float32, device=device)
        U0 = A0 * torch.exp(1j * phase)

        if learn_monotone_gain:
            gain_knots_x, gain_knots_y, gain_input_scale = get_learned_monotone_gain_knots_torch(
                gain_increment_param,
                gain_log_input_scale,
                dtype=torch.float32,
                device=device
            )
            identity_knots = gain_knots_x / gain_input_scale.clamp_min(1e-12)
            loss_gain_reg = monotone_gain_weight * torch.mean((gain_knots_y - identity_knots) ** 2)
            if gain_knots_y.numel() >= 3:
                second_diff = gain_knots_y[2:] - 2 * gain_knots_y[1:-1] + gain_knots_y[:-2]
                loss_gain_smooth = monotone_gain_smoothness_weight * torch.mean(second_diff ** 2)
            else:
                loss_gain_smooth = torch.tensor(0.0, dtype=torch.float32, device=device)
        else:
            loss_gain_reg = torch.tensor(0.0, dtype=torch.float32, device=device)
            loss_gain_smooth = torch.tensor(0.0, dtype=torch.float32, device=device)

        loss_data = 0
        loss_corr = 0
        loss_ssim = 0
        loss_norm = 0.0
        progress = it / max(n_iter - 1, 1)
        corr_weight = resolve_weight_schedule(correlation_loss_weight, progress)
        ssim_weight = resolve_weight_schedule(ssim_loss_weight, progress)
        phase_smoothness_weight = resolve_weight_schedule(phase_smoothness_weight, progress)
        z_weight = resolve_weight_schedule(z_weight, progress)
        monotonic_weight = resolve_weight_schedule(monotonic_weight, progress)

        if include_reference_plane_in_loss:
            I0_linear = torch.abs(U0) ** 2
            if learn_monotone_gain:
                I0_meas_domain = apply_learned_monotone_gain_torch(
                    I0_linear,
                    gain_increment_param,
                    gain_log_input_scale
                )
            else:
                I0_meas_domain = I0_linear

            ref_weight = float(reference_plane_weight)
            loss_data += ref_weight * compute_intensity_loss(
                I0_meas_domain,
                Img_list[0],
                mode=data_loss_mode,
                weight_power=data_loss_weight_power
            )
            loss_corr += ref_weight * compute_correlation_loss(I0_meas_domain, Img_list[0])
            loss_ssim += ref_weight * compute_ssim_loss(I0_meas_domain, Img_list[0])
            loss_norm += ref_weight

        stage_label = "all"
        if curriculum == True:
            available_planes, stage_label = resolve_curriculum_planes(
                progress,
                n_planes,
                device=device,
                curriculum_plan=curriculum_plan
            )
            if stochastic:
                num_select = max(1, int(len(available_planes) * stoch_k))
                perm = torch.randperm(len(available_planes), device=available_planes.device)[:num_select]
                indices = available_planes[perm]
            else:
                indices = available_planes

            for i in indices:
                count[i] += 1
        else:
            if stochastic:
                available_planes = torch.arange(1, n_planes + 1)
                num_select = max(1, int(len(available_planes) * stoch_k))
                perm = torch.randperm(len(available_planes), device=available_planes.device)[:num_select]
                indices = available_planes[perm]
            else:
                indices = torch.arange(1, n_planes + 1)

        for i in indices:
            U_prop = angular_spectrum_propagate(
                U0,
                wavelength,
                dx,
                z_current[i] - z_current[0],
                include_evanescent=include_evanescent
            )
            I_pred = torch.abs(U_prop) ** 2
            if learn_monotone_gain:
                I_pred_meas_domain = apply_learned_monotone_gain_torch(
                    I_pred,
                    gain_increment_param,
                    gain_log_input_scale
                )
            else:
                I_pred_meas_domain = I_pred
            # Old baseline:
            # loss_data += torch.mean(torch.abs(I_pred - Img_list[i]))
            loss_data += compute_intensity_loss(
                I_pred_meas_domain,
                Img_list[i],
                mode=data_loss_mode,
                weight_power=data_loss_weight_power
            )
            loss_corr += compute_correlation_loss(I_pred_meas_domain, Img_list[i])
            loss_ssim += compute_ssim_loss(I_pred_meas_domain, Img_list[i])
            loss_norm += 1.0

        loss_norm = max(loss_norm, 1.0)
        loss_data = loss_data / loss_norm
        loss_data_weighted = data_weight * loss_data
        loss_corr = loss_corr / loss_norm
        loss_ssim = loss_ssim / loss_norm
        loss_corr_weighted = corr_weight * loss_corr
        loss_ssim_weighted = ssim_weight * loss_ssim
        loss_phase = phase_smoothness_weight * phase_smoothness_loss(
            U0,
            grad_weight=phase_grad_weight,
            curv_weight=phase_curv_weight
        )
        if train_delta_z:
            if z_param_mode == "plane_shift":
                loss_z = z_weight * torch.mean(delta_z_limited ** 2)
            else:
                loss_z = z_weight * torch.mean(spacing_update_limited ** 2)
            spacing = z_current[1:] - z_current[:-1]
            loss_monotonic = monotonic_weight * torch.mean(torch.relu(min_gap - spacing) ** 2)
        else:
            loss_z = torch.tensor(0.0, dtype=torch.float32, device=device)
            loss_monotonic = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss = (
            loss_data_weighted
            + loss_corr_weighted
            + loss_ssim_weighted
            + loss_phase
            + loss_amplitude
            + loss_gain_reg
            + loss_gain_smooth
            + loss_z
            + loss_monotonic
        )

        loss_history["iter"].append(it)
        loss_history["total"].append(loss.item())
        loss_history["data"].append(loss_data_weighted.item())
        loss_history["data_raw"].append(loss_data.item())
        loss_history["corr"].append(loss_corr_weighted.item())
        loss_history["corr_raw"].append(loss_corr.item())
        loss_history["ssim"].append(loss_ssim_weighted.item())
        loss_history["ssim_raw"].append(loss_ssim.item())
        loss_history["phase"].append(loss_phase.item())
        loss_history["amplitude"].append(loss_amplitude.item())
        loss_history["gain_reg"].append(loss_gain_reg.item())
        loss_history["gain_smooth"].append(loss_gain_smooth.item())
        loss_history["z"].append(loss_z.item())
        loss_history["monotonic"].append(loss_monotonic.item())

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss_data_weighted.item())

        if it % 100 == 0:  # print loss terms
            print(
                f"Iter {it:4d} | "
                f"Loss {loss.item():.4e} | "
                f"Data {loss_data_weighted.item():.4e} | "
                f"Corr {loss_corr_weighted.item():.4e} | "
                f"SSIM {loss_ssim_weighted.item():.4e} | "
                f"Phase {loss_phase.item():.4e} | "
                f"Amp {loss_amplitude.item():.4e} | "
                f"GainReg {loss_gain_reg.item():.4e} | "
                f"GainSmooth {loss_gain_smooth.item():.4e} | "
                f"Z {loss_z.item():.4e} | "
                f"Mono {loss_monotonic.item():.4e} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e} | "
                f"Stage {stage_label} | "
            )
            # print(
            #     f"Data loss mode: {data_loss_mode} | "
            #     f"data_weight: {data_weight:.2e} "
            #     f"(raw {loss_data.item():.4e}) | "
            #     f"corr_weight: {corr_weight:.2e} "
            #     f"(raw {loss_corr.item():.4e}) | "
            #     f"ssim_weight: {ssim_weight:.2e} "
            #     f"(raw {loss_ssim.item():.4e}) | "
            #     f"amp_limit: {amplitude_update_limit:.2e} | "
            #     f"amp_weight: {amplitude_weight:.2e}"
            # )
            if curriculum == True:
                print(f"Planes: {indices}")
            if train_delta_z == True:
                if z_param_mode == "plane_shift":
                    print(
                        "Learned delta_z (mm):",
                        np.round(delta_z_full.detach().cpu().numpy() * 1e3, 2)
                    )
                else:
                    print(
                        "Learned delta_spacing (mm):",
                        np.round(spacing_update_limited.detach().cpu().numpy() * 1e3, 2)
                    )
                    print(
                        "Implied delta_z (mm):",
                        np.round(delta_z_full.detach().cpu().numpy() * 1e3, 2)
                    )

    phase_final = build_phase_field(phase_param, (ny, nx))
    U_final = (A0 * torch.exp(1j * phase_final)).detach().cpu().numpy()
    z_optimized = z_current.detach().cpu().numpy()
    delta_z_final = delta_z_full.detach().cpu().numpy()
    if learn_monotone_gain:
        learned_gain_curve = export_learned_monotone_gain_curve(
            gain_increment_param,
            gain_log_input_scale
        )
    else:
        learned_gain_curve = None
    if curriculum == True:
        print(count)
    return U_final, z_optimized, delta_z_final, loss_history, learned_gain_curve


# ====== main function ======
if __name__ == "__main__":
    # === training z distance list ===
    train_z_list = [
        0.340, 
        0.341, 
        # 0.342, 
        0.343, 
        # 0.344, 
        # 0.345, 
        0.346, 
        # 0.347, 
        # 0.348, 
        0.349, 
        # 0.350,
    ]
    # === training image paths ===
    # train_img_paths = [
    #     # "other_data/NVLab260130_fixed/0.2THz_34.0cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.1cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.2cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.3cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.4cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.5cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.6cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.7cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.8cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.9cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_35.0cm_1.bmp",
    # ]
    train_img_paths = [
        "other_data/NVLab260130_fixed/0.2THz_34.0cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.1cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.2cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.3cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.4cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.5cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.6cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.7cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.8cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.9cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_35.0cm_2.bmp",
    ]

    # train_img_paths = [
    #     "other_data/NVLab260130_fixed/0.2THz_34.0cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.1cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.2cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.3cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.4cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.5cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.6cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.7cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.8cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.9cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_35.0cm_3.bmp",
    # ]

    # train_img_paths = [
    #     "other_data/NVLab260130_fixed/0.12THz_34.0cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.1cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.2cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.3cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.4cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.5cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.6cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.7cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.8cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.9cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_35.0cm_1.bmp",
    # ]

    # === validation-only data ===
    # These planes are excluded from backpropagation but still appear in metrics and summary plots.
    val_z_list = [
        # 0.340,
        # 0.341,
        0.342,
        # 0.343,
        0.344,
        0.345,
        # 0.346,
        0.347,
        0.348,
        # 0.349,
        0.350,
    ]
    val_img_paths = [
        # "other_data/NVLab260130_fixed/0.2THz_34.0cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.1cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.2cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.3cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.4cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.5cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.6cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.7cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.8cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.9cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_35.0cm_2.bmp",
    ]

    output_dir = "other_data/NVLab260130_results"
    os.makedirs(output_dir, exist_ok=True)

    # TODO: balance the weighting
    # TODO: create xy axis displacement (?
    # TODO: iteration method to produce initial phase (?
    # TODO: radial and stripe combined initial phase (?
    # TODO: consider camera effect, restore real image before phase retrieval (!!
    # TODO: model the initial phase with point source assumption (!!
    # TODO: gain modefication

    # ====== Hyperparameters ======
    wavelength = 2.998e8 / 0.2004e12
    dx = 35e-6
    crop_size = None  # int for square crop, (h, w) for rectangular crop, None for no crop
    spatial_binning_factor = 1  # block-average neighboring pixels before propagation; effective dx becomes dx * binning / scale
    scale = 1  # upsampling
    # dark_current_eff_drift = "other_data/NVLab260417/con50_bri-100_noise_analysis/all_bursts_pixelwise_average.txt"  # scalar normalized offset e.g. 27.75 / 255, txt path, or full-size drift matrix subtracted before all other processing
    dark_current_eff_drift = 27.75 / 255 # set eff_drift as constant seems having better performance
    number_iter = 15000
    learning_rate = 1e-4  # lr if no specific lr_phase and lr_z
    include_evanescent = False  # include evanescent wave

    # === phase coarse-grid ===
    phase_grid_shape = (384 // 6, 288 // 6)  # None, (128, 128), or (64, 64), or (384, 288)

    # === learning rate strategy ===
    use_lr_scheduler = True  # ReduceLROnPlateau
    lr_decay_gamma = 0.5
    lr_plateau_patience = 3000
    lr_plateau_threshold = 1e-4
    lr_min = 1e-10

    # === data loss ===
    # "l1": baseline L1 loss
    # "weighted_l1": emphasize bright informative regions
    # "normalized_l1": compare normalized intensity only
    # "weighted_normalized_l1": emphasize bright informative regions after normalization
    data_weight = 0  # 1e-2
    data_loss_mode = "weighted_l2"
    data_loss_weight_power = 1.5  # for weighted mode: brightness weight power

    # === Correlation loss === global correlation loss
    correlation_loss_weight = 5e-2  # focuses on whether the diffraction pattern shape matches.

    # === SSIM loss === including local correlation loss
    ssim_loss_weight = 0#1e1  # focuses on local structural similarity.

    # === phase smoothness ===
    phase_smoothness_weight = 1e-2  # ratio to added to loss
    lr_phase = 1e-4
    phase_grad_weight = 1.5  # gradient weight(adjacent)
    phase_curv_weight = 0.5  # curvature weight(adjacent of adjacent)

    # === amplitude correction on Img_list[0] ===
    train_amplitude = True  # learn a small multiplicative correction on the first-plane amplitude
    lr_amp = 1e-4
    amplitude_grid_shape = None  # None for full-resolution correction, or coarse grid like (64, 64)
    amplitude_update_limit = 0.1 # amplitude multiplier stays within about 1 +/- limit
    amplitude_weight = 1e-3  # keep the learned amplitude close to the measured baseline

    # === monotone camera gain ===
    learn_monotone_gain = True  # jointly learn a monotone mapping from linear intensity to measured pixel value
    lr_gain = 1e-4
    monotone_gain_num_knots = 16
    monotone_gain_weight = 1e-4  # mild identity prior so the learned gain does not drift arbitrarily
    monotone_gain_smoothness_weight = 1e-4
    include_reference_plane_in_loss = True  # also match z0 in measurement domain so gain and source amplitude co-adapt
    reference_plane_weight = 1.0

    # === z distance ===
    train_delta_z = False  # whether to optimize z distances
    z_param_mode = "spacing"  # "plane_shift" or "spacing"
    lr_z = 1e-7
    z_weight = 1e3
    monotonic_weight = 1e5
    max_z_update = 1e-4  # m

    # === phase initialization ===
    # "random", "zero", "radial", "stripe"
    phase_init_mode = "radial"
    stripe_phase_num_y = 6  # horizontal stripes
    stripe_phase_num_x = 7  # vertical stripes
    radial_phase_positive = False  # increasing or decreasing phase difference
    processed_pixel_scale = scale / max(spatial_binning_factor, 1)
    phase_init_center = (
        (151 + (288 - 256) / 2) * processed_pixel_scale,
        (131 + (384 - 256) / 2) * processed_pixel_scale
    )  # e.g. (cy, cx)=(128, 128) or None, expressed in processed-image pixels
    phase_init_scale = math.pi / 8  # max initial value

    # === curriculum training ===
    curriculum = False
    curriculum_plan = None
    # curriculum_plan = [
    #     {"until": 1/3, "planes": [3, 6, 9], "label": "369"},
    #     {"until": 2/3, "planes": [3, 6, 9, 4, 7, 10], "label": "3467910"},
    #     {"until": 1.01, "planes": None, "label": "all"},
    # ]
    # None is default

    # === stochastic training ===
    stochastic = False
    stoch_k = 0.8  # stochastic ratio

    if len(train_z_list) != len(train_img_paths):
        raise ValueError(
            f"train_z_list and train_img_paths must have the same length, got {len(train_z_list)} and {len(train_img_paths)}"
        )
    if len(val_z_list) != len(val_img_paths):
        raise ValueError(
            f"val_z_list and val_img_paths must have the same length, got {len(val_z_list)} and {len(val_img_paths)}"
        )
    if len(train_img_paths) == 0:
        raise ValueError("At least one training image is required.")

    train_imgs = load_preprocessed_images(
        train_img_paths,
        crop_size=crop_size,
        scale=scale,
        dark_current_eff_drift=dark_current_eff_drift,
        spatial_binning_factor=spatial_binning_factor
    )
    val_imgs = load_preprocessed_images(
        val_img_paths,
        crop_size=crop_size,
        scale=scale,
        dark_current_eff_drift=dark_current_eff_drift,
        spatial_binning_factor=spatial_binning_factor
    )

    effective_dx = dx * spatial_binning_factor / scale

    # Img_list_aligned, shifts = align_img_list(train_imgs)  # for image center alignment

    train_imgs_torch = [torch.tensor(img, dtype=torch.float32, device=device) for img in train_imgs]

    initial_phase = build_initial_phase(
        train_imgs[0].shape[0],
        train_imgs[0].shape[1],
        device='cpu',
        phase_init_mode=phase_init_mode,
        stripe_phase_num_y=stripe_phase_num_y,
        stripe_phase_num_x=stripe_phase_num_x,
        radial_phase_positive=radial_phase_positive,
        phase_init_center=phase_init_center,
        phase_init_scale=phase_init_scale
    ).detach().cpu().numpy()

    U_recon, z_optimized, delta_z_final, loss_history, learned_gain_curve = multi_plane_gradient(
        train_imgs_torch,
        train_z_list,
        wavelength,
        effective_dx,
        n_iter=number_iter,
        lr=learning_rate,
        lr_phase=lr_phase,
        lr_z=lr_z,
        lr_amp=lr_amp,
        lr_gain=lr_gain,
        use_lr_scheduler=use_lr_scheduler,
        lr_decay_gamma=lr_decay_gamma,
        lr_plateau_patience=lr_plateau_patience,
        lr_plateau_threshold=lr_plateau_threshold,
        lr_min=lr_min,
        phase_grid_shape=phase_grid_shape,
        amplitude_grid_shape=amplitude_grid_shape,
        train_amplitude=train_amplitude,
        amplitude_update_limit=amplitude_update_limit,
        amplitude_weight=amplitude_weight,
        learn_monotone_gain=learn_monotone_gain,
        monotone_gain_num_knots=monotone_gain_num_knots,
        monotone_gain_weight=monotone_gain_weight,
        monotone_gain_smoothness_weight=monotone_gain_smoothness_weight,
        include_reference_plane_in_loss=include_reference_plane_in_loss,
        reference_plane_weight=reference_plane_weight,
        train_delta_z=train_delta_z,
        z_param_mode=z_param_mode,
        data_weight=data_weight,
        data_loss_mode=data_loss_mode,
        data_loss_weight_power=data_loss_weight_power,
        correlation_loss_weight=correlation_loss_weight,
        ssim_loss_weight=ssim_loss_weight,
        phase_smoothness_weight=phase_smoothness_weight,
        phase_grad_weight=phase_grad_weight,
        phase_curv_weight=phase_curv_weight,
        z_weight=z_weight,
        monotonic_weight=monotonic_weight,
        max_z_update=max_z_update,
        include_evanescent=include_evanescent,
        phase_init_mode=phase_init_mode,
        stripe_phase_num_y=stripe_phase_num_y,
        stripe_phase_num_x=stripe_phase_num_x,
        radial_phase_positive=radial_phase_positive,
        phase_init_center=phase_init_center,
        phase_init_scale=phase_init_scale,
        curriculum=curriculum,
        curriculum_plan=curriculum_plan,
        stochastic=stochastic,
        stoch_k=stoch_k,
        device=device
    )
    
    phase = np.angle(U_recon)
    amplitude = np.abs(U_recon)
    I_recon = amplitude**2
    print("Optimization finished.")
    print("Training z_list (mm):", np.round(np.array(train_z_list) * 1e3, 0))
    if val_z_list:
        print("Validation z_list (mm):", np.round(np.array(val_z_list) * 1e3, 0))
    print("spatial_binning_factor:", spatial_binning_factor)
    print("effective_dx (um):", effective_dx * 1e6)
    print("dark_current_eff_drift:", dark_current_eff_drift)
    print("train_amplitude:", train_amplitude)
    print("learn_monotone_gain:", learn_monotone_gain)
    print("include_reference_plane_in_loss:", include_reference_plane_in_loss)
    print("reference_plane_weight:", reference_plane_weight)
    print("data_weight:", data_weight)
    print("data_loss_mode:", data_loss_mode)
    print("correlation_loss_weight:", correlation_loss_weight)
    print("ssim_loss_weight:", ssim_loss_weight)
    if train_delta_z == True:
        print("z_param_mode:", z_param_mode)
        print("Optimized z_list (mm):", np.round(z_optimized * 1e3, 2))
        print("delta_z (mm):", np.round(delta_z_final * 1e3, 2))

    loss_keys_to_plot = ["total", "data", "corr", "ssim", "phase", "amplitude", "gain_reg", "gain_smooth"]  # the losses to be plot
    plot_loss_history(
        loss_history,
        keys_to_plot=loss_keys_to_plot,
        save_path=os.path.join(output_dir, "loss_history.png"),
        show_plot=True
    )

    train_eval_entries = build_plane_evaluation_entries(
        U_recon,
        train_imgs[0],
        train_z_list[0],
        train_imgs[1:],
        train_z_list[1:],
        wavelength,
        effective_dx,
        include_evanescent=include_evanescent,
        split_label="train",
        learned_gain_curve=learned_gain_curve
    )
    val_eval_entries = build_plane_evaluation_entries(
        U_recon,
        train_imgs[0],
        train_z_list[0],
        val_imgs,
        val_z_list,
        wavelength,
        effective_dx,
        include_evanescent=include_evanescent,
        split_label="val",
        learned_gain_curve=learned_gain_curve
    )
    all_eval_entries = train_eval_entries + val_eval_entries

    if learned_gain_curve is not None:
        save_learned_monotone_gain_curve(learned_gain_curve, output_dir)
        print("\n===== Learned Monotone Gain Curve =====")
        print("Jointly optimized with phase retrieval in the measurement domain.")
        print("input_scale:", learned_gain_curve["input_scale"])
        print("num_knots:", learned_gain_curve["num_knots"])

    metric_rows = []
    for entry in all_eval_entries:
        gen_metrics = evaluate_intensity_metrics(entry["i_gen"], entry["i_meas"])
        zero_metrics = evaluate_intensity_metrics(entry["i_zero"], entry["i_meas"])

        metric_rows.append({
            "split": entry["split"],
            "plane": entry["plane"],
            "z_mm": entry["z_m"] * 1e3,
            "ssim_gen": gen_metrics["ssim"],
            "ssim_zero": zero_metrics["ssim"],
            "ssim_center256_gen": gen_metrics["ssim_center256"],
            "ssim_center256_zero": zero_metrics["ssim_center256"],
            "corr_gen": gen_metrics["corr"],
            "corr_zero": zero_metrics["corr"],
            "corr_center256_gen": gen_metrics["corr_center256"],
            "corr_center256_zero": zero_metrics["corr_center256"],
            "mae_gen": gen_metrics["mae"],
            "mae_zero": zero_metrics["mae"],
            "mae_center256_gen": gen_metrics["mae_center256"],
            "mae_center256_zero": zero_metrics["mae_center256"],
            "energy_err_gen": gen_metrics["energy_err"],
            "energy_err_zero": zero_metrics["energy_err"],
            "energy_err_center256_gen": gen_metrics["energy_err_center256"],
            "energy_err_center256_zero": zero_metrics["energy_err_center256"],
            "ssim_masked_gen": gen_metrics["ssim_masked"],
            "ssim_masked_zero": zero_metrics["ssim_masked"],
            "ssim_masked_center256_gen": gen_metrics["ssim_masked_center256"],
            "ssim_masked_center256_zero": zero_metrics["ssim_masked_center256"],
            "corr_masked_gen": gen_metrics["corr_masked"],
            "corr_masked_zero": zero_metrics["corr_masked"],
            "corr_masked_center256_gen": gen_metrics["corr_masked_center256"],
            "corr_masked_center256_zero": zero_metrics["corr_masked_center256"],
            "mae_masked_gen": gen_metrics["mae_masked"],
            "mae_masked_zero": zero_metrics["mae_masked"],
            "mae_masked_center256_gen": gen_metrics["mae_masked_center256"],
            "mae_masked_center256_zero": zero_metrics["mae_masked_center256"],
            "mask_ratio": gen_metrics["mask_ratio"],
            "mask_ratio_center256": gen_metrics["mask_ratio_center256"],
        })

    if metric_rows:
        metric_summary = {
            "ssim_gen_mean": float(np.nanmean([row["ssim_gen"] for row in metric_rows])),
            "ssim_zero_mean": float(np.nanmean([row["ssim_zero"] for row in metric_rows])),
            "ssim_center256_gen_mean": float(np.nanmean([row["ssim_center256_gen"] for row in metric_rows])),
            "ssim_center256_zero_mean": float(np.nanmean([row["ssim_center256_zero"] for row in metric_rows])),
            "corr_gen_mean": float(np.nanmean([row["corr_gen"] for row in metric_rows])),
            "corr_zero_mean": float(np.nanmean([row["corr_zero"] for row in metric_rows])),
            "corr_center256_gen_mean": float(np.nanmean([row["corr_center256_gen"] for row in metric_rows])),
            "corr_center256_zero_mean": float(np.nanmean([row["corr_center256_zero"] for row in metric_rows])),
            "mae_gen_mean": float(np.nanmean([row["mae_gen"] for row in metric_rows])),
            "mae_zero_mean": float(np.nanmean([row["mae_zero"] for row in metric_rows])),
            "mae_center256_gen_mean": float(np.nanmean([row["mae_center256_gen"] for row in metric_rows])),
            "mae_center256_zero_mean": float(np.nanmean([row["mae_center256_zero"] for row in metric_rows])),
            "energy_err_gen_mean": float(np.nanmean([row["energy_err_gen"] for row in metric_rows])),
            "energy_err_zero_mean": float(np.nanmean([row["energy_err_zero"] for row in metric_rows])),
            "energy_err_center256_gen_mean": float(np.nanmean([row["energy_err_center256_gen"] for row in metric_rows])),
            "energy_err_center256_zero_mean": float(np.nanmean([row["energy_err_center256_zero"] for row in metric_rows])),
            "ssim_masked_gen_mean": float(np.nanmean([row["ssim_masked_gen"] for row in metric_rows])),
            "ssim_masked_zero_mean": float(np.nanmean([row["ssim_masked_zero"] for row in metric_rows])),
        }

        print("\n===== Metric Summary =====")
        print(
            f"SSIM mean | Generated {metric_summary['ssim_gen_mean']:.4f} | "
            f"Zero-phase {metric_summary['ssim_zero_mean']:.4f}"
        )
        print(
            f"CenterCrop256 SSIM mean | Generated {metric_summary['ssim_center256_gen_mean']:.4f} | "
            f"Zero-phase {metric_summary['ssim_center256_zero_mean']:.4f}"
        )
        print(
            f"Corr mean | Generated {metric_summary['corr_gen_mean']:.4f} | "
            f"Zero-phase {metric_summary['corr_zero_mean']:.4f}"
        )
        print(
            f"CenterCrop256 Corr mean | Generated {metric_summary['corr_center256_gen_mean']:.4f} | "
            f"Zero-phase {metric_summary['corr_center256_zero_mean']:.4f}"
        )
        print(
            f"MAE mean | Generated {metric_summary['mae_gen_mean']:.4e} | "
            f"Zero-phase {metric_summary['mae_zero_mean']:.4e}"
        )
        print(
            f"CenterCrop256 MAE mean | Generated {metric_summary['mae_center256_gen_mean']:.4e} | "
            f"Zero-phase {metric_summary['mae_center256_zero_mean']:.4e}"
        )
        print(
            f"EnergyErr mean | Generated {metric_summary['energy_err_gen_mean']:.4e} | "
            f"Zero-phase {metric_summary['energy_err_zero_mean']:.4e}"
        )
        print(
            f"CenterCrop256 EnergyErr mean | Generated {metric_summary['energy_err_center256_gen_mean']:.4e} | "
            f"Zero-phase {metric_summary['energy_err_center256_zero_mean']:.4e}"
        )
        print(
            f"Masked SSIM mean | Generated {metric_summary['ssim_masked_gen_mean']:.4f} | "
            f"Zero-phase {metric_summary['ssim_masked_zero_mean']:.4f}"
        )

        metrics_csv_path = os.path.join(output_dir, "intensity_metrics.csv")
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)

        summary_csv_path = os.path.join(output_dir, "intensity_metrics_summary.csv")
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in metric_summary.items():
                writer.writerow([key, value])

    images = []
    measured_z0_linear = (
        apply_inverse_learned_monotone_gain_numpy(train_imgs[0], learned_gain_curve)
        if learned_gain_curve is not None else train_imgs[0]
    )
    zero_phase_title_z0 = make_gain_corrected_title(
        "Zero_phase_intensity_z0",
        None,
        learned_gain_curve=learned_gain_curve
    )
    generated_title_z0 = make_gain_corrected_title(
        "Generated_intensity_z0",
        None,
        learned_gain_curve=learned_gain_curve
    )
    measured_title_z0 = make_gain_corrected_title(
        "Measured_intensity_z0",
        dark_current_eff_drift,
        learned_gain_curve=learned_gain_curve
    )
    generated_z0_img = I_recon

    # === append fixed image ===
    images.append((np.zeros_like(train_imgs[0]), "Blank"))
    images.append((np.zeros_like(train_imgs[0]), "Blank"))
    images.append((initial_phase, "Initial_phase"))
    images.append((np.zeros_like(train_imgs[0]), "Blank"))
    images.append((np.zeros_like(train_imgs[0]), "Blank"))
    images.append((phase, "Reconstructed_phase"))
    
    images.append((measured_z0_linear, zero_phase_title_z0))
    images.append((generated_z0_img, generated_title_z0))
    images.append((measured_z0_linear, measured_title_z0))
    
    # === add train/validation image triplets ===
    for entry in train_eval_entries:
        zero_title = make_gain_corrected_title(
            f"Train_zero_phase_intensity_z{entry['plane']}",
            None,
            learned_gain_curve=learned_gain_curve
        )
        generated_title = make_gain_corrected_title(
            f"Train_generated_intensity_z{entry['plane']}",
            None,
            learned_gain_curve=learned_gain_curve
        )
        measured_title = make_gain_corrected_title(
            f"Train_measured_intensity_z{entry['plane']}",
            dark_current_eff_drift,
            learned_gain_curve=learned_gain_curve
        )
        images.append((entry["i_zero_linear"], zero_title))
        images.append((entry["i_gen_linear"], generated_title))
        images.append((entry["i_meas_linear"], measured_title))

    for entry in val_eval_entries:
        zero_title = make_gain_corrected_title(
            f"Val_zero_phase_intensity_z{entry['plane']}",
            None,
            learned_gain_curve=learned_gain_curve
        )
        generated_title = make_gain_corrected_title(
            f"Val_generated_intensity_z{entry['plane']}",
            None,
            learned_gain_curve=learned_gain_curve
        )
        measured_title = make_gain_corrected_title(
            f"Val_measured_intensity_z{entry['plane']}",
            dark_current_eff_drift,
            learned_gain_curve=learned_gain_curve
        )
        images.append((entry["i_zero_linear"], zero_title))
        images.append((entry["i_gen_linear"], generated_title))
        images.append((entry["i_meas_linear"], measured_title))

    # === save as PNG ===
    for img_array, title in images:
        save_png(img_array, title)

    # === summary plot ===
    n_summary_columns = len(images) // 3
    fig, axs = plt.subplots(3, n_summary_columns, figsize=(max(24, 3 * n_summary_columns), 8))
    axs = axs.flatten(order='F')

    for ax, (img_array, title) in zip(axs, images):
        if title == "Blank":
            ax.axis("off")
            continue
        if title in ["Initial_phase", "Reconstructed_phase"]:
            phase_display = np.angle(np.exp(1j * img_array))
            im = ax.imshow(phase_display, cmap="twilight", vmin=-np.pi, vmax=np.pi)  # for wrapping
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(img_array*255, cmap="gray", vmin=0, vmax=255)  # 

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    
# TODO 




# The effect of better comments
# * sdfas
# ! 
# ?
# TODO

