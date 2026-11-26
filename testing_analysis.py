"""Shared reconstruction, frequency, latent and ONN diagnostics.

Importing this module does not initialize models, datasets or process groups.
"""
import csv
import math
import os
import random
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import get_dataloaders
import torch.nn.functional as F
import torchvision.utils as vutils
from model.opticalSimulation import MaterialLayer, SensorLayer


def get_model_device(model):
    return next(model.parameters()).device


def compute_psnr(mse, max_pixel=1.0):
    if mse <= 0:
        return float("inf")
    return 20 * math.log10(max_pixel) - 10 * math.log10(mse)


def ssim_pt(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    padding = window_size // 2
    weight = torch.ones((1, 1, window_size, window_size), device=img1.device) / (
        window_size**2
    )

    mu1 = F.conv2d(img1, weight, padding=padding)
    mu2 = F.conv2d(img2, weight, padding=padding)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, weight, padding=padding) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, weight, padding=padding) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, weight, padding=padding) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def get_model_core(model):
    return model.module if hasattr(model, "module") else model


def project_latent_for_viz(latent, config, target_size=None):
    if isinstance(latent, (tuple, list)):
        latent = latent[0]

    if latent is None:
        return None

    if torch.is_complex(latent):
        latent = torch.abs(latent)

    if latent.dim() == 4:
        projection_mode = config.get("latent_projection_mode", "mean_abs")
        if projection_mode == "first_channel":
            latent_map = latent[:, 0:1, :, :]
        elif projection_mode == "max_abs":
            latent_map = latent.abs().amax(dim=1, keepdim=True)
        else:
            latent_map = latent.abs().mean(dim=1, keepdim=True)
    elif latent.dim() == 3:
        latent_map = latent.unsqueeze(1)
    elif latent.dim() == 2:
        latent_map = latent.unsqueeze(0).unsqueeze(0)
    else:
        latent_map = latent.reshape(latent.shape[0], 1, -1, 1)

    if target_size is not None and latent_map.shape[-2:] != target_size:
        latent_map = F.interpolate(
            latent_map.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    return latent_map.detach().cpu()


def get_latent_cmap(config):
    return config.get("latent_cmap", "gray")


def get_latent_map(model, imgs, config, target_size=None):
    net = get_model_core(model)
    with torch.no_grad():
        encoder_out = net.encoder(imgs)
        latent = encoder_out[0] if isinstance(encoder_out, (tuple, list)) else encoder_out
        latent_map = project_latent_for_viz(latent, config, target_size=target_size)
    return latent_map


def save_frequency_comparison(
    all_imgs, all_recons, indices, model_name, config, group_name,
    mse_values=None,
    latent_images=None,
    model=None,
    output_subdir="frequency_domain_panels",
):
    if not indices:
        return

    frequency_root = os.path.join(
        config["results_save_dir"], output_subdir
    )
    os.makedirs(frequency_root, exist_ok=True)

    for idx in indices:
        gt = all_imgs[idx].detach().float().squeeze().cpu()
        recon = all_recons[idx].detach().float().squeeze().cpu()
        #! [HF-RECOVERY] Modified: Convert intensity images to field amplitude before FFT
        gt_field = torch.sqrt(torch.clamp(gt, min=0.0))
        recon_field = torch.sqrt(torch.clamp(recon, min=0.0))
        gt_spectrum = torch.fft.fftshift(torch.fft.fft2(gt_field))
        recon_spectrum = torch.fft.fftshift(torch.fft.fft2(recon_field))
        gt_magnitude = torch.log1p(torch.abs(gt_spectrum)).numpy()
        recon_magnitude = torch.log1p(torch.abs(recon_spectrum)).numpy()
        spectrum_difference = abs(gt_magnitude - recon_magnitude)

        latent_zero_amp = latent_field_amp = latent_phase = latent_spatial_phase = latent_difference = None
        teacher_channels = None
        teacher_magnitude_name = "Amplitude"
        camera_energy_sum = full_energy_sum = None
        if model is not None:
            net = get_model_core(model)
            sensor_layer = next((layer for layer in net.encoder.layers if isinstance(layer, SensorLayer)), None)
            if sensor_layer is not None:
                with torch.no_grad():
                    x_probe = all_imgs[idx:idx + 1].to(get_model_device(model))
                    for layer in net.encoder.layers:
                        x_probe = layer(x_probe)
                        if isinstance(x_probe, (tuple, list)):
                            x_probe = x_probe[0]
                        if layer is sensor_layer:
                            sensor_energy = x_probe[0, 0].detach().float().cpu()
                            field = sensor_layer.last_field[0, 0].detach().cpu()
                            break
                if config.get("show_teacher_input_channels", False):
                    phase = torch.angle(field)
                    magnitude = field.abs()
                    if getattr(net, "teacher_use_energy", False):
                        magnitude = magnitude.square()
                        teacher_magnitude_name = "Intensity (amplitude^2)"
                    teacher_channels = [magnitude.numpy(), phase.sin().numpy(), phase.cos().numpy()]
                zero_fft = torch.fft.fftshift(torch.fft.fft2(torch.sqrt(torch.clamp(sensor_energy, min=0.0))))
                field_fft = torch.fft.fftshift(torch.fft.fft2(field))
                latent_zero_amp = torch.log1p(torch.abs(zero_fft)).numpy()
                latent_field_amp = torch.log1p(torch.abs(field_fft)).numpy()
                latent_phase = ((torch.angle(field_fft) + math.pi) / (2 * math.pi)).numpy()
                latent_spatial_phase = ((torch.angle(field) + math.pi) / (2 * math.pi)).numpy()
                # Total optical energy arriving inside the camera-aligned latent area.
                # Use the complex conjugate explicitly: E * E.conj() = |E|^2.
                camera_energy_sum = float((field * torch.conj(field)).real.sum().item())
                full_field = sensor_layer.last_full_field[0, 0].detach().cpu()
                full_energy_sum = float((full_field * torch.conj(full_field)).real.sum().item())
                latent_difference = np.abs(latent_zero_amp - latent_field_amp)

        #! [HF-RECOVERY] Added: Radial frequency curves in cycles per pixel
        height, width = gt.shape[-2:]
        freq_y = torch.fft.fftshift(torch.fft.fftfreq(height))
        freq_x = torch.fft.fftshift(torch.fft.fftfreq(width))
        freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing="ij")
        radial_frequency = torch.sqrt(freq_grid_x ** 2 + freq_grid_y ** 2)
        radial_bins = torch.linspace(0.0, 0.5, 65)
        radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
        gt_log_tensor = torch.from_numpy(gt_magnitude)
        recon_log_tensor = torch.from_numpy(recon_magnitude)
        radial_gt = []
        radial_recon = []
        for bin_start, bin_end in zip(radial_bins[:-1], radial_bins[1:]):
            mask = (radial_frequency >= bin_start) & (radial_frequency < bin_end)
            radial_gt.append(gt_log_tensor[mask].mean().item() if mask.any() else float("nan"))
            radial_recon.append(recon_log_tensor[mask].mean().item() if mask.any() else float("nan"))
        #! [HF-RECOVERY] Modified: Signed-log GT minus reconstruction frequency error
        radial_difference = [
            math.copysign(math.log1p(abs(gt_value - recon_value)), gt_value - recon_value)
            if math.isfinite(gt_value) and math.isfinite(recon_value)
            else float("nan")
            for gt_value, recon_value in zip(radial_gt, radial_recon)
        ]

        if mse_values is not None:
            mse_value = float(mse_values[idx])
        else:
            mse_value = torch.mean((all_imgs[idx].float() - all_recons[idx].float()) ** 2).item()
        psnr_value = compute_psnr(mse_value)
        safe_psnr = "inf" if not math.isfinite(psnr_value) else f"{psnr_value:.2f}dB"
        filename = f"PSNR_{safe_psnr}_{idx:06d}.png"
        save_path = os.path.join(frequency_root, filename)

        shared_max = max(float(gt_magnitude.max()), float(recon_magnitude.max()))
        #! [HF-RECOVERY] Added: Spatial-domain comparison beside frequency-domain views
        spatial_difference = abs(gt.numpy() - recon.numpy())
        spatial_max = max(float(gt.max()), float(recon.max()))
        latent_image = None
        if latent_images is not None:
            latent_image = latent_images[idx]
            if torch.is_tensor(latent_image):
                latent_image = latent_image.detach().float().squeeze().cpu().numpy()
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes[0, 0].imshow(gt.numpy(), cmap="gray", vmin=0.0, vmax=spatial_max)
        axes[0, 0].set_title("GT Spatial")
        axes[0, 1].imshow(recon.numpy(), cmap="gray", vmin=0.0, vmax=spatial_max)
        axes[0, 1].set_title("Reconstruction Spatial")
        if latent_image is not None:
            axes[0, 2].imshow(latent_image, cmap=get_latent_cmap(config))
            if camera_energy_sum is not None:
                axes[0, 2].set_title(
                    "Latent Feature\n"
                    f"Camera energy={camera_energy_sum:.6g}\n"
                    f"Full simulation energy={full_energy_sum:.6g}"
                )
            else:
                axes[0, 2].set_title("Latent Feature")
        else:
            axes[0, 2].axis("off")
        if latent_spatial_phase is not None:
            axes[0, 3].imshow(latent_spatial_phase, cmap="twilight", vmin=0.0, vmax=1.0)
            axes[0, 3].set_title("Latent Spatial Phase")
            axes[0, 3].set_xlabel("phase mapped to [0, 1]")
        else:
            axes[0, 3].axis("off")
        axes[0, 4].imshow(spatial_difference, cmap="magma")
        axes[0, 4].set_title("Spatial Absolute Difference")
        axes[1, 0].imshow(gt_magnitude, cmap="magma", vmin=0.0, vmax=shared_max)
        axes[1, 0].set_title("GT Frequency Magnitude")
        axes[1, 1].imshow(recon_magnitude, cmap="magma", vmin=0.0, vmax=shared_max)
        axes[1, 1].set_title("Reconstruction Frequency Magnitude")
        axes[1, 2].imshow(spectrum_difference, cmap="viridis")
        axes[1, 2].set_title("Frequency Absolute Difference")
        axes[1, 3].axis("off")
        axes[1, 4].axis("off")
        axes[3, 0].plot(radial_centers.numpy(), radial_gt, label="GT", linewidth=2)
        axes[3, 0].plot(radial_centers.numpy(), radial_recon, label="Reconstruction", linewidth=2)
        axes[3, 0].plot(
            radial_centers.numpy(), radial_difference,
            label="signed-log(GT - Reconstruction)", linewidth=2, linestyle="-",
        )
        axes[3, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axes[3, 0].set_title("GT vs Reconstruction Radial Spectrum")
        axes[3, 0].set_xlabel("Spatial frequency (cycles/pixel)")
        axes[3, 0].set_ylabel("Mean log magnitude / signed-log difference")
        axes[3, 0].set_xlim(0.0, 0.5)
        axes[3, 0].grid(alpha=0.3)
        axes[3, 0].legend(fontsize=8)
        if latent_zero_amp is not None:
            axes[2, 0].imshow(latent_zero_amp, cmap="magma")
            axes[2, 0].set_title("sqrt(Energy) FFT Amplitude")
            axes[2, 1].imshow(latent_field_amp, cmap="magma")
            axes[2, 1].set_title("Latent Field FFT Amplitude")
            axes[2, 2].imshow(latent_phase, cmap="twilight", vmin=0.0, vmax=1.0)
            axes[2, 2].set_title("Latent Field FFT Phase")
            h, w = latent_zero_amp.shape
            yy, xx = np.indices((h, w))
            radius = np.rint(np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)).astype(int)
            curves = []
            for image in (latent_zero_amp, latent_field_amp, latent_difference):
                curves.append(np.bincount(radius.ravel(), image.ravel()) / np.maximum(np.bincount(radius.ravel()), 1))
            axes[2, 3].plot(curves[0])
            axes[2, 3].plot(curves[1])
            axes[2, 3].plot(curves[2])
            axes[2, 3].set_title("Latent FFT Amplitude Comparison")
            axes[2, 3].set_yscale("log")
            axes[2, 3].legend(["sqrt(Energy)", "Field", "Difference"], fontsize=8)
            axes[2, 0].set_xlabel("bright = high amplitude")
            axes[2, 1].set_xlabel("bright = high amplitude")
            axes[2, 2].set_xlabel("phase mapped to [0, 1]")
            axes[2, 4].axis("off")
        else:
            for axis in axes[2, :]:
                axis.axis("off")
        axes[3, 1].axis("off")
        axes[3, 2].axis("off")
        axes[3, 3].axis("off")
        axes[3, 4].axis("off")
        if teacher_channels is not None:
            for axis, channel, title in zip(
                [axes[1, 3], axes[1, 4], axes[2, 4]], teacher_channels,
                [f"Teacher Input: {teacher_magnitude_name}", "Teacher Input: sin(phase)", "Teacher Input: cos(phase)"],
            ):
                axis.imshow(channel, cmap="coolwarm" if "phase)" in title else "gray")
                axis.set_title(title)
                axis.axis("off")
        for axis in axes[0, :]:
            axis.axis("off")
        fig.suptitle(f"Sample {idx} | PSNR={psnr_value:.4f} dB")
        plt.tight_layout()
        plt.savefig(save_path, dpi=180)
        plt.close(fig)

    print(f"Frequency-domain panels saved at {frequency_root} ({group_name})")


def visualize_results(model, all_imgs, all_recons, model_name, num_image, config, all_mses=None):
    os.makedirs(config["results_save_dir"], exist_ok=True)
    num_image = min(num_image, len(all_imgs), len(all_recons))
    if num_image <= 0:
        return

    imgs = all_imgs[:num_image]
    recons = all_recons[:num_image]
    latent_enabled = config.get("save_latent_panel", True)
    latent_maps = None
    if latent_enabled:
        latent_maps = get_latent_map(
            model,
            imgs.to(get_model_device(model)),
            config,
            target_size=recons.shape[-2:] if config.get("latent_resize_to_output", True) else None,
        )
        latent_cmap = get_latent_cmap(config)

    rows = 3 if latent_enabled else 2
    fig, axes = plt.subplots(rows, num_image, figsize=(num_image * 2, rows * 2))
    if num_image == 1:
        axes = axes.reshape(rows, 1)
    for i in range(num_image):
        axes[0, i].imshow(imgs[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if latent_enabled:
            latent_img = latent_maps[i].squeeze().numpy()
            axes[1, i].imshow(latent_img, cmap=latent_cmap)
            axes[1, i].axis("off")
            axes[2, i].imshow(recons[i].squeeze(), cmap="gray")
            axes[2, i].axis("off")
        else:
            axes[1, i].imshow(recons[i].squeeze(), cmap="gray")
            axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    if latent_enabled:
        axes[1, 0].set_ylabel("Latent", fontsize=12)
        axes[2, 0].set_ylabel("Reconstructed", fontsize=12)
    else:
        axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(config["results_save_dir"], f"{model_name}_image.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Visualization saved at {save_path}")
    #! [HF-RECOVERY] Export FFT comparisons for the first-N panel samples
    if config.get("save_frequency_domain_panels", True):
        save_frequency_comparison(
            all_imgs, all_recons, list(range(num_image)), model_name, config, "first_n",
            mse_values=all_mses,
            latent_images={idx: latent_maps[idx] for idx in range(num_image)} if latent_maps is not None else None,
            model=model,
        )


def visualize_results_by_mse(all_imgs, all_recons, all_mses, model_name, config, model=None):
    if all_imgs is None or all_recons is None or not all_mses:
        return

    save_root = os.path.join(config["results_save_dir"], "mse_ranked")
    os.makedirs(save_root, exist_ok=True)

    mse_values = list(all_mses)
    num_samples = min(len(all_imgs), len(all_recons), len(mse_values))
    if num_samples == 0:
        return

    indexed = list(enumerate(mse_values[:num_samples]))
    indexed.sort(key=lambda item: item[1])

    panel_size = max(1, int(config.get("mse_panel_size", 12)))

    def parse_quantile_range(value, default_start, default_end):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            start_q, end_q = float(value[0]), float(value[1])
        else:
            start_q, end_q = default_start, default_end
        start_q = max(0.0, min(1.0, start_q))
        end_q = max(0.0, min(1.0, end_q))
        if end_q < start_q:
            start_q, end_q = end_q, start_q
        if end_q == start_q:
            end_q = min(1.0, start_q + 1e-6)
        return start_q, end_q

    def select_from_quantile_range(q_range, desired_count, mode="head"):
        start_q, end_q = q_range
        start_idx = int(math.floor(start_q * num_samples))
        end_idx = int(math.ceil(end_q * num_samples))
        start_idx = max(0, min(num_samples, start_idx))
        end_idx = max(start_idx + 1, min(num_samples, end_idx))
        subset = indexed[start_idx:end_idx]
        if not subset:
            subset = indexed

        desired = min(desired_count, len(subset))
        if desired <= 0:
            return []

        if len(subset) <= desired:
            return [idx for idx, _ in subset]

        if mode == "tail":
            subset = subset[-desired:]
        elif mode == "middle":
            center = len(subset) // 2
            half = desired // 2
            sub_start = max(0, center - half)
            sub_end = min(len(subset), sub_start + desired)
            sub_start = max(0, sub_end - desired)
            subset = subset[sub_start:sub_end]
        else:
            subset = subset[:desired]

        return [idx for idx, _ in subset]

    best_range = parse_quantile_range(config.get("mse_best_quantile_range"), 0.0, 0.1)
    middle_range = parse_quantile_range(config.get("mse_middle_quantile_range"), 0.45, 0.55)
    bottom_range = parse_quantile_range(config.get("mse_bottom_quantile_range"), 0.9, 1.0)

    best_indices = select_from_quantile_range(best_range, panel_size, mode="head")
    middle_indices = select_from_quantile_range(middle_range, panel_size, mode="middle")
    bottom_indices = select_from_quantile_range(bottom_range, panel_size, mode="tail")

    def save_group(indices, title, filename):
        if not indices:
            return

        cols = len(indices)
        latent_enabled = config.get("save_latent_panel", True) and model is not None
        rows = 4 if latent_enabled else 3
        fig, axes = plt.subplots(rows, cols, figsize=(max(6, cols * 2.2), rows * 2))
        if cols == 1:
            axes = axes.reshape(rows, 1)

        selected_imgs = all_imgs[indices].to(get_model_device(model))
        latent_maps = None
        if latent_enabled:
            latent_maps = get_latent_map(
                model,
                selected_imgs,
                config,
                target_size=all_recons[indices].shape[-2:] if config.get("latent_resize_to_output", True) else None,
            )
            latent_cmap = get_latent_cmap(config)

        for col, idx in enumerate(indices):
            img = all_imgs[idx].squeeze().numpy()
            recon = all_recons[idx].squeeze().numpy()
            diff = abs(img - recon)
            mse_val = mse_values[idx]
            psnr_val = compute_psnr(mse_val)

            axes[0, col].imshow(img, cmap="gray")
            axes[0, col].set_title(
                f"#{idx}\nMSE={mse_val:.4g}\nPSNR={psnr_val:.2f}",
                fontsize=9,
            )

            if latent_enabled:
                latent_img = latent_maps[col].squeeze().numpy()
                axes[1, col].imshow(latent_img, cmap=latent_cmap)
                axes[2, col].imshow(recon, cmap="gray")
                axes[3, col].imshow(diff, cmap="magma")
            else:
                axes[1, col].imshow(recon, cmap="gray")
                axes[2, col].imshow(diff, cmap="magma")

            for row in range(rows):
                axes[row, col].axis("off")

        axes[0, 0].set_ylabel("GT", fontsize=11)
        if latent_enabled:
            axes[1, 0].set_ylabel("Latent", fontsize=11)
            axes[2, 0].set_ylabel("Recon", fontsize=11)
            axes[3, 0].set_ylabel("Abs Diff", fontsize=11)
        else:
            axes[1, 0].set_ylabel("Recon", fontsize=11)
            axes[2, 0].set_ylabel("Abs Diff", fontsize=11)
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(save_root, filename)
        plt.savefig(save_path, dpi=180)
        plt.close(fig)
        print(f"MSE-ranked visualization saved at {save_path}")
        #! [HF-RECOVERY] Export FFT comparisons for every MSE panel sample
        if config.get("save_frequency_domain_panels", True):
            save_frequency_comparison(
                all_imgs, all_recons, indices, model_name, config, filename.rsplit(".", 1)[0],
                mse_values=mse_values,
                latent_images={idx: latent_maps[col] for col, idx in enumerate(indices)} if latent_maps is not None else None,
                model=model,
            )

    if config.get("save_mse_best_panel", False):
        save_group(best_indices, "Best cases by MSE", f"{model_name}_best_case.png")
    if config.get("save_mse_middle_panel", True):
        save_group(middle_indices, "Middle cases by MSE", f"{model_name}_middle_case.png")
    if config.get("save_mse_bottom_panel", True):
        save_group(bottom_indices, "Bottom cases by MSE", f"{model_name}_bottom_case.png")

    if config.get("save_mse_ranked_csv", True):
        csv_path = os.path.join(save_root, f"{model_name}_mse_ranking.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank_ascending", "sample_index", "mse", "psnr"])
            for rank, (idx, mse_val) in enumerate(indexed):
                writer.writerow([rank, idx, mse_val, compute_psnr(mse_val)])
        print(f"MSE ranking saved at {csv_path}")


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]

    pos = (len(sorted_values) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_values[lower]
    lower_val = sorted_values[lower]
    upper_val = sorted_values[upper]
    return lower_val * (upper - pos) + upper_val * (pos - lower)


def save_psnr_distribution_table(all_mses, model_name, config):
    if not all_mses:
        return

    save_root = os.path.join(config["results_save_dir"], "mse_ranked")
    os.makedirs(save_root, exist_ok=True)

    psnr_values = [compute_psnr(mse_val) for mse_val in all_mses]
    finite_psnrs = [value for value in psnr_values if math.isfinite(value)]
    inf_count = len(psnr_values) - len(finite_psnrs)

    summary_path = os.path.join(save_root, f"{model_name}_psnr_distribution_summary.csv")
    bins_path = os.path.join(save_root, f"{model_name}_psnr_distribution_bins.csv")

    quantile_points = config.get("psnr_quantile_points", [0.1, 0.25, 0.5, 0.75, 0.9])
    finite_sorted = sorted(finite_psnrs)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stat", "value"])
        writer.writerow(["count_total", len(psnr_values)])
        writer.writerow(["count_finite", len(finite_psnrs)])
        writer.writerow(["count_inf", inf_count])
        if finite_sorted:
            mean_val = sum(finite_sorted) / len(finite_sorted)
            variance = sum((x - mean_val) ** 2 for x in finite_sorted) / len(finite_sorted)
            std_val = math.sqrt(variance)
            writer.writerow(["mean", mean_val])
            writer.writerow(["std", std_val])
            writer.writerow(["min", finite_sorted[0]])
            for q in quantile_points:
                writer.writerow([f"q{int(float(q) * 100)}", percentile(finite_sorted, float(q))])
            writer.writerow(["max", finite_sorted[-1]])

    bin_edges = config.get("psnr_bin_edges", [0, 5, 10, 15, 20, 25, 30, 35, 40, 100])
    try:
        bin_edges = [float(edge) for edge in bin_edges]
    except (TypeError, ValueError):
        bin_edges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 100]

    if len(bin_edges) < 2:
        bin_edges = [0, 100]

    bins = [0 for _ in range(len(bin_edges) - 1)]
    for value in finite_psnrs:
        placed = False
        for i in range(len(bin_edges) - 1):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            if i == len(bin_edges) - 2:
                if lower <= value <= upper:
                    bins[i] += 1
                    placed = True
                    break
            elif lower <= value < upper:
                bins[i] += 1
                placed = True
                break
        if not placed and value < bin_edges[0]:
            bins[0] += 1

    total_finite = max(1, len(finite_psnrs))
    with open(bins_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_index", "lower_bound", "upper_bound", "count", "ratio"])
        for i, count in enumerate(bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            writer.writerow([i, lower, upper, count, count / total_finite])
        writer.writerow(["inf", "", "", inf_count, inf_count / max(1, len(psnr_values))])

    if config.get("save_psnr_distribution_histogram", True):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(finite_psnrs, bins=bin_edges, color="#4C78A8", edgecolor="black", alpha=0.85)
        ax.set_title("PSNR Distribution")
        ax.set_xlabel("PSNR")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        hist_path = os.path.join(save_root, f"{model_name}_psnr_distribution_hist.png")
        plt.savefig(hist_path, dpi=180)
        plt.close(fig)
        print(f"PSNR histogram saved at {hist_path}")

    print(f"PSNR summary saved at {summary_path}")
    print(f"PSNR bins saved at {bins_path}")


def onn_output_debug(model, config, dataset):
    net = model.module if hasattr(model, "module") else model

    debug_dir = os.path.join(config["results_save_dir"], "ONN_debug")
    os.makedirs(debug_dir, exist_ok=True)

    split_method = config.get("ONN_input_select", "fix")
    seed = config.get("seed", None)

    if split_method == "fix":
        idx = config["ONN_input_idx"]
    else:
        if seed is not None:
            random.seed(seed)
        idx = random.randint(0, len(dataset) - 1)

    img, _ = dataset[idx]
    img = img.unsqueeze(0).to(get_model_device(model))

    vutils.save_image(img, f"{debug_dir}/input_{split_method}.png", normalize=True)
    print(f"[ONN DEBUG] Saved input image to {debug_dir}/input_{split_method}.png")

    x = img
    for i, layer in enumerate(net.encoder.layers):
        layer_name = (
            net.encoder.layer_names[i]
            if hasattr(net.encoder, "layer_names")
            else f"layer_{i}"
        )

        x = layer(x)
        if not isinstance(x, (tuple, list)):
            x = (x,)
        x, *rest = x

        out = x
        abs_out = torch.abs(out) ** 2 if torch.is_complex(out) else out
        vutils.save_image(
            abs_out[:, 0:1, :, :].cpu(),
            os.path.join(debug_dir, f"{layer_name}_abs.png"),
            normalize=True,
        )
        print(f"[ONN DEBUG] Saved layer '{layer_name}' E field output")

        if isinstance(layer, SensorLayer) and layer.last_field is not None:
            field = layer.last_field[0, 0].detach().cpu()
            energy = out[0, 0].detach().float().cpu()
            zero_fft = torch.fft.fftshift(torch.fft.fft2(torch.sqrt(torch.clamp(energy, min=0))))
            field_fft = torch.fft.fftshift(torch.fft.fft2(field))
            zero_amp = torch.log1p(torch.abs(zero_fft))
            field_amp = torch.log1p(torch.abs(field_fft))
            field_phase = torch.angle(field_fft)
            vutils.save_image(zero_amp.unsqueeze(0), os.path.join(debug_dir, f"{layer_name}_sqrt_energy_fft_amplitude.png"), normalize=True)
            vutils.save_image(field_amp.unsqueeze(0), os.path.join(debug_dir, f"{layer_name}_field_fft_amplitude.png"), normalize=True)
            vutils.save_image(((field_phase + math.pi) / (2 * math.pi)).unsqueeze(0), os.path.join(debug_dir, f"{layer_name}_field_fft_phase.png"), normalize=False)

            h, w = zero_fft.shape
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            radius = torch.round(torch.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)).long()
            bins = torch.arange(radius.max().item() + 1)
            zero_curve = torch.stack([torch.abs(zero_fft)[radius == b].mean() for b in bins])
            field_curve = torch.stack([torch.abs(field_fft)[radius == b].mean() for b in bins])
            difference = torch.abs(zero_curve - field_curve)
            plt.figure(figsize=(8, 5))
            plt.plot(zero_curve.numpy(), label="sqrt(energy), zero phase")
            plt.plot(field_curve.numpy(), label="sensor-aligned complex field")
            plt.plot(difference.numpy(), label="amplitude difference")
            plt.yscale("log")
            plt.xlabel("Frequency radius (FFT pixels)")
            plt.ylabel("Mean |F|")
            plt.title("SensorLayer latent FFT comparison")
            plt.grid(True, which="both", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, f"{layer_name}_fft_comparison_radial.png"), dpi=150)
            plt.close()

        if isinstance(layer, MaterialLayer):
            phase_image = layer.phase.detach().cpu()
            phase_for_stats = phase_image[0] if phase_image.dim() == 4 else phase_image
            phase_chan = phase_for_stats[0] if phase_for_stats.dim() == 3 else phase_for_stats

            dx = phase_chan[:, 1:] - phase_chan[:, :-1]
            dy = phase_chan[1:, :] - phase_chan[:-1, :]
            diffs = torch.abs(torch.cat([dx.flatten(), dy.flatten()], dim=0))

            mean_val = diffs.mean().item()
            median_val = diffs.median().item()
            max_val = diffs.max().item()
            min_val = diffs.min().item()
            std_val = diffs.std().item()
            q25_val = torch.quantile(diffs, 0.25).item()
            q75_val = torch.quantile(diffs, 0.75).item()

            print(f"[ONN DEBUG] {layer_name} phase diff stats:")
            print(
                f"Mean={mean_val:.6f}, max={max_val:.6f}, min={min_val:.6f}"
            )
            print(
                f"q25={q25_val:.6f}, median={median_val:.6f}, q75={q75_val:.6f}, std={std_val:.6f}"
            )

            np_phase = phase_for_stats.squeeze().numpy()
            plt.imshow(np_phase, cmap="viridis")
            plt.colorbar()
            plt.title(f"{layer_name} Phase")
            plt.savefig(os.path.join(debug_dir, f"{layer_name}_phase.png"))
            plt.close()
            print(f"[ONN DEBUG] Saved layer '{layer_name}' phase weight")

            plt.hist(diffs.numpy(), bins=50)
            plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean={mean_val:.4f}")
            plt.axvline(
                median_val, color="green", linestyle="--", label=f"Median={median_val:.4f}"
            )
            plt.axvline(q25_val, color="orange", linestyle="--", label=f"Q25={q25_val:.4f}")
            plt.axvline(q75_val, color="purple", linestyle="--", label=f"Q75={q75_val:.4f}")
            plt.title(f"{layer_name} Phase Diffs Distribution")
            plt.xlabel("Absolute Diff")
            plt.ylabel("Count")
            plt.legend()
            plt.savefig(os.path.join(debug_dir, f"{layer_name}_diffs_hist.png"))
            plt.close()
            print(f"[ONN DEBUG] Saved layer '{layer_name}' diffs distribution")

    print(f"[ONN DEBUG] All layer outputs saved in {debug_dir}")



def shift_tensor_zero_fill(x, dy, dx):
    result = torch.zeros_like(x)
    height, width = x.shape[-2:]
    y0, y1 = max(0, -dy), min(height, height - dy)
    x0, x1 = max(0, -dx), min(width, width - dx)
    if y1 > y0 and x1 > x0:
        result[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx] = x[..., y0:y1, x0:x1]
    return result


def run_model_test(model_factory, checkpoint_loader, dataset_config, settings, model_label):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = bool(settings.get("distributed", False))
    if world_size > 1 and not distributed:
        raise ValueError('Set TESTING_CONFIG["distributed"] = True when using multi-process torchrun.')
    if distributed and world_size == 1:
        print("[TEST] One process available; testing on a single device.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    initialized = False
    try:
        if world_size > 1:
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
            initialized = True
        rank = dist.get_rank() if initialized else 0
        model = model_factory().to(device)
        checkpoint = Path(settings["weight_save_dir"]) / settings["weight_save_name"]
        checkpoint_loader(model, checkpoint, device)
        evaluate_reconstruction(model, checkpoint, dataset_config, settings, device,
                                rank, world_size, model_label)
    finally:
        if initialized:
            dist.destroy_process_group()


def evaluate_reconstruction(model, checkpoint, dataset_config, settings, device,
                            rank=0, world_size=1, model_label="Model"):
    global_batch_size = int(settings.get("batch_size", 32))
    if global_batch_size < world_size:
        raise ValueError("TESTING_CONFIG batch_size must be at least the number of processes.")
    if global_batch_size % world_size != 0:
        raise ValueError("TESTING_CONFIG batch_size must be divisible by the number of processes.")
    per_gpu_batch_size = global_batch_size // world_size
    if rank == 0:
        print(f"[TEST] global batch_size={global_batch_size}, "
              f"GPUs/processes={world_size}, per-process batch_size={per_gpu_batch_size}")
    model.eval()
    shift = tuple(settings.get("input_shift", (0, 0)))
    if len(shift) != 2 or any(int(value) != value for value in shift):
        raise ValueError("input_shift must contain two integer pixel offsets.")
    dy, dx = map(int, shift)
    _, _, loader = get_dataloaders(
        dataset_config, per_gpu_batch_size,
        settings.get("num_workers", 0), False,
    )
    dataset = loader.dataset
    # Strided indices avoid DistributedSampler's padding/duplicate test samples.
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(
        dataset, batch_size=per_gpu_batch_size,
        sampler=indices, num_workers=settings.get("num_workers", 0),
        pin_memory=device.type == "cuda", drop_last=False,
    )
    seed = int(settings.get("fixed_comparison_seed", 42))
    count = min(max(0, int(settings.get("fixed_comparison_count", 10))), len(loader.dataset))
    selected = set(random.Random(seed).sample(range(len(loader.dataset)), count))
    first_n = max(0, int(settings.get("save_first_n_reconstructions", 10)))
    results = Path(settings["results_save_dir"])
    rows, examples = [], {}
    local_imgs, local_recons = [], []
    save_analysis_images = first_n > 0 or settings.get("save_frequency_domain_panels", True) or settings.get("save_mse_ranked_panels", True)
    offset = 0
    with torch.inference_mode():
        for imgs, _ in tqdm(loader, desc=f"Testing {model_label.lower()}", disable=rank != 0):
            imgs = imgs.to(device)
            imgs = shift_tensor_zero_fill(imgs, dy, dx)
            output = model(imgs)
            if output.shape != imgs.shape:
                raise ValueError(f"Prediction/GT shape mismatch: {output.shape} vs {imgs.shape}")
            if settings.get("binary_output_for_metrics", False):
                output = (output >= settings.get("binary_threshold", 0.5)).to(output.dtype)
            mses = F.mse_loss(output, imgs, reduction="none").flatten(1).mean(1).cpu().tolist()
            for i, mse in enumerate(mses):
                index = indices[offset + i]
                psnr = -10 * math.log10(mse) if mse > 0 else float("inf")
                ssim = ssim_pt(imgs[i:i+1], output[i:i+1])
                rows.append((index, mse, psnr, ssim))
                if index in selected or index < first_n:
                    examples[index] = (imgs[i].cpu(), output[i].cpu(), psnr)
            if save_analysis_images:
                local_imgs.append(imgs.cpu())
                local_recons.append(output.cpu())
            offset += len(imgs)
    if world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        # CPU images are retained for the same analyses as test.py.
        dist.gather_object((rows, examples, local_imgs, local_recons, indices), gathered, dst=0)
        if rank != 0:
            return
        rows, examples = [], {}
        for rank_rows, rank_examples, _, _, _ in gathered:
            rows.extend(rank_rows)
            examples.update(rank_examples)
    else:
        gathered = [(rows, examples, local_imgs, local_recons, indices)]
    rows.sort(key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(len(dataset))):
        raise RuntimeError("Testing indices contain missing or duplicate samples.")
    if not rows:
        raise RuntimeError("The testing dataset is empty.")
    results.mkdir(parents=True, exist_ok=True)
    all_imgs = all_recons = None
    if save_analysis_images:
        # Reorder strided GPU partitions into original dataset order so the
        # fixed-seed examples and sample IDs stay identical across GPU counts.
        image_map, recon_map = {}, {}
        for _, _, image_batches, recon_batches, sample_indices in gathered:
            position = 0
            for image_batch, recon_batch in zip(image_batches, recon_batches):
                for image, recon in zip(image_batch, recon_batch):
                    index = sample_indices[position]
                    image_map[index], recon_map[index] = image, recon
                    position += 1
        all_imgs = torch.stack([image_map[i] for i in range(len(dataset))])
        all_recons = torch.stack([recon_map[i] for i in range(len(dataset))])
        del image_map, recon_map
    del gathered, local_imgs, local_recons
    mse = sum(row[1] for row in rows) / len(rows)
    metrics = dict(MSE=mse, PSNR=-10 * math.log10(mse) if mse > 0 else float("inf"),
                   SSIM=sum(row[3] for row in rows) / len(rows), count=len(rows),
                   checkpoint=str(checkpoint), evaluated_model=model_label)
    if model_label == "Teacher":
        metrics["teacher_input"] = (
            "intensity+phase" if getattr(model, "teacher_use_energy", False) else "amplitude+phase"
        )
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (results / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "mse", "psnr", "ssim"])
        writer.writerows(rows)
    for index, (gt, reconstruction, psnr) in examples.items():
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for axis, data, title in zip(axes, [gt, reconstruction, (gt-reconstruction).abs()],
                                     ["GT", model_label, "Absolute error"]):
            axis.imshow(data.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(f"PSNR={psnr:.4f} dB | ({index:06d})")
        fig.tight_layout()
        for group, enabled in [(f"fixed_comparison_seed_{seed}", index in selected),
                                ("first_n", index < first_n)]:
            if enabled:
                folder = results / group
                folder.mkdir(exist_ok=True)
                fig.savefig(folder / f"PSNR_{psnr:.2f}_({index:06d}).png", dpi=180)
        plt.close(fig)
    print(f"{model_label} test MSE: {metrics['MSE']:.8f}, "
          f"PSNR: {metrics['PSNR']:.4f} dB, SSIM: {metrics['SSIM']:.6f}")
    print(f"Results: {results}")
    export_analyses(model, dataset, all_imgs, all_recons, rows, checkpoint.name, settings)


def export_analyses(model, dataset, all_imgs, all_recons, rows, model_name, settings):
    """Run the same analysis functions used by baseline test.py, on rank zero."""
    all_mses = [row[1] for row in rows]
    if settings.get("onn_debug", False):
        with torch.no_grad():
            onn_output_debug(model, settings, dataset)
    if all_imgs is not None:
        first_n = max(0, int(settings.get("save_first_n_reconstructions", 10)))
        if first_n:
            visualize_results(model, all_imgs, all_recons, model_name, first_n,
                              settings, all_mses=all_mses)
        if settings.get("save_frequency_domain_panels", True):
            seed = int(settings.get("fixed_comparison_seed", 42))
            count = min(max(0, int(settings.get("fixed_comparison_count", 10))), len(rows))
            indices = sorted(random.Random(seed).sample(range(len(rows)), count))
            if indices:
                latent_maps = get_latent_map(
                    model, all_imgs[indices].to(next(model.parameters()).device), settings,
                    target_size=all_recons.shape[-2:]
                    if settings.get("latent_resize_to_output", True) else None,
                )
                save_frequency_comparison(
                    all_imgs, all_recons, indices, model_name, settings, "fixed",
                    mse_values=all_mses,
                    latent_images=dict(zip(indices, latent_maps)), model=model,
                    output_subdir=f"fixed_comparison_seed_{seed}",
                )
        if settings.get("save_mse_ranked_panels", True):
            visualize_results_by_mse(all_imgs, all_recons, all_mses,
                                     model_name, settings, model=model)
    if settings.get("save_psnr_distribution_table", True):
        save_psnr_distribution_table(all_mses, model_name, settings)

