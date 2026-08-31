import importlib.util
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from matplotlib.colors import TwoSlopeNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model.Restormer260803 import Restormer
from model.autoencoder import Autoencoder
from model.opticalSimulation import ONN
from dataset import get_dataloaders


COMPARE_CONFIG = {
    "baseline_config_path": "config_lists/Baseline_4F_Restormer8_v1_0809_config.py",
    "onn_config_path": "config_lists/ONN_Restormer8_v1_0824_config.py",
    "results_save_dir": "results/latent_analysis_compare",
    "analysis_name": "baseline_vs_onn",
    "distributed": False,
    "num_workers": 0,
    "batch_size": 16,
    "max_eval_samples": 12500,
    # occupancy analysis settings
    "occupancy_threshold": 0.0,  # > 0 uses non-zero pixel ratio; 0.0 uses sum/HW average intensity
    "occupancy_bins": 10,
    "save_scatter": True,
    "save_binned_curve": True,
    "save_correlation": True,

    # optional region masking analysis kept as secondary output
    "region_rows": 8,
    "region_cols": 8,
    "region_ids_to_analyze": None,
    "area_subset_sizes": [1, 2, 4, 8, 16, 32],
    "area_trials_per_size": 12,
    "area_seed": 1234,
    "save_region_panels": False,
    "save_region_limit": 6,
    "save_heatmaps": False,
    "save_area_curve": False,
    "save_full_metrics": True,
}


def load_module_from_path(module_path):
    module_path = os.path.abspath(module_path)
    module_name = os.path.splitext(os.path.basename(module_path))[0] + "_" + str(abs(hash(module_path)))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline_cfg = load_module_from_path(COMPARE_CONFIG["baseline_config_path"])
onn_cfg = load_module_from_path(COMPARE_CONFIG["onn_config_path"])


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def is_main():
    return get_rank() == 0


distributed = COMPARE_CONFIG.get(
    "distributed",
    ("LOCAL_RANK" in os.environ or "RANK" in os.environ),
)
local_rank = int(os.environ.get("LOCAL_RANK", 0))

if distributed:
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed evaluation requires CUDA.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
else:
    local_rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device} | Distributed: {distributed}, Local Rank: {local_rank}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def select_reconstruction(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def maybe_real(tensor):
    return torch.abs(tensor) if torch.is_complex(tensor) else tensor


def build_model_from_bundle(bundle):
    encoder = ONN(bundle["ENCODER_CONFIG"]).to(device)
    decoder = Restormer(bundle["RESTORMER_CONFIG"]).to(device)
    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        config=bundle["AUTOENCODER_CONFIG"],
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    if hasattr(model, "module"):
        missing, unexpected = model.module.load_state_dict(state_dict, strict=False)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.eval()
    if is_main():
        print(f"Model loaded from {model_path}")
        if missing:
            print(f"[load_model] Missing keys ignored: {missing}")
        if unexpected:
            print(f"[load_model] Unexpected keys ignored: {unexpected}")
    return model


def prepare_test_loader(dataset_config, batch_size, num_workers):
    _, _, base_test_loader = get_dataloaders(
        dataset_config,
        batch_size,
        num_workers=num_workers,
        distributed=distributed,
    )

    test_dataset = base_test_loader.dataset
    if not distributed:
        return base_test_loader, test_dataset

    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, test_dataset


def get_encoder_output(model, img):
    encoder = model.module.encoder if hasattr(model, "module") else model.encoder
    x = img
    for layer in encoder.layers:
        x = layer(x)
        if isinstance(x, (tuple, list)):
            x = x[0]
    return x


def decode_latent(model, latent):
    decoder = model.module.decoder if hasattr(model, "module") else model.decoder
    recons = decoder(latent)
    return select_reconstruction(recons)


def build_region_map(height, width, rows, cols):
    row_bins = np.array_split(np.arange(height), rows)
    col_bins = np.array_split(np.arange(width), cols)
    region_map = torch.full((height, width), -1, dtype=torch.long, device=device)
    specs = []
    region_id = 0
    for r, row_idx in enumerate(row_bins):
        for c, col_idx in enumerate(col_bins):
            r0 = int(row_idx[0])
            r1 = int(row_idx[-1]) + 1
            c0 = int(col_idx[0])
            c1 = int(col_idx[-1]) + 1
            region_map[r0:r1, c0:c1] = region_id
            specs.append(
                {
                    "region_id": region_id,
                    "row": r,
                    "col": c,
                    "bbox": [r0, r1, c0, c1],
                    "area": int((r1 - r0) * (c1 - c0)),
                    "label": f"r{r:02d}_c{c:02d}",
                }
            )
            region_id += 1
    return region_map, specs


def build_region_mask(region_map, region_ids, latent):
    mask_dtype = latent.real.dtype if torch.is_complex(latent) else latent.dtype
    mask_2d = torch.ones(region_map.shape, device=latent.device, dtype=mask_dtype)
    for rid in region_ids:
        mask_2d = mask_2d.masked_fill(region_map == int(rid), 0.0)
    return mask_2d.unsqueeze(0).unsqueeze(0).expand(latent.size(0), 1, *region_map.shape).clone()


def summarize_metrics(gt, recon):
    recon = maybe_real(recon)
    mse = F.mse_loss(recon, gt, reduction="mean").item()
    ssim = ssim_pt(gt, recon)
    psnr = compute_psnr(mse)
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}


def add_sum_state(state, gt, recon):
    recon = maybe_real(recon)
    state["sse"] += F.mse_loss(recon, gt, reduction="sum").item()
    state["pixel_count"] += gt.numel()
    for i in range(gt.size(0)):
        state["ssim_sum"] += ssim_pt(gt[i : i + 1], recon[i : i + 1])
        state["ssim_count"] += 1
    state["sample_count"] += gt.size(0)


def finalize_state(state):
    mse = state["sse"] / max(state["pixel_count"], 1.0)
    psnr = compute_psnr(mse)
    ssim = state["ssim_sum"] / max(state["ssim_count"], 1.0)
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim, "samples": state["sample_count"]}


def init_state():
    return {"sse": 0.0, "pixel_count": 0.0, "ssim_sum": 0.0, "ssim_count": 0.0, "sample_count": 0.0}


def clone_metric_grid(rows, cols):
    return np.full((rows, cols), np.nan, dtype=np.float64)


def save_heatmap(grid, save_path, title, metric_name, annotate=False):
    ensure_dir(os.path.dirname(save_path))
    values = np.array(grid, dtype=np.float64)
    masked = np.ma.masked_invalid(values)
    plt.figure(figsize=(8, 7))
    cmap = "coolwarm" if "gain" in metric_name or metric_name.startswith("delta_") else "viridis"
    norm = None
    finite = values[np.isfinite(values)]
    if finite.size and ("gain" in metric_name or metric_name.startswith("delta_")):
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmin < 0 < vmax:
            norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    im = plt.imshow(masked, cmap=cmap, norm=norm)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Region Col")
    plt.ylabel("Region Row")
    if annotate:
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                if np.isfinite(values[r, c]):
                    plt.text(c, r, f"{values[r, c]:.3g}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_curve(results, save_path, title, x_key, y_key, yerr_key=None):
    ensure_dir(os.path.dirname(save_path))
    results = sorted(results, key=lambda x: x[x_key])
    xs = [r[x_key] for r in results]
    ys = [r[y_key] for r in results]
    plt.figure(figsize=(8, 5))
    if yerr_key is not None:
        yerr = [r[yerr_key] for r in results]
        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3)
    else:
        plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_panel(gt, base_recon, onn_recon, save_path, title=None, max_items=4):
    n = min(max_items, len(gt), len(base_recon), len(onn_recon))
    if n <= 0:
        return
    diff = torch.abs(onn_recon - base_recon)
    fig, axes = plt.subplots(4, n, figsize=(3 * n, 10), squeeze=False)
    rows = [gt, base_recon, onn_recon, diff]
    labels = ["GT", "Baseline", "ONN", "|ONN - Baseline|"]
    cmaps = ["gray", "gray", "gray", "magma"]
    for r in range(4):
        for i in range(n):
            axes[r, i].imshow(rows[r][i].squeeze().detach().cpu(), cmap=cmaps[r])
            axes[r, i].axis("off")
        axes[r, 0].set_ylabel(labels[r], fontsize=12)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def compute_occupancy_ratio(gt, threshold=0.0):
    gt = gt.detach()
    if threshold > 0:
        occupied = (gt > threshold).float().sum(dim=(1, 2, 3))
    else:
        occupied = gt.sum(dim=(1, 2, 3))
    hw = gt.size(-1) * gt.size(-2)
    return (occupied / max(hw, 1)).cpu().tolist()


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def spearman_corr(x, y):
    return pearson_corr(rankdata(x), rankdata(y))


def bin_by_x(records, x_key, y_keys, num_bins=10):
    records = sorted(records, key=lambda r: r[x_key])
    if not records:
        return []

    xs = np.array([r[x_key] for r in records], dtype=np.float64)
    bins = np.linspace(xs.min(), xs.max(), num_bins + 1)
    if np.allclose(bins[0], bins[-1]):
        bins = np.array([xs.min(), xs.max() + 1e-12], dtype=np.float64)

    out = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        if i == len(bins) - 2:
            bucket = [r for r in records if left <= r[x_key] <= right]
        else:
            bucket = [r for r in records if left <= r[x_key] < right]
        if not bucket:
            continue
        row = {
            "bin_left": float(left),
            "bin_right": float(right),
            "bin_center": float((left + right) / 2.0),
            "count": len(bucket),
        }
        for key in y_keys:
            vals = np.array([r[key] for r in bucket], dtype=np.float64)
            row[f"{key}_mean"] = float(vals.mean())
            row[f"{key}_std"] = float(vals.std())
        row[f"{x_key}_mean"] = float(np.mean([r[x_key] for r in bucket]))
        row[f"{x_key}_std"] = float(np.std([r[x_key] for r in bucket]))
        out.append(row)
    return out


def save_scatter(records, save_path, title, x_key, y_key, color_key=None):
    ensure_dir(os.path.dirname(save_path))
    xs = np.array([r[x_key] for r in records], dtype=np.float64)
    ys = np.array([r[y_key] for r in records], dtype=np.float64)
    plt.figure(figsize=(7, 5))
    if color_key is not None:
        cs = np.array([r[color_key] for r in records], dtype=np.float64)
        sc = plt.scatter(xs, ys, c=cs, cmap="viridis", s=24, alpha=0.8)
        plt.colorbar(sc, label=color_key)
    else:
        plt.scatter(xs, ys, s=24, alpha=0.8)
    if len(xs) >= 2:
        coef = np.polyfit(xs, ys, 1)
        xs_line = np.linspace(xs.min(), xs.max(), 200)
        plt.plot(xs_line, np.polyval(coef, xs_line), color="red", linewidth=2, label="linear fit")
        plt.legend()
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def validate_configs():
    if baseline_cfg.DATASET_CONFIG != onn_cfg.DATASET_CONFIG:
        raise ValueError("Baseline and ONN DATASET_CONFIG are different. Please align them before comparison.")


def load_bundle(module, name):
    return {
        "name": name,
        "DATASET_CONFIG": module.DATASET_CONFIG,
        "ENCODER_CONFIG": module.ENCODER_CONFIG,
        "RESTORMER_CONFIG": module.RESTORMER_CONFIG,
        "AUTOENCODER_CONFIG": module.AUTOENCODER_CONFIG,
        "TESTING_CONFIG": module.TESTING_CONFIG,
    }


baseline_bundle = load_bundle(baseline_cfg, "baseline")
onn_bundle = load_bundle(onn_cfg, "onn")
validate_configs()


def compare_models():
    dataset_config = baseline_bundle["DATASET_CONFIG"]
    batch_size = COMPARE_CONFIG.get("batch_size") or min(
        baseline_bundle["TESTING_CONFIG"].get("batch_size", 8),
        onn_bundle["TESTING_CONFIG"].get("batch_size", 8),
    )
    num_workers = COMPARE_CONFIG.get("num_workers", 0)
    test_loader, _ = prepare_test_loader(dataset_config, batch_size, num_workers)

    baseline_model = build_model_from_bundle(baseline_bundle)
    onn_model = build_model_from_bundle(onn_bundle)

    baseline_path = os.path.join(
        baseline_bundle["TESTING_CONFIG"]["weight_save_dir"],
        baseline_bundle["TESTING_CONFIG"]["weight_save_name"],
    )
    onn_path = os.path.join(
        onn_bundle["TESTING_CONFIG"]["weight_save_dir"],
        onn_bundle["TESTING_CONFIG"]["weight_save_name"],
    )

    baseline_model = load_model(baseline_model, baseline_path)
    onn_model = load_model(onn_model, onn_path)

    region_rows = int(COMPARE_CONFIG.get("region_rows", 8))
    region_cols = int(COMPARE_CONFIG.get("region_cols", 8))
    region_ids_to_analyze = COMPARE_CONFIG.get("region_ids_to_analyze", None)
    save_region_limit = int(COMPARE_CONFIG.get("save_region_limit", 6))
    area_subset_sizes = list(COMPARE_CONFIG.get("area_subset_sizes", [1, 2, 4, 8]))
    area_trials_per_size = int(COMPARE_CONFIG.get("area_trials_per_size", 12))
    area_seed = COMPARE_CONFIG.get("area_seed", 1234)
    rng = random.Random(area_seed)

    base_out_dir = os.path.join(
        COMPARE_CONFIG["results_save_dir"],
        COMPARE_CONFIG.get("analysis_name", "baseline_vs_onn"),
    )
    ensure_dir(base_out_dir)

    base_states = {
        "baseline": init_state(),
        "onn": init_state(),
    }

    single_region_states = {}
    area_sweep_states = {}
    region_map = None
    region_specs = None
    selected_specs = None
    preview_written = set()
    processed_samples = 0

    with torch.inference_mode():
        for imgs, _ in tqdm(test_loader, desc="Comparing models", disable=not is_main()):
            imgs = imgs.to(device)

            baseline_latent = get_encoder_output(baseline_model, imgs)
            onn_latent = get_encoder_output(onn_model, imgs)

            if region_map is None:
                if baseline_latent.shape != onn_latent.shape:
                    raise RuntimeError(
                        f"Baseline and ONN latent shapes differ: {baseline_latent.shape} vs {onn_latent.shape}"
                    )
                _, _, h, w = baseline_latent.shape
                region_map, region_specs = build_region_map(h, w, region_rows, region_cols)
                if region_ids_to_analyze is None:
                    selected_specs = region_specs
                else:
                    selected_set = {int(x) for x in region_ids_to_analyze}
                    selected_specs = [spec for spec in region_specs if spec["region_id"] in selected_set]

                for spec in selected_specs:
                    rid = int(spec["region_id"])
                    single_region_states[rid] = {
                        "baseline": init_state(),
                        "onn": init_state(),
                        "base_diff_l1_sum": 0.0,
                        "base_diff_mse_sum": 0.0,
                        "sample_count": 0.0,
                    }

                for k in area_subset_sizes:
                    k = int(k)
                    if k <= 0:
                        continue
                    if k > len(selected_specs):
                        continue
                    area_sweep_states[k] = []

                if is_main():
                    with open(os.path.join(base_out_dir, "region_specs.json"), "w") as f:
                        json.dump(
                            {
                                "region_rows": region_rows,
                                "region_cols": region_cols,
                                "all_regions": region_specs,
                                "selected_region_ids": [int(s["region_id"]) for s in selected_specs],
                                "area_subset_sizes": area_subset_sizes,
                                "area_trials_per_size": area_trials_per_size,
                            },
                            f,
                            indent=2,
                        )

            baseline_full = decode_latent(baseline_model, baseline_latent)
            onn_full = decode_latent(onn_model, onn_latent)
            baseline_full = maybe_real(baseline_full)
            onn_full = maybe_real(onn_full)

            add_sum_state(base_states["baseline"], imgs, baseline_full)
            add_sum_state(base_states["onn"], imgs, onn_full)

            if is_main() and COMPARE_CONFIG.get("save_region_panels", True) and not preview_written:
                baseline_dir = os.path.join(base_out_dir, "baseline_full")
                onn_dir = os.path.join(base_out_dir, "onn_full")
                ensure_dir(baseline_dir)
                ensure_dir(onn_dir)
                save_panel(
                    imgs.cpu(),
                    baseline_full.cpu(),
                    onn_full.cpu(),
                    os.path.join(base_out_dir, "full_comparison_panel.png"),
                    title="Full reconstruction comparison",
                    max_items=int(COMPARE_CONFIG.get("save_region_limit", 6)),
                )
                preview_written.add("full")

            for spec in selected_specs:
                rid = int(spec["region_id"])
                state = single_region_states[rid]
                mask = build_region_mask(region_map, [rid], baseline_latent)

                baseline_masked = decode_latent(baseline_model, baseline_latent * mask)
                onn_masked = decode_latent(onn_model, onn_latent * mask)
                baseline_masked = maybe_real(baseline_masked)
                onn_masked = maybe_real(onn_masked)

                add_sum_state(state["baseline"], imgs, baseline_masked)
                add_sum_state(state["onn"], imgs, onn_masked)
                state["base_diff_l1_sum"] += torch.mean(torch.abs(baseline_masked - onn_masked), dim=(1, 2, 3)).sum().item()
                state["base_diff_mse_sum"] += torch.mean((baseline_masked - onn_masked) ** 2, dim=(1, 2, 3)).sum().item()

                if is_main() and COMPARE_CONFIG.get("save_region_panels", True) and len(preview_written) < int(COMPARE_CONFIG.get("save_region_limit", 6)) + 1 and rid not in preview_written:
                    region_dir = os.path.join(base_out_dir, "region_panels")
                    ensure_dir(region_dir)
                    save_panel(
                        imgs.cpu(),
                        baseline_masked.cpu(),
                        onn_masked.cpu(),
                        os.path.join(region_dir, f"region_{rid:03d}_panel.png"),
                        title=f"Region {rid}",
                        max_items=int(COMPARE_CONFIG.get("save_region_limit", 6)),
                    )
                    preview_written.add(rid)

            for k, records in area_sweep_states.items():
                if not records and k > 0:
                    pass
                for _ in range(area_trials_per_size):
                    subset = rng.sample([int(spec["region_id"]) for spec in selected_specs], k)
                    subset = tuple(sorted(subset))
                    if len(records) >= area_trials_per_size:
                        break
                    mask = build_region_mask(region_map, subset, baseline_latent)
                    baseline_masked = decode_latent(baseline_model, baseline_latent * mask)
                    onn_masked = decode_latent(onn_model, onn_latent * mask)
                    baseline_masked = maybe_real(baseline_masked)
                    onn_masked = maybe_real(onn_masked)

                    base_full_mse = F.mse_loss(baseline_full, imgs, reduction="mean").item()
                    onn_full_mse = F.mse_loss(onn_full, imgs, reduction="mean").item()
                    base_mask_mse = F.mse_loss(baseline_masked, imgs, reduction="mean").item()
                    onn_mask_mse = F.mse_loss(onn_masked, imgs, reduction="mean").item()

                    base_penalty = base_mask_mse - base_full_mse
                    onn_penalty = onn_mask_mse - onn_full_mse
                    abs_gain = base_mask_mse - onn_mask_mse
                    penalty_gain = base_penalty - onn_penalty
                    subset_area_pixels = int(sum(next(spec for spec in selected_specs if int(spec["region_id"]) == rid)["area"] for rid in subset))

                    records.append(
                        {
                            "subset_size_regions": k,
                            "removed_pixels": subset_area_pixels,
                            "baseline_full_mse": base_full_mse,
                            "onn_full_mse": onn_full_mse,
                            "baseline_mask_mse": base_mask_mse,
                            "onn_mask_mse": onn_mask_mse,
                            "absolute_gain_mse": abs_gain,
                            "penalty_gain_mse": penalty_gain,
                            "baseline_penalty_mse": base_penalty,
                            "onn_penalty_mse": onn_penalty,
                        }
                    )
                    area_sweep_states[k] = records
                    if len(records) >= area_trials_per_size:
                        break

            processed_samples += imgs.size(0)
            if COMPARE_CONFIG.get("max_eval_samples") is not None and processed_samples >= int(COMPARE_CONFIG["max_eval_samples"]):
                break

    baseline_full_metrics = finalize_state(base_states["baseline"])
    onn_full_metrics = finalize_state(base_states["onn"])

    heatmap_rows = region_rows
    heatmap_cols = region_cols
    heatmap_grids = {
        "absolute_gain_mse": clone_metric_grid(heatmap_rows, heatmap_cols),
        "penalty_gain_mse": clone_metric_grid(heatmap_rows, heatmap_cols),
        "baseline_penalty_mse": clone_metric_grid(heatmap_rows, heatmap_cols),
        "onn_penalty_mse": clone_metric_grid(heatmap_rows, heatmap_cols),
    }

    region_records = []
    for spec in selected_specs:
        rid = int(spec["region_id"])
        state = single_region_states[rid]
        baseline_metrics = finalize_state(state["baseline"])
        onn_metrics = finalize_state(state["onn"])

        absolute_gain_mse = baseline_metrics["MSE"] - onn_metrics["MSE"]
        baseline_penalty_mse = baseline_metrics["MSE"] - baseline_full_metrics["MSE"]
        onn_penalty_mse = onn_metrics["MSE"] - onn_full_metrics["MSE"]
        penalty_gain_mse = baseline_penalty_mse - onn_penalty_mse
        absolute_gain_psnr = onn_metrics["PSNR"] - baseline_metrics["PSNR"]
        penalty_gain_psnr = (onn_metrics["PSNR"] - onn_full_metrics["PSNR"]) - (
            baseline_metrics["PSNR"] - baseline_full_metrics["PSNR"]
        )
        absolute_gain_ssim = onn_metrics["SSIM"] - baseline_metrics["SSIM"]
        penalty_gain_ssim = (onn_metrics["SSIM"] - onn_full_metrics["SSIM"]) - (
            baseline_metrics["SSIM"] - baseline_full_metrics["SSIM"]
        )

        record = {
            "region_id": rid,
            "row": int(spec["row"]),
            "col": int(spec["col"]),
            "bbox": [int(v) for v in spec["bbox"]],
            "area_pixels": int(spec["area"]),
            "baseline_masked_MSE": baseline_metrics["MSE"],
            "baseline_masked_PSNR": baseline_metrics["PSNR"],
            "baseline_masked_SSIM": baseline_metrics["SSIM"],
            "onn_masked_MSE": onn_metrics["MSE"],
            "onn_masked_PSNR": onn_metrics["PSNR"],
            "onn_masked_SSIM": onn_metrics["SSIM"],
            "baseline_penalty_MSE": baseline_penalty_mse,
            "onn_penalty_MSE": onn_penalty_mse,
            "absolute_gain_MSE": absolute_gain_mse,
            "penalty_gain_MSE": penalty_gain_mse,
            "absolute_gain_PSNR": absolute_gain_psnr,
            "penalty_gain_PSNR": penalty_gain_psnr,
            "absolute_gain_SSIM": absolute_gain_ssim,
            "penalty_gain_SSIM": penalty_gain_ssim,
        }
        region_records.append(record)

        heatmap_grids["absolute_gain_mse"][record["row"], record["col"]] = record["absolute_gain_MSE"]
        heatmap_grids["penalty_gain_mse"][record["row"], record["col"]] = record["penalty_gain_MSE"]
        heatmap_grids["baseline_penalty_mse"][record["row"], record["col"]] = record["baseline_penalty_MSE"]
        heatmap_grids["onn_penalty_mse"][record["row"], record["col"]] = record["onn_penalty_MSE"]

    area_curve_records = []
    for k, records in area_sweep_states.items():
        if not records:
            continue
        values = {
            key: [r[key] for r in records]
            for key in ["absolute_gain_mse", "penalty_gain_mse", "baseline_mask_mse", "onn_mask_mse"]
        }
        area_curve_records.append(
            {
                "subset_size_regions": k,
                "removed_pixels_mean": float(np.mean([r["removed_pixels"] for r in records])),
                "removed_pixels_std": float(np.std([r["removed_pixels"] for r in records])),
                "num_trials": len(records),
                "absolute_gain_mse_mean": float(np.mean(values["absolute_gain_mse"])),
                "absolute_gain_mse_std": float(np.std(values["absolute_gain_mse"])),
                "penalty_gain_mse_mean": float(np.mean(values["penalty_gain_mse"])),
                "penalty_gain_mse_std": float(np.std(values["penalty_gain_mse"])),
                "baseline_mask_mse_mean": float(np.mean(values["baseline_mask_mse"])),
                "onn_mask_mse_mean": float(np.mean(values["onn_mask_mse"])),
            }
        )

    if is_main():
        ensure_dir(base_out_dir)
        summary = {
            "baseline_config_path": COMPARE_CONFIG["baseline_config_path"],
            "onn_config_path": COMPARE_CONFIG["onn_config_path"],
            "baseline_checkpoint": baseline_path,
            "onn_checkpoint": onn_path,
            "baseline_full_metrics": baseline_full_metrics,
            "onn_full_metrics": onn_full_metrics,
            "region_records": region_records,
            "area_curve_records": area_curve_records,
        }
        if COMPARE_CONFIG.get("save_full_metrics", True):
            with open(os.path.join(base_out_dir, "comparison_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

        with open(os.path.join(base_out_dir, "region_records.json"), "w") as f:
            json.dump(region_records, f, indent=2)
        with open(os.path.join(base_out_dir, "area_curve_records.json"), "w") as f:
            json.dump(area_curve_records, f, indent=2)

        if COMPARE_CONFIG.get("save_heatmaps", True):
            save_heatmap(
                heatmap_grids["penalty_gain_mse"],
                os.path.join(base_out_dir, "penalty_gain_mse_heatmap.png"),
                title="Penalty gain MSE heatmap (baseline - ONN mask penalty)",
                metric_name="penalty_gain_mse",
                annotate=False,
            )
            save_heatmap(
                heatmap_grids["absolute_gain_mse"],
                os.path.join(base_out_dir, "absolute_gain_mse_heatmap.png"),
                title="Absolute gain MSE heatmap (baseline - ONN)",
                metric_name="absolute_gain_mse",
                annotate=False,
            )

        if COMPARE_CONFIG.get("save_area_curve", True) and area_curve_records:
            save_curve(
                area_curve_records,
                os.path.join(base_out_dir, "area_curve_penalty_gain_mse.png"),
                title="Penalty gain vs masked region count",
                x_key="removed_pixels_mean",
                y_key="penalty_gain_mse_mean",
                yerr_key="penalty_gain_mse_std",
            )
            save_curve(
                area_curve_records,
                os.path.join(base_out_dir, "area_curve_absolute_gain_mse.png"),
                title="Absolute gain vs masked region count",
                x_key="removed_pixels_mean",
                y_key="absolute_gain_mse_mean",
                yerr_key="absolute_gain_mse_std",
            )

        print("\n===== Baseline vs ONN Comparison =====")
        print(
            f"Baseline full | MSE={baseline_full_metrics['MSE']:.6f} | PSNR={baseline_full_metrics['PSNR']:.4f} | SSIM={baseline_full_metrics['SSIM']:.4f}"
        )
        print(
            f"ONN full      | MSE={onn_full_metrics['MSE']:.6f} | PSNR={onn_full_metrics['PSNR']:.4f} | SSIM={onn_full_metrics['SSIM']:.4f}"
        )
        print(f"Saved results to: {base_out_dir}")

    return baseline_full_metrics, onn_full_metrics, region_records, area_curve_records


def compare_by_occupancy():
    dataset_config = baseline_bundle["DATASET_CONFIG"]
    batch_size = COMPARE_CONFIG.get("batch_size") or min(
        baseline_bundle["TESTING_CONFIG"].get("batch_size", 8),
        onn_bundle["TESTING_CONFIG"].get("batch_size", 8),
    )
    num_workers = COMPARE_CONFIG.get("num_workers", 0)
    test_loader, _ = prepare_test_loader(dataset_config, batch_size, num_workers)

    baseline_model = build_model_from_bundle(baseline_bundle)
    onn_model = build_model_from_bundle(onn_bundle)

    baseline_path = os.path.join(
        baseline_bundle["TESTING_CONFIG"]["weight_save_dir"],
        baseline_bundle["TESTING_CONFIG"]["weight_save_name"],
    )
    onn_path = os.path.join(
        onn_bundle["TESTING_CONFIG"]["weight_save_dir"],
        onn_bundle["TESTING_CONFIG"]["weight_save_name"],
    )
    baseline_model = load_model(baseline_model, baseline_path)
    onn_model = load_model(onn_model, onn_path)

    base_out_dir = os.path.join(
        COMPARE_CONFIG["results_save_dir"],
        COMPARE_CONFIG.get("analysis_name", "baseline_vs_onn"),
        "occupancy",
    )
    ensure_dir(base_out_dir)

    occupancy_threshold = float(COMPARE_CONFIG.get("occupancy_threshold", 0.0))
    num_bins = int(COMPARE_CONFIG.get("occupancy_bins", 10))
    max_eval_samples = COMPARE_CONFIG.get("max_eval_samples", None)
    sample_records: List[Dict[str, float]] = []
    base_states = {
        "baseline": init_state(),
        "onn": init_state(),
    }
    processed_samples = 0
    preview_written = False

    with torch.inference_mode():
        for imgs, _ in tqdm(test_loader, desc="Occupancy compare", disable=not is_main()):
            imgs = imgs.to(device)
            baseline_full = maybe_real(decode_latent(baseline_model, get_encoder_output(baseline_model, imgs)))
            onn_full = maybe_real(decode_latent(onn_model, get_encoder_output(onn_model, imgs)))

            add_sum_state(base_states["baseline"], imgs, baseline_full)
            add_sum_state(base_states["onn"], imgs, onn_full)

            occupancy_ratios = compute_occupancy_ratio(imgs, threshold=occupancy_threshold)
            baseline_mse = torch.mean((baseline_full - imgs) ** 2, dim=(1, 2, 3)).detach().cpu().tolist()
            onn_mse = torch.mean((onn_full - imgs) ** 2, dim=(1, 2, 3)).detach().cpu().tolist()
            baseline_psnr = [compute_psnr(mse) for mse in baseline_mse]
            onn_psnr = [compute_psnr(mse) for mse in onn_mse]

            for i in range(imgs.size(0)):
                sample_records.append(
                    {
                        "sample_index": processed_samples + i,
                        "gt_occupancy_ratio": float(occupancy_ratios[i]),
                        "baseline_mse": float(baseline_mse[i]),
                        "onn_mse": float(onn_mse[i]),
                        "improvement_mse": float(baseline_mse[i] - onn_mse[i]),
                        "baseline_psnr": float(baseline_psnr[i]),
                        "onn_psnr": float(onn_psnr[i]),
                        "improvement_psnr": float(onn_psnr[i] - baseline_psnr[i]),
                        "baseline_ssim": float(ssim_pt(imgs[i : i + 1], baseline_full[i : i + 1])),
                        "onn_ssim": float(ssim_pt(imgs[i : i + 1], onn_full[i : i + 1])),
                    }
                )
                sample_records[-1]["improvement_ssim"] = sample_records[-1]["onn_ssim"] - sample_records[-1]["baseline_ssim"]

            if is_main() and COMPARE_CONFIG.get("save_region_panels", False) and not preview_written:
                save_panel(
                    imgs.cpu(),
                    baseline_full.cpu(),
                    onn_full.cpu(),
                    os.path.join(base_out_dir, "full_comparison_panel.png"),
                    title="Full reconstruction comparison",
                    max_items=int(COMPARE_CONFIG.get("save_region_limit", 6)),
                )
                preview_written = True

            processed_samples += imgs.size(0)
            if max_eval_samples is not None and processed_samples >= int(max_eval_samples):
                break

    baseline_full_metrics = finalize_state(base_states["baseline"])
    onn_full_metrics = finalize_state(base_states["onn"])

    if is_main():
        correlation = {
            "pearson_improvement_mse": pearson_corr(
                [r["gt_occupancy_ratio"] for r in sample_records],
                [r["improvement_mse"] for r in sample_records],
            ),
            "spearman_improvement_mse": spearman_corr(
                [r["gt_occupancy_ratio"] for r in sample_records],
                [r["improvement_mse"] for r in sample_records],
            ),
            "pearson_improvement_psnr": pearson_corr(
                [r["gt_occupancy_ratio"] for r in sample_records],
                [r["improvement_psnr"] for r in sample_records],
            ),
            "pearson_improvement_ssim": pearson_corr(
                [r["gt_occupancy_ratio"] for r in sample_records],
                [r["improvement_ssim"] for r in sample_records],
            ),
        }

        binned = bin_by_x(
            sample_records,
            x_key="gt_occupancy_ratio",
            y_keys=["improvement_mse", "improvement_psnr", "improvement_ssim", "onn_mse", "baseline_mse"],
            num_bins=num_bins,
        )

        summary = {
            "baseline_config_path": COMPARE_CONFIG["baseline_config_path"],
            "onn_config_path": COMPARE_CONFIG["onn_config_path"],
            "baseline_checkpoint": baseline_path,
            "onn_checkpoint": onn_path,
            "occupancy_threshold": occupancy_threshold,
            "baseline_full_metrics": baseline_full_metrics,
            "onn_full_metrics": onn_full_metrics,
            "correlation": correlation,
            "sample_count": len(sample_records),
            "binned_records": binned,
        }
        if COMPARE_CONFIG.get("save_full_metrics", True):
            with open(os.path.join(base_out_dir, "occupancy_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

        with open(os.path.join(base_out_dir, "occupancy_samples.json"), "w") as f:
            json.dump(sample_records, f, indent=2)
        with open(os.path.join(base_out_dir, "occupancy_binned.json"), "w") as f:
            json.dump(binned, f, indent=2)

        if COMPARE_CONFIG.get("save_scatter", True):
            save_scatter(
                sample_records,
                os.path.join(base_out_dir, "occupancy_vs_improvement_mse.png"),
                title="GT occupancy ratio vs ONN improvement (MSE)",
                x_key="gt_occupancy_ratio",
                y_key="improvement_mse",
            )
            save_scatter(
                sample_records,
                os.path.join(base_out_dir, "occupancy_vs_improvement_psnr.png"),
                title="GT occupancy ratio vs ONN improvement (PSNR)",
                x_key="gt_occupancy_ratio",
                y_key="improvement_psnr",
            )

        if COMPARE_CONFIG.get("save_binned_curve", True):
            save_curve(
                binned,
                os.path.join(base_out_dir, "occupancy_binned_improvement_mse.png"),
                title="Binned occupancy vs improvement (MSE)",
                x_key="gt_occupancy_ratio_mean",
                y_key="improvement_mse_mean",
                yerr_key="improvement_mse_std",
            )
            save_curve(
                binned,
                os.path.join(base_out_dir, "occupancy_binned_improvement_psnr.png"),
                title="Binned occupancy vs improvement (PSNR)",
                x_key="gt_occupancy_ratio_mean",
                y_key="improvement_psnr_mean",
                yerr_key="improvement_psnr_std",
            )

        if COMPARE_CONFIG.get("save_correlation", True):
            print("\n===== Occupancy vs Improvement =====")
            print(
                f"Baseline full | MSE={baseline_full_metrics['MSE']:.6f} | PSNR={baseline_full_metrics['PSNR']:.4f} | SSIM={baseline_full_metrics['SSIM']:.4f}"
            )
            print(
                f"ONN full      | MSE={onn_full_metrics['MSE']:.6f} | PSNR={onn_full_metrics['PSNR']:.4f} | SSIM={onn_full_metrics['SSIM']:.4f}"
            )
            print(
                f"Pearson corr(occupancy, improvement_mse) = {correlation['pearson_improvement_mse']:.4f}"
            )
            print(
                f"Spearman corr(occupancy, improvement_mse) = {correlation['spearman_improvement_mse']:.4f}"
            )
            print(f"Saved results to: {base_out_dir}")

    return baseline_full_metrics, onn_full_metrics, sample_records, binned


if __name__ == "__main__":
    baseline_full_metrics, onn_full_metrics, sample_records, binned = compare_by_occupancy()

    if distributed and is_distributed():
        dist.barrier()
        dist.destroy_process_group()
