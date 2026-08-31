import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import (
    AUTOENCODER_CONFIG,
    DATASET_CONFIG,
    ENCODER_CONFIG,
    LATENT_ANALYSIS_V2_CONFIG,
    RESTORMER_CONFIG,
)
from dataset import get_dataloaders
from model.autoencoder import Autoencoder
from model.opticalSimulation import ONN
from model.Restormer260803 import Restormer


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def is_main():
    return get_rank() == 0


distributed = LATENT_ANALYSIS_V2_CONFIG.get(
    "distributed",
    ("LOCAL_RANK" in os.environ or "RANK" in os.environ),
)
local_rank = int(os.environ.get("LOCAL_RANK", 0))

if distributed:
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed testing requires CUDA.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend=LATENT_ANALYSIS_V2_CONFIG.get("backend", "nccl"))
else:
    local_rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device} | Distributed: {distributed}, Local Rank: {local_rank}")


def build_model():
    encoder = ONN(ENCODER_CONFIG).to(device)
    decoder = Restormer(RESTORMER_CONFIG).to(device)
    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        config=AUTOENCODER_CONFIG,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    return model


def get_model_core(model):
    return model.module if hasattr(model, "module") else model


def prepare_test_loader():
    per_gpu_batch = LATENT_ANALYSIS_V2_CONFIG.get("batch_size", 64)
    num_workers = LATENT_ANALYSIS_V2_CONFIG.get("num_workers", 4)

    _, _, base_test_loader = get_dataloaders(
        DATASET_CONFIG,
        per_gpu_batch,
        num_workers=num_workers,
        distributed=distributed,
    )

    test_dataset = base_test_loader.dataset
    if not distributed:
        return base_test_loader, test_dataset

    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=per_gpu_batch,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, test_dataset


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


def select_decoder_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def get_onn_output(model, img):
    encoder = get_model_core(model).encoder

    x = img
    for layer in encoder.layers:
        x = layer(x)
        if isinstance(x, (tuple, list)):
            x = x[0]

    return x


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def build_region_layout(height, width, rows, cols):
    row_bins = np.array_split(np.arange(height), rows)
    col_bins = np.array_split(np.arange(width), cols)

    region_map = torch.full((height, width), -1, dtype=torch.long, device=device)
    specs: List[Dict[str, object]] = []

    region_id = 0
    for r, row_idx in enumerate(row_bins):
        for c, col_idx in enumerate(col_bins):
            row_start = int(row_idx[0])
            row_end = int(row_idx[-1]) + 1
            col_start = int(col_idx[0])
            col_end = int(col_idx[-1]) + 1
            region_map[row_start:row_end, col_start:col_end] = region_id
            specs.append(
                {
                    "region_id": region_id,
                    "row": r,
                    "col": c,
                    "bbox": [row_start, row_end, col_start, col_end],
                    "label": f"r{r:02d}_c{c:02d}",
                    "area": int((row_end - row_start) * (col_end - col_start)),
                }
            )
            region_id += 1

    return region_map, specs


_REGION_CACHE: Dict[Tuple[int, int, int, int, str, Optional[int]], Tuple[torch.Tensor, List[Dict[str, object]]]] = {}


def get_region_layout(height, width, rows, cols):
    key = (height, width, rows, cols, device.type, device.index)
    if key not in _REGION_CACHE:
        _REGION_CACHE[key] = build_region_layout(height, width, rows, cols)
    return _REGION_CACHE[key]


def build_region_specs(config, latent_shape):
    _, _, height, width = latent_shape
    rows = int(config.get("region_rows", 8))
    cols = int(config.get("region_cols", 8))
    region_map, all_specs = get_region_layout(height, width, rows, cols)

    selected_ids = config.get("region_ids_to_analyze", None)
    if selected_ids is None:
        selected_ids = [spec["region_id"] for spec in all_specs]
    else:
        selected_ids = [int(rid) for rid in selected_ids]

    selected = []
    by_id = {int(spec["region_id"]): spec for spec in all_specs}
    for rid in selected_ids:
        if rid not in by_id:
            raise ValueError(
                f"Region id {rid} is out of range for a {rows} x {cols} grid."
            )
        spec = dict(by_id[rid])
        spec["rows"] = rows
        spec["cols"] = cols
        spec["mask_region_ids"] = [rid]
        selected.append(spec)

    return region_map, all_specs, selected


def build_region_mask(region_map, region_ids, latent):
    mask_dtype = latent.real.dtype if torch.is_complex(latent) else latent.dtype
    mask_2d = torch.ones(region_map.shape, device=latent.device, dtype=mask_dtype)
    for rid in region_ids:
        mask_2d = mask_2d.masked_fill(region_map == int(rid), 0.0)
    return mask_2d.unsqueeze(0).unsqueeze(0).expand(latent.size(0), 1, *region_map.shape).clone()


def decode_latent(model, latent):
    decoder = get_model_core(model).decoder
    recons = decoder(latent)
    return select_decoder_output(recons)


def maybe_complex_to_real(tensor):
    return torch.abs(tensor) if torch.is_complex(tensor) else tensor


def save_two_panel(inputs, outputs, save_path, title=None, max_items=5, ylabel_left="Input", ylabel_right="Output"):
    n = min(max_items, len(inputs), len(outputs))
    if n <= 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(inputs[i].squeeze().detach().cpu(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(outputs[i].squeeze().detach().cpu(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel(ylabel_left, fontsize=12)
    axes[1, 0].set_ylabel(ylabel_right, fontsize=12)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_four_panel(gt, base, masked, diff, save_path, title=None, max_items=4):
    n = min(max_items, len(gt), len(base), len(masked), len(diff))
    if n <= 0:
        return

    fig, axes = plt.subplots(4, n, figsize=(3 * n, 10), squeeze=False)
    rows = [gt, base, masked, diff]
    labels = ["GT", "Baseline", "Masked", "|Masked - Base|"]
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


def save_output_diff_map(diff_map, save_path, title=None):
    if diff_map.ndim == 3 and diff_map.size(0) > 1:
        diff_map = diff_map.mean(dim=0)
    elif diff_map.ndim == 3:
        diff_map = diff_map.squeeze(0)

    plt.figure(figsize=(5, 5))
    plt.imshow(diff_map.detach().cpu(), cmap="magma")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_heatmap(grid, save_path, title, metric_name, annotate=False):
    values = np.array(grid, dtype=np.float64)
    masked = np.ma.masked_invalid(values)

    plt.figure(figsize=(8, 7))
    cmap = "coolwarm" if metric_name.startswith("delta_") else "viridis"
    if np.any(np.isfinite(values)) and metric_name.startswith("delta_"):
        finite = values[np.isfinite(values)]
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmin < 0 < vmax:
            norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
        else:
            norm = None
    else:
        norm = None

    im = plt.imshow(masked, cmap=cmap, norm=norm)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Region Col")
    plt.ylabel("Region Row")

    if annotate:
        rows, cols = values.shape
        for r in range(rows):
            for c in range(cols):
                if np.isfinite(values[r, c]):
                    plt.text(
                        c,
                        r,
                        f"{values[r, c]:.3g}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if metric_name.startswith("delta_") and values[r, c] < 0 else "black",
                    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def metric_value_from_record(record, metric_name):
    if metric_name in record:
        return record[metric_name]
    raise ValueError(f"Unsupported heatmap metric: {metric_name}")


def compute_region_record(base_state, region_state, region_spec):
    base_mse = base_state["sse"] / max(base_state["pixel_count"], 1.0)
    base_psnr = compute_psnr(base_mse)
    base_ssim = base_state["ssim_sum"] / max(base_state["ssim_count"], 1.0)

    masked_mse = region_state["sse"] / max(region_state["pixel_count"], 1.0)
    masked_psnr = compute_psnr(masked_mse)
    masked_ssim = region_state["ssim_sum"] / max(region_state["ssim_count"], 1.0)

    delta_mse = masked_mse - base_mse
    delta_psnr = masked_psnr - base_psnr
    delta_ssim = masked_ssim - base_ssim
    relative_mse = delta_mse / (base_mse + 1e-12)

    recon_diff_l1 = region_state["recon_diff_l1_sum"] / max(region_state["sample_count"], 1.0)
    recon_diff_mse = region_state["recon_diff_mse_sum"] / max(region_state["sample_count"], 1.0)

    record = {
        "region_id": int(region_spec["region_id"]),
        "label": str(region_spec["label"]),
        "row": int(region_spec["row"]),
        "col": int(region_spec["col"]),
        "bbox": [int(v) for v in region_spec["bbox"]],
        "area": int(region_spec["area"]),
        "base_mse": base_mse,
        "base_psnr": base_psnr,
        "base_ssim": base_ssim,
        "masked_mse": masked_mse,
        "masked_psnr": masked_psnr,
        "masked_ssim": masked_ssim,
        "delta_mse": delta_mse,
        "delta_psnr": delta_psnr,
        "delta_ssim": delta_ssim,
        "relative_mse": relative_mse,
        "recon_diff_l1": recon_diff_l1,
        "recon_diff_mse": recon_diff_mse,
        "removed_pixels": int(region_spec["area"]),
    }
    return record


def analyze_regions(model, test_loader, save_dir, config):
    ensure_dir(save_dir)
    model.eval()

    preview_ids = config.get("preview_region_ids", None)
    preview_limit = int(config.get("preview_region_limit", 6))
    save_baseline_panel = bool(config.get("save_baseline_panel", True))
    save_region_previews = bool(config.get("save_region_previews", True))
    save_output_diff_maps = bool(config.get("save_output_diff_maps", True))
    save_output_diff_maps_limit = int(config.get("save_output_diff_maps_limit", 6))
    annotate_heatmap = bool(config.get("annotate_heatmap", False))
    heatmap_metrics = list(
        config.get(
            "heatmap_metrics",
            ["delta_psnr", "delta_mse", "relative_mse", "recon_diff_l1"],
        )
    )

    random_preview_cap = set()
    if preview_ids is not None:
        random_preview_cap = {int(x) for x in preview_ids}

    region_map = None
    region_specs = None
    selected_region_specs = None

    base_state = {
        "sse": 0.0,
        "pixel_count": 0.0,
        "ssim_sum": 0.0,
        "ssim_count": 0.0,
        "sample_count": 0.0,
    }

    region_states = {}
    preview_written = set()
    diff_map_written = set()
    baseline_written = False

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Latent region analysis", disable=not is_main()):
            imgs = imgs.to(device)
            latent = get_onn_output(model, imgs)

            if region_map is None:
                _, _, height, width = latent.shape
                region_map, region_specs, selected_region_specs = build_region_specs(config, latent.shape)
                region_rows = int(config.get("region_rows", 8))
                region_cols = int(config.get("region_cols", 8))

                for spec in selected_region_specs:
                    key = int(spec["region_id"])
                    region_states[key] = {
                        "sse": 0.0,
                        "pixel_count": 0.0,
                        "ssim_sum": 0.0,
                        "ssim_count": 0.0,
                        "recon_diff_l1_sum": 0.0,
                        "recon_diff_mse_sum": 0.0,
                        "sample_count": 0.0,
                        "diff_map_sum": None,
                    }

                if is_main():
                    with open(os.path.join(save_dir, "region_specs.json"), "w") as f:
                        json.dump(
                            {
                                "region_rows": region_rows,
                                "region_cols": region_cols,
                                "selected_region_ids": [int(s["region_id"]) for s in selected_region_specs],
                                "all_regions": region_specs,
                            },
                            f,
                            indent=2,
                        )

            base_recon = decode_latent(model, latent)
            base_recon = maybe_complex_to_real(base_recon)

            base_state["sse"] += F.mse_loss(base_recon, imgs, reduction="sum").item()
            base_state["pixel_count"] += imgs.numel()
            base_state["sample_count"] += imgs.size(0)
            for i in range(imgs.size(0)):
                base_state["ssim_sum"] += ssim_pt(imgs[i : i + 1], base_recon[i : i + 1])
                base_state["ssim_count"] += 1

            if is_main() and save_baseline_panel and not baseline_written:
                baseline_dir = os.path.join(save_dir, "baseline")
                ensure_dir(baseline_dir)
                save_two_panel(
                    imgs.cpu(),
                    base_recon.cpu(),
                    os.path.join(baseline_dir, "baseline_panel.png"),
                    title="Baseline reconstruction",
                    max_items=int(config.get("visualize_samples", 5)),
                    ylabel_left="GT",
                    ylabel_right="Baseline",
                )
                baseline_written = True

            output_diff_shape = (1,) + tuple(base_recon.shape[1:])
            for spec in selected_region_specs:
                region_id = int(spec["region_id"])
                region_state = region_states[region_id]
                if region_state["diff_map_sum"] is None:
                    region_state["diff_map_sum"] = torch.zeros(
                        output_diff_shape, device=device, dtype=torch.float64
                    )

            for spec in selected_region_specs:
                region_id = int(spec["region_id"])
                region_state = region_states[region_id]

                masked_latent = latent * build_region_mask(region_map, [region_id], latent)
                masked_recon = decode_latent(model, masked_latent)
                masked_recon = maybe_complex_to_real(masked_recon)

                region_state["sse"] += F.mse_loss(masked_recon, imgs, reduction="sum").item()
                region_state["pixel_count"] += imgs.numel()
                region_state["sample_count"] += imgs.size(0)

                for i in range(imgs.size(0)):
                    region_state["ssim_sum"] += ssim_pt(
                        imgs[i : i + 1],
                        masked_recon[i : i + 1],
                    )
                    region_state["ssim_count"] += 1

                diff = masked_recon - base_recon
                region_state["recon_diff_l1_sum"] += (
                    torch.mean(torch.abs(diff), dim=(1, 2, 3)).sum().item()
                )
                region_state["recon_diff_mse_sum"] += (
                    torch.mean(diff**2, dim=(1, 2, 3)).sum().item()
                )
                region_state["diff_map_sum"] += torch.abs(diff).sum(dim=0, keepdim=True).double()

                if is_main():
                    if save_region_previews and (
                        (preview_ids is not None and region_id in random_preview_cap)
                        or (preview_ids is None and len(preview_written) < preview_limit)
                    ) and region_id not in preview_written:
                        region_dir = os.path.join(save_dir, "region_previews")
                        ensure_dir(region_dir)
                        diff_map = torch.abs(diff).mean(dim=0)
                        save_four_panel(
                            imgs.cpu(),
                            base_recon.cpu(),
                            masked_recon.cpu(),
                            diff_map.cpu().unsqueeze(0),
                            os.path.join(region_dir, f"region_{region_id:03d}_panel.png"),
                            title=f"Region {region_id}",
                            max_items=int(config.get("visualize_samples", 5)),
                        )
                        preview_written.add(region_id)

                    if save_output_diff_maps and region_id not in diff_map_written:
                        if len(diff_map_written) < save_output_diff_maps_limit:
                            diff_dir = os.path.join(save_dir, "output_diff_maps")
                            ensure_dir(diff_dir)
                            diff_map = region_state["diff_map_sum"] / max(region_state["sample_count"], 1.0)
                            save_output_diff_map(
                                diff_map.squeeze(0),
                                os.path.join(diff_dir, f"region_{region_id:03d}_diff_map.png"),
                                title=f"Region {region_id} output diff",
                            )
                            diff_map_written.add(region_id)

    if region_map is None or region_specs is None or selected_region_specs is None:
        raise RuntimeError("No region specs were created. Check the configuration.")

    if distributed:
        base_tensor = torch.tensor(
            [
                base_state["sse"],
                base_state["pixel_count"],
                base_state["ssim_sum"],
                base_state["ssim_count"],
                base_state["sample_count"],
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(base_tensor, op=dist.ReduceOp.SUM)
        (
            base_sse,
            base_pixel_count,
            base_ssim_sum,
            base_ssim_count,
            base_sample_count,
        ) = base_tensor.tolist()
    else:
        base_sse = base_state["sse"]
        base_pixel_count = base_state["pixel_count"]
        base_ssim_sum = base_state["ssim_sum"]
        base_ssim_count = base_state["ssim_count"]
        base_sample_count = base_state["sample_count"]

    base_mse = base_sse / max(base_pixel_count, 1.0)
    base_psnr = compute_psnr(base_mse)
    base_ssim = base_ssim_sum / max(base_ssim_count, 1.0)
    base_record = {
        "MSE": base_mse,
        "PSNR": base_psnr,
        "SSIM": base_ssim,
        "sample_count": base_sample_count,
    }

    results = []
    heatmap_grid_cache = {
        metric: np.full((int(config.get("region_rows", 8)), int(config.get("region_cols", 8))), np.nan, dtype=np.float64)
        for metric in heatmap_metrics
    }

    for spec in selected_region_specs:
        region_id = int(spec["region_id"])
        state = region_states[region_id]

        if distributed:
            region_tensor = torch.tensor(
                [
                    state["sse"],
                    state["pixel_count"],
                    state["ssim_sum"],
                    state["ssim_count"],
                    state["recon_diff_l1_sum"],
                    state["recon_diff_mse_sum"],
                    state["sample_count"],
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(region_tensor, op=dist.ReduceOp.SUM)
            (
                masked_sse,
                masked_pixel_count,
                masked_ssim_sum,
                masked_ssim_count,
                recon_diff_l1_sum,
                recon_diff_mse_sum,
                sample_count,
            ) = region_tensor.tolist()
            diff_map_sum = state["diff_map_sum"].clone()
            dist.all_reduce(diff_map_sum, op=dist.ReduceOp.SUM)
        else:
            masked_sse = state["sse"]
            masked_pixel_count = state["pixel_count"]
            masked_ssim_sum = state["ssim_sum"]
            masked_ssim_count = state["ssim_count"]
            recon_diff_l1_sum = state["recon_diff_l1_sum"]
            recon_diff_mse_sum = state["recon_diff_mse_sum"]
            sample_count = state["sample_count"]
            diff_map_sum = state["diff_map_sum"]

        masked_mse = masked_sse / max(masked_pixel_count, 1.0)
        masked_psnr = compute_psnr(masked_mse)
        masked_ssim = masked_ssim_sum / max(masked_ssim_count, 1.0)

        record = {
            **spec,
            "base_MSE": base_mse,
            "base_PSNR": base_psnr,
            "base_SSIM": base_ssim,
            "masked_MSE": masked_mse,
            "masked_PSNR": masked_psnr,
            "masked_SSIM": masked_ssim,
            "delta_MSE": masked_mse - base_mse,
            "delta_PSNR": masked_psnr - base_psnr,
            "delta_SSIM": masked_ssim - base_ssim,
            "relative_MSE": (masked_mse - base_mse) / (base_mse + 1e-12),
            "recon_diff_L1": recon_diff_l1_sum / max(sample_count, 1.0),
            "recon_diff_MSE": recon_diff_mse_sum / max(sample_count, 1.0),
            "removed_pixels": int(spec["area"]),
        }
        results.append(record)

        row = int(spec["row"])
        col = int(spec["col"])
        for metric in heatmap_metrics:
            key = metric.lower()
            if key == "delta_psnr":
                heatmap_grid_cache[metric][row, col] = record["delta_PSNR"]
            elif key == "delta_mse":
                heatmap_grid_cache[metric][row, col] = record["delta_MSE"]
            elif key == "relative_mse":
                heatmap_grid_cache[metric][row, col] = record["relative_MSE"]
            elif key == "delta_ssim":
                heatmap_grid_cache[metric][row, col] = record["delta_SSIM"]
            elif key == "masked_psnr":
                heatmap_grid_cache[metric][row, col] = record["masked_PSNR"]
            elif key == "masked_mse":
                heatmap_grid_cache[metric][row, col] = record["masked_MSE"]
            elif key == "masked_ssim":
                heatmap_grid_cache[metric][row, col] = record["masked_SSIM"]
            elif key == "recon_diff_l1":
                heatmap_grid_cache[metric][row, col] = record["recon_diff_L1"]
            elif key == "recon_diff_mse":
                heatmap_grid_cache[metric][row, col] = record["recon_diff_MSE"]
            else:
                raise ValueError(f"Unsupported heatmap metric: {metric}")

        if is_main() and bool(config.get("save_region_json", True)):
            region_dir = os.path.join(save_dir, "region_metrics")
            ensure_dir(region_dir)
            with open(os.path.join(region_dir, f"region_{region_id:03d}.json"), "w") as f:
                json.dump(record, f, indent=2)

    if is_main():
        summary = {
            "model_path": os.path.join(
                config["weight_save_dir"],
                config["weight_save_name"],
            ),
            "base_metrics": base_record,
            "results": results,
        }
        with open(os.path.join(save_dir, "latent_analysis_v2_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        for metric, grid in heatmap_grid_cache.items():
            save_heatmap(
                grid,
                os.path.join(save_dir, f"{metric}_heatmap.png"),
                title=f"{metric} heatmap",
                metric_name=metric,
                annotate=annotate_heatmap,
            )

        with open(os.path.join(save_dir, "latent_analysis_v2_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    return base_record, results


if __name__ == "__main__":
    model_path = os.path.join(
        LATENT_ANALYSIS_V2_CONFIG["weight_save_dir"],
        LATENT_ANALYSIS_V2_CONFIG["weight_save_name"],
    )
    model_name = os.path.basename(model_path)

    save_dir = os.path.join(
        LATENT_ANALYSIS_V2_CONFIG["results_save_dir"],
        LATENT_ANALYSIS_V2_CONFIG.get("analysis_name", "region_heatmap"),
        os.path.splitext(model_name)[0],
    )

    model = build_model()
    model = load_model(model, model_path)

    test_loader, _ = prepare_test_loader()

    base_record, results = analyze_regions(
        model,
        test_loader,
        save_dir,
        LATENT_ANALYSIS_V2_CONFIG,
    )

    if is_main():
        print("\n===== Latent Region Analysis V2 =====")
        print(
            f"Baseline | MSE={base_record['MSE']:.6f} | PSNR={base_record['PSNR']:.4f} | SSIM={base_record['SSIM']:.4f}"
        )
        for record in results:
            print(
                f"Region {record['region_id']:03d} ({record['label']}) | "
                f"delta_PSNR={record['delta_PSNR']:.4f} | delta_MSE={record['delta_MSE']:.6f} | "
                f"delta_SSIM={record['delta_SSIM']:.4f} | recon_diff_L1={record['recon_diff_L1']:.6f}"
            )
        print(f"\nAll latent analysis v2 results saved to:\n{save_dir}")

    if distributed and is_distributed():
        dist.barrier()
        dist.destroy_process_group()
