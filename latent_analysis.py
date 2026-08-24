import json
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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
    LATENT_ANALYSIS_CONFIG,
    RESTORMER_CONFIG,
)
from dataset import get_dataloaders
from model.autoencoder import Autoencoder
from model.opticalSimulation import ONN
from model.restormer250724 import Restormer


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def is_main():
    return get_rank() == 0


distributed = LATENT_ANALYSIS_CONFIG.get(
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
        dist.init_process_group(backend=LATENT_ANALYSIS_CONFIG.get("backend", "nccl"))
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
    per_gpu_batch = LATENT_ANALYSIS_CONFIG.get("batch_size", 64)
    num_workers = LATENT_ANALYSIS_CONFIG.get("num_workers", 4)

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


def format_ratio_label(value):
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def make_generator():
    if device.type == "cuda":
        return torch.Generator(device=device)
    return torch.Generator()


_RADIAL_CACHE: Dict[Tuple[int, int, str, Optional[int]], Dict[str, torch.Tensor]] = {}
_RING_CACHE: Dict[Tuple[int, int, int, str, Optional[int]], Dict[str, torch.Tensor]] = {}


def get_radial_geometry(height, width):
    key = (height, width, device.type, device.index)
    if key not in _RADIAL_CACHE:
        y = torch.linspace(-1.0, 1.0, height, device=device)
        x = torch.linspace(-1.0, 1.0, width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx**2 + yy**2)
        _RADIAL_CACHE[key] = {
            "radius": radius,
            "sorted_indices": torch.argsort(radius.reshape(-1)),
        }
    return _RADIAL_CACHE[key]


def get_ring_geometry(height, width, ring_num):
    key = (height, width, ring_num, device.type, device.index)
    if key not in _RING_CACHE:
        radial = get_radial_geometry(height, width)["radius"].reshape(-1)
        bounds = torch.quantile(
            radial,
            torch.linspace(0.0, 1.0, ring_num + 1, device=device),
        )
        ring_ids = torch.bucketize(radial, bounds[1:-1], right=False).view(height, width)
        _RING_CACHE[key] = {
            "ring_ids": ring_ids,
            "ring_num": torch.tensor(ring_num, device=device),
        }
    return _RING_CACHE[key]


def build_analysis_specs(config, latent_shape):
    _, _, height, width = latent_shape
    modes = config.get(
        "analysis_modes",
        ["mask_outer", "mask_center", "ring_by_ring", "random_mask"],
    )
    mask_ratios = list(config.get("mask_ratios", [0.0, 0.25, 0.5, 0.75, 1.0]))
    ring_num = int(config.get("ring_num", 8))
    ring_steps = list(config.get("ring_mask_steps", list(range(1, ring_num + 1))))
    ring_patterns = config.get("ring_mask_patterns", None)

    specs = []
    for mode in modes:
        if mode in {"mask_outer", "mask_center", "random_mask"}:
            for ratio in mask_ratios:
                ratio = float(ratio)
                specs.append(
                    {
                        "mode": mode,
                        "label": f"{mode}_r{format_ratio_label(ratio)}",
                        "ratio": ratio,
                    }
                )
            continue

        if mode == "ring_by_ring":
            if ring_patterns is not None:
                for pattern_idx, pattern in enumerate(ring_patterns):
                    if isinstance(pattern, int):
                        masked_rings = list(range(max(0, ring_num - int(pattern)), ring_num))
                        pattern_label = f"k{int(pattern):02d}"
                    else:
                        masked_rings = sorted({int(r) for r in pattern})
                        pattern_label = "-".join(str(r) for r in masked_rings)
                    specs.append(
                        {
                            "mode": mode,
                            "label": f"ring_by_ring_{pattern_idx:02d}_{pattern_label}",
                            "ring_num": ring_num,
                            "masked_rings": masked_rings,
                        }
                    )
            else:
                for step in ring_steps:
                    step = int(step)
                    step = max(0, min(step, ring_num))
                    masked_rings = list(range(max(0, ring_num - step), ring_num))
                    specs.append(
                        {
                            "mode": mode,
                            "label": f"ring_by_ring_k{step:02d}",
                            "ring_num": ring_num,
                            "masked_rings": masked_rings,
                        }
                    )
            continue

        raise ValueError(f"Unsupported analysis mode: {mode}")

    if not specs:
        raise ValueError("No latent analysis specifications were generated.")

    return specs


def build_mask_for_spec(latent, spec, random_generator=None):
    if latent.ndim != 4:
        raise ValueError(f"Expected latent tensor with shape [B, C, H, W], got {latent.shape}")

    batch_size, _, height, width = latent.shape
    num_pixels = height * width
    mask_dtype = latent.real.dtype if torch.is_complex(latent) else latent.dtype
    mask = torch.ones((batch_size, 1, height, width), device=latent.device, dtype=mask_dtype)

    mode = spec["mode"]

    if mode in {"mask_outer", "mask_center"}:
        ratio = float(spec["ratio"])
        num_mask = int(round(ratio * num_pixels))
        if num_mask <= 0:
            return mask
        if num_mask >= num_pixels:
            return torch.zeros_like(mask)

        radial = get_radial_geometry(height, width)
        sorted_indices = radial["sorted_indices"]
        if mode == "mask_outer":
            mask_indices = sorted_indices[-num_mask:]
        else:
            mask_indices = sorted_indices[:num_mask]

        mask_flat = mask.view(batch_size, -1)
        mask_flat[:, mask_indices] = 0.0
        return mask_flat.view(batch_size, 1, height, width)

    if mode == "random_mask":
        ratio = float(spec["ratio"])
        num_mask = int(round(ratio * num_pixels))
        if num_mask <= 0:
            return mask
        if num_mask >= num_pixels:
            return torch.zeros_like(mask)

        scores = torch.rand(
            (batch_size, num_pixels),
            device=latent.device,
            generator=random_generator,
        )
        mask_flat = torch.ones((batch_size, num_pixels), device=latent.device, dtype=mask_dtype)
        mask_indices = scores.topk(num_mask, dim=1, largest=True).indices
        mask_flat.scatter_(1, mask_indices, 0.0)
        return mask_flat.view(batch_size, 1, height, width)

    if mode == "ring_by_ring":
        ring_num = int(spec["ring_num"])
        ring_ids = get_ring_geometry(height, width, ring_num)["ring_ids"]
        mask_2d = torch.ones((height, width), device=latent.device, dtype=mask_dtype)
        for ring_id in spec.get("masked_rings", []):
            mask_2d = mask_2d.masked_fill(ring_ids == int(ring_id), 0.0)
        return mask_2d.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width).clone()

    raise ValueError(f"Unsupported analysis mode: {mode}")


def decode_latent(model, latent):
    decoder = get_model_core(model).decoder
    recons = decoder(latent)
    return select_decoder_output(recons)


def save_reconstruction_panel(inputs, recons, save_path, title=None, max_items=5):
    n = min(max_items, len(inputs), len(recons))
    if n <= 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(inputs[i].squeeze().detach().cpu(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i].squeeze().detach().cpu(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_mask_preview(mask, save_path, title=None):
    preview = mask[0, 0].detach().cpu()
    plt.figure(figsize=(5, 5))
    plt.imshow(preview, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_metric_curve(results, save_path, title):
    if not results:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ordered = sorted(results, key=lambda item: item["masked_fraction"])
    xs = [item["masked_fraction"] for item in ordered]
    mse = [item["MSE"] for item in ordered]
    psnr = [item["PSNR"] for item in ordered]
    ssim = [item["SSIM"] for item in ordered]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(xs, mse, marker="o")
    axes[0].set_xlabel("Masked Fraction")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, psnr, marker="o")
    axes[1].set_xlabel("Masked Fraction")
    axes[1].set_ylabel("PSNR")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs, ssim, marker="o")
    axes[2].set_xlabel("Masked Fraction")
    axes[2].set_ylabel("SSIM")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_removed_pixels_curve(results, save_path, title):
    if not results:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ordered = sorted(results, key=lambda item: item["removed_pixels"])
    xs = [item["removed_pixels"] for item in ordered]
    mse = [item["MSE"] for item in ordered]
    psnr = [item["PSNR"] for item in ordered]
    ssim = [item["SSIM"] for item in ordered]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(xs, mse, marker="o")
    axes[0].set_xlabel("Removed latent pixels")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, psnr, marker="o")
    axes[1].set_xlabel("Removed latent pixels")
    axes[1].set_ylabel("PSNR")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs, ssim, marker="o")
    axes[2].set_xlabel("Removed latent pixels")
    axes[2].set_ylabel("SSIM")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def summarize_spec(spec):
    summary = {
        "mode": spec["mode"],
        "label": spec["label"],
    }
    if "ratio" in spec:
        summary["ratio"] = float(spec["ratio"])
    if "masked_rings" in spec:
        summary["masked_rings"] = [int(r) for r in spec["masked_rings"]]
    if "ring_num" in spec:
        summary["ring_num"] = int(spec["ring_num"])
    return summary


def analyze_latent_modes(model, test_loader, save_dir, config):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    random_generator = make_generator()
    seed = config.get("mask_seed", None)
    if seed is not None:
        random_generator.manual_seed(int(seed))

    specs = None
    metric_states = {}
    panel_written = set()

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Latent analysis", disable=not is_main()):
            imgs = imgs.to(device)
            latent = get_onn_output(model, imgs)

            if specs is None:
                specs = build_analysis_specs(config, latent.shape)
                for spec in specs:
                    key = spec["label"]
                    metric_states[key] = {
                        "sse": 0.0,
                        "pixel_count": 0.0,
                        "ssim_sum": 0.0,
                        "ssim_count": 0.0,
                        "masked_fraction_sum": 0.0,
                        "removed_pixel_sum": 0.0,
                        "sample_count": 0.0,
                    }

                if is_main():
                    with open(os.path.join(save_dir, "analysis_specs.json"), "w") as f:
                        json.dump([summarize_spec(spec) for spec in specs], f, indent=2)

            for spec in specs:
                key = spec["label"]
                masked_latent, mask = apply_mask_and_return(latent, spec, random_generator)
                recons = decode_latent(model, masked_latent)

                if torch.is_complex(recons):
                    recons = torch.abs(recons)

                state = metric_states[key]
                state["sse"] += F.mse_loss(recons, imgs, reduction="sum").item()
                state["pixel_count"] += imgs.numel()
                state["masked_fraction_sum"] += (1.0 - mask.float().mean().item()) * imgs.size(0)
                state["removed_pixel_sum"] += (1.0 - mask.float()).sum(dim=(1, 2, 3)).sum().item()
                state["sample_count"] += imgs.size(0)

                for i in range(imgs.size(0)):
                    state["ssim_sum"] += ssim_pt(imgs[i : i + 1], recons[i : i + 1])
                    state["ssim_count"] += 1

                if is_main() and key not in panel_written:
                    mode_dir = os.path.join(save_dir, spec["mode"])
                    sample_dir = os.path.join(mode_dir, "samples")
                    mask_dir = os.path.join(mode_dir, "mask_previews")
                    os.makedirs(sample_dir, exist_ok=True)
                    os.makedirs(mask_dir, exist_ok=True)

                    if config.get("save_sample_panels", True):
                        save_reconstruction_panel(
                            imgs.cpu(),
                            recons.cpu(),
                            os.path.join(sample_dir, f"{key}_panel.png"),
                            title=key,
                            max_items=config.get("visualize_samples", 5),
                        )
                    if config.get("save_mask_previews", True):
                        save_mask_preview(
                            mask.cpu(),
                            os.path.join(mask_dir, f"{key}_mask.png"),
                            title=key,
                        )
                    panel_written.add(key)

    if specs is None:
        raise RuntimeError("No analysis specs were created. Check the configuration.")

    results = []
    for spec in specs:
        key = spec["label"]
        state = metric_states[key]

        if distributed:
            stats = torch.tensor(
                [
                    state["sse"],
                    state["pixel_count"],
                    state["ssim_sum"],
                    state["ssim_count"],
                    state["masked_fraction_sum"],
                    state["removed_pixel_sum"],
                    state["sample_count"],
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            (
                sse,
                pixel_count,
                ssim_sum,
                ssim_count,
                masked_fraction_sum,
                removed_pixel_sum,
                sample_count,
            ) = stats.tolist()
        else:
            sse = state["sse"]
            pixel_count = state["pixel_count"]
            ssim_sum = state["ssim_sum"]
            ssim_count = state["ssim_count"]
            masked_fraction_sum = state["masked_fraction_sum"]
            removed_pixel_sum = state["removed_pixel_sum"]
            sample_count = state["sample_count"]

        mse = sse / max(pixel_count, 1.0)
        psnr = compute_psnr(mse)
        ssim = ssim_sum / max(ssim_count, 1.0)
        masked_fraction = masked_fraction_sum / max(sample_count, 1.0)
        removed_pixels = removed_pixel_sum / max(sample_count, 1.0)

        record = {
            **summarize_spec(spec),
            "masked_fraction": masked_fraction,
            "removed_pixels": removed_pixels,
            "MSE": mse,
            "PSNR": psnr,
            "SSIM": ssim,
        }
        results.append(record)

        if is_main():
            mode_dir = os.path.join(save_dir, spec["mode"])
            metric_dir = os.path.join(mode_dir, "metrics")
            os.makedirs(metric_dir, exist_ok=True)
            with open(os.path.join(metric_dir, f"{key}.json"), "w") as f:
                json.dump(record, f, indent=2)

    if is_main():
        summary_path = os.path.join(save_dir, "latent_analysis_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "model_path": os.path.join(
                        config["weight_save_dir"],
                        config["weight_save_name"],
                    ),
                    "results": results,
                },
                f,
                indent=2,
            )

        if config.get("save_metric_curves", True):
            by_mode: Dict[str, List[Dict[str, float]]] = {}
            for record in results:
                by_mode.setdefault(record["mode"], []).append(record)
            for mode, mode_results in by_mode.items():
                save_metric_curve(
                    mode_results,
                    os.path.join(save_dir, mode, f"{mode}_curves.png"),
                    title=mode,
                )
                save_removed_pixels_curve(
                    mode_results,
                    os.path.join(save_dir, mode, f"{mode}_removed_pixels_curves.png"),
                    title=f"{mode} - removed pixels",
                )

    return results


def apply_mask_and_return(latent, spec, random_generator):
    mask = build_mask_for_spec(latent, spec, random_generator)
    return latent * mask, mask


if __name__ == "__main__":
    model_path = os.path.join(
        LATENT_ANALYSIS_CONFIG["weight_save_dir"],
        LATENT_ANALYSIS_CONFIG["weight_save_name"],
    )
    model_name = os.path.basename(model_path)

    save_dir = os.path.join(
        LATENT_ANALYSIS_CONFIG["results_save_dir"],
        "latent_analysis",
        os.path.splitext(model_name)[0],
    )

    model = build_model()
    model = load_model(model, model_path)

    test_loader, _ = prepare_test_loader()

    results = analyze_latent_modes(
        model,
        test_loader,
        save_dir,
        LATENT_ANALYSIS_CONFIG,
    )

    if is_main():
        print("\n===== Latent Mask Analysis =====")
        for record in results:
            print(
                f"{record['mode']:<14} | {record['label']:<24} | "
                f"mask={record['masked_fraction']:.4f} | removed={record['removed_pixels']:.2f} | "
                f"MSE={record['MSE']:.6f} | PSNR={record['PSNR']:.4f} | SSIM={record['SSIM']:.4f}"
            )
        print(f"\nAll latent analysis results saved to:\n{save_dir}")

    if distributed and is_distributed():
        dist.barrier()
        dist.destroy_process_group()
