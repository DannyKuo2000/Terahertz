import csv
import json
import math
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import (
    AUTOENCODER_CONFIG,
    DATASET_CONFIG,
    ENCODER_CONFIG,
    RESTORMER_CONFIG,
    TESTING_CONFIG,
)
from dataset import get_dataloaders
from model.autoencoder import Autoencoder
from model.opticalSimulation import MaterialLayer, ONN
# from model.restormer250724 import Restormer
from model.Restormer260803 import Restormer

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def is_main():
    return get_rank() == 0


distributed = TESTING_CONFIG.get(
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
        dist.init_process_group(backend=TESTING_CONFIG.get("backend", "nccl"))
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


def prepare_test_loader():
    per_gpu_batch = TESTING_CONFIG.get("batch_size", 64)
    num_workers = TESTING_CONFIG.get("num_workers", 4)

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


test_loader, test_dataset = prepare_test_loader()


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


def select_reconstruction(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def test_model(model):
    model.eval()
    local_imgs, local_recons = [], []
    local_mses = []

    local_sse = 0.0
    local_pixel_count = 0
    local_ssim_sum = 0.0
    local_ssim_count = 0

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Testing", disable=not is_main()):
            imgs = imgs.to(device)
            outputs = model(imgs)
            recons = select_reconstruction(outputs)

            if torch.is_complex(recons):
                recons = torch.abs(recons)

            batch_mse_sum = F.mse_loss(recons, imgs, reduction="none").flatten(1).mean(dim=1)
            local_mses.extend(batch_mse_sum.detach().cpu().tolist())

            local_sse += F.mse_loss(recons, imgs, reduction="sum").item()
            local_pixel_count += imgs.numel()

            for i in range(imgs.size(0)):
                local_ssim_sum += ssim_pt(imgs[i : i + 1], recons[i : i + 1])
                local_ssim_count += 1

            local_imgs.append(imgs.cpu())
            local_recons.append(recons.cpu())

    if distributed:
        stats = torch.tensor(
            [local_sse, float(local_pixel_count), local_ssim_sum, float(local_ssim_count)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        global_mse = (stats[0] / stats[1]).item()
        global_ssim = (stats[2] / stats[3]).item()
    else:
        global_mse = local_sse / local_pixel_count
        global_ssim = local_ssim_sum / max(1, local_ssim_count)

    global_psnr = compute_psnr(global_mse)

    if distributed:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(
            gathered,
            {
                "imgs": local_imgs,
                "recons": local_recons,
                "mses": local_mses,
            },
        )
    else:
        gathered = [
            {
                "imgs": local_imgs,
                "recons": local_recons,
                "mses": local_mses,
            }
        ]

    if is_main():
        all_imgs = []
        all_recons = []
        all_mses = []
        for payload in gathered:
            all_imgs.extend(payload["imgs"])
            all_recons.extend(payload["recons"])
            all_mses.extend(payload["mses"])

        if all_imgs:
            all_imgs = torch.cat(all_imgs, dim=0)
            all_recons = torch.cat(all_recons, dim=0)
        else:
            all_imgs = None
            all_recons = None
            all_mses = []

        print(f"Test MSE: {global_mse:.6f}, PSNR: {global_psnr:.4f}, SSIM: {global_ssim:.4f}")
        return all_imgs, all_recons, all_mses, global_mse, global_psnr, global_ssim

    return None, None, None, global_mse, global_psnr, global_ssim


def visualize_results(all_imgs, all_recons, model_name, num_image, config):
    os.makedirs(config["results_save_dir"], exist_ok=True)
    num_image = min(num_image, len(all_imgs), len(all_recons))

    imgs = all_imgs[:num_image]
    recons = all_recons[:num_image]

    fig, axes = plt.subplots(2, num_image, figsize=(num_image * 2, 4))
    for i in range(num_image):
        axes[0, i].imshow(imgs[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(config["results_save_dir"], f"{model_name}_image.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Visualization saved at {save_path}")


def visualize_results_by_mse(all_imgs, all_recons, all_mses, model_name, config):
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
        fig, axes = plt.subplots(3, cols, figsize=(max(6, cols * 2.2), 6))
        if cols == 1:
            axes = axes.reshape(3, 1)

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
            axes[1, col].imshow(recon, cmap="gray")
            axes[2, col].imshow(diff, cmap="magma")
            for row in range(3):
                axes[row, col].axis("off")

        axes[0, 0].set_ylabel("GT", fontsize=11)
        axes[1, 0].set_ylabel("Recon", fontsize=11)
        axes[2, 0].set_ylabel("Abs Diff", fontsize=11)
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(save_root, filename)
        plt.savefig(save_path, dpi=180)
        plt.close(fig)
        print(f"MSE-ranked visualization saved at {save_path}")

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


def onn_output_debug(model):
    net = model.module if hasattr(model, "module") else model

    debug_dir = os.path.join(TESTING_CONFIG["results_save_dir"], "ONN_debug")
    os.makedirs(debug_dir, exist_ok=True)

    split_method = TESTING_CONFIG.get("ONN_input_select", "fix")
    seed = TESTING_CONFIG.get("seed", None)

    if split_method == "fix":
        idx = TESTING_CONFIG["ONN_input_idx"]
    else:
        if seed is not None:
            random.seed(seed)
        idx = random.randint(0, len(test_dataset) - 1)

    img, _ = test_dataset[idx]
    img = img.unsqueeze(0).to(device)

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


if __name__ == "__main__":
    model_path = os.path.join(TESTING_CONFIG["weight_save_dir"], TESTING_CONFIG["weight_save_name"])
    model_name = os.path.basename(model_path)

    model = build_model()
    model = load_model(model, model_path)

    if TESTING_CONFIG.get("onn_debug", False) and is_main():
        onn_output_debug(model)

    all_imgs, all_recons, all_mses, mse, psnr, ssim = test_model(model)

    if is_main():
        if all_imgs is not None and all_recons is not None:
            visualize_results(
                all_imgs,
                all_recons,
                model_name,
                num_image=TESTING_CONFIG.get("save_first_n_reconstructions", 10),
                config=TESTING_CONFIG,
            )

            if TESTING_CONFIG.get("save_mse_ranked_panels", True):
                visualize_results_by_mse(
                    all_imgs,
                    all_recons,
                    all_mses,
                    model_name,
                    TESTING_CONFIG,
                )

            if TESTING_CONFIG.get("save_psnr_distribution_table", True):
                save_psnr_distribution_table(all_mses, model_name, TESTING_CONFIG)

        os.makedirs(TESTING_CONFIG["results_save_dir"], exist_ok=True)
        metrics_path = os.path.join(
            TESTING_CONFIG["results_save_dir"],
            f"{model_name}{TESTING_CONFIG['results_save_name_suffix']}",
        )
        with open(metrics_path, "w") as f:
            json.dump({"MSE": mse, "PSNR": psnr, "SSIM": ssim}, f, indent=2)
        print(f"Metrics saved at {metrics_path}")

    if distributed and is_distributed():
        dist.barrier()
        dist.destroy_process_group()
