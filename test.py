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
from model.restormer250724 import Restormer


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
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    if is_main():
        print(f"Model loaded from {model_path}")
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
    all_imgs, all_recons = [], []

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

            local_sse += F.mse_loss(recons, imgs, reduction="sum").item()
            local_pixel_count += imgs.numel()

            for i in range(imgs.size(0)):
                local_ssim_sum += ssim_pt(imgs[i : i + 1], recons[i : i + 1])
                local_ssim_count += 1

            if is_main():
                all_imgs.append(imgs.cpu())
                all_recons.append(recons.cpu())

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

    if is_main():
        if all_imgs:
            all_imgs = torch.cat(all_imgs, dim=0)
            all_recons = torch.cat(all_recons, dim=0)
        else:
            all_imgs = None
            all_recons = None

        print(f"Test MSE: {global_mse:.6f}, PSNR: {global_psnr:.4f}, SSIM: {global_ssim:.4f}")
        return all_imgs, all_recons, global_mse, global_psnr, global_ssim

    return None, None, global_mse, global_psnr, global_ssim


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

    all_imgs, all_recons, mse, psnr, ssim = test_model(model)

    if is_main():
        if all_imgs is not None and all_recons is not None:
            visualize_results(all_imgs, all_recons, model_name, num_image=10, config=TESTING_CONFIG)

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
