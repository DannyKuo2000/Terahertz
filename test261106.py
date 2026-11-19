"""Student-only evaluation for the 261106 teacher/student experiment."""
import json
import math
import os
import random
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as cfg
from dataset import get_dataloaders
from model.Restormer261106 import Restormer
from model.opticalSimulation import ONN


def ssim_pt(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    padding = window_size // 2
    weight = torch.ones(
        (1, 1, window_size, window_size), device=img1.device
    ) / (window_size ** 2)
    mu1 = F.conv2d(img1, weight, padding=padding)
    mu2 = F.conv2d(img2, weight, padding=padding)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, weight, padding=padding) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, weight, padding=padding) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, weight, padding=padding) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def _remove_prefix(state_dict, prefix):
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def load_student_checkpoint(checkpoint, encoder, decoder):
    """Load both legacy split checkpoints and train261106 full checkpoints."""
    # `last_model.pth` / `best_model.pth` may be a raw state_dict rather than
    # a checkpoint dictionary containing model_state_dict.
    if "model_state_dict" not in checkpoint and "encoder_state_dict" not in checkpoint:
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            checkpoint = {"model_state_dict": checkpoint}

    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    elif "model_state_dict" in checkpoint:
        state = _remove_prefix(checkpoint["model_state_dict"], "module.")
        encoder_state = {
            key[len("encoder."):]: value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise KeyError("No encoder.* parameters found in model_state_dict")
        encoder.load_state_dict(encoder_state, strict=True)
    else:
        raise KeyError("Checkpoint has neither encoder_state_dict nor model_state_dict")

    if "student_state_dict" in checkpoint:
        student_state = checkpoint["student_state_dict"]
    elif "decoder_state_dict" in checkpoint:
        student_state = checkpoint["decoder_state_dict"]
    elif "model_state_dict" in checkpoint:
        state = _remove_prefix(checkpoint["model_state_dict"], "module.")
        student_state = {
            key[len("decoder.student."):]: value
            for key, value in state.items()
            if key.startswith("decoder.student.")
        }
        if not student_state:
            raise KeyError("No decoder.student.* parameters found in model_state_dict")
    else:
        raise KeyError("Checkpoint has no student/decoder state")

    decoder.load_student_state_dict(student_state, strict=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ONN(cfg.ENCODER_CONFIG).to(device)
    decoder = Restormer(cfg.RESTORMER_261106_CONFIG).to(device)
    checkpoint_path = os.path.join(
        cfg.TESTING_CONFIG["weight_save_dir"],
        cfg.TESTING_CONFIG.get("weight_save_name", "last_checkpoint.pth"),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_student_checkpoint(checkpoint, encoder, decoder)
    decoder.eval(); encoder.eval()
    _, _, test_loader = get_dataloaders(cfg.DATASET_CONFIG, cfg.TESTING_CONFIG.get("batch_size", 32), cfg.TESTING_CONFIG.get("num_workers", 0), False)
    total_sse = 0.0; total_count = 0; total_ssim = 0.0; sample_count = 0
    per_sample = []; per_sample_ssim = []
    all_imgs = []; all_outputs = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            energy = encoder(imgs)
            if isinstance(energy, (tuple, list)):
                energy = energy[0]
            output = decoder.student(energy)
            output = output[0] if isinstance(output, (tuple, list)) else output
            total_sse += F.mse_loss(output, imgs, reduction="sum").item()
            total_count += imgs.numel()
            batch_mse = F.mse_loss(output, imgs, reduction="none").flatten(1).mean(1)
            per_sample.extend(batch_mse.cpu().tolist())
            for index in range(imgs.size(0)):
                sample_ssim = ssim_pt(imgs[index:index + 1], output[index:index + 1])
                total_ssim += sample_ssim
                per_sample_ssim.append(sample_ssim)
                sample_count += 1
            all_imgs.append(imgs.cpu()); all_outputs.append(output.cpu())
    mse = total_sse / total_count
    psnr = float("inf") if mse <= 0 else -10.0 * torch.log10(torch.tensor(mse)).item()
    ssim = total_ssim / max(1, sample_count)
    print(f"Student test MSE: {mse:.8f}, PSNR: {psnr:.4f} dB, SSIM: {ssim:.6f}")

    all_imgs = torch.cat(all_imgs, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    results_dir = cfg.TESTING_CONFIG["results_save_dir"]
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump({"MSE": mse, "PSNR": psnr, "SSIM": ssim, "count": len(per_sample)}, f, indent=2)

    #! [TEACHER-STUDENT-261106] Added: Fixed-seed student comparison images
    seed = int(cfg.TESTING_CONFIG.get("fixed_comparison_seed", 42))
    count = min(int(cfg.TESTING_CONFIG.get("fixed_comparison_count", 10)), len(all_imgs))
    indices = sorted(random.Random(seed).sample(range(len(all_imgs)), count))
    fixed_dir = os.path.join(results_dir, f"fixed_comparison_seed_{seed}")
    os.makedirs(fixed_dir, exist_ok=True)
    for idx in indices:
        image = all_imgs[idx].squeeze().numpy()
        output = all_outputs[idx].squeeze().numpy()
        sample_mse = per_sample[idx]
        sample_psnr = float("inf") if sample_mse <= 0 else -10 * math.log10(sample_mse)
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(image, cmap="gray"); axes[0].set_title("GT")
        axes[1].imshow(output, cmap="gray"); axes[1].set_title("Student")
        axes[2].imshow(abs(image - output), cmap="magma"); axes[2].set_title("Abs Diff")
        for axis in axes: axis.axis("off")
        fig.suptitle(f"PSNR={sample_psnr:.4f} dB | ({idx:06d})")
        plt.tight_layout()
        plt.savefig(os.path.join(fixed_dir, f"PSNR_{sample_psnr:.2f}dB_({idx:06d}).png"), dpi=180)
        plt.close(fig)

    with open(os.path.join(results_dir, "per_sample_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["index", "mse", "psnr", "ssim"])
        for idx, sample_mse in enumerate(per_sample):
            writer.writerow([
                idx,
                sample_mse,
                float("inf") if sample_mse <= 0 else -10 * math.log10(sample_mse),
                per_sample_ssim[idx],
            ])


if __name__ == "__main__":
    main()
