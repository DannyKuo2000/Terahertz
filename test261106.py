"""Student-only evaluation for the 261106 teacher/student experiment."""
import json
import math
import os
import random
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import config as cfg
from dataset import get_dataloaders
from model.Restormer261106 import Restormer
from model.opticalSimulation import ONN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ONN(cfg.ENCODER_CONFIG).to(device)
    decoder = Restormer(cfg.RESTORMER_261106_CONFIG).to(device)
    checkpoint_path = os.path.join(
        cfg.TESTING_CONFIG["weight_save_dir"],
        cfg.TESTING_CONFIG.get("weight_save_name", "last_checkpoint.pth"),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_student_state_dict(checkpoint.get("student_state_dict", checkpoint["decoder_state_dict"]))
    decoder.eval(); encoder.eval()
    _, _, test_loader = get_dataloaders(cfg.DATASET_CONFIG, cfg.TESTING_CONFIG.get("batch_size", 32), cfg.TESTING_CONFIG.get("num_workers", 0), False)
    total_sse = 0.0; total_count = 0; per_sample = []
    all_imgs = []; all_outputs = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            energy = encoder(imgs)
            if isinstance(energy, (tuple, list)):
                energy = energy[0]
            output = decoder.student(energy)
            output = output[0] if isinstance(output, (tuple, list)) else output
            total_sse += F.mse_loss(output, imgs, reduction="sum").item()
            total_count += imgs.numel()
            per_sample.extend(
                F.mse_loss(output, imgs, reduction="none").flatten(1).mean(1).cpu().tolist()
            )
            all_imgs.append(imgs.cpu()); all_outputs.append(output.cpu())
    mse = total_sse / total_count
    psnr = float("inf") if mse <= 0 else -10.0 * torch.log10(torch.tensor(mse)).item()
    print(f"Student test MSE: {mse:.8f}, PSNR: {psnr:.4f} dB")

    all_imgs = torch.cat(all_imgs, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    results_dir = cfg.TESTING_CONFIG["results_save_dir"]
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump({"MSE": mse, "PSNR": psnr, "count": len(per_sample)}, f, indent=2)

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
        writer = csv.writer(f); writer.writerow(["index", "mse", "psnr"])
        for idx, sample_mse in enumerate(per_sample):
            writer.writerow([idx, sample_mse, float("inf") if sample_mse <= 0 else -10 * math.log10(sample_mse)])


if __name__ == "__main__":
    main()
