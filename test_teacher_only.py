"""Evaluate teacher-only checkpoints using config_teacher_only.py.

Supports raw best_model/last_model weights and full training checkpoints.
Teacher input matches train_teacher_only.py: amplitude, sin(phase), cos(phase).
"""
import csv
import json
import math
import os
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_teacher_only as cfg
from dataset import get_dataloaders
from model.RestormerOfficial import Restormer
from model.opticalSimulation import ONN
from test261106 import ssim_pt


class TeacherOnlyAutoencoder(nn.Module):
    """Matches the training model's state_dict names and forward computation."""
    def __init__(self):
        super().__init__()
        settings = dict(cfg.TEACHER_ONLY_CONFIG)
        settings.update(
            dim=settings.get("teacher_dim", 16),
            heads=settings.get("teacher_heads", [1, 2, 4, 8]),
            num_blocks=settings.get("teacher_num_blocks", [1, 2, 2, 2]),
            num_refinement_blocks=settings.get("teacher_num_refinement_blocks", 2),
            inp_channels=3,
        )
        self.encoder = ONN(cfg.ENCODER_CONFIG)
        self.teacher = Restormer(settings)
        self.teacher_output_projection = nn.Conv2d(3, settings["out_channels"], 1)

    def forward(self, imgs):
        self.encoder(imgs)
        field = next((layer.last_field for layer in self.encoder.layers
                      if getattr(layer, "last_field", None) is not None), None)
        if field is None:
            raise RuntimeError("SensorLayer.last_field is empty after ONN forward.")
        phase = torch.angle(field)
        teacher_input = torch.cat(
            [field.abs(), phase.sin(), phase.cos()], dim=1
        )
        return self.teacher_output_projection(self.teacher(teacher_input))


def load_teacher_checkpoint(model, path, device):
    # Full checkpoints contain Python/NumPy RNG objects; only load trusted files.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = [
        f"{key}: checkpoint={tuple(state[key].shape)}, model={tuple(expected[key].shape)}"
        for key in sorted(set(expected) & set(state))
        if state[key].shape != expected[key].shape
    ]
    if missing or unexpected or mismatched:
        details = [f"Checkpoint/model mismatch: {path}",
                   f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}, shape mismatches: {len(mismatched)}"]
        for label, values in [("Missing", missing), ("Unexpected", unexpected),
                              ("Shape mismatch", mismatched)]:
            if values:
                details.append(label + ":\n  " + "\n  ".join(values[:8]))
        details.append(
            "Use a train_teacher_only.py checkpoint and the same architecture settings "
            "in config_teacher_only.py as during training. No weights were skipped."
        )
        raise RuntimeError("\n".join(details))
    model.load_state_dict(state, strict=True)


def main():
    settings = cfg.TESTING_CONFIG
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
        evaluate(settings, device, rank, world_size)
    finally:
        if initialized:
            dist.destroy_process_group()


def evaluate(settings, device, rank=0, world_size=1):
    global_batch_size = int(settings.get("batch_size", 32))
    if global_batch_size < world_size:
        raise ValueError("TESTING_CONFIG batch_size must be at least the number of processes.")
    if global_batch_size % world_size != 0:
        raise ValueError("TESTING_CONFIG batch_size must be divisible by the number of processes.")
    per_gpu_batch_size = global_batch_size // world_size
    if rank == 0:
        print(f"[TEST] global batch_size={global_batch_size}, "
              f"GPUs/processes={world_size}, per-process batch_size={per_gpu_batch_size}")
    model = TeacherOnlyAutoencoder().to(device)
    checkpoint = Path(settings["weight_save_dir"]) / settings["weight_save_name"]
    load_teacher_checkpoint(model, checkpoint, device)
    model.eval()
    shift = tuple(settings.get("input_shift", (0, 0)))
    if shift != (0, 0):
        raise ValueError("Teacher-only testing currently requires input_shift=(0, 0).")
    _, _, loader = get_dataloaders(
        cfg.DATASET_CONFIG, per_gpu_batch_size,
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
    offset = 0
    with torch.inference_mode():
        for imgs, _ in tqdm(loader, desc="Testing teacher", disable=rank != 0):
            imgs = imgs.to(device)
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
            offset += len(imgs)
    if world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        # Transfer only scalar metrics and selected CPU image examples.
        dist.gather_object((rows, examples), gathered, dst=0)
        if rank != 0:
            return
        rows, examples = [], {}
        for rank_rows, rank_examples in gathered:
            rows.extend(rank_rows)
            examples.update(rank_examples)
    rows.sort(key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(len(dataset))):
        raise RuntimeError("Testing indices contain missing or duplicate samples.")
    if not rows:
        raise RuntimeError("The testing dataset is empty.")
    results.mkdir(parents=True, exist_ok=True)
    mse = sum(row[1] for row in rows) / len(rows)
    metrics = dict(MSE=mse, PSNR=-10 * math.log10(mse) if mse > 0 else float("inf"),
                   SSIM=sum(row[3] for row in rows) / len(rows), count=len(rows),
                   checkpoint=str(checkpoint), teacher_input="amplitude+phase")
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (results / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "mse", "psnr", "ssim"])
        writer.writerows(rows)
    for index, (gt, reconstruction, psnr) in examples.items():
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for axis, data, title in zip(axes, [gt, reconstruction, (gt-reconstruction).abs()],
                                     ["GT", "Teacher", "Absolute error"]):
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
    print(f"Teacher test MSE: {metrics['MSE']:.8f}, "
          f"PSNR: {metrics['PSNR']:.4f} dB, SSIM: {metrics['SSIM']:.6f}")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
