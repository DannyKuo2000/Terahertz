"""Evaluate teacher-only checkpoints using config_teacher_only.py.

Supports raw best_model/last_model weights and full training checkpoints.
Teacher input matches training: amplitude or intensity, sin(phase), cos(phase).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn

import config_teacher_only as cfg
from model.RestormerOfficial import Restormer
from model.opticalSimulation import ONN
from testing_analysis import run_model_test, evaluate_reconstruction





class TeacherOnlyAutoencoder(nn.Module):
    """Matches the training model's state_dict names and forward computation."""
    def __init__(self):
        super().__init__()
        settings = dict(cfg.TEACHER_ONLY_CONFIG)
        self.teacher_use_energy = settings.get("teacher_use_energy", False)
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
        magnitude = field.abs()
        if self.teacher_use_energy:
            magnitude = magnitude.square()
        teacher_input = torch.cat(
            [magnitude, phase.sin(), phase.cos()], dim=1
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








def evaluate(settings, device, rank=0, world_size=1):
    model = TeacherOnlyAutoencoder().to(device)
    checkpoint = Path(settings["weight_save_dir"]) / settings["weight_save_name"]
    load_teacher_checkpoint(model, checkpoint, device)
    return evaluate_reconstruction(model, checkpoint, cfg.DATASET_CONFIG, settings,
                                   device, rank, world_size, "Teacher")


def main():
    run_model_test(TeacherOnlyAutoencoder, load_teacher_checkpoint,
                   cfg.DATASET_CONFIG, cfg.TESTING_CONFIG, "Teacher")


if __name__ == "__main__":
    main()
