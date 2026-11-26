"""Student-only evaluation for the 261106 teacher/student experiment."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn

import config as cfg
from model.RestormerOfficial import Restormer
from model.opticalSimulation import ONN
from testing_analysis import run_model_test, evaluate_reconstruction, ssim_pt


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
        encoder.load_state_dict(_remove_prefix(checkpoint["encoder_state_dict"], "module."), strict=True)
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

    student_state = _remove_prefix(student_state, "module.")
    if any(key.startswith("student.") for key in student_state):
        student_state = {key[len("student."):]: value for key, value in student_state.items()
                         if key.startswith("student.")}
    student = getattr(decoder, "student", decoder)
    student.load_state_dict(student_state, strict=True)


class StudentOnlyAutoencoder(nn.Module):
    """Evaluate the energy-only student without constructing a teacher."""
    def __init__(self):
        super().__init__()
        self.encoder = ONN(cfg.ENCODER_CONFIG)
        self.decoder = Restormer(cfg.RESTORMER_261106_CONFIG)

    def forward(self, imgs):
        energy = self.encoder(imgs)
        if isinstance(energy, (tuple, list)):
            energy = energy[0]
        output = self.decoder(energy)
        return output[0] if isinstance(output, (tuple, list)) else output


def load_student_model(model, path, device):
    # Full training checkpoints contain Python/NumPy RNG state.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    load_student_checkpoint(checkpoint, model.encoder, model.decoder)


def evaluate(settings, device, rank=0, world_size=1):
    model = StudentOnlyAutoencoder().to(device)
    checkpoint = Path(settings["weight_save_dir"]) / settings["weight_save_name"]
    load_student_model(model, checkpoint, device)
    return evaluate_reconstruction(model, checkpoint, cfg.DATASET_CONFIG, settings,
                                   device, rank, world_size, "Student")


def main():
    settings = dict(cfg.TESTING_CONFIG)
    # Optical phase remains available for simulation diagnostics, but is
    # never an input to the student's reconstruction forward.
    settings["show_teacher_input_channels"] = False
    run_model_test(StudentOnlyAutoencoder, load_student_model,
                   cfg.DATASET_CONFIG, settings, "Student")


if __name__ == "__main__":
    main()
