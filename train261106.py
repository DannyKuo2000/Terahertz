# =========================================================
# Running
# =========================================================
# Single GPU: python train.py
# Multi GPU: torchrun --nproc_per_node=2 train.py

import os
import csv
import time
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import amp

from model.opticalSimulation import ONN
from model.Restormer261106 import Restormer
from dataset import get_dataloaders
from shared_training_utils import capture_rng_state
from config import (
    DATASET_CONFIG,
    ENCODER_CONFIG,
    RESTORMER_261106_CONFIG as RESTORMER_CONFIG,
    AUTOENCODER_CONFIG,
    TRAINING_CONFIG
)

# =========================================================
# Debug utilities
# =========================================================
def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def unwrap_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

def time_forward(module, x):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    y = module(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return y, end - start

def profile_named_children(module, child_names, x):
    timings = {}
    handles = []

    def make_pre_hook(name):
        def _pre_hook(_mod, _inputs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings[name] = time.perf_counter()
        return _pre_hook

    def make_post_hook(name):
        def _post_hook(_mod, _inputs, _output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings[name] = time.perf_counter() - timings[name]
        return _post_hook

    for name in child_names:
        child = getattr(module, name, None)
        if isinstance(child, nn.Module):
            handles.append(child.register_forward_pre_hook(make_pre_hook(name)))
            handles.append(child.register_forward_hook(make_post_hook(name)))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = module(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = time.perf_counter() - start

    for handle in handles:
        handle.remove()

    return out, total, timings

# =========================================================
# Distributed utilities
# =========================================================
def is_distributed():
    return dist.is_available() and dist.is_initialized()  # return if the PyTorch provide distributed and if dist.init_process_group() been done

def get_world_size():
    return dist.get_world_size() if is_distributed() else 1

def get_rank():
    return dist.get_rank() if is_distributed() else 0  # return rank

def is_main():
    return get_rank() == 0  # return true or false if you are rank 0 or not

# =========================================================
# DDP initialization (CRITICAL FIX)
# =========================================================
# Terminology explaination:
# Process: real process
# Rank: the label that DDP gave to process
# local_rank: the label of process on this server. It means local_rank != Rank if there are multiple server
distributed = TRAINING_CONFIG["distributed"]  # switch

if distributed:
    local_rank = int(os.environ["LOCAL_RANK"])

    print(
        f"[BEFORE CUDA] "
        f"RANK={os.environ.get('RANK')} "
        f"LOCAL_RANK={local_rank}",
        flush=True
    )

    torch.cuda.set_device(local_rank)

    print(
        f"[GPU DEBUG] "
        f"RANK={os.environ.get('RANK')} "
        f"LOCAL_RANK={local_rank} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"current_device={torch.cuda.current_device()}",
        flush=True
    )

    dist.init_process_group(
        backend="nccl",   # IMPORTANT: use nccl for GPU
        #init_method="env://"
    )
else:
    local_rank = 0


device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


# =========================================================
# TensorBoard (rank 0 only)
# =========================================================
tb_purge_step = TRAINING_CONFIG.get("tensorboard_purge_step")
tb_purge_step = int(tb_purge_step) if tb_purge_step is not None else None
writer = (
    SummaryWriter(TRAINING_CONFIG["writer_save_path"], purge_step=tb_purge_step)
    if is_main() else None
)
if is_main() and tb_purge_step is not None:
    print(f"[TensorBoard] Purging old scalar events from step {tb_purge_step} onward")

tb_log_lr = TRAINING_CONFIG.get("tb_log_lr", True)
tb_log_epoch_time = TRAINING_CONFIG.get("tb_log_epoch_time", True)
tb_log_gpu_memory = TRAINING_CONFIG.get("tb_log_gpu_memory", True)
tb_log_recon_every_n_epochs = TRAINING_CONFIG.get("tb_log_recon_every_n_epochs", 5)
tb_log_recon_num_images = TRAINING_CONFIG.get("tb_log_recon_num_images", 8)

csv_log_enabled = TRAINING_CONFIG.get("csv_log_enabled", True)
csv_log_path = TRAINING_CONFIG.get(
    "csv_log_path",
    os.path.join(TRAINING_CONFIG["checkpoints_weights_save_dir"], "training_log.csv")
)

best_model_name = TRAINING_CONFIG.get("best_model_name", "best_model.pth")
last_model_name = TRAINING_CONFIG.get("last_model_name", "last_model.pth")
best_checkpoint_name = TRAINING_CONFIG.get("best_checkpoint_name", "best_checkpoint.pth")
last_checkpoint_name = TRAINING_CONFIG.get("last_checkpoint_name", "last_checkpoint.pth")


# =========================================================
# Model
# =========================================================
def get_sensor_field(encoder):
    encoder_core = encoder.module if isinstance(encoder, DDP) else encoder
    for layer in encoder_core.layers:
        if hasattr(layer, "last_field") and layer.last_field is not None:
            return layer.last_field
    raise RuntimeError("SensorLayer.last_field is empty after ONN forward.")


encoder = ONN(ENCODER_CONFIG).to(device)
decoder = Restormer(RESTORMER_CONFIG).to(device)

class TeacherStudentAutoencoder(nn.Module):
    #! [TEACHER-STUDENT-261106] Added: DDP-safe joint optical/teacher/student forward
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        energy = self.encoder(x)
        if isinstance(energy, (tuple, list)):
            energy = energy[0]
        field = get_sensor_field(self.encoder)
        teacher_input = teacher_input_from_field(field)
        return self.decoder.forward_teacher_student(energy, teacher_input)


def teacher_input_from_field(field):
    amplitude = torch.abs(field)
    phase = torch.angle(field)
    return torch.cat([amplitude, torch.sin(phase), torch.cos(phase)], dim=1)


model = TeacherStudentAutoencoder(encoder, decoder).to(device)

if distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

if is_main():
    encoder_module = model.module.encoder if distributed else model.encoder
    decoder_module = model.module.decoder if distributed else model.decoder
    model_module = model.module if distributed else model

    enc_total, enc_trainable = count_parameters(encoder_module)
    dec_total, dec_trainable = count_parameters(decoder_module)
    student_total, student_trainable = count_parameters(decoder_module.student)
    teacher_total, teacher_trainable = count_parameters(decoder_module.teacher)
    mdl_total, mdl_trainable = count_parameters(model_module)

    print(f"[PARAMS] encoder   total={enc_total:,} trainable={enc_trainable:,}")
    print(f"[PARAMS] decoder   total={dec_total:,} trainable={dec_trainable:,}")
    print(f"[PARAMS] student   total={student_total:,} trainable={student_trainable:,}")
    print(f"[PARAMS] teacher   total={teacher_total:,} trainable={teacher_trainable:,}")
    print(f"[PARAMS] autoenc   total={mdl_total:,} trainable={mdl_trainable:,}")

#! [TEACHER-STUDENT-261106] Added: every DDP rank needs the decoder for
# intermediate-feature projection during its local loss computation.
decoder_module = model.module.decoder if distributed else model.decoder


# =========================================================
# Batch size
# =========================================================
global_batch = TRAINING_CONFIG.get("global_batch_size", 64)

if is_distributed():
    per_gpu_batch = max(1, global_batch // get_world_size())
else:
    per_gpu_batch = global_batch


# =========================================================
# Dataset
# =========================================================
train_loader, valid_loader, test_loader = get_dataloaders(
    DATASET_CONFIG,
    per_gpu_batch,
    num_workers=TRAINING_CONFIG["num_workers"],
    distributed=distributed
)


# =========================================================
# Optimizer / Loss
# =========================================================
# Loss
if TRAINING_CONFIG["loss"] == "L1":
    criterion = nn.L1Loss()
elif TRAINING_CONFIG["loss"] == "MSE":
    criterion = nn.MSELoss()
else:
    raise ValueError(f"Unsupported loss: {TRAINING_CONFIG['loss']}")

# Optimizer
optimizer_name = TRAINING_CONFIG["optimizer"]

if optimizer_name == "AdamW":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        **TRAINING_CONFIG["optimizer_params"],
    )

else:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


# =========================================================
# Scheduler
# =========================================================
scheduler = None
if TRAINING_CONFIG.get("use_scheduler", False):
    scheduler_type = TRAINING_CONFIG["scheduler_type"]
    scheduler_params = TRAINING_CONFIG["scheduler_params"]

    if scheduler_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer,**scheduler_params)

    elif scheduler_type == "StepLR":
        scheduler = StepLR(optimizer,**scheduler_params)

    elif scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer,**scheduler_params)

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")


# =========================================================
# AMP
# =========================================================
# use_amp = TRAINING_CONFIG.get("use_amp", True)
# scaler = amp.GradScaler(enabled=use_amp)

# =========================================================
# Debug profile
# =========================================================
# enable_profiling = TRAINING_CONFIG.get("enable_profiling", False)
# profile_steps = TRAINING_CONFIG.get("profile_steps", 0) if enable_profiling else 0
# profile_decoder_blocks = [
#     "patch_embed",
#     "encoder_level1",
#     "down1_2",
#     "encoder_level2",
#     "down2_3",
#     "encoder_level3",
#     "down3_4",
#     "latent",
#     "up4_3",
#     "reduce_chan_level3",
#     "decoder_level3",
#     "up3_2",
#     "reduce_chan_level2",
#     "decoder_level2",
#     "up2_1",
#     "decoder_level1",
#     "refinement",
#     "output",
# ]


# =========================================================
# Save model
# =========================================================
def save_model(
    model,
    epoch,
    val_loss,
    optimizer=None,
    scheduler=None,
    global_step=0,
    best_loss=None,
):

    if not is_main():
        return

    base_dir = TRAINING_CONFIG["checkpoints_weights_save_dir"]
    os.makedirs(base_dir, exist_ok=True)

    weights_dir = os.path.join(base_dir, "weights")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")

    state = model.module.state_dict() if distributed else model.state_dict()

    weight_path = os.path.join(
        weights_dir,
        f"epoch{epoch+1}_loss{val_loss:.4f}_{ts}.pth"
    )
    torch.save(state, weight_path)

    print(f"[SAVE] {weight_path}")

    ckpt = {
        "model_state_dict": state,
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "best_loss": val_loss if best_loss is None else best_loss,
        "rng_state": capture_rng_state(),
    }

    if optimizer:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    ckpt_path = os.path.join(
        ckpt_dir,
        f"epoch{epoch+1}_loss{val_loss:.4f}_{ts}.pth"
    )

    torch.save(ckpt, ckpt_path)
    print(f"[SAVE] {ckpt_path}")


def get_model_state_dict(model):
    return model.module.state_dict() if distributed else model.state_dict()


def save_named_weights(filename):
    if not is_main():
        return

    base_dir = TRAINING_CONFIG["checkpoints_weights_save_dir"]
    os.makedirs(base_dir, exist_ok=True)

    path = os.path.join(base_dir, filename)
    torch.save(get_model_state_dict(model), path)
    print(f"[SAVE] {path}")


def save_named_checkpoint(
    filename,
    epoch,
    val_loss,
    global_step,
    best_loss,
    optimizer=None,
    scheduler=None,
):
    if not is_main():
        return

    base_dir = TRAINING_CONFIG["checkpoints_weights_save_dir"]
    os.makedirs(base_dir, exist_ok=True)

    checkpoint = {
        # Model
        "model_state_dict": get_model_state_dict(model),

        # Training progress
        "epoch": epoch,
        "global_step": global_step,

        # Validation / best model tracking
        "val_loss": val_loss,
        "best_loss": best_loss,
        # Reproduce the next epoch's stochastic operations when resuming.
        "rng_state": capture_rng_state(),
    }

    # Optimizer state
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # Scheduler state
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    path = os.path.join(base_dir, filename)

    torch.save(checkpoint, path)

    print(f"[SAVE] {path}")

def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    device="cuda",
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    # Checkpoints are saved from the unwrapped model, while DDP exposes the
    # same model under the additional "module." prefix.  Load into the
    # underlying module to support both single-GPU and DDP resumes.
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    if optimizer is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
        else:
            print("[WARNING] Optimizer state not found.")

    # ---------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------

    if scheduler is not None:
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

    # ---------------------------------------------------------
    # Random states
    # ---------------------------------------------------------
    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "torch" in rng_state:
            torch_rng = rng_state["torch"]
            # Older checkpoints may contain RNG state as a list/array rather
            # than the ByteTensor required by torch.set_rng_state().
            if not (torch.is_tensor(torch_rng) and torch_rng.dtype == torch.uint8):
                torch_rng = torch.as_tensor(torch_rng, dtype=torch.uint8)
            try:
                torch.set_rng_state(torch_rng.detach().cpu().contiguous())
            except (TypeError, RuntimeError) as exc:
                print(f"[WARNING] Could not restore CPU RNG state; continuing: {exc}")
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            cuda_rng = rng_state["cuda"]
            cuda_rng = [
                state if (torch.is_tensor(state) and state.dtype == torch.uint8)
                else torch.as_tensor(state, dtype=torch.uint8)
                for state in cuda_rng
            ]
            try:
                torch.cuda.set_rng_state_all(
                    [state.detach().cpu().contiguous() for state in cuda_rng]
                )
            except (TypeError, RuntimeError) as exc:
                print(f"[WARNING] Could not restore CUDA RNG state; continuing: {exc}")
        else:
            print("[WARNING] Scheduler state not found.")

    # ---------------------------------------------------------
    # Training progress
    # ---------------------------------------------------------

    start_epoch = checkpoint.get("epoch", -1) + 1
    global_step = checkpoint.get("global_step", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))

    print(
        f"[RESUME] Loaded checkpoint: {path}"
    )

    print(
        f"[RESUME] Starting from epoch={start_epoch}, "
        f"global_step={global_step}, "
        f"best_loss={best_loss:.6f}"
    )

    return start_epoch, global_step, best_loss


def init_csv_log():
    if not is_main() or not csv_log_enabled:
        return

    log_dir = os.path.dirname(csv_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(csv_log_path):
        with open(csv_log_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "lr",
                "epoch_time_sec",
                "gpu_mem_allocated_mb",
                "gpu_mem_reserved_mb",
                "timestamp",
            ])


def append_csv_log(epoch, train_loss, val_loss, lr, epoch_time_sec, gpu_mem_allocated_mb, gpu_mem_reserved_mb):
    if not is_main() or not csv_log_enabled:
        return

    with open(csv_log_path, "a", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            epoch + 1,
            train_loss,
            val_loss,
            lr,
            epoch_time_sec,
            gpu_mem_allocated_mb,
            gpu_mem_reserved_mb,
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])


def get_current_lr():
    return optimizer.param_groups[0]["lr"]


def get_hf_recovery_gamma_values(model):
    model_core = model.module if hasattr(model, "module") else model
    decoder = getattr(model_core, "decoder", None)
    decoder_core = decoder.module if distributed else decoder
    student_decoder = getattr(decoder_core, "student", decoder_core)
    gammas = getattr(student_decoder, "refinement_recovery_gammas", None)
    if gammas is None:
        return {}
    return {
        f"gamma_{idx + 1}": gamma.detach().item()
        for idx, gamma in enumerate(gammas)
    }


def log_reconstruction_to_tensorboard(epoch):
    if not writer or tb_log_recon_every_n_epochs <= 0:
        return
    if (epoch + 1) % tb_log_recon_every_n_epochs != 0:
        return

    was_training = model.training
    model.eval()

    sample_imgs = None
    sample_recons = None

    with torch.no_grad():
        for imgs, _ in valid_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            recons = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            if torch.is_complex(recons):
                recons = torch.abs(recons)

            sample_imgs = imgs[:tb_log_recon_num_images].detach().cpu()
            sample_recons = recons[:tb_log_recon_num_images].detach().cpu()
            break

    if sample_imgs is not None and sample_recons is not None:
        img_grid = vutils.make_grid(sample_imgs, normalize=True, scale_each=True)
        recon_grid = vutils.make_grid(sample_recons, normalize=True, scale_each=True)
        writer.add_image("reconstruction/input", img_grid, epoch)
        writer.add_image("reconstruction/output", recon_grid, epoch)
        writer.flush()

    if was_training:
        model.train()


def log_epoch_to_tensorboard(epoch, train_loss, val_loss, epoch_time_sec, gpu_mem_allocated_mb, gpu_mem_reserved_mb, hf_recovery_gammas=None):
    if not writer:
        return

    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("val/loss", val_loss, epoch)

    #! [HF-RECOVERY] Log each refinement injection strength to TensorBoard
    for gamma_name, gamma_value in (hf_recovery_gammas or {}).items():
        writer.add_scalar(f"hf_recovery/{gamma_name}", gamma_value, epoch)

    if tb_log_lr:
        writer.add_scalar("lr", get_current_lr(), epoch)
    if tb_log_epoch_time:
        writer.add_scalar("time/epoch_sec", epoch_time_sec, epoch)
    if tb_log_gpu_memory and torch.cuda.is_available():
        writer.add_scalar("gpu/memory_allocated_mb", gpu_mem_allocated_mb, epoch)
        writer.add_scalar("gpu/memory_reserved_mb", gpu_mem_reserved_mb, epoch)

    writer.flush()


# =========================================================
# Loss
# =========================================================
def local_contrast_loss(phase_list, sigma):

    all_phases = torch.cat([p.unsqueeze(0) for p in phase_list], dim=0)

    wrapped = torch.atan2(torch.sin(all_phases), torch.cos(all_phases))

    dx = wrapped[:, :, 1:] - wrapped[:, :, :-1]
    dy = wrapped[:, 1:, :] - wrapped[:, :-1, :]

    dx = torch.atan2(torch.sin(dx), torch.cos(dx))
    dy = torch.atan2(torch.sin(dy), torch.cos(dy))

    return dx.abs().mean() + dy.abs().mean()


def get_reconstruction(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


#! [HF-RECOVERY] Begin: Latent shift consistency training
def shift_tensor_zero_fill(x, dy, dx):
    shifted = torch.zeros_like(x)
    height, width = x.shape[-2:]
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_x0 = max(0, dx)

    if src_y1 > src_y0 and src_x1 > src_x0:
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    return shifted


def shifted_latent_reconstruction(model, imgs, max_pixels):
    model_core = model.module if hasattr(model, "module") else model
    encoded = model_core.encoder(imgs)
    latent = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
    dy = int(torch.randint(-max_pixels, max_pixels + 1, (), device=imgs.device).item())
    dx = int(torch.randint(-max_pixels, max_pixels + 1, (), device=imgs.device).item())
    shifted_latent = shift_tensor_zero_fill(latent, dy, dx)
    return model_core.decoder(shifted_latent), (dy, dx)
#! [HF-RECOVERY] End: Latent shift consistency training


# =========================================================
# Training loop
# =========================================================
def train_model(
        start_epoch=0,
        global_step=0,
        best_loss=float("inf"),
    ):

    init_csv_log()

    # One global_step = one optimizer update
    max_iterations = TRAINING_CONFIG["max_iterations"]

    for epoch in range(start_epoch, TRAINING_CONFIG["epochs"]):
        
        epoch_t0 = time.perf_counter()

        if global_step >= max_iterations:
            break

        if torch.cuda.is_available() and tb_log_gpu_memory:
            torch.cuda.reset_peak_memory_stats(device)

        if distributed:
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        model.train()

        total_loss = 0.0

        grad_accum_steps = TRAINING_CONFIG.get("grad_accum_steps", 1)

        optimizer.zero_grad(set_to_none=True)
        accum_counter = 0

        for step_idx, (imgs, _) in enumerate(
            tqdm(train_loader, disable=not is_main())
        ):

            # Stop when reaching the maximum number of
            # optimizer updates
            if global_step >= max_iterations:
                break

            imgs = imgs.to(device)

            # =====================================================
            # Forward
            # =====================================================

            #! [TEACHER-STUDENT-261106] Modified: Joint teacher/student forward and loss
            student_output, teacher_output, student_features, teacher_features = model(imgs)
            student_out = get_reconstruction(student_output)
            teacher_out = get_reconstruction(teacher_output)
            if teacher_out.shape[-2:] != imgs.shape[-2:]:
                teacher_out = F.interpolate(
                    teacher_out, size=imgs.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            student_loss = criterion(student_out, imgs)
            teacher_loss = criterion(teacher_out, imgs)
            distill_loss = criterion(student_out, teacher_out.detach())
            feature_distill_loss = torch.zeros((), device=imgs.device)
            for feature_name, student_feature in student_features.items():
                teacher_feature = teacher_features[feature_name].detach()
                teacher_feature = decoder_module.teacher_feature_projections[feature_name](teacher_feature)
                if teacher_feature.shape[-2:] != student_feature.shape[-2:]:
                    teacher_feature = F.interpolate(
                        teacher_feature, size=student_feature.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                feature_distill_loss = feature_distill_loss + criterion(
                    student_feature, teacher_feature
                )
            if student_features:
                feature_distill_loss = feature_distill_loss / len(student_features)
            loss = (
                student_loss
                + RESTORMER_CONFIG.get("teacher_loss_weight", 1.0) * teacher_loss
                + RESTORMER_CONFIG.get("distillation_loss_weight", 0.05) * distill_loss
                + RESTORMER_CONFIG.get("feature_distillation_weight", 0.05) * feature_distill_loss
            )

            # =====================================================
            # Backward
            # =====================================================

            loss.backward()

            accum_counter += 1

            # =====================================================
            # Gradient accumulation
            # =====================================================

            # Update when:
            # 1. accumulation window is full, or
            # 2. this is the last batch of the epoch
            if (
                accum_counter == grad_accum_steps
                or step_idx == len(train_loader) - 1
            ):

                # Average gradients over the actual
                # accumulation window
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.div_(accum_counter)

                # =================================================
                # Optimizer update
                # =================================================

                optimizer.step()

                # =================================================
                # Scheduler update
                # =================================================

                if scheduler is not None:
                    scheduler.step()

                # Resume/scheduler diagnostic: print the actual LR at high
                # precision so small cosine changes are visible in terminal.
                if (
                    scheduler is not None
                    and is_main()
                    and global_step % 100 == 0
                ):
                    print(
                        f"[LR DEBUG] step={global_step}, "
                        f"scheduler={type(scheduler).__name__}, "
                        f"last_epoch={scheduler.last_epoch}, "
                        f"base_lrs={scheduler.base_lrs}, "
                        f"eta_min={getattr(scheduler, 'eta_min', None)}, "
                        f"T_max={getattr(scheduler, 'T_max', None)}, "
                        f"optimizer_lr={optimizer.param_groups[0]['lr']:.12e}"
                    )

                # Prepare for next accumulation window
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

                # One optimizer update = one global iteration
                global_step += 1

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        epoch_time_sec = time.perf_counter() - epoch_t0

        if is_main():
            print(
                f"Epoch {epoch + 1} | "
                f"Iteration {global_step}/{max_iterations} | "
                f"Loss: {avg_loss:.4f}"
            )

        # =========================================================
        # Validation
        # =========================================================

        val_loss = validate_model(epoch) if is_main() else float("inf")

        # =========================================================
        # GPU memory
        # =========================================================

        if torch.cuda.is_available():
            gpu_mem_allocated_mb = (
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 2)
            )

            gpu_mem_reserved_mb = (
                torch.cuda.max_memory_reserved(device)
                / (1024 ** 2)
            )

        else:
            gpu_mem_allocated_mb = 0.0
            gpu_mem_reserved_mb = 0.0

        # =========================================================
        # Logging / Checkpoint
        # =========================================================

        if is_main():

            lr = get_current_lr()
            #! [HF-RECOVERY] Capture learned injection strengths once per epoch
            hf_recovery_gammas = get_hf_recovery_gamma_values(model)
            if hf_recovery_gammas:
                gamma_summary = " ".join(
                    f"{name}={value:.6f}"
                    for name, value in hf_recovery_gammas.items()
                )
                print(f"[HF-RECOVERY] {gamma_summary}")

            log_epoch_to_tensorboard(
                epoch,
                avg_loss,
                val_loss,
                epoch_time_sec,
                gpu_mem_allocated_mb,
                gpu_mem_reserved_mb,
                hf_recovery_gammas,
            )

            append_csv_log(
                epoch,
                avg_loss,
                val_loss,
                lr,
                epoch_time_sec,
                gpu_mem_allocated_mb,
                gpu_mem_reserved_mb,
            )

            # -----------------------------------------------------
            # Save best model
            # -----------------------------------------------------

            if val_loss < best_loss:

                best_loss = val_loss

                save_model(
                    model,
                    epoch,
                    val_loss,
                    optimizer,
                    scheduler,
                    global_step,
                    best_loss,
                )

                save_named_weights(best_model_name)

                save_named_checkpoint(
                    best_checkpoint_name,
                    epoch,
                    val_loss,
                    global_step,
                    best_loss,
                    optimizer,
                    scheduler,
                )

            # -----------------------------------------------------
            # Save latest model
            # -----------------------------------------------------

            save_named_weights(last_model_name)

            save_named_checkpoint(
                last_checkpoint_name,
                epoch,
                val_loss,
                global_step,
                best_loss,
                optimizer,
                scheduler,
            )

            log_reconstruction_to_tensorboard(epoch)


# =========================================================
# Validation
# =========================================================
def validate_model(epoch):

    model.eval()
    total = 0.0

    with torch.no_grad():
        for imgs, _ in valid_loader:
            imgs = imgs.to(device)
            student_output, _, _, _ = model(imgs)
            out = get_reconstruction(student_output)

            total += criterion(out, imgs).item()

    loss = total / len(valid_loader)

    print(f"[VAL] epoch {epoch+1}: {loss:.4f}")

    return loss


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if TRAINING_CONFIG.get("resume_training", False):
        resume_path = TRAINING_CONFIG.get("resume_checkpoint_path")
        if not resume_path:
            raise ValueError(
                "resume_training=True but resume_checkpoint_path is empty."
            )
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )
        start_epoch, global_step, best_loss = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        # A checkpoint restores its original optimizer/scheduler state.  Apply
        # an explicit LR override after loading so a corrected LR is effective.
        resume_lr = TRAINING_CONFIG.get("resume_learning_rate")
        if resume_lr is not None:
            resume_lr = float(resume_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = resume_lr

            if scheduler is not None and TRAINING_CONFIG.get("reset_scheduler_on_resume", False):
                # scheduler.load_state_dict() above restores the old
                # eta_min/T_max from the checkpoint.  Explicitly restore the
                # current config values when resetting the scheduler.
                current_scheduler_params = TRAINING_CONFIG.get("scheduler_params", {})
                if hasattr(scheduler, "eta_min") and "eta_min" in current_scheduler_params:
                    scheduler.eta_min = float(current_scheduler_params["eta_min"])
                if hasattr(scheduler, "T_max") and "T_max" in current_scheduler_params:
                    scheduler.T_max = int(current_scheduler_params["T_max"])
                scheduler.last_epoch = -1
                scheduler._step_count = 0
                scheduler.base_lrs = [resume_lr for _ in optimizer.param_groups]
                scheduler._last_lr = [resume_lr for _ in optimizer.param_groups]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = resume_lr
                print(f"[RESUME] Scheduler reset; learning_rate={resume_lr:.3e}")
            else:
                print(f"[RESUME] Optimizer learning rate overridden to {resume_lr:.3e}")

    train_model(
        start_epoch=start_epoch,
        global_step=global_step,
        best_loss=best_loss,
    )

    if writer is not None:
        writer.close()

    if distributed:
        dist.destroy_process_group()
