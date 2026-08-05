# =========================================================
# Running
# =========================================================
# Single GPU: python train.py
# Multi GPU: torchrun --nproc_per_node=2 train.py
import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import amp

from model.autoencoder import Autoencoder
from model.opticalSimulation import ONN
# from model.restormer250724 import Restormer
from model.Restormer260803 import Restormer
from dataset import get_dataloaders

from config import (
    DATASET_CONFIG,
    ENCODER_CONFIG,
    RESTORMER_CONFIG,
    AUTOENCODER_CONFIG,
    TRAINING_CONFIG
)

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR


# =========================================================
# Distributed utilities
# =========================================================
def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_distributed() else 0

def get_world_size():
    return dist.get_world_size() if is_distributed() else 1

def is_main():
    return get_rank() == 0

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
# DDP initialization (CRITICAL FIX)
# =========================================================
distributed = TRAINING_CONFIG["distributed"]

if distributed:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",   # IMPORTANT: use nccl for GPU
        init_method="env://"
    )
else:
    local_rank = 0


device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


# =========================================================
# TensorBoard (rank 0 only)
# =========================================================
writer = SummaryWriter(TRAINING_CONFIG["writer_save_path"]) if is_main() else None

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
encoder = ONN(ENCODER_CONFIG).to(device)
decoder = Restormer(RESTORMER_CONFIG).to(device)

model = Autoencoder(
    encoder=encoder,
    decoder=decoder,
    config=AUTOENCODER_CONFIG
).to(device)

if distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

if is_main():
    encoder_module = model.module.encoder if distributed else model.encoder
    decoder_module = model.module.decoder if distributed else model.decoder
    model_module = model.module if distributed else model

    enc_total, enc_trainable = count_parameters(encoder_module)
    dec_total, dec_trainable = count_parameters(decoder_module)
    mdl_total, mdl_trainable = count_parameters(model_module)

    print(f"[PARAMS] encoder   total={enc_total:,} trainable={enc_trainable:,}")
    print(f"[PARAMS] decoder   total={dec_total:,} trainable={dec_trainable:,}")
    print(f"[PARAMS] autoenc   total={mdl_total:,} trainable={mdl_trainable:,}")


# =========================================================
# Batch size
# =========================================================
global_batch = TRAINING_CONFIG.get("batch_size", 64)

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])


# =========================================================
# Scheduler
# =========================================================
scheduler = None
if TRAINING_CONFIG.get("use_scheduler", False):
    t = TRAINING_CONFIG["scheduler_type"]
    p = TRAINING_CONFIG["scheduler_params"]

    if t == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, **p)
    elif t == "StepLR":
        scheduler = StepLR(optimizer, **p)
    elif t == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, **p)


# =========================================================
# AMP
# =========================================================
use_amp = TRAINING_CONFIG.get("use_amp", True)
scaler = amp.GradScaler(enabled=use_amp)

# =========================================================
# Debug profile
# =========================================================
enable_profiling = TRAINING_CONFIG.get("enable_profiling", False)
profile_steps = TRAINING_CONFIG.get("profile_steps", 0) if enable_profiling else 0
profile_decoder_blocks = [
    "patch_embed",
    "encoder_level1",
    "down1_2",
    "encoder_level2",
    "down2_3",
    "encoder_level3",
    "down3_4",
    "latent",
    "up4_3",
    "reduce_chan_level3",
    "decoder_level3",
    "up3_2",
    "reduce_chan_level2",
    "decoder_level2",
    "up2_1",
    "decoder_level1",
    "refinement",
    "output",
]


# =========================================================
# Save model
# =========================================================
def save_model(model, epoch, val_loss, optimizer=None, scheduler=None):

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
        "val_loss": val_loss,
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


def save_named_checkpoint(filename, epoch, val_loss, optimizer=None, scheduler=None):
    if not is_main():
        return

    base_dir = TRAINING_CONFIG["checkpoints_weights_save_dir"]
    os.makedirs(base_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": get_model_state_dict(model),
        "epoch": epoch,
        "val_loss": val_loss,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    path = os.path.join(base_dir, filename)
    torch.save(checkpoint, path)
    print(f"[SAVE] {path}")


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


def log_epoch_to_tensorboard(epoch, train_loss, val_loss, epoch_time_sec, gpu_mem_allocated_mb, gpu_mem_reserved_mb):
    if not writer:
        return

    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("val/loss", val_loss, epoch)

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


# =========================================================
# Training loop
# =========================================================
def train_model():

    best_loss = float("inf")

    init_csv_log()

    for epoch in range(TRAINING_CONFIG["epochs"]):
        epoch_start = time.perf_counter()

        if torch.cuda.is_available() and tb_log_gpu_memory:
            torch.cuda.reset_peak_memory_stats(device)

        if distributed:
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0

        optimizer.zero_grad()

        for step_idx, (imgs, _) in enumerate(tqdm(train_loader, disable=not is_main())):
            step_t0 = time.perf_counter()
            imgs = imgs.to(device)
            step_t1 = time.perf_counter()

            if is_main() and step_idx < profile_steps:  # for debug
                model_ref = model.module if distributed else model
                encoder_ref = model_ref.encoder
                decoder_ref = model_ref.decoder

                enc_out, enc_time = time_forward(encoder_ref, imgs)
                enc_x = unwrap_output(enc_out)

                dec_out, dec_time, dec_timings = profile_named_children(
                    decoder_ref,
                    profile_decoder_blocks,
                    enc_x,
                )
                out = unwrap_output(dec_out)

                with amp.autocast(device_type="cuda", enabled=use_amp):
                    loss = criterion(out, imgs)
                step_t2 = time.perf_counter()

                loss = loss / TRAINING_CONFIG.get("grad_accum_steps", 1)

                with torch.autograd.profiler.profile(
                    use_cuda=torch.cuda.is_available(),
                    record_shapes=False,
                ) as prof:
                    scaler.scale(loss).backward()
                step_t3 = time.perf_counter()

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_t4 = time.perf_counter()

                total_loss += loss.item()

                block_times = sorted(dec_timings.items(), key=lambda kv: kv[1], reverse=True)
                top_sort = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"

                print(
                    f"[PROFILE] step {step_idx + 1}: "
                    f"to(device)={step_t1 - step_t0:.3f}s, "
                    f"encoder={enc_time:.3f}s, "
                    f"decoder_total={dec_time:.3f}s, "
                    f"loss={step_t2 - step_t1 - enc_time - dec_time:.3f}s, "
                    f"backward={step_t3 - step_t2:.3f}s, "
                    f"step={step_t4 - step_t3:.3f}s"
                )

                if block_times:
                    print("[PROFILE] decoder blocks:")
                    for name, sec in block_times[:10]:
                        print(f"  - {name}: {sec:.3f}s")

                print("[PROFILE] backward ops:")
                print(prof.key_averages().table(sort_by=top_sort, row_limit=12))
            else:
                with amp.autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(imgs)
                    out = get_reconstruction(outputs)
                    # ====== debug ======
                    if not torch.isfinite(imgs).all():
                        print(f"❌ INPUT NaN/Inf at epoch={epoch+1}, step={step_idx}")
                        print("imgs min:", imgs.nan_to_num().min().item())
                        print("imgs max:", imgs.nan_to_num().max().item())
                        raise RuntimeError("Input contains NaN/Inf")
                    if not torch.isfinite(out).all():
                        print(f"❌ OUTPUT NaN/Inf at epoch={epoch+1}, step={step_idx}")
                        print("out min:", out.nan_to_num().min().item())
                        print("out max:", out.nan_to_num().max().item())
                        raise RuntimeError("Model output contains NaN/Inf")
                    


                    loss = criterion(out, imgs)

                    # ====== debug ======
                    if not torch.isfinite(loss):
                        print(f"❌ LOSS NaN/Inf at epoch={epoch+1}, step={step_idx}")
                        print("loss:", loss.item())
                        raise RuntimeError("Loss contains NaN/Inf")

                loss = loss / TRAINING_CONFIG.get("grad_accum_steps", 1)

                # ====== debug ======
                if step_idx % 500 == 0:
                    print(
                        f"[DEBUG] step={step_idx} "
                        f"loss={loss.item():.6e}, "
                        f"out_min={out.min().item():.6e}, "
                        f"out_max={out.max().item():.6e}, "
                        f"out_mean={out.mean().item():.6e}, "
                        f"out_std={out.std().item():.6e}"
                    )

                scaler.scale(loss).backward()
                if (step_idx + 1) % TRAINING_CONFIG.get("grad_accum_steps", 1) == 0:  # accumulation steps
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if is_main():
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # ================= validation =================
        val_loss = validate_model(epoch) if is_main() else float("inf")

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        epoch_time_sec = time.perf_counter() - epoch_start
        if torch.cuda.is_available():
            gpu_mem_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            gpu_mem_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        else:
            gpu_mem_allocated_mb = 0.0
            gpu_mem_reserved_mb = 0.0

        if is_main():
            lr = get_current_lr()
            log_epoch_to_tensorboard(
                epoch,
                avg_loss,
                val_loss,
                epoch_time_sec,
                gpu_mem_allocated_mb,
                gpu_mem_reserved_mb,
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

            if val_loss < best_loss:
                best_loss = val_loss
                save_model(model, epoch, val_loss, optimizer, scheduler)
                save_named_weights(best_model_name)
                save_named_checkpoint(best_checkpoint_name, epoch, val_loss, optimizer, scheduler)

            save_named_weights(last_model_name)
            save_named_checkpoint(last_checkpoint_name, epoch, val_loss, optimizer, scheduler)
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
            outputs = model(imgs)
            out = get_reconstruction(outputs)

            total += criterion(out, imgs).item()

    loss = total / len(valid_loader)

    print(f"[VAL] epoch {epoch+1}: {loss:.4f}")

    return loss


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    train_model()

    if writer is not None:
        writer.close()

    if distributed:
        dist.destroy_process_group()
