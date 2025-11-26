import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import os
import time

# ==== 匯入平行化(DDP)模組 ====
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# === 匯入AMP模組
from torch import amp

# ==== 匯入自定義模組 ====
from model.autoencoder import Autoencoder
from model.opticalSimulation import ONN
from model.restormer250724 import Restormer
from dataset import get_dataloaders
from config import DATASET_CONFIG, ENCODER_CONFIG, RESTORMER_CONFIG, AUTOENCODER_CONFIG, TRAINING_CONFIG
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR


# === Parallel setup & Device ===
distributed = TRAINING_CONFIG.get("distributed", False)
local_rank = 0
if distributed:
    local_rank = int(os.environ["LOCAL_RANK"])  # torchrun 會傳入
    torch.cuda.set_device(local_rank)  # 綁定 GPU
    if not dist.is_initialized():  # 避免重複初始化
        dist.init_process_group(backend=TRAINING_CONFIG.get("backend", "nccl"))

device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# === Tensorboard ===
if distributed:
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=TRAINING_CONFIG["writer_save_path"])
    else:
        writer = None
else:
    writer = SummaryWriter(log_dir=TRAINING_CONFIG["writer_save_path"])

# === Model ===
encoder = ONN(ENCODER_CONFIG).to(device)
decoder = Restormer(RESTORMER_CONFIG).to(device)
model = Autoencoder(encoder=encoder, decoder=decoder, config=AUTOENCODER_CONFIG).to(device)

if distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# --- batch scaling（必須在建立 DataLoader 前） ---
global_batch = TRAINING_CONFIG.get("batch_size", 64)  # 你想要的有效 global batch
if distributed:
    world_size = dist.get_world_size()
    per_gpu_batch = max(1, global_batch // world_size)
    print(f"🟢 Distributed training detected — world_size={world_size}, per-GPU batch={per_gpu_batch}")
else:
    per_gpu_batch = global_batch

# === Dataset ===
train_loader, valid_loader, test_loader = get_dataloaders(DATASET_CONFIG, per_gpu_batch, num_workers=TRAINING_CONFIG["num_workers"], distributed=TRAINING_CONFIG["distributed"])

# === Optimizer & Loss ===
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])

# === Scheduler ===
scheduler = None
if TRAINING_CONFIG.get("use_scheduler", False):
    sched_type = TRAINING_CONFIG.get("scheduler_type", "ReduceLROnPlateau")
    params = TRAINING_CONFIG.get("scheduler_params", {})

    if sched_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, **params)
    elif sched_type == "StepLR":
        scheduler = StepLR(optimizer, **params)
    elif sched_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, **params)
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")

#==== Save model ====
def save_model(model, epoch, val_loss, optimizer=None, scheduler=None, learning_rate=None,
               save_extra=False, base_dir=TRAINING_CONFIG["checkpoints_weights_save_dir"]):
    # ======
    # 自動分開儲存：
    # weights/      -> 存純模型權重
    # checkpoints/  -> 存完整 checkpoint (包含 optimizer, scheduler, )
    # ======
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    weights_dir = os.path.join(base_dir, "weights")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 1️⃣ 儲存純模型權重
    weight_path = os.path.join(weights_dir, f"epoch{epoch+1}_valLoss{val_loss:.4f}_{timestamp}.pth")
    # 儲存純模型權重（統一儲存 module 的 state_dict，並在非 DDP 時直接使用 model.state_dict()）
    if distributed and hasattr(model, "module"):
        torch.save(model.module.state_dict(), weight_path)
    else:
        torch.save(model.state_dict(), weight_path)
    print(f"Model weights saved at {weight_path}")

    # 2️⃣ 儲存完整 checkpoint（如果 save_extra=True）
    # 儲存完整 checkpoint（建議也把 model_state_dict 一致性處理）
    if save_extra:
        if distributed and hasattr(model, "module"):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        checkpoint = {
            'model_state_dict': model_state,
            'epoch': epoch,
            'val_loss': val_loss
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if learning_rate is not None:
            checkpoint['learning_rate'] = learning_rate

        checkpoint_path = os.path.join(checkpoints_dir, f"epoch{epoch+1}_valLoss{val_loss:.4f}_{timestamp}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint with extra info saved at {checkpoint_path}")


# ==== ONN Material Phase Difference Loss Calculation ====
def local_contrast_loss(phase_list: list[torch.Tensor], sigma, use_weight=True, loss_mode: str = "local", margin) -> torch.Tensor:
    """
    計算多層 phase matrix 的 Local Contrast Loss，完全 vectorized。
    loss_mode:
        "local"：原本形式（|dx| + |dy|）
        "square"：平方懲罰（dx^2 + dy^2）
        "std"：局部變異數（更強調差異）
    """
    # 將多層 phase concat 起來 -> (N, H, W)
    all_phases = torch.cat([p.unsqueeze(0) for p in phase_list], dim=0)

    # wrap phase 到 [-pi, pi]
    phase_wrapped = torch.atan2(torch.sin(all_phases), torch.cos(all_phases))
    
    # 計算水平、垂直方向差分
    dx = phase_wrapped[:, :, 1:] - phase_wrapped[:, :, :-1]
    dy = phase_wrapped[:, 1:, :] - phase_wrapped[:, :-1, :]
    dx = torch.atan2(torch.sin(dx), torch.cos(dx))
    dy = torch.atan2(torch.sin(dy), torch.cos(dy))
    
    # 權重
    if use_weight:
        N, C, H, W = all_phases.shape
        def make_gaussian(size_x, size_y, sigma):
            x = torch.linspace(-1, 1, size_x)
            y = torch.linspace(-1, 1, size_y)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            gaussian = gaussian / gaussian.mean()
            return gaussian.to(all_phases.device)
        
        w_dx = make_gaussian(H, W-1, sigma).unsqueeze(0).unsqueeze(0)
        w_dy = make_gaussian(H-1, W, sigma).unsqueeze(0).unsqueeze(0)
        dx = dx * w_dx
        dy = dy * w_dy

    # ====== Loss mode switch ======
    if loss_mode == "mean":
        loss = (dx.abs().mean() + dy.abs().mean())

    elif loss_mode == "margin":
        # ---- Margin-based Gradient Loss (Hinge style) ----
        # penalty = max(0, margin - |grad|)
        dx_penalty = torch.relu(margin - dx.abs())
        dy_penalty = torch.relu(margin - dy.abs())
        # mean penalty
        loss = (dx_penalty.mean() + dy_penalty.mean()) * 0.5

    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")
    # ======================================

    return loss

# === AMP 設定 ===
use_amp = TRAINING_CONFIG.get("use_amp", True)
# 正確的初始化方式：不要把 "cuda" 當 positional arg 傳入
scaler = amp.GradScaler(enabled=use_amp)
# 建議：在使用 autocast 時用 device_type 明確指定
_amp_device_type = "cuda" if torch.cuda.is_available() and device.type.startswith("cuda") else "cpu"

# === Gradient Accumulation 設定 ===
accum_steps = TRAINING_CONFIG.get("grad_accum_steps", 1)  # 梯度累積步數，1 表示不累積


#==== Train model ====
def train_model():
    best_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0 

    # === Resume training 功能 ===
    if TRAINING_CONFIG.get("resume_training", False):
        resume_path = TRAINING_CONFIG.get("resume_checkpoint_path", None)
        if resume_path and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            if distributed and hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])

            if "optimizer_state_dict" in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = checkpoint.get("epoch", 0) + 1
            best_loss = checkpoint.get("val_loss", float('inf'))
            print(f"✅ Resumed training from checkpoint: {resume_path}")
            print(f"👉 Starting from epoch {start_epoch+1}, previous best val_loss={best_loss:.4f}")
        else:
            print("⚠️ Resume requested but checkpoint path not found — starting fresh training.")


    for epoch in range(TRAINING_CONFIG["epochs"]):
        # Parallel
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_plc_loss = 0
        
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            imgs, _ = batch
            imgs = imgs.to(device)

            with amp.autocast("cuda", enabled=use_amp):
                if TRAINING_CONFIG["return_phases"]:  # calculating PLC loss
                    recon, phase_lists = model(imgs)
                    plc_loss = local_contrast_loss(phase_lists, sigma=TRAINING_CONFIG["plc_sigma"], use_weight=TRAINING_CONFIG["use_weight"])
                    recon_loss = criterion(recon, imgs)  # reconstruction loss
                    total_loss = recon_loss + TRAINING_CONFIG["plc_loss_weight"] * plc_loss  # total loss
                else:
                    recon = model(imgs)
                    total_loss = criterion(recon, imgs)

                # 平均 loss 避免累積爆大
                loss_scaled = total_loss / accum_steps

                # 主動檢查 NaN，並在 distributed 時嘗試取得 rank（非 distributed 則顯示 rank=0）
                if torch.isnan(loss_scaled):
                    rank_for_msg = dist.get_rank() if distributed and dist.is_initialized() else 0
                    print(f"[Rank {rank_for_msg}] NaN Detected at Iter {i}, skip batch")
                    optimizer.zero_grad(set_to_none=True)
                    continue

            # backward with AMP
            scaler.scale(loss_scaled).backward()

            # 更新權重
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # DDP loss reduction（只有在 distributed 啟用時做 all_reduce）
            with torch.no_grad():
                if distributed and dist.is_initialized():
                    total_loss_reduced = total_loss.clone()
                    dist.all_reduce(total_loss_reduced, op=dist.ReduceOp.SUM)
                    total_loss_reduced = total_loss_reduced / max(1, dist.get_world_size())

                    if TRAINING_CONFIG["return_phases"]:
                        recon_loss_reduced = recon_loss.clone()
                        plc_loss_reduced = plc_loss.clone()
                        dist.all_reduce(recon_loss_reduced, op=dist.ReduceOp.SUM)
                        dist.all_reduce(plc_loss_reduced, op=dist.ReduceOp.SUM)
                        recon_loss_reduced = recon_loss_reduced / max(1, dist.get_world_size())
                        plc_loss_reduced = plc_loss_reduced / max(1, dist.get_world_size())
                else:
                    # 非分散式直接把原本的 tensor 當做 reduced 結果
                    total_loss_reduced = total_loss.detach()
                    if TRAINING_CONFIG["return_phases"]:
                        recon_loss_reduced = recon_loss.detach()
                        plc_loss_reduced = plc_loss.detach()

            # 只在 rank 0 累加（在非分散式時 rank 視為 0）
            rank_for_accum = dist.get_rank() if distributed and dist.is_initialized() else 0
            if rank_for_accum == 0:
                epoch_loss += total_loss_reduced.item()
                if TRAINING_CONFIG["return_phases"]:
                    epoch_recon_loss += recon_loss_reduced.item()
                    epoch_plc_loss += plc_loss_reduced.item()

        if TRAINING_CONFIG["return_phases"] and dist.get_rank() == 0:
            print(f"Ratio of recon and plc loss: {epoch_recon_loss/epoch_plc_loss:.4f}")

        avg_loss = epoch_loss / len(train_loader)
        if (not distributed) or (dist.get_rank() == 0):
            print(f"Epoch {epoch+1}, Total Loss: {avg_loss:.4f}")
        if writer is not None and ((not distributed) or dist.get_rank() == 0):
            writer.add_scalar("Loss/train", avg_loss, epoch)


        # === Validation (only rank 0 runs; then broadcast val_loss to all ranks) ===
        rank = dist.get_rank() if distributed and dist.is_initialized() else 0

        if rank == 0:
            val_loss = validate_model(epoch)
            val_tensor = torch.tensor([val_loss], device=device)
        else:
            val_tensor = torch.tensor([0.0], device=device)

        # sync: 確保所有 ranks 都到這裡再廣播（避免 race）
        if distributed and dist.is_initialized():
            dist.barrier()
            dist.broadcast(val_tensor, src=0)
            dist.barrier()

        val_loss = float(val_tensor.item())

        # === Scheduler step ===
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # === Early stopping check ===
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            if (not distributed) or (dist.get_rank() == 0):
                # 儲存最佳模型
                save_model(model, epoch, val_loss, optimizer=optimizer, scheduler=scheduler, learning_rate=TRAINING_CONFIG["learning_rate"], base_dir=TRAINING_CONFIG["checkpoints_weights_save_dir"], save_extra=True) 
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= TRAINING_CONFIG["patience"]:
                if (not distributed) or (dist.get_rank() == 0):
                    save_model(model, epoch, val_loss, optimizer=optimizer, scheduler=scheduler, learning_rate=TRAINING_CONFIG["learning_rate"], base_dir=TRAINING_CONFIG["checkpoints_weights_save_dir"], save_extra=True)
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # === Last save ===
        if epoch + 1 == TRAINING_CONFIG["epochs"]:
            if (not distributed) or (dist.get_rank() == 0):
                save_model(model, epoch, val_loss, optimizer=optimizer, scheduler=scheduler, learning_rate=TRAINING_CONFIG["learning_rate"], base_dir=TRAINING_CONFIG["checkpoints_weights_save_dir"], save_extra=True)

        # === Logging images to Tensorboard ===
        with torch.no_grad():
            sample_imgs = imgs[:8]
            if TRAINING_CONFIG["return_phases"]:
                recon_imgs, _ = model(sample_imgs)
            else:
                recon_imgs = model(sample_imgs)

            img_grid = vutils.make_grid(sample_imgs.cpu(), normalize=True, scale_each=True)
            recon_grid = vutils.make_grid(recon_imgs.cpu(), normalize=True, scale_each=True)
            if local_rank == 0:
                if writer is not None and (not distributed or (distributed and dist.is_initialized() and dist.get_rank() == 0)):
                    writer.add_image('Original', img_grid, global_step=epoch)
                    writer.add_image('Reconstructed', recon_grid, global_step=epoch)
                    print(f"Rank {local_rank}: recon max {recon_imgs.max()}, min {recon_imgs.min()}")

    if writer is not None:
        writer.close()
    if distributed: # 訓練完解除DDP
        dist.destroy_process_group()


#==== Validate model ====
def validate_model(epoch):
    if distributed and dist.get_rank() != 0:
        return float('inf')
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            imgs, _ = batch
            imgs = imgs.to(device)
            if TRAINING_CONFIG["return_phases"]:
                recon, phase_lists = model(imgs)
            else: 
                recon = model(imgs)
            loss = criterion(recon, imgs)
            epoch_loss += loss.item()

    val_loss = epoch_loss / len(valid_loader)
    if (not distributed) or (dist.get_rank() == 0):
        print(f"Validation Loss at Epoch {epoch+1}: {val_loss:.4f}")
    if writer is not None and ((not distributed) or dist.get_rank() == 0):
        writer.add_scalar("Loss/validation", val_loss, epoch)
    return val_loss

if __name__ == "__main__":
    train_model()
