import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

import os
import time

# ==== 匯入自定義模組 ====
from model.autoencoder import Autoencoder
from model.opticalSimulation import ONN
from model.restormer250724 import Restormer
from dataset import get_dataloaders
from config import DATASET_CONFIG, ENCODER_CONFIG, RESTORMER_CONFIG, AUTOENCODER_CONFIG, TRAINING_CONFIG
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ========= Training Set Up =========
writer = SummaryWriter(log_dir=TRAINING_CONFIG["writer_save_path"])

# === Dataset ===
train_loader, valid_loader, test_loader = get_dataloaders(DATASET_CONFIG)

# === Model ===
encoder = ONN(ENCODER_CONFIG).to(device)

decoder = Restormer(RESTORMER_CONFIG).to(device)

model = Autoencoder(encoder=encoder, decoder=decoder, config=AUTOENCODER_CONFIG).to(device)

# === Optimizer & Loss ===
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])

# === 建立 scheduler（若啟用） ===
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
               save_extra=False, base_dir=TRAINING_CONFIG["checkpoints_weight_save_dir"]):
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
    torch.save(model.state_dict(), weight_path)
    print(f"Model weights saved at {weight_path}")

    # 2️⃣ 儲存完整 checkpoint（如果 save_extra=True）
    if save_extra:
        checkpoint = {
            'model_state_dict': model.state_dict(),
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
def local_contrast_loss(phase: torch.Tensor, sigma, use_weight=True) -> torch.Tensor:
    """
    計算 phase matrix 的 Local Contrast Loss。
    Phase 會自動 wrap 到 [-pi, pi]，並計算相鄰元素差分。
    
    Args:
        phase: (B, C, H, W) tensor，float32 或 float64，相位值（可以超過 [-pi, pi]）
        
    Returns:
        loss: 標量 tensor，局部對比損失
    """
    H, W = phase.shape
    # 將 phase 壓回 [-pi, pi]
    phase_wrapped = torch.atan2(torch.sin(phase), torch.cos(phase))

    # 計算水平、垂直方向相鄰差分
    dx = phase_wrapped[:, 1:] - phase_wrapped[:, :-1]
    dy = phase_wrapped[1:, :] - phase_wrapped[:-1, :]

    # 差分後也 wrap 回 [-pi, pi], 處理掉跳躍點的問題
    dx = torch.atan2(torch.sin(dx), torch.cos(dx))
    dy = torch.atan2(torch.sin(dy), torch.cos(dy))

    if use_weight:
        # ---- 建立對應大小的 Gaussian 權重 ----
        def make_gaussian(size_x, size_y, sigma):
            x = torch.linspace(-1, 1, size_x)
            y = torch.linspace(-1, 1, size_y)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            gaussian = gaussian / gaussian.mean()  # normalize
            return gaussian.to(phase.device)

        w_dx = make_gaussian(H, W-1, sigma)   # 對應 dx 的空間
        w_dy = make_gaussian(H-1, W, sigma)   # 對應 dy 的空間

        # 加上 batch 和 channel 維度
        w_dx = w_dx.unsqueeze(0).unsqueeze(0)
        w_dy = w_dy.unsqueeze(0).unsqueeze(0)

        # 乘上權重
        dx = dx * w_dx
        dy = dy * w_dy

    # loss 計算
    loss = (dx.abs().mean() + dy.abs().mean()) / 2.0
    return loss

#==== Train model ====
def train_model(patience=5, scheduler=None):
    best_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0  # === 新增 === 起始 epoch


    # === Resume training 功能 ===
    if TRAINING_CONFIG.get("resume_training", False):
        resume_path = TRAINING_CONFIG.get("resume_checkpoint_path", None)
        if resume_path and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
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
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_plc_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, _ = batch
            imgs = imgs.to(device)

            if TRAINING_CONFIG["return_phases"]:
                recon, phase_lists = model(imgs)

                plc_loss = 0.0  # Phase local loss
                for phase in phase_lists:  # calculating phase local loss for each ONN material 
                    plc_loss += local_contrast_loss(phase, sigma=TRAINING_CONFIG["plc_sigma"], use_weight=TRAINING_CONFIG["use_weight"])

                recon_loss = criterion(recon, imgs)  # reconstruction loss
                total_loss = recon_loss + TRAINING_CONFIG["plc_loss_weight"] * plc_loss  # total loss
            else:
                recon = model(imgs)
                total_loss = criterion(recon, imgs)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if TRAINING_CONFIG["return_phases"]:
                epoch_recon_loss += recon_loss.item()
                epoch_plc_loss += plc_loss.item()
            epoch_loss += total_loss.item()

        if TRAINING_CONFIG["return_phases"]:
            print(f"Ratio of recon and plc loss: {epoch_recon_loss/epoch_plc_loss:.4f}")
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Total Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # === Validate ===
        val_loss = validate_model(epoch)

        # === 如果有 scheduler，就更新 ===
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # === Early stopping check ===
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            save_model(model, epoch, val_loss, save_extra=False)  # 儲存最佳模型
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                save_model(model, epoch, val_loss, save_extra=True)
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # === Last save ===
        if epoch + 1 == TRAINING_CONFIG["epochs"]:
            save_model(model, epoch, val_loss, optimizer=optimizer, scheduler=scheduler, learning_rate=TRAINING_CONFIG["learning_rate"], base_dir=TRAINING_CONFIG["checkpoints_weight_save_dir"], save_extra=True)

        # === Logging images to Tensorboard ===
        with torch.no_grad():
            sample_imgs = imgs[:8]
            recon_imgs = model(sample_imgs)

            img_grid = vutils.make_grid(sample_imgs.cpu(), normalize=True, scale_each=True)
            recon_grid = vutils.make_grid(recon_imgs.cpu(), normalize=True, scale_each=True)

            writer.add_image('Original', img_grid, global_step=epoch)
            writer.add_image('Reconstructed', recon_grid, global_step=epoch)

    writer.close()

#==== Validate model ====
def validate_model(epoch, scheduler=scheduler):
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
    print(f"Validation Loss at Epoch {epoch+1}: {val_loss:.4f}")
    writer.add_scalar("Loss/validation", val_loss, epoch)
    return val_loss

if __name__ == "__main__":
    train_model(patience=TRAINING_CONFIG["patience"], scheduler=scheduler)
