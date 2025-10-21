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

#==== Save model ====
def save_model(model, epoch, val_loss, save_dir=TRAINING_CONFIG["weight_save_dir"]):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    weight_path = os.path.join(save_dir, f"epoch{epoch}_valLoss{val_loss}_{timestamp}")
    torch.save(model.state_dict(), weight_path)
    print(f"Model saved at {weight_path}")

#==== Load model ====
def load_model(model, weight_path):
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    print(f"Model loaded from {weight_path}")
    return model

#==== Train model ====
def train_model(patience=5):
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(TRAINING_CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, _ = batch
            imgs = imgs.to(device)

            recon = model(imgs)
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # === Validate ===
        val_loss = validate_model(epoch)

        # === Early stopping check ===
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            save_model(model, epoch, val_loss)  # 儲存最佳模型
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

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
def validate_model(epoch):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            imgs, _ = batch
            imgs = imgs.to(device)
            recon = model(imgs)
            loss = criterion(recon, imgs)
            epoch_loss += loss.item()

    val_loss = epoch_loss / len(valid_loader)
    print(f"Validation Loss at Epoch {epoch+1}: {val_loss:.4f}")
    writer.add_scalar("Loss/validation", val_loss, epoch)
    return val_loss

if __name__ == "__main__":
    train_model(patience=TRAINING_CONFIG["patience"])
    save_model(model)
