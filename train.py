import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ONN, Sensor, ConditionedUNet, DiffusionDecoder, Autoencoder
from tqdm import tqdm
from dataset import get_dataloaders
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
註解符號說明:
    ###說明概念
    #說明程式碼
"""

# ========= 訓練設定 =========
writer = SummaryWriter(log_dir="runs/ddpm_autoencoder")

# Training Hyperparameters
batch_size = 64
epochs = 25
learning_rate = 0.001
timesteps = 1000

# Dataset Hyperparameters
img_shape = (1, 128, 128)

# 載入 dataset
train_loader, valid_loader, test_loader = get_dataloaders(batch_size=batch_size)

# 創建 Autoencoder 模型
encoder = ONN(num_layers=3, num_size=128)
sensor = Sensor()
unet = ConditionedUNet(img_channels=1, t_dim=64, latent_channels=1, base_channels=64).to(device)
decoder = DiffusionDecoder(model=unet, timesteps=timesteps, image_shape=img_shape).to(device)
model = Autoencoder(encoder, sensor, decoder).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 小批量測試：預先測試模型是否能運行
def test_model_with_sample_data():
    model.eval()
    imgs, _ = next(iter(train_loader))
    imgs = imgs[:8].to(device)

    with torch.no_grad():
        pred_noise, true_noise = model(imgs, torch.randint(0, timesteps, (imgs.size(0),), device=device).long(), mode='train')
        print(f"Small batch test - Prediction shape: {pred_noise.shape}, True noise shape: {true_noise.shape}")
    model.train()

# 儲存模型並測試重新加載
def save_and_load_model():
    torch.save(model.state_dict(), 'autoencoder_model.pth')
    print("Model saved!")

    model_loaded = Autoencoder(encoder, sensor, decoder).to(device)
    model_loaded.load_state_dict(torch.load('autoencoder_model.pth'))
    model_loaded.eval()
    print("Model loaded and ready for inference!")

# 訓練與驗證函數
def train_model():
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, _ = batch
            imgs = imgs.to(device)
            t = torch.randint(0, timesteps, (imgs.size(0),), device=device).long()
            pred_noise, true_noise = model(imgs, t, mode='train')
            loss = F.mse_loss(pred_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")
        writer.add_scalar("Loss/train", epoch_loss / len(train_loader), epoch)

        validate_model(epoch)

        with torch.no_grad():
            sample_imgs = imgs[:8]
            recon_imgs = model(sample_imgs, mode='sample')

            img_grid = vutils.make_grid(sample_imgs.cpu(), normalize=True, scale_each=True)
            recon_grid = vutils.make_grid(recon_imgs.cpu(), normalize=True, scale_each=True)

            writer.add_image('Original', img_grid, global_step=epoch)
            writer.add_image('Reconstructed', recon_grid, global_step=epoch)

    writer.close()

# 驗證模型
def validate_model(epoch):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            imgs, _ = batch
            imgs = imgs.to(device)
            t = torch.randint(0, timesteps, (imgs.size(0),), device=device).long()
            pred_noise, true_noise = model(imgs, t, mode='train')
            loss = F.mse_loss(pred_noise, true_noise)
            epoch_loss += loss.item()

    print(f"Validation Loss at Epoch {epoch+1}: {epoch_loss / len(valid_loader):.4f}")
    writer.add_scalar("Loss/validation", epoch_loss / len(valid_loader), epoch)

if __name__ == "__main__":
    #test_model_with_sample_data()
    train_model()
    save_and_load_model()








''' 參考文件架構
project_root/
├── model/          # 模型相關的程式碼
│   ├── __init__.py
│   ├── model.py    # 定義神經網路結構
│   ├── loss.py     # 定義損失函數
│   ├── utils.py    # 其他輔助函數
│   ├── checkpoints/ # 存放模型權重
│       ├── best_model.pth
│       ├── latest.pth
├── train.py        # 訓練程式
├── test.py         # 測試/驗證程式
├── dataset/        # 數據處理相關
│   ├── dataloader.py
│   ├── preprocess.py
├── configs/        # 超參數和設定檔
│   ├── config.yaml
├── scripts/        # 可能的執行腳本
│   ├── run_training.sh
│   ├── evaluate.sh
├── logs/           # 訓練時的log文件
'''