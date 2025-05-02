import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ONN, Sensor, ConditionedUNet, DiffusionDecoder, Autoencoder  
from tqdm import tqdm
from dataset import get_dataloaders
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import math

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
latent_dim = 8*8
timesteps = 1000

# Dataset Hyperparameters
img_shape = (1, 128, 128)

# 載入 dataset
train_loader, test_loader = get_dataloaders(batch_size=batch_size)

# 創建 Autoencoder 模型
encoder = ONN(num_layers=3, num_size=128)
sensor = Sensor()
unet = ConditionedUNet(img_channels=1, t_dim=64, latent_channels=1, base_channels=64).to(device)
decoder = DiffusionDecoder(model=unet, timesteps=timesteps, image_shape=img_shape).to(device)

model = Autoencoder(encoder, sensor, decoder).to(device)  # 使用 GPU，如果有的話
criterion = nn.MSELoss()  # 使用均方誤差損失函數
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用 Adam 優化器

def train_model():
    for epoch in range(epochs):
        model.train() # 在 PyTorch 中，有些操作（如 dropout 或 batch normalization）在訓練和測試階段的行為會有所不同，所以在訓練時需要使用 train()。
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, _ = batch # 通常是圖片（imgs）和對應的標籤（_，因為我們這裡的標籤不需要用到）。
            imgs = imgs.to(device)
            t = torch.randint(0, timesteps, (imgs.size(0),), device=device).long()
            pred_noise, true_noise = model(imgs, t, mode='train')
            loss = F.mse_loss(pred_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")

        ### for tensorboard
        # 記錄 loss
        writer.add_scalar("Loss/train", epoch_loss / len(train_loader), epoch)

        # 原圖與重建圖（最多顯示前 8 張）
        with torch.no_grad():
            sample_imgs = imgs[:8]
            recon_imgs = model(sample_imgs, mode='sample')

            img_grid = vutils.make_grid(sample_imgs.cpu(), normalize=True, scale_each=True)
            recon_grid = vutils.make_grid(recon_imgs.cpu(), normalize=True, scale_each=True)

            writer.add_image('Original', img_grid, global_step=epoch)
            writer.add_image('Reconstructed', recon_grid, global_step=epoch)
    writer.close()

# 顯示原圖與重建圖
def show_images(original, reconstructed):
    fig, axs = plt.subplots(2, len(original), figsize=(12, 3))
    for i in range(len(original)):
        axs[0, i].imshow(original[i].cpu().squeeze(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        axs[1, i].axis('off')
    axs[0, 0].set_ylabel('Original', fontsize=12)
    axs[1, 0].set_ylabel('Reconstructed', fontsize=12)
    plt.tight_layout()
    plt.show()

def validate_model():
    model.eval()
    imgs, _ = next(iter(train_loader))
    imgs = imgs[:8].to(device)
    with torch.no_grad():
        recon = model(imgs, mode='sample')

if __name__ == "__main__":
    train_model()
    validate_model()


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

'''
# 訓練模型
def train_model():
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        # 用 tqdm 包裝 train_loader，顯示每一個 batch 的進度
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)):
            data = data.to(device)  # 保留原始形狀
            #data = data.view(-1, input_dim).to(device)  # 扁平化圖像為一維向量並傳遞到 GPU
            optimizer.zero_grad()

            # 前向傳播
            reconstructed = model(data)
            
            # 將ground truth調整到正確大小
            data_resized = F.interpolate(data, size=(28, 28), mode='bilinear', align_corners=False)
            data_resized = data_resized.view(data_resized.size(0), -1)

            # 計算損失
            loss = criterion(reconstructed, data_resized)
            loss.backward()

            # 更新權重
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # 每個 epoch 結束後輸出平均損失
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
        
        # 在每個epoch後保存模型
        torch.save(model.state_dict(), f'./model/checkpoints/TestingExperiments_epoch_{epoch+1}.pth')

# 驗證模型（用於 training 過程）
def validate_model():
    model.eval()  # 設置為評估模式
    total_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstructed = model(data)

            # ground truth resize
            data_resized = F.interpolate(data, size=(28, 28), mode='bilinear', align_corners=False)
            data_resized = data_resized.view(data_resized.size(0), -1)

            loss = criterion(reconstructed, data_resized)
            total_loss += loss.item()
    
    avg_val_loss = total_loss / len(test_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss
'''

