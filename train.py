import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
#from torch.utils.data import DataLoader
#from torchvision import transforms
from model import ONN, Sensor, SensorNoise, ConditionedUNet, DiffusionDecoder, Autoencoder
from tqdm import tqdm
from dataset import get_dataloaders
#import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
"""
註解符號說明:
    #====大段落=====
    ###說明概念
    #說明程式碼
"""

# ========= Training Set Up =========
writer = SummaryWriter(log_dir="runs/ddpm_autoencoder")

### Training Hyperparameters
batch_size = 64
epochs = 50
learning_rate = 0.001
timesteps = 1000
patience = 5

### Dataset Hyperparameters
img_shape = (1, 128, 128)

### Load dataset
train_loader, valid_loader, test_loader = get_dataloaders(batch_size=batch_size)

### Model
encoder = ONN(
    num_layers=3, 
    num_size=128
).to(device)

sensor = Sensor().to(device)

unet = ConditionedUNet(
    img_channels=1, 
    t_dim=64, 
    latent_channels=1, 
    base_channels=64
).to(device)

decoder = DiffusionDecoder(
    model=unet, 
    timesteps=timesteps, 
    image_shape=img_shape
).to(device)

model = Autoencoder(
    encoder=encoder, 
    decoder=decoder,
    sensor=sensor, 
).to(device)

### Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#==== Test model with sample data ====
def test_model_with_sample_data():
    model.eval()
    # next(iter()): 其實跟for()做的事一樣
    imgs, _ = next(iter(train_loader))  # 從 train_loader 取出一個 batch 的圖像資料，這裡只取圖像 imgs，忽略標籤 _
    imgs = imgs[:8].to(device)  # 只選前 8 張圖像，並傳送到指定的裝置

    with torch.no_grad():  # 表示不需要計算梯度，用於inference
        pred_noise, true_noise = model(imgs, torch.randint(0, timesteps, (imgs.size(0),), device=device).long(), mode='train')
        print(f"Small batch test - Prediction shape: {pred_noise.shape}, True noise shape: {true_noise.shape}")
    model.train()

#==== Save model ====
def save_model(model, save_dir='./checkpoints', name='autoencoder_model.pth'):
    # 確保儲存路徑存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 使用當前時間戳創建子資料夾
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    weight_dir = os.path.join(save_dir, f"weight_{timestamp}")
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    # 儲存模型權重
    weight_path = os.path.join(weight_dir, f"{name}")
    torch.save(model.state_dict(), weight_path)
    print(f"Model saved at {weight_path}")

#==== Load model ====
def load_model(model, weight_path):
    # 加載模型權重
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    print(f"Model loaded from {weight_path}")
    return model

#==== Train model ====
def train_model(patience=5):
    """
    ### check latent code statistic parameters for adjustment of SensorNoiseAdaptive() 
    with torch.no_grad():
        for batch in train_loader:
            imgs, _ = batch
            imgs = imgs.to(device)
            latent_for_SNA = encoder(imgs)
            latent_for_SNA = sensor(latent_for_SNA)

            print("Latent mean:", latent_for_SNA.mean().item())
            print("Latent std:", latent_for_SNA.std().item())
            print("Latent min:", latent_for_SNA.min().item())
            print("Latent max:", latent_for_SNA.max().item())
            break  # 只看一個 batch 就好
    """
    ### parameters for early stop
    best_loss = float('inf')  # set initial value as infinite
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, _ = batch
            imgs = imgs.to(device)
            t = torch.randint(0, timesteps, (imgs.size(0),), device=device).long()
            pred_noise, true_noise = model(imgs, t, mode='train')
            loss = F.mse_loss(pred_noise, true_noise)

            optimizer.zero_grad()  # clear last grad
            loss.backward()  # back propogation
            optimizer.step()  # update parameters
            epoch_loss += loss.item()  # add batch loss to epoch loss

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)  # TensorBoard writer

        # === Validate ===
        val_loss = validate_model(epoch)

        # === Early stopping check ===
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            save_model(model)  # 可選：儲存最佳模型
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # === Logging images to Tensorboard ===
        with torch.no_grad():  # no grad record
            sample_imgs = imgs[:8] 
            recon_imgs = model(sample_imgs, mode='sample')

            # 是 torchvision 提供的工具，用來將一組圖片排成一張網格圖
            # scale_each=True：讓每張圖片單獨做 normalization，而不是整體
            img_grid = vutils.make_grid(sample_imgs.cpu(), normalize=True, scale_each=True)
            recon_grid = vutils.make_grid(recon_imgs.cpu(), normalize=True, scale_each=True)

            # writer: 將原圖與重建圖儲存下來作為可視化輸出
            # global_step=epoch 會把這兩張圖歸到當前 epoch，使你能在 TensorBoard 中追蹤每個 epoch 的結果
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
            t = torch.randint(0, timesteps, (imgs.size(0),), device=device).long()
            pred_noise, true_noise = model(imgs, t, mode='train')
            loss = F.mse_loss(pred_noise, true_noise)
            epoch_loss += loss.item()

    val_loss = epoch_loss / len(valid_loader)
    print(f"Validation Loss at Epoch {epoch+1}: {val_loss:.4f}")
    writer.add_scalar("Loss/validation", val_loss, epoch)
    return val_loss

if __name__ == "__main__":
    train_model(patience)
    save_model(model)
