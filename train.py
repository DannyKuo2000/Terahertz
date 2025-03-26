import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Autoencoder
from tqdm import tqdm
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
註解符號說明:
    ###說明概念
    #說明程式碼
"""

# Set up Hyperparameters
batch_size = 64
epochs = 25
learning_rate = 0.001
latent_dim = 8*8
input_dim = (28 * 4) * (28 * 4)  # fit input size
output_dim = 28 * 28  # Fashion MNIST 是28x28的圖像

# 載入 dataset
train_loader, test_loader = get_dataloaders(batch_size=64)

# 創建 Autoencoder 模型
model = Autoencoder(input_dim, latent_dim, output_dim).to(device)  # 使用 GPU，如果有的話
criterion = nn.MSELoss()  # 使用均方誤差損失函數
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用 Adam 優化器

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