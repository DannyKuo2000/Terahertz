import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Terahertz_model import Autoencoder
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 設定超參數
batch_size = 64
epochs = 10
learning_rate = 0.001
latent_dim = 8*8
input_dim = (28 * 4) * (28 * 4)  # Fashion MNIST 是28x28的圖像
output_dim = 28 * 28 

# 定義資料增強和轉換
transform = transforms.Compose([
    transforms.Resize((28*4, 28*4)), # 貼齊encoder大小
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # 標準化至 [-1, 1]
    transforms.RandomHorizontalFlip(),  # 水平翻轉
    transforms.RandomRotation(10),  # 隨機旋轉
])

# 載入 Fashion MNIST 資料集
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 設定 Dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        torch.save(model.state_dict(), f'autoencoder_epoch_{epoch+1}.pth')

# 測試模型
def test_model():
    model.eval()  # 設置為評估模式
    with torch.no_grad():
        total_loss = 0.0
        # 用 tqdm 包裝 test_loader，顯示測試過程中的進度
        for data, _ in tqdm(test_loader, desc="Testing", ncols=100):
            data = data.to(device)  # 保留原始形狀
            reconstructed = model(data)

            # 將ground truth調整到正確大小
            data_resized = F.interpolate(data, size=(28, 28), mode='bilinear', align_corners=False)
            data_resized = data_resized.view(data_resized.size(0), -1)

            # 計算損失
            loss = criterion(reconstructed, data_resized)
            total_loss += loss.item()
        
        avg_test_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    train_model()
    test_model()
