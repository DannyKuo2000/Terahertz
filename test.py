'''
載入模型
載入測試數據
進行推理
計算評估指標
輸出結果
'''
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Autoencoder
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 只載入測試集
_, test_loader = get_dataloaders(batch_size=64, num_workers=0)

# 載入模型
def load_model(model_path):
    model = Autoencoder(input_dim=(28 * 4) * (28 * 4), latent_dim=8*8, output_dim=28 * 28).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 測試模型
def test_model(model, criterion):
    total_loss = 0.0
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc="Testing", ncols=100):
            data = data.to(device)
            reconstructed = model(data)

            # ground truth resize
            data_resized = F.interpolate(data, size=(28, 28), mode='bilinear', align_corners=False)
            data_resized = data_resized.view(data_resized.size(0), -1)

            loss = criterion(reconstructed, data_resized)
            total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # 儲存測試結果
    with open("test_results.json", "w") as f:
        json.dump({"test_loss": avg_test_loss}, f)

# 顯示部分輸出圖片（可選）
def visualize_results(model):
    model.eval()
    data, _ = next(iter(test_loader))
    data = data.to(device)
    with torch.no_grad():
        reconstructed = model(data)

    # ground truth resize
    data_resized = F.interpolate(data, size=(28, 28), mode='bilinear', align_corners=False)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        # 原圖
        axes[0, i].imshow(data_resized[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")

        # 重建圖
        axes[1, i].imshow(reconstructed[i].cpu().view(28, 28), cmap="gray")
        axes[1, i].axis("off")

    plt.show()

if __name__ == "__main__":
    model_path = "./model/checkpoints/TestingExperiments_epoch_25.pth"
    model = load_model(model_path)
    criterion = torch.nn.MSELoss()
    test_model(model, criterion)
    visualize_results(model)  # 選擇是否要顯示測試圖像
