'''
載入模型
載入測試數據
進行推理
計算評估指標
輸出結果
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from model import ONN, Sensor, ConditionedUNet, DiffusionDecoder, Autoencoder
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from dataset import get_dataloaders
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 只載入測試集
_, _, test_loader = get_dataloaders(batch_size=64)

# 載入模型
def load_model(model_path):
    timesteps = 1000
    img_shape = (1, 128, 128)
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

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 測試模型
def test_model(model, criterion, model_name):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Testing", ncols=100):
            imgs = imgs.to(device)
            t = torch.randint(0, 1000, (imgs.size(0),), device=device).long()
            pred_noise, true_noise = model(imgs, t, mode='train')
            loss = criterion(pred_noise, true_noise)
            total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")


    # 確保 results 資料夾存在
    os.makedirs("./results", exist_ok=True)

    # 依模型名稱儲存測試結果
    result_path = os.path.join("results", f"{model_name}_test_results.json")
    with open(result_path, "w") as f:
        json.dump({"test_loss": avg_test_loss}, f)

# 顯示部分輸出圖片
def visualize_results(model, model_name):
    model.eval()
    imgs, _ = next(iter(test_loader))
    imgs = imgs[:5].to(device)

    with torch.no_grad():
        recon_imgs = model(imgs, mode='sample')

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))  # plot 2*5 subplot
    for i in range(5):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_imgs[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.tight_layout()
    image_path = os.path.join("results", f"{model_name}_image.png")
    plt.savefig(image_path)  # 先儲存圖片
    print("Image saved!")
    plt.show()

if __name__ == "__main__":
    model_path = "./checkpoints/weight_20250504-221802/autoencoder_model.pth"
    model = load_model(model_path)
    criterion = nn.MSELoss()

    model_name = model_path.split("/")[-2]  # extract model name to name results
    test_model(model, criterion, model_name)
    visualize_results(model, model_name)
