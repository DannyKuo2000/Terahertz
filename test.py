'''
載入模型
載入測試數據
進行推理
計算評估指標
輸出結果
'''
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from model import ONN, Sensor, ConditionedUNet, DiffusionDecoder, Autoencoder
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 只載入測試集
_, _, test_loader = get_dataloaders(batch_size=64)

# 載入模型
def load_model(model_path):
    encoder = ONN(num_layers=3, num_size=128)
    sensor = Sensor()
    unet = ConditionedUNet(img_channels=1, t_dim=64, latent_channels=1, base_channels=64).to(device)
    decoder = DiffusionDecoder(model=unet, timesteps=1000, image_shape=(1, 128, 128)).to(device)
    model = Autoencoder(encoder, sensor, decoder).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 測試模型
def test_model(model, criterion):
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
    with open("test_results.json", "w") as f:
        json.dump({"test_loss": avg_test_loss}, f)

# 顯示部分輸出圖片
def visualize_results(model):
    model.eval()
    imgs, _ = next(iter(test_loader))
    imgs = imgs[:5].to(device)

    with torch.no_grad():
        recon_imgs = model(imgs, mode='sample')

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_imgs[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = "./autoencoder_model.pth"
    model = load_model(model_path)
    criterion = torch.nn.MSELoss()
    test_model(model, criterion)
    visualize_results(model)
