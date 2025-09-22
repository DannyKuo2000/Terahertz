import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt

# ==== 自訂模組 ====
from model.autoencoder import Autoencoder
from model.opticalSimulation import ONN
from model.restormer250724 import Restormer
from model.sensor import Sensor, SensorNoise
from dataset import get_dataloaders
from config import DATASET_CONFIG, ENCODER_CONFIG, SENSOR_CONFIG, RESTORMER_CONFIG, AUTOENCODER_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 只載入測試集
_, _, test_loader = get_dataloaders(DATASET_CONFIG)

# ==== 建立模型 ====
def build_model():
    encoder = ONN(ENCODER_CONFIG).to(device)
    sensor = Sensor(SENSOR_CONFIG).to(device)
    sensor_noise = SensorNoise(SENSOR_CONFIG)
    decoder = Restormer(RESTORMER_CONFIG).to(device)
    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        sensor=sensor,
        sensor_noise=sensor_noise,
        config=AUTOENCODER_CONFIG
    ).to(device)
    return model

# ==== 載入模型權重 ====
def load_model(model, model_path):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

# ==== PSNR 計算 ====
def compute_psnr(mse, max_pixel=1.0):
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_pixel)) - 10 * torch.log10(torch.tensor(mse))

# ==== SSIM 計算 ====
def ssim_pt(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """計算單通道灰階圖的 SSIM"""
    padding = window_size // 2
    weight = torch.ones((1, 1, window_size, window_size), device=img1.device) / (window_size**2)
    mu1 = F.conv2d(img1, weight, padding=padding)
    mu2 = F.conv2d(img2, weight, padding=padding)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, weight, padding=padding) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, weight, padding=padding) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, weight, padding=padding) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

# ==== 測試與指標計算 ====
def test_model(model, max_ssim_images=100):
    model.eval()
    total_mse = 0.0
    all_imgs = []
    all_recons = []
    ssim_imgs = []
    ssim_recons = []
    ssim_count = 0

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            recons = model(imgs)
            loss = F.mse_loss(recons, imgs)
            total_mse += loss.item()
            all_imgs.append(imgs.cpu())
            all_recons.append(recons.cpu())

            # 限定部分圖像計算 SSIM
            if ssim_count < max_ssim_images:
                needed = max_ssim_images - ssim_count
                ssim_imgs.append(imgs[:needed].cpu())
                ssim_recons.append(recons[:needed].cpu())
                ssim_count += min(needed, imgs.size(0))

    all_imgs = torch.cat(all_imgs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    ssim_imgs = torch.cat(ssim_imgs, dim=0)
    ssim_recons = torch.cat(ssim_recons, dim=0)

    mse = F.mse_loss(all_recons, all_imgs).item()
    psnr = compute_psnr(mse).item()
    ssim_total = 0.0
    for i in range(ssim_imgs.size(0)):
        ssim_total += ssim_pt(ssim_imgs[i:i+1], ssim_recons[i:i+1])
    ssim = ssim_total / ssim_imgs.size(0)

    print(f"Test MSE: {mse:.6f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

    return all_imgs, all_recons, mse, psnr, ssim

# ==== 視覺化 ====
def visualize_results(all_imgs, all_recons, model_name, num_image=10):
    os.makedirs("./results", exist_ok=True)
    imgs = all_imgs[:num_image]
    recons = all_recons[:num_image]

    fig, axes = plt.subplots(2, num_image, figsize=(num_image*2, 4))
    for i in range(num_image):
        axes[0, i].imshow(imgs[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./results/{model_name}_image.png")
    plt.show()
    print("Visualization saved!")

# ==== 主程式 ====
if __name__ == "__main__":
    model_path = "./checkpoints/weight_20250821-031543/autoencoder_model.pth"
    model_name = model_path.split("/")[-2] + "_Restormer"

    model = build_model()
    model = load_model(model, model_path)

    all_imgs, all_recons, mse, psnr, ssim = test_model(model)
    visualize_results(all_imgs, all_recons, model_name, num_image=10)

    # 儲存指標
    os.makedirs("./results", exist_ok=True)
    metrics_path = f"./results/{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"MSE": mse, "PSNR": psnr, "SSIM": ssim}, f, indent=2)
    print(f"Metrics saved at {metrics_path}")
