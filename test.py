import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import json
import random
import os
import time

# ==== 匯入自定義模組 ====
from model.autoencoder import Autoencoder
from model.opticalSimulation import ONN
from model.restormer250724 import Restormer
from dataset import get_dataloaders
from config import DATASET_CONFIG, ENCODER_CONFIG, RESTORMER_CONFIG, AUTOENCODER_CONFIG, TESTING_CONFIG 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==== 載入測試集 ====
_, _, test_loader = get_dataloaders(DATASET_CONFIG)
test_dataset = test_loader.dataset

# ==== 建立模型 ====
encoder = ONN(ENCODER_CONFIG).to(device)
decoder = Restormer(RESTORMER_CONFIG).to(device)
model = Autoencoder(encoder=encoder, decoder=decoder, config=AUTOENCODER_CONFIG).to(device)

# ==== 載入模型權重 ====
def load_model(model, model_path):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
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

# ==== ONN Material Phase Difference Loss Calculation ====
def local_contrast_loss(phase: torch.Tensor) -> torch.Tensor:
    """
    計算 phase matrix 的 Local Contrast Loss。
    Phase 會自動 wrap 到 [-pi, pi]，並計算相鄰元素差分。
    
    Args:
        phase: (B, C, H, W) tensor，float32 或 float64，相位值（可以超過 [-pi, pi]）
        
    Returns:
        loss: 標量 tensor，局部對比損失
    """
    # 將 phase 壓回 [-pi, pi]
    phase_wrapped = torch.atan2(torch.sin(phase), torch.cos(phase))

    # 計算水平、垂直方向相鄰差分
    dx = phase_wrapped[:, :, :, 1:] - phase_wrapped[:, :, :, :-1]
    dy = phase_wrapped[:, :, 1:, :] - phase_wrapped[:, :, :-1, :]

    # 差分後也 wrap 回 [-pi, pi]
    dx = torch.atan2(torch.sin(dx), torch.cos(dx))
    dy = torch.atan2(torch.sin(dy), torch.cos(dy))

    # Loss: 平均相鄰差分絕對值
    loss = (dx.abs().mean() + dy.abs().mean()) / 2.0
    return loss

# ==== 測試與指標計算 ====
def test_model(model, max_ssim_images=100):
    model.eval()
    all_imgs, all_recons = [], []
    ssim_imgs, ssim_recons = [], []
    ssim_count = 0

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            recons = model(imgs)
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

    mse = F.mse_loss(all_recons, all_imgs).item()
    psnr = compute_psnr(mse).item()

    ssim_total = 0.0
    for i in range(min(max_ssim_images, len(all_imgs))):
        ssim_total += ssim_pt(all_imgs[i:i+1], all_recons[i:i+1])
    ssim = ssim_total / min(max_ssim_images, len(all_imgs))

    print(f"Test MSE: {mse:.6f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
    return all_imgs, all_recons, mse, psnr, ssim



# ==== Output 視覺化 ====
def visualize_results(all_imgs, all_recons, model_name, num_image, config):
    os.makedirs(config["results_save_dir"], exist_ok=True)
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
    save_path = f"{config['results_save_dir']}/{model_name}_image.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Visualization saved at {save_path}")

# ==== ONN debug ====
def onn_debug_run(model):
    debug_dir = os.path.join(TESTING_CONFIG["results_save_dir"], "onn_debug")
    os.makedirs(debug_dir, exist_ok=True)

    split_method = TESTING_CONFIG.get("ONN_input_select", "fix")
    seed = TESTING_CONFIG.get("seed", None)

    # 選取 input
    if split_method == "fix":
        idx = 0
    else:
        if seed is not None:
            random.seed(seed)
        idx = random.randint(0, len(test_dataset) - 1)

    img, _ = test_dataset[idx]
    img = img.unsqueeze(0).to(device)

    vutils.save_image(img, f"{debug_dir}/input_{split_method}.png", normalize=True)
    print(f"[ONN DEBUG] Saved input image to {debug_dir}/input_{split_method}.png")

    # -------------------------------
    # forward 每層並存 output
    # -------------------------------
    x = img
    for i, layer in enumerate(model.encoder.layers):
        x = layer(x)
        out = x

        # 處理 complex
        if torch.is_complex(out):
            out = torch.abs(out)**2

        # 只存第一個 channel
        out_to_save = out[:, 0:1, :, :]

        # 對應名字
        layer_name = model.encoder.layer_names[i]

        save_path = os.path.join(debug_dir, f"{layer_name}_output.png")
        vutils.save_image(out_to_save, save_path, normalize=True)
        print(f"[ONN DEBUG] Saved layer '{layer_name}' output to {save_path}")

    print(f"[ONN DEBUG] All layer outputs saved in {debug_dir}")

    
# ==== 主程式 ====
if __name__ == "__main__":
    model_path = f"{TESTING_CONFIG['weight_save_dir']}/{TESTING_CONFIG['weight_save_name']}"
    model_name = os.path.basename(model_path)

    model = load_model(model, model_path)

    if TESTING_CONFIG.get("onn_debug", False):
        onn_debug_run(model)

    all_imgs, all_recons, mse, psnr, ssim = test_model(model, max_ssim_images=100)
    visualize_results(all_imgs, all_recons, model_name, num_image=10, config=TESTING_CONFIG)

    # 儲存指標
    os.makedirs(TESTING_CONFIG["results_save_dir"], exist_ok=True)
    metrics_path = f"{TESTING_CONFIG['results_save_dir']}/{model_name}{TESTING_CONFIG['results_save_name_suffix']}"
    with open(metrics_path, "w") as f:
        json.dump({"MSE": mse, "PSNR": psnr, "SSIM": ssim}, f, indent=2)
    print(f"Metrics saved at {metrics_path}")
