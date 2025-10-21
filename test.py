import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import matplotlib as plt
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

    if split_method == "fix":
        idx = 0
    else:
        if seed is not None:
            random.seed(seed)
        idx = random.randint(0, len(test_dataset) - 1)

    img, _ = test_dataset[idx]
    img = img.unsqueeze(0).to(device)

    vutils.save_image(img, f"{debug_dir}/input_{split_method}.png", normalize=True)

    # -------------------------------
    # 設定 hook 擷取每層輸出：
    # 幫 encoder 的每一層加上 hook，在 forward 時自動儲存輸出
    # 在不改動模型的情況下，取出中間層資訊
    # -------------------------------
    activations = {}

    def get_hook(name):
        def hook(_, __, output):  # _:該層module本身, __:該層輸入, 
            # 自動處理複數
            if torch.is_complex(output):
                output_vis = torch.abs(output)**2  # 取振幅
            else:
                output_vis = output
            activations[name] = output_vis.detach().cpu()
        return hook

    hooks = []
    for name, layer in model.encoder.named_children():
        hooks.append(layer.register_forward_hook(get_hook(name)))

    # forward encoder
    with torch.no_grad():  # 這一行會自動觸發每層的 hook → 自動蒐集每層輸出。
        _ = model.encoder(img)

    # 移除 hook
    for h in hooks:  # 避免未來再 forward 時繼續觸發（防止 memory leak）。
        h.remove()

    # -------------------------------
    # 儲存每層的 feature map
    # -------------------------------
    for name, feat in activations.items():
        # 若是多通道，只取前 1 個通道顯示
        save_path = os.path.join(debug_dir, f"{name}_output.png")

        # 若數值範圍太大或太小，自動 normalize
        feat_to_save = feat[:, 0:1, :, :]
        vutils.save_image(feat_to_save, save_path, normalize=True)
        print(f"[ONN DEBUG] Saved layer output: {save_path}")

    print(f"[ONN DEBUG] All layer outputs saved in {debug_dir}")

# ==== 主程式 ====
if __name__ == "__main__":
    model_path = f"{TESTING_CONFIG['weight_save_dir']}/{TESTING_CONFIG['weight_save_name']}"
    model_name = os.path.basename(model_path)

    model = load_model(model, model_path)

    if TESTING_CONFIG.get("onn_debug", False):
        onn_debug_run(model)
    else:
        all_imgs, all_recons, mse, psnr, ssim = test_model(model, max_ssim_images=100)
        visualize_results(all_imgs, all_recons, model_name, num_image=10, config=TESTING_CONFIG)

        # 儲存指標
        os.makedirs(TESTING_CONFIG["results_save_dir"], exist_ok=True)
        metrics_path = f"{TESTING_CONFIG['results_save_dir']}/{model_name}{TESTING_CONFIG['results_save_name_suffix']}"
        with open(metrics_path, "w") as f:
            json.dump({"MSE": mse, "PSNR": psnr, "SSIM": ssim}, f, indent=2)
        print(f"Metrics saved at {metrics_path}")
