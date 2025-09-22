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
from model import ONN, Sensor, ConditionedUNet, DiffusionDecoder, Autoencoder, LatentExamination
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from dataset import get_dataloaders
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 只載入測試集
_, _, test_loader = get_dataloaders(batch_size=64)

#==== Build Model ====
def build_model():
    """
    ### Latent code testing
    encoder = ONN(
        num_layers=3, 
        num_size=128
    ).to(device)

    sensor = Sensor().to(device)

    model = LatentExamination(
        encoder=encoder, 
        sensor=sensor 
    ).to(device)

    return model
    """
    """
    ### Whole testing
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
    return model
    """
#==== Load Model Weight ====
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

#==== Self-defined PSNR Computation ====
def compute_psnr(mse, max_pixel=1.0):
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_pixel)) - 10 * torch.log10(torch.tensor(mse))

#==== Pytorch SSIM Computation ====
def ssim_pt(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """計算單通道灰階圖的 SSIM"""
    # 假設輸入 shape: (1, 1, H, W)
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

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

#==== Test Model ====
def test_model(model, criterion, model_name, max_ssim_images=100):
    model.eval()
    total_loss = 0.0
    all_imgs = []
    all_recons = []

    # SSIM 統計
    ssim_imgs = []
    ssim_recons = []
    ssim_count = 0

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Testing", ncols=100):
            imgs = imgs.to(device)
            t = torch.randint(0, 1000, (imgs.size(0),), device=device).long()
            pred_noise, true_noise = model(imgs, t, mode='train')
            loss = criterion(pred_noise, true_noise)
            total_loss += loss.item()

            recons = model(imgs, mode='sample')

            # 累計所有圖像用於 MSE/PSNR
            all_imgs.append(imgs.cpu())
            all_recons.append(recons.cpu())

            # 限定部分圖像做 SSIM (SSIM計算時間非常久)
            if ssim_count < max_ssim_images:
                needed = max_ssim_images - ssim_count
                ssim_imgs.append(imgs[:needed].cpu())
                ssim_recons.append(recons[:needed].cpu())
                ssim_count += min(needed, imgs.size(0))

    # 拼接所有圖像
    all_imgs = torch.cat(all_imgs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)

    ssim_imgs = torch.cat(ssim_imgs, dim=0)
    ssim_recons = torch.cat(ssim_recons, dim=0)

    # 計算指標
    metrics = compute_metrics(all_imgs, all_recons, ssim_imgs, ssim_recons)

    avg_test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # 儲存結果
    os.makedirs("./results", exist_ok=True)
    result_path = os.path.join("results", f"{model_name}_test_results.json")
    with open(result_path, "w") as f:
        json.dump({
            "test_loss": avg_test_loss,
            "mse": metrics["MSE"],
            "psnr": metrics["PSNR"],
            "ssim": metrics["SSIM"]
        }, f, indent=2)

    print("Metrics:")
    print(json.dumps(metrics, indent=2))


#==== MSE / PSNR / SSIM Computation ====
def compute_metrics(all_imgs, all_recons, ssim_imgs, ssim_recons):
    mse = F.mse_loss(all_recons, all_imgs).item()
    psnr = 10 * torch.log10(1.0 / mse)

    # SSIM 計算較慢，限量處理
    ssim_total = 0.0
    for i in range(ssim_imgs.size(0)):
        ssim_val = ssim_torch(ssim_imgs[i:i+1], ssim_recons[i:i+1])
        ssim_total += ssim_val.item()
    ssim = ssim_total / ssim_imgs.size(0)

    return {"MSE": mse, "PSNR": psnr.item(), "SSIM": ssim}


#==== Print statistic stats ====
def print_image_stats(tensor_batch, name="Image"):
    for i, img in enumerate(tensor_batch):
        img_np = img.cpu().numpy()
        stats = {
            "max": img_np.max(),
            "min": img_np.min(),
            "mean": img_np.mean(),
            "std": img_np.std()
        }
        print(f"{name} {i}: {stats}")

#==== Image Visualization ====
def visualize_results(model, model_name, num_image):
    model.eval()
    imgs, _ = next(iter(test_loader))
    imgs = imgs[:num_image].to(device)

    with torch.no_grad():
        recon_imgs = model(imgs, mode='sample')

    # print statistic stats
    print("Original Images Stats:")
    print_image_stats(imgs, name="Original")
    print("\nReconstructed Images Stats:")
    print_image_stats(recon_imgs, name="Reconstructed")

    fig, axes = plt.subplots(2, num_image, figsize=(num_image*2, 4))  # plot 2*num_image subplot
    for i in range(num_image):
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
    
    model_path = "./checkpoints/weight_20250820-235928/autoencoder_model.pth"
    model_name = model_path.split("/")[-2]
    model_name = model_name + "_LatentCode"
    # 假設你已經有一個state_dict變數（通常是從torch.load('model.pth')得到）
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # 篩選出所有以 "encoder." 開頭的參數（也就是 encoder 的子模組權重）
    ONN_state_dict = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
    for k in ONN_state_dict.keys():
        print(k)
    # 建立新的ONN實例並載入該部分state_dict
    model = build_model()
    model.load_state_dict(ONN_state_dict)
    visualize_results(model, model_name, 10)
    


    """
    ### Whole testing
    model_path = "./checkpoints/weight_20250506-202256/autoencoder_model.pth"
    model = build_model()
    load_model(model, model_path)
    criterion = nn.MSELoss()
    model_name = model_path.split("/")[-2]  # extract model name to name results
    test_model(model, criterion, model_name, 10)
    visualize_results(model, model_name, 10)
    """