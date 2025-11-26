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
from model.opticalSimulation import ONN, MaterialLayer
from model.restormer250724 import Restormer
from dataset import get_dataloaders
from config import DATASET_CONFIG, ENCODER_CONFIG, RESTORMER_CONFIG, AUTOENCODER_CONFIG, TESTING_CONFIG 

# --- DDP bootstrap: detect LOCAL_RANK / 初始化 process group ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

distributed = "LOCAL_RANK" in os.environ or "RANK" in os.environ
local_rank = int(os.environ.get("LOCAL_RANK", 0))

if distributed:
    # 綁定對應 GPU 並初始化 process group
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # 若尚未初始化 process group，初始化（torchrun 已經會設定 env）
    if not dist.is_initialized():
        dist.init_process_group(backend=TESTING_CONFIG.get("backend", "nccl"))
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}  |  Distributed: {distributed}, Local Rank: {local_rank}")

# ==== 載入測試集（DDP-aware） ====
# 使用 TESTING_CONFIG 中的 batch_size 作為 per-GPU batch（若沒有則 fallback）
per_gpu_batch = TESTING_CONFIG.get("batch_size", 64)
# get_dataloaders 應該會根據 distributed 參數回傳 DistributedSampler 的 loader
_, _, test_loader = get_dataloaders(DATASET_CONFIG, per_gpu_batch, num_workers=TESTING_CONFIG.get("num_workers", 4), distributed=distributed)
test_dataset = test_loader.dataset

# ==== 建立模型 ====
encoder = ONN(ENCODER_CONFIG).to(device)
decoder = Restormer(RESTORMER_CONFIG).to(device)
model = Autoencoder(encoder=encoder, decoder=decoder, config=AUTOENCODER_CONFIG).to(device)

if distributed:
    # 確保 model 在當前 process 的 GPU
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# ==== 載入模型權重 ====
def load_model(model, model_path):
    # 載入 checkpoint / state_dict（移除不存在的參數）
    checkpoint = torch.load(model_path, map_location=device)
    # 若 checkpoint 是 dict 並包含 'model_state_dict'，使用它；否則假設已是 state_dict
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # 若 model 被 DDP 包裝（有 .module），則把 state_dict 載入到 module
    if distributed and hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    if (not distributed) or (dist.is_initialized() and dist.get_rank() == 0):
        print(f"Model loaded from {model_path}")
    return model

# ==== PSNR 計算 ====
def compute_psnr(mse, max_pixel=1.0):
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_pixel)) - 10 * torch.log10(torch.tensor(mse))

# ==== helper：選取 recon 張量 ====
def select_recon_from_outputs(outputs, imgs):
    if not isinstance(outputs, (tuple, list)):
        return outputs
    recons, *rest = outputs
    return recons

# ==== 測試與指標計算 (僅 MSE / PSNR) ====
def test_model(model):
    model.eval()
    all_imgs, all_recons = [], []

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            outputs = model(imgs)

            if isinstance(outputs, (tuple, list)):
                recons = outputs[0]  # 取第一個 output 作為重建影像
            else:
                recons = outputs

            # 若為複數，取 magnitude
            if torch.is_complex(recons):
                recons_proc = torch.abs(recons)
            else:
                recons_proc = recons
            
            all_imgs.append(imgs.cpu())
            all_recons.append(recons_proc.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)

    mse = F.mse_loss(all_recons, all_imgs).item()
    psnr = compute_psnr(mse).item()

    # 只有 rank 0 印出與回傳可視化資料（其餘 ranks 回傳 None 或空）
    rank = dist.get_rank() if distributed and dist.is_initialized() else 0
    if rank == 0:
        print(f"Test MSE: {mse:.6f}, PSNR: {psnr:.4f}")
        return all_imgs, all_recons, mse, psnr
    else:
        # 其他 ranks 不需要回傳完整 tensors（節省記憶體與 I/O）
        return None, None, mse, psnr

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
def onn_output_debug(model):
    # 如果 model 被 DDP 包裝，取出原始 module
    net = model.module if hasattr(model, "module") else model

    debug_dir = os.path.join(TESTING_CONFIG["results_save_dir"], "ONN_debug")
    os.makedirs(debug_dir, exist_ok=True)

    split_method = TESTING_CONFIG.get("ONN_input_select", "fix")
    seed = TESTING_CONFIG.get("seed", None)

    if split_method == "fix":
        idx = TESTING_CONFIG["ONN_input_idx"]
    else:
        if seed is not None:
            random.seed(seed)
        idx = random.randint(0, len(test_dataset) - 1)

    img, _ = test_dataset[idx]
    img = img.unsqueeze(0).to(device)

    vutils.save_image(img, f"{debug_dir}/input_{split_method}.png", normalize=True)
    print(f"[ONN DEBUG] Saved input image to {debug_dir}/input_{split_method}.png")

    x = img
    # 使用 net.encoder（不是 model.encoder）
    for i, layer in enumerate(net.encoder.layers):
        layer_name = net.encoder.layer_names[i] if hasattr(net.encoder, "layer_names") else f"layer_{i}"
        x = layer(x)
        if not isinstance(x, (tuple, list)):
            x = (x,)
        x, *rest = x

        out = x
        if torch.is_complex(out):
            abs_out = torch.abs(out)**2
        else:
            abs_out = out

        vutils.save_image(abs_out[:, 0:1, :, :].cpu(), 
                          os.path.join(debug_dir, f"{layer_name}_abs.png"), normalize=True)
        print(f"[ONN DEBUG] Saved layer '{layer_name}' E field output")

        # 如果 layer 是 MaterialLayer，輸出 phase 統計
        if isinstance(layer, MaterialLayer):
            # 注意：layer.phase 可能有 batch 維度，視實作而定
            phase_image = layer.phase.detach().cpu()
            # 若 phase_image 的 shape 為 (C,H,W) 或 (B,C,H,W)，下面差分計算需視形狀調整
            if phase_image.dim() == 4:
                # (B, C, H, W) -> 對第一個 batch 做示例
                phase_for_stats = phase_image[0]
            else:
                phase_for_stats = phase_image

            # 計算差分（以 channel 0 為例）
            phase_chan = phase_for_stats[0] if phase_for_stats.dim() == 3 else phase_for_stats
            dx = phase_chan[:, 1:] - phase_chan[:, :-1]
            dy = phase_chan[1:, :] - phase_chan[:-1, :]
            diffs = torch.abs(torch.cat([dx.flatten(), dy.flatten()], dim=0))
            mean_val = diffs.mean().item()
            median_val = diffs.median().item()
            max_val = diffs.max().item()
            min_val = diffs.min().item()
            std_val = diffs.std().item()
            q25_val = torch.quantile(diffs, 0.25).item()
            q75_val = torch.quantile(diffs, 0.75).item()

            print(f"[ONN DEBUG] {layer_name} phase diff stats:")
            print(f"Mean={mean_val:.6f}, max={max_val:.6f}, min={min_val:.6f}")
            print(f"q25={q25_val:.6f}, median={median_val:.6f}, q75={q75_val:.6f}, std={std_val:.6f}")

            np_phase = phase_for_stats.squeeze().numpy()
            plt.imshow(np_phase, cmap='viridis')
            plt.colorbar()
            plt.title(f"{layer_name} Phase")
            plt.savefig(os.path.join(debug_dir, f"{layer_name}_phase.png"))
            plt.close()
            print(f"[ONN DEBUG] Saved layer '{layer_name}' phase weight")

            plt.hist(diffs.numpy(), bins=50)
            plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean={mean_val:.4f}")
            plt.axvline(median_val, color='green', linestyle='--', label=f"Median={median_val:.4f}")
            plt.axvline(q25_val, color='orange', linestyle='--', label=f"Q25={q25_val:.4f}")
            plt.axvline(q75_val, color='purple', linestyle='--', label=f"Q75={q75_val:.4f}")
            plt.title(f"{layer_name} Phase Diffs Distribution")
            plt.xlabel("Absolute Diff")
            plt.ylabel("Count")
            plt.legend()
            plt.savefig(os.path.join(debug_dir, f"{layer_name}_diffs_hist.png"))
            plt.close()
            print(f"[ONN DEBUG] Saved layer '{layer_name}' diffs distribution")

    print(f"[ONN DEBUG] All layer outputs saved in {debug_dir}")
# def onn_output_debug(model):
#     debug_dir = os.path.join(TESTING_CONFIG["results_save_dir"], "ONN_debug")
#     os.makedirs(debug_dir, exist_ok=True)

#     split_method = TESTING_CONFIG.get("ONN_input_select", "fix")
#     seed = TESTING_CONFIG.get("seed", None)

#     if split_method == "fix":
#         idx = TESTING_CONFIG["ONN_input_idx"]
#     else:
#         if seed is not None:
#             random.seed(seed)
#         idx = random.randint(0, len(test_dataset) - 1)

#     img, _ = test_dataset[idx]
#     img = img.unsqueeze(0).to(device)

#     vutils.save_image(img, f"{debug_dir}/input_{split_method}.png", normalize=True)
#     print(f"[ONN DEBUG] Saved input image to {debug_dir}/input_{split_method}.png")

#     x = img
#     for i, layer in enumerate(model.encoder.layers):
#         layer_name = model.encoder.layer_names[i]
#         x = layer(x)
#         if not isinstance(x, (tuple, list)):
#             x = (x,)
#         x, *rest = x
        

#         out = x
#         if torch.is_complex(out):
#             abs_out = torch.abs(out)**2
#         else:
#             abs_out = out

#         vutils.save_image(abs_out[:, 0:1, :, :].cpu(), 
#                           os.path.join(debug_dir, f"{layer_name}_abs.png"), normalize=True)
#         print(f"[ONN DEBUG] Saved layer '{layer_name}' E field output")

#         if isinstance(layer, MaterialLayer):
#             phase_image = layer.phase.detach().cpu()
#             dx = phase_image[:, 1:] - phase_image[:, :-1]
#             dy = phase_image[1:, :] - phase_image[:-1, :]
#             diffs = torch.abs(torch.cat([dx.flatten(), dy.flatten()], dim=0))
#             mean_val = diffs.mean().item()
#             median_val = diffs.median().item()
#             max_val = diffs.max().item()
#             min_val = diffs.min().item()
#             std_val = diffs.std().item()
#             q25_val = torch.quantile(diffs, 0.25).item()
#             q75_val = torch.quantile(diffs, 0.75).item()

#             print(f"[ONN DEBUG] {layer_name} phase diff stats:")
#             print(f"Mean={mean_val:.6f}, max={max_val:.6f}, min={min_val:.6f}")
#             print(f"q25={q25_val:.6f}, median={median_val:.6f}, q75={q75_val:.6f}, std={std_val:.6f}")

#             np_phase = phase_image.squeeze().numpy()
#             plt.imshow(np_phase, cmap='viridis')
#             plt.colorbar()
#             plt.title(f"{layer_name} Phase")
#             plt.savefig(os.path.join(debug_dir, f"{layer_name}_phase.png"))
#             plt.close()
#             print(f"[ONN DEBUG] Saved layer '{layer_name}' phase weight")

#             plt.hist(diffs.numpy(), bins=50, color='skyblue', edgecolor='black')
#             plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean={mean_val:.4f}")
#             plt.axvline(median_val, color='green', linestyle='--', label=f"Median={median_val:.4f}")
#             plt.axvline(q25_val, color='orange', linestyle='--', label=f"Q25={q25_val:.4f}")
#             plt.axvline(q75_val, color='purple', linestyle='--', label=f"Q75={q75_val:.4f}")
#             plt.title(f"{layer_name} Phase Diffs Distribution")
#             plt.xlabel("Absolute Diff")
#             plt.ylabel("Count")
#             plt.legend()
#             plt.savefig(os.path.join(debug_dir, f"{layer_name}_diffs_hist.png"))
#             plt.close()
#             print(f"[ONN DEBUG] Saved layer '{layer_name}' diffs distribution")

#     print(f"[ONN DEBUG] All layer outputs saved in {debug_dir}")

# ==== 主程式 ====
if __name__ == "__main__":
    model_path = f"{TESTING_CONFIG['weight_save_dir']}/{TESTING_CONFIG['weight_save_name']}"
    model_name = os.path.basename(model_path)

    model = load_model(model, model_path)

    if TESTING_CONFIG.get("onn_debug", False) and ((not distributed) or (dist.get_rank() == 0)):
        rank = dist.get_rank() if distributed and dist.is_initialized() else 0
        if rank == 0:
            onn_output_debug(model)

    all_imgs, all_recons, mse, psnr = test_model(model)

    rank = dist.get_rank() if distributed and dist.is_initialized() else 0
    if rank == 0:
        # 只有 rank0 做可視化與儲存
        if all_imgs is not None and all_recons is not None:
            visualize_results(all_imgs, all_recons, model_name, num_image=10, config=TESTING_CONFIG)

        os.makedirs(TESTING_CONFIG["results_save_dir"], exist_ok=True)
        metrics_path = f"{TESTING_CONFIG['results_save_dir']}/{model_name}{TESTING_CONFIG['results_save_name_suffix']}"
        with open(metrics_path, "w") as f:
            json.dump({"MSE": mse, "PSNR": psnr}, f, indent=2)
        print(f"Metrics saved at {metrics_path}")

    # 若使用 distributed，要等待所有 process 並清理
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
