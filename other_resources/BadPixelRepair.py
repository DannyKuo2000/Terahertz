import os
from pathlib import Path
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from imageio import imwrite
from scipy.ndimage import median_filter, binary_dilation, gaussian_filter, sobel
from skimage.restoration import richardson_lucy

# ===============================
# Image Loading
# ===============================
def load_image(path):
    img = Image.open(path)
    img_array = np.asarray(img)
    return img_array

# ===============================
# Pixel repair
# ===============================
def repair_pixel_masked(image, mask, max_iter=10):
    """
    Boundary-driven iterative repair using local gradient interpolation
    image: np.ndarray, float
    mask: boolean array, True = need repair
    kernel: median filter fallback size (for isolated pixels)
    max_iter: maximum number of iterations
    """
    repaired = image.copy()
    mask = mask.copy()  # 所有壞點

    for i in range(max_iter):
        # 找壞點的邊界 (接觸到正常點)
        border = mask & binary_dilation(~mask)

        if not border.any():
            break  # 沒有邊界了就停止

        # 計算梯度
        gx = sobel(repaired, axis=1)  # x方向
        gy = sobel(repaired, axis=0)  # y方向

        # 修復每個 border pixel
        ys, xs = np.where(border)
        for y, x in zip(ys, xs):
            # 找鄰域正常像素 (3x3)
            y0, y1 = max(y-1,0), min(y+2,repaired.shape[0])
            x0, x1 = max(x-1,0), min(x+2,repaired.shape[1])
            local = repaired[y0:y1, x0:x1]
            local_mask = mask[y0:y1, x0:x1]  # True = 需要修復
            normal_pixels = local[~local_mask]

            if normal_pixels.size > 0:
                # 沿梯度最小方向做加權平均
                gx_local = gx[y0:y1, x0:x1][~local_mask]
                gy_local = gy[y0:y1, x0:x1][~local_mask]
                weight = 1 / (np.sqrt(gx_local**2 + gy_local**2) + 1e-6)  # 梯度小的權重大
                repaired[y, x] = np.sum(normal_pixels * weight) / np.sum(weight)
            else:
                # fallback 用 median
                repaired[y, x] = np.median(local)

        # 標記已修復
        mask[border] = False

    return repaired

# -----------------------------
# PSF Gaussian 修復
# -----------------------------
def repair_psf_add_gaussian(image, coords, sigma=1.5, amplitude=10.0, expand=3):
    """
    coords:
        coords[0] = (cx, cy)  -> Gaussian center (float)
        coords[1:] = mask seed coordinates (int)

    sigma: Gaussian sigma
    amplitude: 加回強度
    expand: dilation 次數
    """

    img = image.copy()

    if len(coords) < 1:
        return img

    # -------------------------
    # 1️⃣ Gaussian center
    # -------------------------
    cx, cy = coords[0]

    # -------------------------
    # 2️⃣ 建立 mask (由 seed 決定)
    # -------------------------
    mask = np.zeros_like(img, dtype=bool)

    if len(coords) > 1:
        # 用後面座標當 seed
        for sx, sy in coords[1:]:
            mask[int(sy), int(sx)] = True
    else:
        # 如果沒有給 seed，就用中心最近整數點當 seed
        mask[int(np.round(cy)), int(np.round(cx))] = True

    # dilation
    for _ in range(expand):
        mask = binary_dilation(mask)

    ys, xs = np.where(mask)

    # -------------------------
    # 3️⃣ 加回 subpixel Gaussian
    # -------------------------
    for y, x in zip(ys, xs):

        dx = x - cx
        dy = y - cy

        g = amplitude * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

        img[y, x] += g

    return img

# ===============================
# Command parser
# ===============================
def parse_and_apply_commands(image, txt_path, psf_sigma=10.0, psf_amplitude=3, psf_expand=7, default_dilate=0):
    """
    image: np.ndarray
    txt_path: repair_commands.txt
    pixel_kernel: median filter size
    psf_sigma: gaussian sigma for PSF
    psf_expand: PSF mask dilation iterations
    default_dilate: default binary dilation iterations if TXT沒有指定
    """
    steps = []
    with open(txt_path, "r", encoding="utf-8") as f:
        current_cmd = None
        coords = []
        dilate = default_dilate  # <-- 新增
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 指令切換
            if line.upper() in ("P", "S"):
                if current_cmd and coords:
                    steps.append((current_cmd, coords, dilate))
                current_cmd = line.upper()
                coords = []
                dilate = default_dilate
                continue

            # dilation 指定
            if line.upper().startswith("D"):
                try:
                    dilate = int(line.split()[1])
                except:
                    dilate = default_dilate
                continue

            # 解析座標
            try:
                if current_cmd == "P":
                    x, y = map(int, line.split(","))
                else:  # S 可用小數
                    x, y = map(float, line.split(","))
                coords.append((x, y))
            except ValueError:
                print("Warning: invalid line ignored:", line)

        if current_cmd and coords:
            steps.append((current_cmd, coords, dilate))

    # ----------------------
    # Step 1: repair all pixel points
    # ----------------------
    for step in steps:
        cmd, coords, dilate_iter = step
        if cmd == "P" and coords:  # 一次修一部分的 bad pixels, 需要先整合起來
            xs, ys = zip(*coords)
            mask = np.zeros_like(image, dtype=bool)
            mask[list(ys), list(xs)] = True
            # dilation for P
            for _ in range(dilate_iter):
                mask = binary_dilation(mask)
            image = repair_pixel_masked(image, mask)

        if cmd == "S" and coords:  # 
            # 直接用 Gaussain 修局部
            image = repair_psf_add_gaussian(image, coords, sigma=psf_sigma, amplitude=psf_amplitude, expand=psf_expand)

    return image

# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    input_dir = "other_data/NVLab260130_averaged"
    output_dir = "other_data/NVLab260130_fixed"
    txt_path = "other_resources/repair_commands.txt"

    os.makedirs(output_dir, exist_ok=True)

    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)

    # ----------------------
    # Load images
    # ----------------------
    paths = list(chain(
        input_dir_path.glob("*.bmp"),
        input_dir_path.glob("*.png")
    ))

    print("Found images:", len(paths))

    # Parameters
    psf_sigma = 1.2
    psf_amplitude = 90
    psf_expand = 7

    for idx, path in enumerate(sorted(paths)):

        # ----------------------
        # 讀圖（保留原始 dtype）
        # ----------------------
        image = load_image(path)
        original_dtype = image.dtype

        # 轉成 float 計算（避免溢位）
        image_float = image.astype(np.float32)

        repaired = parse_and_apply_commands(
            image_float.copy(),
            txt_path,
            psf_sigma=psf_sigma,
            psf_amplitude=psf_amplitude,
            psf_expand=psf_expand
        )

        # ----------------------
        # 轉回原始 dtype（不做 scaling）
        # ----------------------
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            repaired = np.clip(repaired, info.min, info.max)
        else:
            info = np.finfo(original_dtype)
            repaired = np.clip(repaired, info.min, info.max)

        repaired = repaired.astype(original_dtype)

        save_path = output_dir_path / path.name

        # 直接寫檔（不做任何 normalization）
        imwrite(save_path, repaired)

        # ----------------------
        # 顯示第一張（固定顯示範圍）
        # ----------------------
        if idx == 0:

            vmin = image.min()
            vmax = image.max()

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            ax[0].set_title("Original")
            ax[0].axis("off")

            ax[1].imshow(repaired, cmap="gray", vmin=vmin, vmax=vmax)
            ax[1].set_title("Repaired")
            ax[1].axis("off")

            plt.tight_layout()
            plt.show()