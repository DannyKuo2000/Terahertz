"""
This file is from senior 葉邦彥
"""
### 把經過繞射效應的圖像，再加了高斯噪聲的作爲輸入，輸出更接近實驗的圖像
import cv2
import numpy as np
import os
from pathlib import Path

# 調整後的轉換函數：模擬圖 → 更灰更模糊的實測風格
def simulate_realistic_image_gray_blurry(img):
    # Step 1: 強模糊
    blurred = cv2.GaussianBlur(img, (15, 15), sigmaX=5)

    # Step 2: 加亮灰底（模擬感測器背景）
    gray_background = np.random.normal(loc=155, scale=5, size=img.shape).astype(np.uint8)
    combined = cv2.addWeighted(blurred, 0.65, gray_background, 0.55, 0)  # 降對比

    # Step 3: 高斯雜訊（小一點）
    noise = np.random.normal(loc=0, scale=10, size=img.shape).astype(np.int16)
    noisy = np.clip(combined.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return noisy

# 處理資料夾中的所有圖像
def process_directory(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.png"))  # 或 *.jpg
    print(f"Found {len(image_files)} images.")

    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read {img_path}")
            continue

        simulated = simulate_realistic_image_gray_blurry(img)

        output_path = output_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(output_path), simulated)

    print(f"All images saved to: {output_dir}")

# ========== 使用方式 ==========

# 指定輸入與輸出資料夾（你可以改為自己的路徑）
input_folder = "data\MNIST_train_diffracted_images_2"       # 放原始圖（圖一）
output_folder = "data\MNIST_Final_2"      # 轉換後存這裡（像圖二）

# 執行批次處理
process_directory(input_folder, output_folder)
