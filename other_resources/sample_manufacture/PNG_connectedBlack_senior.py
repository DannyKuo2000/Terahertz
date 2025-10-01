"""
The first version of this file is from senior 葉邦彥
"""
import cv2
import numpy as np
import os
from tqdm import tqdm  # ✅ 進度條套件

input_folder =  '../sample_data/EMNIST/28pixels_PNG'
output_folder = '../sample_data/EMNIST/28pixels_connected_PNG'
os.makedirs(output_folder, exist_ok=True)

# 取得所有 PNG 檔案清單
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# 使用 tqdm 包裹迴圈，顯示進度
for filename in tqdm(image_files, desc="處理中"):
    img_path = os.path.join(input_folder, filename)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black_mask = (binary == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(black_mask, connectivity=8)

    for label in range(1, num_labels):  # label 0 是背景
        component_mask = (labels == label).astype(np.uint8)

        touches_edge = np.any(component_mask[0, :]) or np.any(component_mask[-1, :]) or \
                       np.any(component_mask[:, 0]) or np.any(component_mask[:, -1])

        if not touches_edge:
            ys, xs = np.where(component_mask == 1)
            if len(ys) == 0:
                continue
            cy = int(np.mean(ys))
            binary[cy, :] = 0

    binary_resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_NEAREST)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, binary_resized)

print("✅ 處理完成：已對真正封閉的洞橫向畫線。")

