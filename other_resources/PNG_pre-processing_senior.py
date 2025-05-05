"""
The first version of this file is from senior 葉邦彥
"""
import cv2
import numpy as np
import os

input_folder = 'MNIST60'   # 放原始圖像的資料夾
output_folder = 'output_images' # 存儲處理後圖像的資料夾
os.makedirs(output_folder, exist_ok=True)



for filename in os.listdir(input_folder):
    if not filename.endswith('.png'):
        continue

    img_path = os.path.join(input_folder, filename)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: 二值化（白字 = 255，黑底 = 0）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: 建立黑區域掩碼（0 = 黑）
    black_mask = (binary == 0).astype(np.uint8)

    # Step 3: 找出所有黑區連通塊
    num_labels, labels = cv2.connectedComponents(black_mask, connectivity=8)

    h, w = binary.shape

    for label in range(1, num_labels):  # label 0 是背景
        component_mask = (labels == label).astype(np.uint8)

        # 檢查這個黑區是否接觸邊界
        touches_edge = False
        if np.any(component_mask[0, :]) or np.any(component_mask[-1, :]) or \
           np.any(component_mask[:, 0]) or np.any(component_mask[:, -1]):
            touches_edge = True

        if not touches_edge:
            # 是洞 → 找其bounding box，計算中心線
            ys, xs = np.where(component_mask == 1)
            if len(ys) == 0:
                continue
            cy = int(np.mean(ys))

            # 畫一條從整張圖左到右的黑線（像素值 0）
            binary[cy, :] = 0

    # Resize 為 28x28（可選）
    binary_resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_NEAREST)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, binary_resized)

print("處理完成：已對真正封閉的洞橫向畫線。")
