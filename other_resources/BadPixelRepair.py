import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import median_filter, binary_dilation
from itertools import chain

def load_image(path, cut=(288, 288), size=(160, 160)):
    img = Image.open(path).convert("L")
    print(f"Original size {img.size}")
    
    #img_array = np.array(img, dtype=np.float32) / 255.0
    return img

def load_dead_pixel_txt(txt_path, image_shape, dilation_iter=0): # 被動使用txt檔
    """
    txt_path: path to txt file
    image_shape: (H, W)
    dilation_iter: 是否對指定點做擴張（0 表示不擴張）
    """
    mask = np.zeros(image_shape, dtype=bool)

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = map(int, line.split(","))
            if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                mask[y, x] = True   # 注意 numpy: [row=y, col=x]

    for _ in range(dilation_iter):
        mask = binary_dilation(mask)

    return mask

def detect_dead_pixels(image, threshold=10, kernel_size=3): # 主動偵測
    """
    image: float image (H, W)
    threshold: value considered as dead (near zero)
    """
    median_img = median_filter(image, size=kernel_size)  # 中位數

    # dead pixel本身非常暗 or 
    dead_mask = (image < threshold) & (np.abs(image - median_img) > threshold)

    repaired = image.copy()
    repaired[dead_mask] = median_img[dead_mask]

    return repaired, dead_mask


def detect_and_repair_dead_pixels(image, low_threshold, contrast_threshold, kernel_size=3, max_iter=20):
    """
    image: numpy array (H, W), float or uint
    low_threshold: 判定「非常暗」的門檻
    contrast_threshold: 與鄰域差異門檻
    kernel_size: median filter size (odd)
    max_iter: 最大修補迭代次數
    """

    # ---------- Step 0: 型別處理 ----------
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    image = image.astype(np.float32)

    # ---------- Handmade Mask ----------
    handmade_mask = np.zeros_like(image)
    handmade_mask = handmade_mask.astype(bool)
    points1 = [(173, 53), (173, 54)]
    for x, y in points1:   # 注意 numpy 的順序是 row, col
        handmade_mask[y, x] = 1
    for _ in range(3):
        handmade_mask = binary_dilation(handmade_mask)

    points2 = [(169, 104), (170, 104), (170, 105), (170, 106), (169, 106)]
    for x, y in points2:   # 注意 numpy 的順序是 row, col
        handmade_mask[y, x] = 1
    for _ in range(2):
        handmade_mask = binary_dilation(handmade_mask)


    # ---------- Step 1: 保守初始偵測 ----------
    # 找特別暗 且 與周圍差距巨大的pixel 作為初始mask
    local_median = median_filter(image, size=kernel_size)
    initial_dead_mask = ((image < low_threshold) & (np.abs(image - local_median) > contrast_threshold))

    initial_dead_mask = (initial_dead_mask | handmade_mask)
    final_dead_mask = initial_dead_mask

    # ---------- Step 2: Iterative 修補 ----------
    # Method Name: boundary-driven propagation or morphological inpainting

    repaired = image.copy()
    mask = initial_dead_mask.copy() # 標記所有需要被修復的點

    for i in range(max_iter):
        # 找壞點中「接觸到正常區域」的那一圈
        border = mask & binary_dilation(~mask)  # binary_dilation: 把區域外擴一格

        # if not border.any():  # 如果沒有壞點就停止
        #     break

        # 用中位數作為正確值來補
        local_median = median_filter(repaired, size=kernel_size)
        repaired[border] = local_median[border]
        mask[border] = False

        # 找出藏在壞點群中間的
        local_median = median_filter(repaired, size=kernel_size)
        mask = mask | ((repaired < int(low_threshold + i*2)) & (np.abs(repaired - local_median) > contrast_threshold))  # 隨著iteration加大low_threshold(穩定了再改成好看的格式)


        # 將新的點加入final_dead_mask
        final_dead_mask = final_dead_mask | mask

    return repaired, initial_dead_mask, final_dead_mask



if __name__ == "__main__":
    # parameters
    low_threshold = 40
    contrast_threshold = 20

    input_dir = "other_data/NVLab260116_processed"
    input_dir_path = Path("other_data/NVLab260116_processed")
    output_dir = "other_data/NVLab260116_fixed"
    os.makedirs(output_dir, exist_ok=True)
    output_dir_path = Path("other_data/NVLab260116_fixed")

    # 支援 bmp 與 png
    paths = list(chain(input_dir_path.glob("*.bmp"), input_dir_path.glob("*.png")))

    for idx, path in enumerate(sorted(paths)):
        image = np.asarray(load_image(path))
        
        repaired, initial_dead_mask, final_dead_mask = detect_and_repair_dead_pixels(
            image, low_threshold=low_threshold, contrast_threshold=contrast_threshold,
            kernel_size=3, max_iter=20
        )

        # 儲存處理後影像
        save_path = output_dir_path / path.name
        plt.imsave(save_path, repaired, cmap="gray")

        # 顯示第一張
        if idx == 0:
            fig, ax = plt.subplots(1, 4, figsize=(15, 5))
            ax[0].imshow(image, cmap="gray")
            ax[0].set_title("Original")
            ax[0].axis("off")

            ax[1].imshow(initial_dead_mask, cmap="gray")
            ax[1].set_title("Initial Dead Pixels")
            ax[1].axis("off")

            ax[2].imshow(final_dead_mask, cmap="gray")
            ax[2].set_title("Final Dead Pixels")
            ax[2].axis("off")

            ax[3].imshow(repaired, cmap="gray")
            ax[3].set_title("Repaired")
            ax[3].axis("off")

            plt.tight_layout()
            plt.show()