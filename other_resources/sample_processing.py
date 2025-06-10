import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # 進度條套件


# ====== Parameters ======
# ubyte2PNG
ubyte_folder = '../data/EMNIST/raw/emnist-balanced-train-images-idx3-ubyte'  # MNIST 圖像檔
pixels_28_folder = '../sample_data/EMNIST/28pixels_PNG'  # 輸出資料夾
os.makedirs(pixels_28_folder, exist_ok=True)

# PNG_connectedBlack
pixels_28_connected_folder = '../sample_data/EMNIST/28pixels_connected_PNG'
os.makedirs(pixels_28_connected_folder, exist_ok=True)

# PNG_resize
pixels_400_folder = '../sample_data/EMNIST/400pixels_connected_PNG'
scale_factor = 250/28
border = 75  # pixel
resample_method = Image.BICUBIC
os.makedirs(pixels_400_folder, exist_ok=True)

# PNG_splice
spliced_folder = '../sample_data/EMNIST/spliced_PNG'
padding = 250                          # 外圍 padding（像素）
images_per_group = 25                 # 每張拼接圖包含的圖片數
grid_size = (5, 5)                    # 拼圖的行列數（rows, cols）
os.makedirs(spliced_folder, exist_ok=True)


# ====== ubyte2PNG ======
def ubyte2PNG(input_folder, output_folder):
    # ======= 載入 MNIST 圖像資料 =======
    with open(input_folder, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')       # magic number
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape((num_images, rows, cols))

    # ======= 轉成 PNG 圖檔（加進度條）=======
    for i, img_array in enumerate(tqdm(images, desc="轉換中", unit="張")):
        img = Image.fromarray(img_array, mode='L')  # 'L' = 8-bit 灰階
        img.save(os.path.join(output_folder, f'{i:05}.png'))

    print(f"✅ 共儲存 {num_images} 張圖到：{output_folder}")

# ====== PNG_connectedBlack ======
def PNG_connectedBlack(input_folder, output_folder):
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

# ====== PNG_resize ======
def PNG_resize(input_folder, output_folder):
    # ======= 取得所有檔案 =======
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

    # ======= 處理圖像並顯示進度條 =======
    for filename in tqdm(image_files, desc="處理圖片中"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        resized_img = img.resize(new_size, resample=resample_method)

        threshold = 128
        resized_img = resized_img.convert("L").point(lambda x: 255 if x > threshold else 0, '1')

        bordered_size = (resized_img.width + 2 * border, resized_img.height + 2 * border)
        bordered_img = Image.new('1', bordered_size, 0)
        bordered_img.paste(resized_img, (border, border))

        output_path = os.path.join(output_folder, filename)
        bordered_img.save(output_path)

    print("✅ 所有圖片已放大、加邊框並儲存到：", output_folder)

# ====== PNG_splicing ======
def PNG_splice(input_folder, output_folder):
    # ===== 讀取圖片路徑 =====
    image_files = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ])

    # ===== 分組處理每25張圖片，加入 tqdm 進度條 =====
    num_groups = (len(image_files) + images_per_group - 1) // images_per_group
    for group_idx in tqdm(range(num_groups), desc="Processing groups"):
        group = image_files[group_idx * images_per_group : (group_idx + 1) * images_per_group]
        images = [Image.open(img_path) for img_path in group]

        # 取得統一的圖片尺寸
        img_w, img_h = images[0].size

        collage_w = grid_size[1] * img_w
        collage_h = grid_size[0] * img_h

        # 建立大圖，加上 padding
        total_w = collage_w + 2 * padding
        total_h = collage_h + 2 * padding
        collage = Image.new('RGB', (total_w, total_h), color='black')  # padding black

        # 將圖片貼到 collage 上
        for idx, img in enumerate(images):
            row = idx // grid_size[1]
            col = idx % grid_size[1]
            x = padding + col * img_w
            y = padding + row * img_h
            collage.paste(img, (x, y))

        # 儲存拼接圖
        output_path = os.path.join(output_folder, f"group_{group_idx*images_per_group+1}~{(group_idx+1)*images_per_group}.png")
        collage.save(output_path)

    print("✅ 完成拼接與儲存！")

if __name__ == "__main__":
    #ubyte2PNG(ubyte_folder, pixels_28_folder)
    #PNG_connectedBlack(pixels_28_folder, pixels_28_connected_folder)
    PNG_resize(pixels_28_connected_folder, pixels_400_folder)
    PNG_splice(pixels_400_folder, spliced_folder)