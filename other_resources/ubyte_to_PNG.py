import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # 進度條套件

# ======= 設定 =======
input_file = '../data/EMNIST/raw/emnist-balanced-train-images-idx3-ubyte'  # MNIST 圖像檔
output_dir = '../sample_data/EMNIST/28pixels_PNG'  # 輸出資料夾

os.makedirs(output_dir, exist_ok=True)

# ======= 載入 MNIST 圖像資料 =======
with open(input_file, 'rb') as f:
    _ = int.from_bytes(f.read(4), 'big')       # magic number
    num_images = int.from_bytes(f.read(4), 'big')
    rows = int.from_bytes(f.read(4), 'big')
    cols = int.from_bytes(f.read(4), 'big')
    image_data = np.frombuffer(f.read(), dtype=np.uint8)
    images = image_data.reshape((num_images, rows, cols))

# ======= 轉成 PNG 圖檔（加進度條）=======
for i, img_array in enumerate(tqdm(images, desc="轉換中", unit="張")):
    img = Image.fromarray(img_array, mode='L')  # 'L' = 8-bit 灰階
    img.save(os.path.join(output_dir, f'{i:05}.png'))

print(f"✅ 共儲存 {num_images} 張圖到：{output_dir}")

