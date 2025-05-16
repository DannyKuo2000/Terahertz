from PIL import Image
import os
from tqdm import tqdm  # ✅ 加入進度條套件

# ======= 設定參數 =======
input_folder = '../sample_data/EMNIST/28pixels_connected_PNG'
output_folder = '../sample_data/EMNIST/380pixels_connected_PNG'
scale_factor = 10
border = 50
resample_method = Image.BICUBIC

# ======= 建立輸出資料夾 =======
os.makedirs(output_folder, exist_ok=True)

# ======= 取得所有檔案 =======
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

# ======= 處理圖像並顯示進度條 =======
for filename in tqdm(image_files, desc="處理圖片中"):
    img_path = os.path.join(input_folder, filename)
    img = Image.open(img_path)

    new_size = (img.width * scale_factor, img.height * scale_factor)
    resized_img = img.resize(new_size, resample=resample_method)

    threshold = 128
    resized_img = resized_img.convert("L").point(lambda x: 255 if x > threshold else 0, '1')

    bordered_size = (resized_img.width + 2 * border, resized_img.height + 2 * border)
    bordered_img = Image.new('1', bordered_size, 0)
    bordered_img.paste(resized_img, (border, border))

    output_path = os.path.join(output_folder, filename)
    bordered_img.save(output_path)

print("✅ 所有圖片已放大、加邊框並儲存到：", output_folder)
