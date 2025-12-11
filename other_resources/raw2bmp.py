from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

raw_name = "other_data/1cm_correct_new.raw"
save_name = "other_data/1cm_correct_new_handmade_normalization.png"
width = 384
height = 288
dtype = np.float32  # 改成你的實際 dtype

data = np.fromfile(raw_name, dtype=dtype)

expected_size = width * height
if data.size > expected_size:
    print(f"⚠ 檔案有 {data.size} 個數值，比期望多 {data.size - expected_size} 個，將忽略多餘部分。")
    data = data[:expected_size]
elif data.size < expected_size:
    raise ValueError("檔案數據不足，請檢查dtype或影像尺寸。")

image = data.reshape((height, width))

plt.imshow(image, cmap='gray')
plt.title("RAW Image")
plt.axis('off')
plt.show()

def normalize_like_camera(img, gamma=0.6, clip_percent=0.01):
    """
    模擬相機的 auto-contrast + gamma 處理
    """
    img = img.astype(np.float32)

    # 1. clip 兩側極值（避免噪聲讓圖變暗）
    low = np.percentile(img, clip_percent * 100)
    high = np.percentile(img, (1 - clip_percent) * 100)
    img = np.clip(img, low, high)

    # 2. normalize 到 0~1
    img = (img - low) / (high - low + 1e-8)

    # 3. gamma 調整（亮部更亮）
    img = img ** gamma

    # 4. scale 到 0~255
    img = (img * 255).astype(np.uint8)
    return img

# 使用normalization，但其實和相機內部所使用的normalization不同，圖片結構相同但亮度不同，僅為參考使用
img = normalize_like_camera(image)

img = Image.fromarray(img)
if img.mode != 'L':
    img = img.convert('L')
img.save(save_name)