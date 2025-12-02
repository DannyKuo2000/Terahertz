# import numpy as np
# from PIL import Image

# # 你的 raw 影像資訊
# w = 384
# h = 288
# expected_size = w * h
# raw_path = "other_data/1cm_correct_new.raw"
# bmp_path = "other_data/1cm_free_space_source.bmp"

# # 讀 raw
# raw = np.fromfile(raw_path, dtype=np.float32)
# print(raw.shape)



# if raw.size > expected_size:
#     print(f"⚠ 檔案有 {raw.size} 個數值，比期望多 {raw.size - expected_size} 個，將忽略多餘部分。")
#     raw = raw[:expected_size]
# elif raw.size < expected_size:
#     raise ValueError("檔案數據不足，請檢查dtype或影像尺寸。")

# img = raw.reshape((h, w))

# img = Image.fromarray(img)
# if img.mode != 'L':
#     img = img.convert('L')
# img.save(f"other_data/channel_1.png")
    


import numpy as np
import matplotlib.pyplot as plt

raw_file = "other_data/1cm_correct_new.raw"
width = 384
height = 288
dtype = np.float32  # 改成你的實際 dtype

data = np.fromfile(raw_file, dtype=dtype)

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