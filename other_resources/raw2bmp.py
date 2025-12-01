import numpy as np
from PIL import Image

# 你的 raw 影像資訊
w = 384
h = 288
raw_path = "other_data/1cm_correct_new.raw"
bmp_path = "other_data/1cm_free_space_source.bmp"

# 讀 raw
raw = np.fromfile(raw_path, dtype=np.uint8)
print(raw.shape)

raw = raw[8:]
raw4 = raw.reshape((h, w, 4))             # reshape 成 4 通道

for i in range(4):
    channel = raw4[:, :, i]
    img = Image.fromarray(channel)
    if img.mode != 'L':
        img = img.convert('L')
    img.save(f"other_data/channel_{i}.bmp")