import os
from pathlib import Path

input_dir = "other_data/NVLab260130_fixed"  # 改成你的資料夾
output_dir = "other_data/NVLab260130_fixed"
os.makedirs(output_dir, exist_ok=True)

# 參數
start_dist = 34.0
end_dist = 35.0
step_dist = 0.1
repeat_per_dist = 3
freq_first33 = "0.2THz"   # 前33張
freq_last33 = "0.12THz"   # 後33張

input_dir_path = Path(input_dir)
paths = sorted(input_dir_path.glob("*.*"))  # 保留原始順序

# 生成距離序列
dist_values = [round(start_dist + i*step_dist, 1) for i in range(int((end_dist - start_dist)/step_dist)+1)]

if len(paths) != 66:
    print(f"警告：總共 {len(paths)} 張圖，預期 66 張")

for idx, path in enumerate(paths):
    if idx < 33:  # 前 33 張
        dist_idx = idx // repeat_per_dist
        num_idx = (idx % repeat_per_dist) + 1
        dist_str = f"{dist_values[dist_idx]:.1f}cm"
        freq = freq_first33
    else:         # 後 33 張
        idx2 = idx - 33
        dist_idx = idx2 // repeat_per_dist
        num_idx = (idx2 % repeat_per_dist) + 1
        dist_str = f"{dist_values[dist_idx]:.1f}cm"
        freq = freq_last33

    new_name = f"{freq}_{dist_str}_{num_idx}{path.suffix}"
    save_path = Path(output_dir) / new_name
    path.rename(save_path)
    print(f"{path.name} -> {new_name}")