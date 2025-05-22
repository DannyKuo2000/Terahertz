import numpy as np
import matplotlib.pyplot as plt

# 波長與波數
wavelength = 500e-9  # 500 nm
k = 2 * np.pi / wavelength

# 光源位置
source_pos = np.array([0, 0, 0])

# 接收面中心位置
receiver_center = np.array([0, 0, 0.01])  # 1 cm away

# 接收面尺寸
area_size = 0.001  # 1 mm x 1 mm
num_samples = 50   # sample grid 50x50 points
grid = np.linspace(-area_size/2, area_size/2, num_samples)
X, Y = np.meshgrid(grid, grid)

# 所有點的接收位置 (flatten 成列表)
receiver_points = np.stack([X.ravel(), Y.ravel(), np.full(X.size, receiver_center[2])], axis=1)

# 計算每個點到光源的距離與 phasor
vectors = receiver_points - source_pos
r = np.linalg.norm(vectors, axis=1)
phasors = np.exp(1j * k * r) / r

# 平均場 (模擬積分)
avg_field = np.mean(phasors)
intensity_avg = np.abs(avg_field)**2

# 中央點近似
r_center = np.linalg.norm(receiver_center - source_pos)
phasor_center = np.exp(1j * k * r_center) / r_center
intensity_center = np.abs(phasor_center)**2

# 結果比較
print(f"中央點近似強度：      {intensity_center:.5e}")
print(f"面積平均後的強度：    {intensity_avg:.5e}")
print(f"誤差百分比：          {100 * abs(intensity_avg - intensity_center) / intensity_avg:.5f}%")
