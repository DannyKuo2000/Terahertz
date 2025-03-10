"""
This part is a demonstration of:
torch.fft.fft2() and
torch.fft.fftshift()
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
# 生成一個更複雜的圖像 (多個亮點和線條)
size = 128
image = torch.zeros((size, size))

# 中心亮點
image[size//2, size//2] = 1.0

# 增加幾個離散亮點
image[size//4, size//4] = 1.0
image[3*size//4, 3*size//4] = 1.0

# 增加一條水平線
image[size//2, :] = 1.0

# 進行2D傅立葉轉換
fft_image = torch.fft.fft2(image)
fft_image_shifted = torch.fft.fftshift(fft_image)

# 取絕對值並轉換為對數尺度，方便觀察
magnitude_spectrum_not_shifted = torch.log(1 + torch.abs(fft_image))
magnitude_spectrum = torch.log(1 + torch.abs(fft_image_shifted))

# 視覺化原圖和頻譜
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image.numpy(), cmap='gray')

plt.subplot(2, 2, 2)
plt.title("FFT image")
plt.imshow(magnitude_spectrum_not_shifted.numpy(), cmap='gray')

plt.subplot(2, 2, 3)
plt.title("Magnitude Spectrum")
plt.imshow(magnitude_spectrum.numpy(), cmap='gray')

plt.show()


"""
This part is a demonstration of:
torch.fft.fftfreq() and
np.meshgrid()
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
# 生成一個更複雜的圖像 (多個亮點和線條)
size = 128
dx = 0.1  # 取樣間隔
image = torch.zeros((size, size))

# 中心亮點
image[size//2, size//2] = 1.0

# 增加幾個離散亮點
image[size//4, size//4] = 1.0
image[3*size//4, 3*size//4] = 1.0

# 增加一條水平線
image[size//2, :] = 1.0

# 進行2D傅立葉轉換
fft_image = torch.fft.fft2(image)
fft_image_shifted = torch.fft.fftshift(fft_image)

# 取絕對值並轉換為對數尺度，方便觀察
magnitude_spectrum = torch.log(1 + torch.abs(fft_image_shifted))

# 計算頻率軸
fx = np.fft.fftshift(np.fft.fftfreq(size, d=dx))
fxx, fyy = np.meshgrid(fx, fx)

# 視覺化原圖、頻譜與頻率軸
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image.numpy(), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Magnitude Spectrum")
plt.imshow(magnitude_spectrum.numpy(), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Frequency Axes")
plt.contourf(fxx, fyy, np.sqrt(fxx**2 + fyy**2), cmap='viridis')
plt.colorbar(label="Frequency Magnitude")

plt.show()