import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.colors import Normalize
import os
from scipy.signal import correlate2d



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_phase_structure(matrix, image_path):
    """
    顯示相位矩陣的 circular 自相關圖與傅立葉頻譜，用於檢查是否具有週期性或重複結構。
    matrix: 2D tensor，相位值應落在 [0, 2π)
    """
    matrix = matrix.view(128, 128).cpu().numpy()

    # 將相位轉為複數形式 e^{iθ}
    phase_complex = np.exp(1j * matrix)

    # Circular 自相關（透過 FFT 實現）
    fft_phase = np.fft.fft2(phase_complex)
    power_spectrum = np.abs(fft_phase) ** 2
    autocorr = np.fft.ifft2(power_spectrum).real
    autocorr = np.fft.fftshift(autocorr)

    # 傅立葉轉換頻譜（取模長）
    magnitude_spectrum = np.abs(np.fft.fftshift(fft_phase))

    # 繪圖
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axs[0].imshow(autocorr, cmap='viridis')
    axs[0].set_title("Circular Phase Autocorrelation")
    axs[0].axis('off')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(np.log1p(magnitude_spectrum), cmap='gray')
    axs[1].set_title("Phase FFT Spectrum")
    axs[1].axis('off')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(image_path, exist_ok=True)
    full_path = os.path.join(image_path, "structure.png")
    plt.savefig(full_path)
    plt.show()


if __name__ == "__main__":
    
    weight_path = "../checkpoints/weight_20250506-202256/autoencoder_model.pth"
    weight_name = weight_path.split("/")[-2]
    image_path = "./ONN_weightExtractor_result"
    image_path = os.path.join(f"{image_path}", f"{weight_name}.png")

    # 假設你已經有一個state_dict變數（通常是從torch.load('model.pth')得到）
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    
    # 篩選出所有以 "encoder." 開頭的參數（也就是 encoder 的子模組權重）
    ONN_state_dict = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
    for k in ONN_state_dict.keys():
        print(k)
    
    names = [
        "encoder.layers.1.phase",
        "encoder.layers.3.phase",
        "encoder.layers.5.phase"
    ]

    fig, axs = plt.subplots(1, len(names), figsize=(5 * len(names), 5))

    # 建立 HSV colormap 和 normalization 對應到 [0, 2π]
    cmap = plt.cm.hsv
    norm = Normalize(vmin=0, vmax=2 * np.pi)

    for i, name in enumerate(names):
        weight = ONN_state_dict[name]
        img = weight.view(128, 128).cpu().numpy()

        im = axs[i].imshow(img, cmap=cmap, norm=norm)
        axs[i].set_title(f"Weight: {name}")
        axs[i].axis('off')

    # 加一條共用 colorbar
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal', shrink=0.8, pad=0.05)
    cbar.set_label("Phase (radians)")

    plt.savefig(image_path)
    plt.show()

    visualize_phase_structure(weight, "./ONN_weightExtractor_result")
    