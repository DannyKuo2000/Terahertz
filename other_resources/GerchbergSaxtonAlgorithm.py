import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from scipy.ndimage import zoom


def load_intensity_image(path):
    img = imageio.imread(path)

    # 如果是 RGB，轉灰階
    if img.ndim == 3:
        img = img.mean(axis=2)

    img = img.astype(np.float64)

    # 避免 0（phase retrieval 很怕）
    img = img - img.min()
    img = img / (img.max() + 1e-12)

    return img

def center_crop(img, crop_size):
    """
    crop_size : int or (h, w)
    """
    h, w = img.shape

    if isinstance(crop_size, int):
        ch = cw = crop_size
    else:
        ch, cw = crop_size

    y0 = h // 2 - ch // 2
    x0 = w // 2 - cw // 2

    return img[y0:y0 + ch, x0:x0 + cw]

# === IMPROVEMENT 1: low-pass filter utility ===
def lowpass_filter(field, sigma):
    """
    Apply Gaussian low-pass filter in Fourier domain
    sigma: normalized frequency width (e.g. 0.05 ~ 0.15)
    """
    H, W = field.shape
    fx = np.fft.fftfreq(W)
    fy = np.fft.fftfreq(H)
    FX, FY = np.meshgrid(fx, fy)

    G = np.exp(-(FX**2 + FY**2) / (2 * sigma**2))

    F = np.fft.fft2(field)
    return np.fft.ifft2(F * G)

# === IMPROVEMENT 2: pupil / NA mask ===
def apply_pupil(U, wavelength, dx, NA):
    """
    Apply circular pupil (low-pass) in Fourier domain
    """
    H, W = U.shape
    fx = np.fft.fftfreq(W, d=dx)
    fy = np.fft.fftfreq(H, d=dx)
    FX, FY = np.meshgrid(fx, fy)

    f_cutoff = NA / wavelength
    pupil = (FX**2 + FY**2) <= f_cutoff**2

    F = np.fft.fft2(U)
    return np.fft.ifft2(F * pupil)

def angular_spectrum_propagate(U, wavelength, dx, z, n=1.0, pad_factor=1, mask_evanescent=True):
    """
    Angular Spectrum Propagation (NumPy, 保留原本簡單結果)
    
    U               : complex field (2D np.ndarray)
    wavelength      : wavelength (m)
    dx              : pixel size (m)
    z               : propagation distance (m, 可為負)
    n               : refractive index
    pad_factor      : zero-padding 倍數 (整數 >=1)
    mask_evanescent : 是否遮掉 evanescent
    """
    ny, nx = U.shape
    
    # Step 0: padding
    if pad_factor > 1:
        pad_x = (nx * (pad_factor - 1)) // 2
        pad_y = (ny * (pad_factor - 1)) // 2
        U = np.pad(U, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0.0)
    
    ny_p, nx_p = U.shape
    
    # Step 1: spatial frequency axes (不 shift)
    fx = np.fft.fftfreq(nx_p, d=dx)
    fy = np.fft.fftfreq(ny_p, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    
    # Step 2: wave numbers
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    k = 2 * np.pi * n / wavelength
    
    # Step 3: longitudinal wave number
    argument = k**2 - kx**2 - ky**2
    kz = np.sqrt(np.maximum(0.0, argument))  # 丟掉 evanescent
    if not mask_evanescent:
        # 保留 evanescent 虛數部分
        kz = np.sqrt(np.abs(argument))
        kz = np.where(argument >= 0, kz, 1j * kz)
    
    # Step 4: propagation factor
    H = np.exp(1j * kz * z)
    
    # Step 5: FFT propagation (不 shift)
    F = np.fft.fft2(U)
    U_prop = np.fft.ifft2(F * H)
    
    # Step 6: crop 回原始大小
    if pad_factor > 1:
        start_y = (ny_p - ny) // 2
        start_x = (nx_p - nx) // 2
        U_prop = U_prop[start_y:start_y + ny, start_x:start_x + nx]
    
    return U_prop

# def angular_spectrum_propagate(U, wavelength, dx, z, n=1.0, pad_factor=1, mask_evanescent=True, window=None):
#     """
#     Full-featured Angular Spectrum Propagation (NumPy)
    
#     U               : complex field (2D np.ndarray)
#     wavelength      : wavelength (m)
#     dx              : pixel size (m)
#     z               : propagation distance (m)
#     n               : refractive index
#     pad_factor      : zero-padding 倍數
#     mask_evanescent : 是否遮掉 evanescent
#     reverse_z       : 是否反向傳播 (-z)
#     window          : "hann" or None
#     """
    
#     # -------------------------------
#     # Step 0: padding
#     # -------------------------------
#     ny, nx = U.shape
#     if pad_factor > 1:
#         pad_x = (nx * (pad_factor - 1)) // 2
#         pad_y = (ny * (pad_factor - 1)) // 2
#         U = np.pad(U, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0.0)
    
#     ny_p, nx_p = U.shape
    
#     # -------------------------------
#     # Step 1: spatial frequency axes
#     # -------------------------------
#     fx = np.fft.fftshift(np.fft.fftfreq(nx_p, d=dx))
#     fy = np.fft.fftshift(np.fft.fftfreq(ny_p, d=dx))
#     FX, FY = np.meshgrid(fx, fy)
    
#     # -------------------------------
#     # Step 2: wave numbers
#     # -------------------------------
#     kx = 2 * np.pi * FX
#     ky = 2 * np.pi * FY
#     k = 2 * np.pi * n / wavelength
    
#     # -------------------------------
#     # Step 3: longitudinal wave number
#     # -------------------------------
#     argument = k**2 - kx**2 - ky**2
#     kz = np.sqrt(np.abs(argument))
#     kz = np.where(argument >= 0, kz, 1j * kz)  # evanescent components

#     # -------------------------------
#     # Step 4: propagation factor
#     # -------------------------------
#     H = np.exp(1j * kz * z)
    
#     if mask_evanescent:
#         H = np.where(argument >= 0, H, 0.0)

#     # -------------------------------
#     # Step 5: FFT propagation
#     # -------------------------------
#     F = np.fft.fftshift(np.fft.fft2(U))
#     U_prop = np.fft.ifft2(np.fft.ifftshift(F * H))

#     # -------------------------------
#     # Step 6: optional window
#     # -------------------------------
#     if pad_factor > 1:
#         if window == "hann":
#             wy = np.hanning(ny_p)
#             wx = np.hanning(nx_p)
#             w2d = wy[:, None] * wx[None, :]
#             U_prop = U_prop * w2d
#         # crop 回原始大小
#         start_y = (ny_p - ny) // 2
#         start_x = (nx_p - nx) // 2
#         U_prop = U_prop[start_y:start_y + ny, start_x:start_x + nx]

#     return U_prop

def upscale_intensity(img, scale_factor=2, order=3):
    """
    img: 2D numpy array
    scale_factor: 放大倍率 (例如 2, 4)
    order: interpolation order
           1 = bilinear
           3 = bicubic (推薦)
    """
    return zoom(img, zoom=scale_factor, order=order)

def multi_plane_gs(
    Img_list, z_list, wavelength, dx,
    n_iter=100,
    init_phase="random",
    phase_lp_sigma=0.08,     # === IMPROVEMENT ===
    smooth_every=1,          # === IMPROVEMENT ===
    NA=None                  # === IMPROVEMENT ===
):
    """
    Multi-plane Gerchberg–Saxton with physical regularization
    """

    N = len(Img_list)
    amp = [np.sqrt(Img) for Img in Img_list]

    # === IMPROVEMENT 1: low-pass phase initialization ===
    if init_phase == "lowpass":
        phase0 = np.random.randn(*amp[0].shape)
        phase0 = lowpass_filter(phase0, phase_lp_sigma).real
        phase0 = phase0 / np.std(phase0) * 0.3  # 控制強度
    elif init_phase == "random":
        phase0 = np.random.randn(*amp[0].shape)
    else:
        phase0 = np.zeros_like(amp[0])

    U = amp[0] * np.exp(1j * phase0)

    # for it in range(n_iter):

    #     # -------- forward --------
    #     for i in range(N - 1):
    #         dz = z_list[i + 1] - z_list[i]
    #         U = angular_spectrum_propagate(U, wavelength, dx, dz)

    #         # === IMPROVEMENT 2: apply pupil (PSF) ===
    #         if NA is not None:
    #             U = apply_pupil(U, wavelength, dx, NA)

    #         U = amp[i + 1] * np.exp(1j * np.angle(U))

    #     # -------- backward --------
    #     for i in range(N - 1, 0, -1):
    #         dz = z_list[i - 1] - z_list[i]
    #         U = angular_spectrum_propagate(U, wavelength, dx, dz)

    #         if NA is not None:
    #             U = apply_pupil(U, wavelength, dx, NA)

    #         U = amp[i - 1] * np.exp(1j * np.angle(U))

    #     # === IMPROVEMENT 3: phase smoothing ===
    #     if it % smooth_every == 0:
    #         phase = np.angle(U)
    #         phase = lowpass_filter(phase, phase_lp_sigma).real
    #         U = np.abs(U) * np.exp(1j * phase)

    #     if it % 10 == 0:
    #         print(f"Iteration {it}/{n_iter}")
    for it in range(n_iter):

        # -------- forward --------
        for i in range(N - 1):
            dz = z_list[i + 1] - z_list[i]
            U = angular_spectrum_propagate(U, wavelength, dx, dz)

            if NA is not None:
                U = apply_pupil(U, wavelength, dx, NA)

            U = amp[i + 1] * np.exp(1j * np.angle(U))

            # === 隨機層濾波 ===
            if np.random.rand() < 0.2:   # 機率濾波
                phase = np.angle(U)
                phase = lowpass_filter(phase, phase_lp_sigma).real
                # phase = np.angle(
                #     gaussian_filter(np.cos(phase), sigma=phase_lp_sigma) + 1j * gaussian_filter(np.sin(phase), sigma=phase_lp_sigma)
                # )
                U = np.abs(U) * np.exp(1j * phase)

        # -------- backward --------
        for i in range(N - 1, 0, -1):
            dz = z_list[i - 1] - z_list[i]
            U = angular_spectrum_propagate(U, wavelength, dx, dz)

            if NA is not None:
                U = apply_pupil(U, wavelength, dx, NA)

            U = amp[i - 1] * np.exp(1j * np.angle(U))

            # === 隨機層濾波 ===
            if np.random.rand() < 0.2:   # 機率濾波
                phase = np.angle(U)
                phase = lowpass_filter(phase, phase_lp_sigma).real
                # phase = np.angle(
                #     gaussian_filter(np.cos(phase), sigma=phase_lp_sigma) + 1j * gaussian_filter(np.sin(phase), sigma=phase_lp_sigma)
                # )
                U = np.abs(U) * np.exp(1j * phase)
        if it % 10 == 0:
            print(f"Iteration {it}/{n_iter}")
    return U


if __name__ == "__main__":
    # 量測距離（一定要對應圖片順序）
    z_list = [
        0.340, 
        0.341, 
        0.342, 
        0.343, 
        0.344, 
        # 0.345, 
        # 0.346, 
        # 0.347, 
        # 0.348, 
        # 0.349, 
        # 0.350,
    ]

    # 圖片路徑
    img_paths = [
        # "other_data/NVLab260130_fixed/0.2THz_34.0cm_1.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.1cm_1.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.2cm_1.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.3cm_1.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.4cm_1.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.5cm_1.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.6cm_1.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.7cm_1.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.8cm_1.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.9cm_1.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_35.0cm_1.bmp",
    ]
    img_paths = [
        "other_data/NVLab260130_fixed/0.2THz_34.0cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.1cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.2cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.3cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.4cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.5cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.6cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.7cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.8cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_34.9cm_2.bmp",
        # "other_data/NVLab260130_fixed/0.2THz_35.0cm_2.bmp",
    ]
    # img_paths = [
    #     "other_data/NVLab260130_fixed/0.12THz_34.0cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.1cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.2cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.3cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.4cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.5cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.6cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.7cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.8cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_34.9cm_1.bmp",
    #     "other_data/NVLab260130_fixed/0.12THz_35.0cm_1.bmp",
    # ]

    # === 設定輸出資料夾 ===
    output_dir = "other_data/NVLab260130_results"
    os.makedirs(output_dir, exist_ok=True)

    # === 參數設定 ===
    wavelength = 2.998e8 / 0.2e12   # THz
    dx = 35e-6                         # pixel size (m)
    scale = 2
    crop_size = 256  # 原圖384*288裁成256*256避免不穩定邊界
    n_iter = 200
    phase_lp_sigma = 0.05
    # phase_lp_sigma = 0.00001
    # phase_lp_sigma = 0.12

    # === 讀取圖片並裁切 ===
    Img_list = []
    for p in img_paths:
        Img = load_intensity_image(p)           # 你自己定義的讀圖函數
        Img = center_crop(Img, crop_size)       # 你自己定義的裁切函數
        if scale is not None:
            Img = upscale_intensity(Img, scale_factor=scale)
        Img_list.append(Img)

    # === 多平面 GS 重建 ===
    U_recon = multi_plane_gs(
        Img_list,
        z_list,
        wavelength,
        dx/scale,
        n_iter=n_iter,
        init_phase="random",
        phase_lp_sigma=phase_lp_sigma,
        smooth_every=1,
        NA=None
    )

    # === 提取幅度與相位 ===
    phase = np.angle(U_recon)
    amplitude = np.abs(U_recon)
    I_recon = amplitude**2

    # === 預測目標平面 ===
    U_test = angular_spectrum_propagate(U_recon, wavelength, dx, z_list[4] - z_list[0])
    I_test = np.abs(U_test)**2

    # === 零相位推估 ===
    U_zero = np.sqrt(Img_list[0]) * np.exp(1j * 0)
    U_test2 = angular_spectrum_propagate(U_zero, wavelength, dx, z_list[4] - z_list[0])
    I_test2 = np.abs(U_test2)**2

    # === 真實量測的中間平面 ===
    I_meas = Img_list[4]
    I_start = Img_list[0]

    # === 正規化 (避免亮度 scale 不一致) ===
    I_test_n = I_test / (I_test.max() + 1e-12)
    I_test2_n = I_test2 / (I_test2.max() + 1e-12)
    I_meas_n = I_meas / (I_meas.max() + 1e-12)
    I_start_n = I_start / (I_start.max() + 1e-12)

    # === 差異圖 ===
    diff = I_test_n - I_meas_n
    diff2 = I_test2_n - I_meas_n
    diff_1_2 = I_test_n - I_test2_n

    # === 誤差指標 ===
    mse = np.mean(diff**2)
    rel_err = np.linalg.norm(diff) / np.linalg.norm(I_meas_n)
    mse2 = np.mean(diff2**2)
    rel_err2 = np.linalg.norm(diff2) / np.linalg.norm(I_meas_n)

    # === SSIM 指標 ===
    ssim_1 = ssim(I_meas_n, I_test_n, data_range=1.0)
    ssim_2 = ssim(I_meas_n, I_test2_n, data_range=1.0)
    ssim_3 = ssim(I_test2_n, I_test_n, data_range=1.0)

    print(f"MSE: {mse:.3e}, Relative error: {rel_err:.3e}, SSIM: {ssim_1:.4f}")
    print(f"MSE_2: {mse2:.3e}, Relative error_2: {rel_err2:.3e}, SSIM_2: {ssim_2:.4f}")
    print(f"SSIM (zero phase vs reconstructed): {ssim_3:.4f}")

    # === 可逆存檔 npz ===
    np.savez(f"{output_dir}/reconstruction_results.npz",
            U_recon=U_recon,
            phase=phase,
            amplitude=amplitude,
            I_test=I_test,
            I_test2=I_test2,
            I_meas=I_meas,
            I_start=I_start,
            diff=diff,
            diff2=diff2,
            diff_1_2=diff_1_2,
            mse=mse,
            rel_err=rel_err,
            mse2=mse2,
            rel_err2=rel_err2,
            ssim_1=ssim_1,
            ssim_2=ssim_2,
            ssim_3=ssim_3
    )
    # === 儲存 metric 到 txt ===
    with open(f"{output_dir}/reconstruction_metrics.txt", "w") as f:
        f.write("=== Reconstruction Metrics ===\n")
        f.write(f"MSE (measured vs predicted): {mse:.6e}\n")
        f.write(f"Relative error: {rel_err:.6e}\n")
        f.write(f"SSIM (measured vs predicted): {ssim_1:.4f}\n\n")
        
        f.write(f"MSE_2 (measured vs zero phase): {mse2:.6e}\n")
        f.write(f"Relative error_2: {rel_err2:.6e}\n")
        f.write(f"SSIM_2 (measured vs zero phase): {ssim_2:.4f}\n\n")
        
        f.write(f"SSIM_3 (zero phase vs reconstructed): {ssim_3:.4f}\n")

    # === PNG 存檔 (視覺化用) ===
    def save_png(img, title):
        img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        Image.fromarray(img_norm).save(f"{output_dir}/{title}.png")

    images = [
        (I_meas, "Measured_intensity"),
        (I_test, "Predicted_intensity"),
        (diff, "Difference_measured_vs_predicted"),
        (phase, "Reconstructed_phase"),
        (I_start, "Start_image_intensity"),
        (I_test2, "Zero_phase_intensity"),
        (diff2, "Difference_measured_vs_zero_phase"),
        (diff_1_2, "Difference_zero_vs_reconstructed")
    ]

    for img_array, title in images:
        save_png(img_array, title)

    # === 視覺化 ===
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()

    for ax, (img_array, title) in zip(axs, images):
        ax.imshow(img_array, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
