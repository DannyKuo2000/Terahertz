import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

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

    for it in range(n_iter):

        # -------- forward --------
        for i in range(N - 1):
            dz = z_list[i + 1] - z_list[i]
            U = angular_spectrum_propagate(U, wavelength, dx, dz)

            # === IMPROVEMENT 2: apply pupil (PSF) ===
            if NA is not None:
                U = apply_pupil(U, wavelength, dx, NA)

            U = amp[i + 1] * np.exp(1j * np.angle(U))

        # -------- backward --------
        for i in range(N - 1, 0, -1):
            dz = z_list[i - 1] - z_list[i]
            U = angular_spectrum_propagate(U, wavelength, dx, dz)

            if NA is not None:
                U = apply_pupil(U, wavelength, dx, NA)

            U = amp[i - 1] * np.exp(1j * np.angle(U))

        # # === IMPROVEMENT 3: phase smoothing ===
        # if it % smooth_every == 0:
        #     phase = np.angle(U)
        #     phase = lowpass_filter(phase, phase_lp_sigma).real
        #     U = np.abs(U) * np.exp(1j * phase)

        if it % 10 == 0:
            print(f"Iteration {it}/{n_iter}")

    return U


if __name__ == "__main__":
    wavelength = 2.998e8 / 0.2004e12   # THz
    dx = 35e-6                         # pixel size (m)

    # 量測距離（一定要對應圖片順序）
    z_list = [0.341, 0.342, 0.343, 0.344, 0.345]

    # 圖片路徑
    img_paths = [
        "other_data/NVLab251224_fixed/camera_34.1.bmp",
        "other_data/NVLab251224_fixed/camera_34.2.bmp",
        "other_data/NVLab251224_fixed/camera_34.3.bmp",
        "other_data/NVLab251224_fixed/camera_34.4.bmp",
        "other_data/NVLab251224_fixed/camera_34.5.bmp",
    ]

    crop_size = 256 # 原圖384*288 裁成256*256能避開不穩的邊界

    Img_list = []
    for p in img_paths:
        Img = load_intensity_image(p)
        Img = center_crop(Img, crop_size)
        Img_list.append(Img)

    U_recon = multi_plane_gs(
        Img_list,
        z_list,
        wavelength,
        dx,
        n_iter=150,
        init_phase="random",
        phase_lp_sigma=0.08,   # 建議 0.05 ~ 0.12
        smooth_every=1,
        NA=0.5               # === 相機 NA，沒概念就 0.1~0.2 試 ===
    )

    phase = np.angle(U_recon)
    amplitude = np.abs(U_recon)
    I_recon = np.abs(U_recon)**2

    # 用重建場預測中間平面
    U_test = angular_spectrum_propagate(U_recon, wavelength, dx, z_list[3] - z_list[0])
    I_test = np.abs(U_test)**2

    U_zero = np.sqrt(Img_list[0]) * np.exp(1j * 0)
    U_test2 = angular_spectrum_propagate(U_zero, wavelength, dx, z_list[3] - z_list[0])
    I_test2 = np.abs(U_test2)**2

    # 真實量測的中間平面
    I_meas = Img_list[3]
    I_start = Img_list[0]

    # 正規化（避免 scale 不一致誤導）
    I_test_n = I_test / (I_test.max() + 1e-12)
    I_test2_n = I_test / (I_test2.max() + 1e-12)
    I_meas_n = I_meas / (I_meas.max() + 1e-12)
    I_start_n = I_start / (I_start.max() + 1e-12)
    # 差異圖
    diff = I_test_n - I_meas_n
    diff2 = I_test2_n - I_meas_n
    diff_1_2 = I_test_n - I_test2_n
    # 誤差指標
    mse = np.mean(diff**2)
    rel_err = np.linalg.norm(diff) / np.linalg.norm(I_meas_n)
    mse2 = np.mean(diff2**2)
    rel_err2 = np.linalg.norm(diff2) / np.linalg.norm(I_meas_n)

    print(f"MSE: {mse:.3e}")
    print(f"Relative error: {rel_err:.3e}")
    print(f"MSE_2: {mse2:.3e}")
    print(f"Relative error_2: {rel_err2:.3e}")

    # 視覺化
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()   # ← 關鍵修正

    axs[0].imshow(I_meas, cmap="gray")
    axs[0].set_title("Measured intensity")
    axs[0].axis("off")

    axs[1].imshow(I_test, cmap="gray")
    axs[1].set_title("Predicted intensity")
    axs[1].axis("off")

    axs[2].imshow(diff, cmap="bwr")
    axs[2].set_title("Difference between measured & reconstructed")
    axs[2].axis("off")

    axs[3].imshow(phase, cmap="gray")
    axs[3].set_title("Reconstructed phase")
    axs[3].axis("off")

    # im = axs[3].imshow(np.log(I_test_n + 1e-6), cmap="gray")
    # axs[3].set_title("Predicted (log scale)")
    # axs[3].axis("off")

    axs[4].imshow(I_start, cmap="gray")
    axs[4].set_title("Start image intensity")
    axs[4].axis("off")

    axs[5].imshow(I_test2, cmap="gray")
    axs[5].set_title("Zero phase intensity")
    axs[5].axis("off")

    axs[6].imshow(diff2, cmap="bwr")
    axs[6].set_title("Difference between measured & zero phase")
    axs[6].axis("off")

    axs[7].imshow(diff_1_2, cmap="bwr")
    axs[7].set_title("Difference between zero phase & reconstructed")
    axs[7].axis("off")
    


    plt.tight_layout()
    plt.show()
