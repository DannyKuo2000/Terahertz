import math
import torch
import torch.fft
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import zoom
import cv2
from scipy.ndimage import shift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_intensity_image(path): 
    img = imageio.imread(path) 
    if img.ndim == 3: 
        img = img.mean(axis=2) 
    img = img.astype(np.float64) 
    img = img - img.min() 
    img = img / (img.max() + 1e-12) 
    return img

# def load_intensity_image(path):  # 假設相機沒有做gamma correction (不確定是否有做)
#     img = imageio.imread(path)
#     if img.ndim == 3:
#         img = img.mean(axis=2)
#     img = img.astype(np.float64)
#     # print(np.max(img))
#     return img / 255

def center_crop(img, crop_size):
    h, w = img.shape
    y0 = h//2 - crop_size//2
    x0 = w//2 - crop_size//2
    return img[y0:y0+crop_size, x0:x0+crop_size]

def find_center_robust(img, blur_ksize=11, threshold_ratio=0.3):
    """
    img: 2D numpy array (intensity)
    return: (cy, cx)
    """
    img = img.astype(np.float64)

    # 1️⃣ Gaussian blur 去除高頻雜訊
    img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    fig, axs = plt.subplots(1,2, figsize=(10,5))

    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(img_blur, cmap="gray")
    axs[1].set_title("Gaussian Blur")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

    # 2️⃣ threshold 只保留強度高的部分
    threshold = threshold_ratio * np.max(img_blur)
    mask = img_blur > threshold

    if np.sum(mask) == 0:
        raise ValueError("Threshold 太高，沒有有效區域")

    # 3️⃣ 計算質心
    y, x = np.indices(img.shape)

    weighted = img_blur * mask

    cx = np.sum(x * weighted) / np.sum(weighted)
    cy = np.sum(y * weighted) / np.sum(weighted)
    print(f"Y replacement:{cy -  img.shape[0]//2:.1f}, X replacement:{cx - img.shape[1]//2:.1f}")
    return cy, cx

def center_image(img):
    ny, nx = img.shape

    cy, cx = find_center_robust(img)

    # 目標中心
    target_y = ny / 2
    target_x = nx / 2

    shift_y = target_y - cy
    shift_x = target_x - cx

    img_shifted = shift(img, shift=(shift_y, shift_x), order=3)

    return img_shifted, (shift_y, shift_x)

def align_img_list(Img_list):
    """
    Img_list: list of 2D numpy arrays
    return: aligned list
    """

    aligned_list = []
    shifts = []

    for img in Img_list:
        img_aligned, s = center_image(img)
        aligned_list.append(img_aligned)
        shifts.append(s)

    return aligned_list, shifts

def upscale_intensity(img, scale_factor=2, order=3):
    """
    img: 2D numpy array
    scale_factor: 放大倍率 (例如 2, 4)
    order: interpolation order
           1 = bilinear
           3 = bicubic (推薦)
    """
    return zoom(img, zoom=scale_factor, order=order)

def angular_spectrum_propagate(U, wavelength, dx, z, include_evanescent=False):
    # U 必須在 device 上
    ny, nx = U.shape
    fx = torch.fft.fftfreq(nx, d=dx, device=U.device)
    fy = torch.fft.fftfreq(ny, d=dx, device=U.device)
    FY, FX = torch.meshgrid(fx, fy, indexing="ij")  # 代表用x, y的Cartesian為output: x,y => x,y； index="ij": Matrix為output: x,y => y,x

    k = 2 * np.pi / wavelength
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY

    argument = k**2 - kx**2 - ky**2
    kz_real = torch.sqrt(torch.clamp(argument, min=0.0))

    if include_evanescent:
        kz_imag = torch.sqrt(torch.clamp(-argument, min=0.0))
        H = torch.exp(1j * kz_real * z) * torch.exp(-kz_imag * abs(z))
    else:
        H = torch.exp(1j * kz_real * z)

    F = torch.fft.fft2(U)
    U_prop = torch.fft.ifft2(F * H)

    return U_prop

def angular_spectrum_propagate_numpy(U, wavelength, dx, z, include_evanescent=False):
    ny, nx = U.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FY, FX = np.meshgrid(fy, fx)

    k = 2 * np.pi / wavelength
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY

    argument = k**2 - kx**2 - ky**2
    kz_real = np.sqrt(np.maximum(argument, 0.0))

    if include_evanescent:
        kz_imag = np.sqrt(np.maximum(-argument, 0.0))
        H = np.exp(1j * kz_real * z) * np.exp(-kz_imag * abs(z))
    else:
        H = np.exp(1j * kz_real * z)

    F = np.fft.fft2(U)
    U_prop = np.fft.ifft2(F * H)

    return U_prop

def phase_smoothness_loss(U):
    # 一次微分
    dx = torch.angle(U[:, 1:] * torch.conj(U[:, :-1]))
    dy = torch.angle(U[1:, :] * torch.conj(U[:-1, :]))

    # 二次微分
    dxx = dx[:, 1:] - dx[:, :-1]
    dyy = dy[1:, :] - dy[:-1, :]

    return torch.mean(dxx**2) + torch.mean(dyy**2)

    # return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))  # L1 loss
    # return torch.mean(dx**2) + torch.mean(dy**2)  # L2 loss

    # phase = torch.angle(U)
    # dx = phase[:, 1:] - phase[:, :-1]
    # dy = phase[1:, :] - phase[:-1, :]
    # return torch.mean(dx**2) + torch.mean(dy**2)

# ==========================
# Multi-plane optimization version 2
# ==========================
def multi_plane_gradient(Img_list, z_list, wavelength, dx, n_iter=500, lr=5e-3, lambda_phase=1e-3,
        k=0.75,                # planes per iteration
        device='cpu'
):

    ny, nx = Img_list[0].shape

    # amplitude initialization
    A0 = torch.sqrt(Img_list[0])

    # phase initialization
    phase = math.pi * torch.rand((ny, nx), device=device) - 0.5 * math.pi
    phase.requires_grad_(True)

    # optimizer
    optimizer = torch.optim.Adam([phase], lr=lr)

    n_planes = len(z_list) - 1

    count = torch.zeros(n_planes+1)

    for it in range(n_iter):

        optimizer.zero_grad()

        # construct field
        U0 = A0 * torch.exp(1j * phase)

        loss_data = 0

        # ---------------------------
        # curriculum learning stage
        # ---------------------------

        progress = it / n_iter

        if progress < 0.3:
            start_plane = max(1, n_planes - 3)   # far planes
        elif progress < 0.7:
            start_plane = max(1, n_planes - 6)
        else:
            start_plane = 1

        available_planes = torch.arange(start_plane, n_planes + 1)
        num_select = int(len(available_planes) * k)

        # random mini-batch of planes
        perm = torch.randperm(len(available_planes), device=available_planes.device)[:num_select]
        indices = available_planes[perm]

        for i in indices:  # 計算每張圖被選到的次數
            count[i] += 1
        # ---------------------------
        # loss calculation
        # ---------------------------

        for i in indices:

            U_prop = angular_spectrum_propagate(
                U0,
                wavelength,
                dx,
                z_list[i] - z_list[0]
            )

            I_pred = torch.abs(U_prop) ** 2

            loss_data += torch.mean(
                torch.abs(I_pred - Img_list[i])
            )

        loss_data = loss_data / k

        # regularization
        loss_reg = lambda_phase * phase_smoothness_loss(U0)

        loss = loss_data + loss_reg

        loss.backward()
        optimizer.step()

        if it % 100 == 0:

            print(
                f"Iter {it:4d} | "
                f"Loss {loss.item():.4e} | "
                f"Data {loss_data.item():.4e} | "
                f"Reg {loss_reg.item():.4e} | "
                f"Stage start plane {start_plane} | "
            )
            # print(
            #     f"{available_planes} | "
            #     f"{indices} | "
            # )
        # if it % 200 == 0:
        #     print("phase std:", torch.std(phase).item())

    U_final = (A0 * torch.exp(1j * phase)).detach().cpu().numpy()
    print(count)
    return U_final

# ==========================
# Multi-plane optimization version 3
# phase + learnable delta_z
# ==========================
# Legacy fixed-z implementation is kept above unchanged for comparison.
def multi_plane_gradient(Img_list, z_list, wavelength, dx, n_iter=500, lr=5e-3, lambda_phase=1e-3,
        lambda_z=1e3,
        lambda_monotonic=1e5,
        max_z_update=2e-4,
        include_evanescent=False,
        curriculum=False,
        k=0.75,
        device='cpu'
):

    ny, nx = Img_list[0].shape

    # amplitude initialization
    A0 = torch.sqrt(Img_list[0])

    # phase initialization
    phase = math.pi * torch.rand((ny, nx), device=device) - 0.5 * math.pi
    phase.requires_grad_(True)

    # z0 is fixed. We only learn relative corrections for the other planes.
    z_nominal = torch.tensor(z_list, dtype=torch.float32, device=device)  # 名義上的 z
    delta_z = torch.zeros(len(z_list) - 1, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([phase, delta_z], lr=lr)

    n_planes = len(z_list) - 1
    count = torch.zeros(n_planes + 1)

    nominal_spacing = z_nominal[1:] - z_nominal[:-1]
    min_gap = 0.5 * torch.min(nominal_spacing).item()

    for it in range(n_iter):

        optimizer.zero_grad()

        delta_z_limited = max_z_update * torch.tanh(delta_z / max_z_update)
        delta_z_full = torch.cat([  # 加上基準平面
            torch.zeros(1, dtype=torch.float32, device=device),
            delta_z_limited
        ])
        z_current = z_nominal + delta_z_full

        # construct field
        U0 = A0 * torch.exp(1j * phase)

        loss_data = 0

        # --- curriculum training ---
        if curriculum == True:  
            progress = it / n_iter

            if progress < 0.3:
                start_plane = max(1, n_planes - 3)
            elif progress < 0.7:
                start_plane = max(1, n_planes - 6)
            else:
                start_plane = 1

            available_planes = torch.arange(start_plane, n_planes + 1)
            num_select = max(1, int(len(available_planes) * k))
            perm = torch.randperm(len(available_planes), device=available_planes.device)[:num_select]
            indices = available_planes[perm]

            for i in indices:
                count[i] += 1
        else:
            indices = torch.arange(1, n_planes + 1)

        for i in indices:
            U_prop = angular_spectrum_propagate(
                U0,
                wavelength,
                dx,
                z_current[i] - z_current[0],
                include_evanescent=include_evanescent
            )

            I_pred = torch.abs(U_prop) ** 2
            loss_data += torch.mean(torch.abs(I_pred - Img_list[i]))

        loss_data = loss_data / len(indices)

        loss_phase = lambda_phase * phase_smoothness_loss(U0)  # phase smoothness loss
        loss_z = lambda_z * torch.mean(delta_z_limited ** 2)  # replacment loss
        spacing = z_current[1:] - z_current[:-1]
        loss_monotonic = lambda_monotonic * torch.mean(torch.relu(min_gap - spacing) ** 2)
        loss = loss_data + loss_phase + loss_z + loss_monotonic

        loss.backward()
        optimizer.step()

        if it % 100 == 0:
            print(
                f"Iter {it:4d} | "
                f"Loss {loss.item():.4e} | "
                f"Data {loss_data.item():.4e} | "
                f"Phase {loss_phase.item():.4e} | "
                f"Z {loss_z.item():.4e} | "
                f"Mono {loss_monotonic.item():.4e} | "
            )
            if curriculum == True:
                print(f"Stage start plane {start_plane}")
            print(
                "Learned delta_z (mm):",
                np.round(delta_z_full.detach().cpu().numpy() * 1e3, 4)
            )

    U_final = (A0 * torch.exp(1j * phase)).detach().cpu().numpy()
    z_optimized = z_current.detach().cpu().numpy()
    delta_z_final = delta_z_full.detach().cpu().numpy()
    if curriculum == True:
        print(count)
    return U_final, z_optimized, delta_z_final

# ==========================
# Multi-plane optimization version 1
# ==========================
# def multi_plane_gradient(
#         Img_list,      # tensor list (on device)
#         z_list,
#         wavelength,
#         dx,
#         n_iter=500,
#         lr=5e-3,
#         lambda_phase=1e-3,
#         device='cpu'
# ):

#     ny, nx = Img_list[0].shape

#     # 已知 amplitude (from first plane)
#     A0 = torch.sqrt(Img_list[0])

#     # 只優化 phase
#     phase = 1 * math.pi * torch.rand((ny, nx), device=device) - 0.5 * math.pi
#     phase.requires_grad_(True)  # 這樣 phase 就是 leaf tensor

#     optimizer = torch.optim.Adam([phase], lr=lr)

#     for it in range(n_iter):
#         optimizer.zero_grad()

#         # Construct field at first plane
#         U0 = A0 * torch.exp(1j * phase)

#         loss_data = 0

#         # if it % 1000 == 0:  # energy conservation check
#         #     print(torch.sum(torch.abs(U0)**2))
            


#         # 從 z0 propagate 到其他 planes
#         for i in range(1, len(z_list)):

#             U_prop = angular_spectrum_propagate(U0, wavelength, dx, z_list[i] - z_list[0])

#             I_pred = torch.abs(U_prop) ** 2

#             loss_data += torch.mean(torch.abs((I_pred - Img_list[i])))  # !!!

#             # if it % 1000 == 0:  # energy conservation check, seems to be OK
#             #     print(torch.sum(torch.abs(U_prop)**2))
            

#         # phase smoothness regularization
#         loss_reg = lambda_phase * phase_smoothness_loss(U0)

#         loss = loss_data + loss_reg

#         loss.backward()
#         optimizer.step()

#         if it % 100 == 0:
#             print(
#                 f"Iter {it:4d} | "
#                 f"Loss {loss.item():.4e} | "
#                 f"Data {loss_data.item():.4e} | "
#                 f"Reg {loss_reg.item():.4e}"
#             )
#             # print("max intensity diff:", torch.max(torch.abs(I_pred - Img_list[i])))
        
#     U_final = (A0 * torch.exp(1j * phase)).detach().cpu().numpy()

#     return U_final


if __name__ == "__main__":

    z_list = [
        0.340, 
        0.341, 
        0.342, 
        0.343, 
        0.344, 
        0.345, 
        0.346, 
        0.347, 
        0.348, 
        0.349, 
        0.350,
    ]
    # img_paths = [
    #     # "other_data/NVLab260130_fixed/0.2THz_34.0cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.1cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.2cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.3cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.4cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.5cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.6cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.7cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.8cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_34.9cm_1.bmp",
    #     # "other_data/NVLab260130_fixed/0.2THz_35.0cm_1.bmp",
    # ]
    img_paths = [
        "other_data/NVLab260130_fixed/0.2THz_34.0cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.1cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.2cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.3cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.4cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.5cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.6cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.7cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.8cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_34.9cm_2.bmp",
        "other_data/NVLab260130_fixed/0.2THz_35.0cm_2.bmp",
    ]

    # img_paths = [
    #     "other_data/NVLab260130_fixed/0.2THz_34.0cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.1cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.2cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.3cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.4cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.5cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.6cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.7cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.8cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_34.9cm_3.bmp",
    #     "other_data/NVLab260130_fixed/0.2THz_35.0cm_3.bmp",
    # ]

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

    output_dir = "other_data/NVLab260130_results"
    os.makedirs(output_dir, exist_ok=True)

    # Hyperparameters
    wavelength = 2.998e8 / 0.2004e12
    dx = 35e-6
    crop_size = 256
    scale = 1  # this doesn't work now...
    number_iter = 20000
    learning_rate = 1e-4
    # lambda_phase = 0
    lambda_phase = 1e-3
    lambda_z = 1e3
    lambda_monotonic = 1e5
    max_z_update = 3e-4  # 0.5 mm
    include_evanescent = False
    curriculum = True

    Img_list = []
    for p in img_paths:
        Img = load_intensity_image(p)           # 你自己定義的讀圖函數
        Img = center_crop(Img, crop_size)       # 你自己定義的裁切函數
        if scale is not None:
            Img = upscale_intensity(Img, scale_factor=scale)
        Img_list.append(Img)

    # 將影像轉成 numpy 後轉成 tensor 時放 device

    # Img_list_aligned, shifts = align_img_list(Img_list)  # for image center alignment

    Img_list_torch = [torch.tensor(img, dtype=torch.float32, device=device) for img in Img_list]

    # Legacy fixed-z call:
    # U_recon = multi_plane_gradient(
    #     Img_list_torch,
    #     z_list,
    #     wavelength,
    #     dx/scale,
    #     n_iter=number_iter,
    #     lr=learning_rate,
    #     lambda_phase=lambda_phase,
    #     device=device
    # )

    U_recon, z_optimized, delta_z_final = multi_plane_gradient(
        Img_list_torch,
        z_list,
        wavelength,
        dx/scale,
        n_iter=number_iter,
        lr=learning_rate,
        lambda_phase=lambda_phase,
        lambda_z=lambda_z,
        lambda_monotonic=lambda_monotonic,
        max_z_update=max_z_update,
        include_evanescent=include_evanescent,
        curriculum=curriculum,
        device=device
    )
    # 用 numpy 可視化前確保在 CPU
    phase = np.angle(U_recon)
    amplitude = np.abs(U_recon)
    I_recon = amplitude**2
    print("Optimization finished.")
    print("Nominal z_list (mm):", np.round(np.array(z_list) * 1e3, 4))
    print("Optimized z_list (mm):", np.round(z_optimized * 1e3, 4))
    print("delta_z (mm):", np.round(delta_z_final * 1e3, 4))

    I_gens = []
    I_zeros = []
    I_meas = []
    for i in range(1, len(z_list)):

        # === 預測目標平面 ===
        U_test = angular_spectrum_propagate_numpy(
            U_recon, wavelength, dx, z_list[i] - z_list[0],
            include_evanescent=include_evanescent
        )
        I_test = np.abs(U_test)**2
        I_test_n = I_test / (I_test.max() + 1e-12)
        I_gens.append(I_test_n)

        # === 零相位推估 ===
        U_zero = np.sqrt(Img_list[0]) * np.exp(1j * 0)
        U_test2 = angular_spectrum_propagate_numpy(
            U_zero, wavelength, dx, z_list[i] - z_list[0],
            include_evanescent=include_evanescent
        )
        I_test2 = np.abs(U_test2)**2
        I_test2_n = I_test2 / (I_test2.max() + 1e-12)
        I_zeros.append(I_test2_n)

        # === 真實量測的中間平面 ===
        I_meas.append(Img_list[i])

    # === 正規化 (避免亮度 scale 不一致) ===
    # I_test_n = I_test / (I_test.max() + 1e-12)
    # I_test2_n = I_test2 / (I_test2.max() + 1e-12)
    # I_meas_n = I_meas / (I_meas.max() + 1e-12)
    # I_start_n = I_start / (I_start.max() + 1e-12)

    # === 差異圖 ===
    # diff = I_test_n - I_meas_n
    # diff2 = I_test2_n - I_meas_n
    # diff_1_2 = I_test_n - I_test2_n

    # === 誤差指標 ===
    # mse = np.mean(diff**2)
    # rel_err = np.linalg.norm(diff) / np.linalg.norm(I_meas_n)
    # mse2 = np.mean(diff2**2)
    # rel_err2 = np.linalg.norm(diff2) / np.linalg.norm(I_meas_n)

    # === SSIM 指標 ===
    # ssim_1 = ssim(I_meas_n, I_test_n, data_range=1.0)
    # ssim_2 = ssim(I_meas_n, I_test2_n, data_range=1.0)
    # ssim_3 = ssim(I_test2_n, I_test_n, data_range=1.0)

    # print(f"MSE: {mse:.3e}, Relative error: {rel_err:.3e}, SSIM: {ssim_1:.4f}")
    # print(f"MSE_2: {mse2:.3e}, Relative error_2: {rel_err2:.3e}, SSIM_2: {ssim_2:.4f}")
    # print(f"SSIM (zero phase vs reconstructed): {ssim_3:.4f}")

    # # === 可逆存檔 npz ===
    # np.savez(f"{output_dir}/reconstruction_results.npz",
    #         U_recon=U_recon,
    #         phase=phase,
    #         amplitude=amplitude,
    #         I_test=I_test,
    #         I_test2=I_test2,
    #         I_meas=I_meas,
    #         I_start=I_start,
    #         diff=diff,
    #         diff2=diff2,
    #         diff_1_2=diff_1_2,
    #         mse=mse,
    #         rel_err=rel_err,
    #         mse2=mse2,
    #         rel_err2=rel_err2,
    #         ssim_1=ssim_1,
    #         ssim_2=ssim_2,
    #         ssim_3=ssim_3
    # )
    # # === 儲存 metric 到 txt ===
    # with open(f"{output_dir}/reconstruction_metrics.txt", "w") as f:
    #     f.write("=== Reconstruction Metrics ===\n")
    #     f.write(f"MSE (measured vs predicted): {mse:.6e}\n")
    #     f.write(f"Relative error: {rel_err:.6e}\n")
    #     f.write(f"SSIM (measured vs predicted): {ssim_1:.4f}\n\n")
        
    #     f.write(f"MSE_2 (measured vs zero phase): {mse2:.6e}\n")
    #     f.write(f"Relative error_2: {rel_err2:.6e}\n")
    #     f.write(f"SSIM_2 (measured vs zero phase): {ssim_2:.4f}\n\n")
        
    #     f.write(f"SSIM_3 (zero phase vs reconstructed): {ssim_3:.4f}\n")

    def save_png(img, title):
        img = img * 255
        img = img.astype(np.uint8)
        Image.fromarray(img).save(f"{output_dir}/{title}.png")

    images = []

    # 固定圖片
    images.append((I_recon, "Reconstructed_intensity"))
    images.append((phase, "Reconstructed_phase"))
    images.append((Img_list[0], "Measured_intensity_z0"))
    
    # 根據 z_list 動態加入
    for i in range(1, len(z_list)):
        images.append((I_gens[i-1],  f"Generated_intensity_z{i}"))
        images.append((I_zeros[i-1], f"Zero_phase_intensity_z{i}"))
        images.append((I_meas[i-1],  f"Measured_intensity_z{i}"))

    for img_array, title in images:
        save_png(img_array, title)

    # === 視覺化 ===
    fig, axs = plt.subplots(3, len(z_list), figsize=(24, 8))
    axs = axs.flatten(order='F')

    for ax, (img_array, title) in zip(axs, images):

        if title == "Reconstructed_phase":
            phase_display = np.angle(np.exp(1j * phase))
            ax.imshow(phase_display, cmap="twilight")
        else:
            ax.imshow(img_array*255, cmap="gray", vmin=0, vmax=255)  # ← 你要自己決定範圍

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary.png", dpi=300, bbox_inches='tight')
    plt.show()
