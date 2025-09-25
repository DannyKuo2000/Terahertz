import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model.opticalSimulation import ResizePadLayer, DiffractiveLayer, LensLayer, RadialAttenuationLayer, CameraLayer


# ============================================================
# Image Loader
# ============================================================
def load_image(path, size=400):
    img = Image.open(path).convert("L")
    print(f"Original size {img.size}")
    ratio = size / min(img.size[0], img.size[1])
    img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.BICUBIC)
    print(f"Resized size {img.size}")
    img_processed = img.crop([img.size[0]//2-size//2, img.size[1]//2-size//2, img.size[0]//2+(size-size//2), img.size[1]//2+(size-size//2)])
    print(f"Processed size {img_processed.size}")
    img_array = np.array(img_processed, dtype=np.float32) / 255.0
    return img_array

# ============================================================
# Image Moving
# ============================================================
import numpy as np

def shift_image(img_array, shift):
    """
    對影像做水平 & 垂直位移
    img_array: numpy 2D (灰階) or 3D (彩色) array, 值域 [0,1]
    shift: (shift_h, shift_w)
        shift_h > 0 向下移, < 0 向上移
        shift_w > 0 向右移, < 0 向左移
    """
    h, w = img_array.shape[:2]
    shift_h, shift_w = shift

    # 建立一張全黑影像
    shifted = np.zeros_like(img_array)

    # 計算有效範圍
    src_y_start = max(0, -shift_h)
    src_y_end   = min(h, h - shift_h)   # 原圖範圍
    dst_y_start = max(0, shift_h)
    dst_y_end   = min(h, h + shift_h)   # 新圖範圍

    src_x_start = max(0, -shift_w)
    src_x_end   = min(w, w - shift_w)
    dst_x_start = max(0, shift_w)
    dst_x_end   = min(w, w + shift_w)

    # 複製有效範圍
    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        img_array[src_y_start:src_y_end, src_x_start:src_x_end]

    return shifted


# ============================================================
# Test Pattern Generator
# ============================================================
def generate_pattern(pattern="circle", size=128):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    if pattern == "circle":
        R = np.sqrt(X**2 + Y**2)
        img = (R < 0.1).astype(np.float32)
    elif pattern == "double_slit":
        img = np.zeros((size, size), dtype=np.float32)
        slit_width = int(size * 0.05)
        slit_height = int(size * 0.6)
        gap = int(size * 0.15)
        y_center = size // 2
        img[y_center - slit_height//2:y_center + slit_height//2,
            size//2 - gap//2 - slit_width:size//2 - gap//2] = 1.0
        img[y_center - slit_height//2:y_center + slit_height//2,
            size//2 + gap//2:size//2 + gap//2 + slit_width] = 1.0
    else:
        raise ValueError("Unknown pattern type. Choose 'circle' or 'double_slit'.")

    return img

# ============================================================
# Main
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### ---- 參數設定 ----
    size = 1024
    frequency = 0.2004e12          # Hz
    #frequency = 500e12 # 可見光
    c = 2.998e8
    wavelength = c / frequency     # ~1.497 mm
    dx = 0.03 / 400               # 模擬平面邊長大小 
    z1 = 0.142                     # 14.2 cm (Sample -> Lens) (Real, 第一部分空氣層寬度)                  
    z2 = 0.041                      # 4.1 cm (Lens -> Camera) (Real, 第二部分空氣層寬度)
    pupil_radius = 0.02375          # 2.54 cm 半徑
    crop_size = 400 #int(size*128/512*1.5) # 特寫寬度
    f = 0.018 # 其中一種理論公式 f = 1 / (1 / z1 + 1 / z2)
    f = 0.029

    ### Sweep (展示多組參數)
    n_scan_x = 5 # x-axis繪製的個數
    plot_list_x = np.linspace(-60, 60, n_scan_x)
    print(f"x-axis: {plot_list_x}")

    n_scan_y = 5 # y-axis繪製的個數
    plot_list_y = np.linspace(z1+0.05, z1+0.1, n_scan_y)
    """plot_list_y = [
        "002",
        "052",
        "102",
        "152",
        "202",
        "252",
        "302",
        "352"
    ]"""
    print(f"y-axis: {plot_list_y}")

    
    


    """
    lens_back = LensLayer(focal_length=float(f),
                         dx=dx,
                         num_size=size,
                         wavelength=float(wavelength),
                         device=device,
                         pupil_type="circular",
                        pupil_radius=pupil_radius,
                         pupil_width=None,
                         phase_model="exact",
                         mode="backward").to(device)
    prop_back1_divide3 = DiffractiveLayer(dx=dx, num_size=size, frequency=frequency, z=z1/3, pad_factor=1, reverse_z=True).to(device)
    prop_back1 = DiffractiveLayer(dx=dx, num_size=size, frequency=frequency, z=z1, pad_factor=1, reverse_z=True).to(device)
    prop_back2 = DiffractiveLayer(dx=dx, num_size=size, frequency=frequency, z=z2, pad_factor=1, reverse_z=True).to(device)
    """
    ### 轉換成電場
    """
    E_no_sample_at_camera = np.sqrt(img_no_sample_at_camera)   # 🔹 取平方根得到電場幅值
    E_no_sample_at_camera = torch.from_numpy(E_no_sample_at_camera).to(device).type(torch.complex64)
    """

    ### 回推金屬板處的原圖像
    """
    E_no_sample_at_sample = prop_back2(E_no_sample_at_camera)
    E_no_sample_at_sample = lens_back(E_no_sample_at_sample)
    E_no_sample_at_sample = prop_back1_divide3(E_no_sample_at_sample)
    E_no_sample_at_sample = prop_back1_divide3(E_no_sample_at_sample)
    E_no_sample_at_sample = prop_back1_divide3(E_no_sample_at_sample)
    """

    ### 建立輸入場（振幅為影像，phase = 0）
    """img_no_sample_at_sample = ((torch.abs(E_no_sample_at_sample)) ** 2).cpu().numpy()
    img_no_sample_at_sample = ((torch.abs(E_no_sample_at_camera)) ** 2).cpu().numpy()
    I0 = img_no_sample_at_sample * img_array"""

    ### 建立可共用的 diffractive, lens, camera layers
    resizepad = ResizePadLayer(resize_size=(400, 400), pad_size=(1024, 1024))
    
    
    prop2 = DiffractiveLayer(dx=dx, num_size=size*2, frequency=frequency, z=z2, 
                             pad_factor=1, keep_pad=True, mask_evanescent=False, multi_step=2,
                             alpha_global=0.0, beta_freq=0.0, use_geom_atten=False).to(device)
    
    camera = CameraLayer(crop_size=size, bin_size=1, flip=True).to(device)
    camera2 = CameraLayer(crop_size=128, bin_size=1, flip=True).to(device)

    
    
    attenuation = RadialAttenuationLayer(R0_ratio=0, exponent=2, min_factor=0)
    
    # 讀入實拍圖像
    print(f"Loading real image")
    img_array = load_image("sample_data/GroundTruth-800-v1/052.png", size=400)
    print(f"Loading ground truth image")
    img_GT = load_image("sample_data/RealDataset-800-v1/052.png")
    #print(f"Loading background image") 
    #img_no_sample_at_camera = load_image("Terahertz/sample_data/Background/Background2025-08-07.png", size=size)
        


    UI_index = 0
    results = []  # list of tuples
    for y in plot_list_y:
        ### 讀入影像 
        prop1 = DiffractiveLayer(dx=dx, num_size=size, frequency=frequency, z=y, 
                             pad_factor=2, keep_pad=True, mask_evanescent=False, multi_step=6,
                             alpha_global=0.0, beta_freq=0.0, use_geom_atten=False).to(device)

        ### 建立 y-axis 不可共用的 layers
        
        
        for x in plot_list_x:
            ### 建立 x-axis 不可共用的 layers
            lens = LensLayer(focal_length=float(f), dx=dx, num_size=size*2,
                     wavelength=wavelength, device=device, pupil_type="circular",
                     pupil_radius=pupil_radius, pupil_width=None, phase_model="exact",
                     mode="forward", outside="one",
                     frame=True, frame_inner=0.02375, frame_outer=0.0254).to(device)

            ### UI
            UI_index += 1
            print(f"===== Running {UI_index}th image =====")
            #results.append(img_array)
            #results.append(img_GT)
            
            
            I0 = shift_image(img_array, (0, int(x)))
            E0 = np.sqrt(I0)   # 🔹 取平方根得到電場幅值
            E0 = torch.from_numpy(E0).to(device).type(torch.complex64)
            E0 = resizepad(E0)

            E = E0
            print(f"Size of the input {E.shape}")

            E = prop1(E)
            print(f"Size of the output of prop1 {E.shape}")

            E = lens(E)
            print(f"Size of the output of lens {E.shape}")
            """E_len_crop = camera2(E)
            I_len = (torch.abs(E) ** 2)
            I_len = I_len.cpu().numpy()
            results.append(I_len)

            I_len_crop = (torch.abs(E_len_crop) ** 2)
            I_len_crop = I_len_crop.cpu().numpy()
            results.append(I_len_crop)"""

            E = prop2(E)
            #print(f"E mean: {E.mean()}")
            #E = attenuation(E)
            #print(f"E mean: {E.mean()}")
            E2 = camera2(E)
            E = camera(E)
            
            I2 = (torch.abs(E2) ** 2)
            I2 = I2.cpu().numpy()
            results.append(I2) 

            #I = (torch.abs(E) ** 2)
            #I = I.cpu().numpy()
            #results.append(I)


    # ---------------------------
    # 繪圖
    # ---------------------------
    ncols = max(len(plot_list_x), 5) # 橫的有幾個
    nrows = max(len(plot_list_y), 5) # 直的有幾個
    print(f"ncols={ncols}, nrows={nrows}")
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))  # 每欄寬與高(吋)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.03, hspace=0.03)

    # Input 放左邊，跨兩列 
    """ax_input = fig.add_subplot(gs[0, 0])
    ax_input.imshow(img_array, cmap="gray")
    ax_input.set_title("Input (original)")
    ax_input.axis("off")

    #ax_input = fig.add_subplot(gs[0, 1])
    #ax_input.imshow(img_no_sample_at_camera, cmap="gray")
    #ax_input.set_title("No sample, view at camera")
    #ax_input.axis("off")

    #ax_input = fig.add_subplot(gs[0, 2])
    #ax_input.imshow(img_no_sample_at_sample, cmap="gray")
    #ax_input.set_title("No sample, real input source at sample")
    #ax_input.axis("off")

    ax_input = fig.add_subplot(gs[0, 3])
    ax_input.imshow(I0, cmap="gray")
    ax_input.set_title("Real input")
    ax_input.axis("off")

    ax_input = fig.add_subplot(gs[0, 4])
    ax_input.imshow(img_GT, cmap="gray")
    ax_input.set_title("Ground Truth")
    ax_input.axis("off")"""
    # 其餘欄：每欄上 full，下 crop
    for i, (full_img) in enumerate(results):
        row_index = i // ncols
        col_index = i % ncols
        print(row_index, col_index)

        ax_full = fig.add_subplot(gs[row_index, col_index])
        ax_full.imshow(full_img, cmap="gray")
        #ax_full.set_title(f"y={plot_list_y[row_index]:.4f}, x={plot_list_x[col_index]:.4f}")
        ax_full.axis("off")
        

    plt.show()


if __name__ == "__main__":
    main()