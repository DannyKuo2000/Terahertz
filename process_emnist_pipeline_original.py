import os
import re
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ==================== 參數設定 ====================
BASE_DIR = './sample_data/EMNIST'

# 1. 原始輸入資料夾
INPUT_DIR = os.path.join(BASE_DIR, '28pixels_choosen')

# 各步驟獨立輸出資料夾（方便微調與手動修圖）
DIR_STEP1 = os.path.join(BASE_DIR, 'step1_resized_250')
DIR_STEP2 = os.path.join(BASE_DIR, 'step2_antialiased')
DIR_STEP3 = os.path.join(BASE_DIR, 'step3_holes_processed')
DIR_STEP4 = os.path.join(BASE_DIR, 'step4_bordered_400')
DIR_STEP5 = os.path.join(BASE_DIR, 'step5_spliced_4x4')

# 拼接相關參數 4*4
BORDER_SIZE = 75              # 步驟 4: 250 + 2*75 = 400
TARGET_INNER_SIZE = (250, 250)
GRID_SIZE = (4, 4)            # 步驟 5: 4x4 拼圖
IMAGES_PER_GROUP = 16         # 4 * 4 = 16
SPLICE_PADDING = 250          # 大圖外圍邊界 (如不需要可改為 0)

# # 拼接相關參數 2*2
# BORDER_SIZE = 100              # 步驟 4: 600 + 2*100 = 800
# TARGET_INNER_SIZE = (250, 250)
# GRID_SIZE = (2, 2)            # 步驟 5: 2x2 拼圖
# IMAGES_PER_GROUP = 4          # 2 * 2 = 4
# SPLICE_PADDING = 250          # 大圖外圍邊界 (如不需要可改為 0)


# ==================== 工具函式 ====================
def natural_sort_key(s):
    """自然排序：避免 10.png 排在 2.png 前面"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_relative_image_paths(root_dir):
    """遞迴取得所有子資料夾內的圖片相對路徑"""
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                full_path = os.path.join(dirpath, f)
                rel_path = os.path.relpath(full_path, root_dir)
                image_paths.append(rel_path)
    return sorted(image_paths, key=natural_sort_key)


# ==================== 步驟 1: 放大至 250x250 ====================
def step1_resize_to_250(input_dir, output_dir):
    print("\n[Step 1] 正在將影像放大至 250x250...")
    rel_files = get_relative_image_paths(input_dir)
    if not rel_files:
        print("⚠️ 找不到輸入圖片！請檢查資料夾路徑。")
        return

    for rel_path in tqdm(rel_files, desc="Step 1 放大中"):
        in_path = os.path.join(input_dir, rel_path)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with Image.open(in_path) as img:
            # 採用 BICUBIC 放大，並二值化保持黑白高對比
            resized = img.convert('L').resize(TARGET_INNER_SIZE, resample=Image.BICUBIC)
            binary = resized.point(lambda p: 255 if p > 128 else 0, mode='L')
            binary.save(out_path)
    print(f"Step 1 完成，儲存於: {output_dir}")

#!  This paragraph is commented and not used
# ==================== 步驟 2: 處理邊緣鋸齒狀 (Anti-aliasing) ====================
# def step2_smooth_edges(input_dir, output_dir):
#     print("\n[Step 2] 正在平滑化與去鋸齒處理...")
#     rel_files = get_relative_image_paths(input_dir)

#     for rel_path in tqdm(rel_files, desc="Step 2 平滑邊緣"):
#         in_path = os.path.join(input_dir, rel_path)
#         out_path = os.path.join(output_dir, rel_path)
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)

#         gray = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

#         # 形態學閉運算平滑邊緣毛刺 + 高斯模糊後重新二值化
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
#         blurred = cv2.GaussianBlur(closed, (5, 5), 0)
#         _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

#         cv2.imwrite(out_path, result)
#     print(f"Step 2 完成，儲存於: {output_dir}")
#     print("提示：如果部分字元邊緣不理想，可在手動修圖後再執行後續步驟。")


def smooth_contour(contour, sigma=2.0):
    """
    對閉合輪廓點序列進行循環高斯濾波，消除階梯鋸齒，生成平滑圓弧。
    sigma 越大越圓潤平滑（建議 1.5 ~ 3.0）。
    """
    if len(contour) < 5:
        return contour
    
    # contour reshape: (N, 1, 2) -> (N, 2)
    pts = contour[:, 0, :].astype(np.float32)
    
    # 閉合輪廓需使用 mode='wrap' 保持首尾無縫連接  #! 對座標進行高斯濾波，不是對圖形，可能可避免兩條黑縫消失
    x_smooth = gaussian_filter1d(pts[:, 0], sigma=sigma, mode='wrap')
    y_smooth = gaussian_filter1d(pts[:, 1], sigma=sigma, mode='wrap')
    
    smoothed = np.stack([x_smooth, y_smooth], axis=1).round().astype(np.int32)
    return smoothed[:, np.newaxis, :]

def step2_vector_smooth(input_dir, output_dir, sigma=2.2):  #! 只能處理黑包白包黑的圖形
    """
    取代原本的形態學/模糊操作：
    1. 抓取文字外輪廓與內部孔洞輪廓
    2. 針對所有輪廓的向量點進行弧度平滑
    3. 重新填色為黑白高對比影像
    """
    print(f"\n[Step 2] 正在進行向量級輪廓平滑處理 (sigma={sigma})...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍歷所有子資料夾與檔案
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            in_path = os.path.join(root, file)
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
            # 取得輪廓拓撲結構（包含外框與內洞）
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            
            # 建立一張空白黑底圖 (與原圖尺寸相同，例如 250x250)
            h, w = binary.shape
            smooth_img = np.zeros((h, w), dtype=np.uint8)
            
            if hierarchy is not None:
                hierarchy = hierarchy[0]
                
                # 第一步：先將平滑後的外輪廓塗成白色 (255)
                for i, c in enumerate(contours):
                    # hierarchy[i][3] == -1 表示最外層輪廓（白字區域）
                    if hierarchy[i][3] == -1:
                        c_smooth = smooth_contour(c, sigma=sigma)
                        cv2.drawContours(smooth_img, [c_smooth], -1, 255, thickness=cv2.FILLED)
                
                # 第二步：再將平滑後的內層輪廓（洞）挖空填回黑色 (0)
                for i, c in enumerate(contours):
                    # hierarchy[i][3] != -1 表示內部孔洞
                    if hierarchy[i][3] != -1:
                        c_smooth = smooth_contour(c, sigma=sigma)
                        cv2.drawContours(smooth_img, [c_smooth], -1, 0, thickness=cv2.FILLED)
            else:
                smooth_img = binary

            cv2.imwrite(out_path, smooth_img)
            
    print(f"Step 2 向量輪廓平滑完成！輸出至: {output_dir}")


# ==================== 步驟 3: 處理孔洞 (如 0, 8 等封閉區域劃線開口) ====================
def step3_process_holes(input_dir, output_dir, cut_line_thickness=5):
    print("\n[Step 3] 正在偵測內部孔洞並橫向切開...")
    rel_files = get_relative_image_paths(input_dir)

    for rel_path in tqdm(rel_files, desc="Step 3 孔洞處理"):
        in_path = os.path.join(input_dir, rel_path)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        gray = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 找黑色連通區域 (0 為黑)
        black_mask = (binary == 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(black_mask, connectivity=8)

        height, width = binary.shape
        half_thick = cut_line_thickness // 2

        for label_id in range(1, num_labels):
            comp_mask = (labels == label_id).astype(np.uint8)

            # 檢查黑區是否接觸圖片四邊
            touches_edge = (np.any(comp_mask[0, :]) or np.any(comp_mask[-1, :]) or
                            np.any(comp_mask[:, 0]) or np.any(comp_mask[:, -1]))

            # 不接觸邊界 = 內部封閉孔洞（如 0、8、B、D、O 等中間的洞）
            if not touches_edge:
                ys, _ = np.where(comp_mask == 1)
                if len(ys) > 0:
                    cy = int(np.mean(ys))
                    # 在孔洞中心畫水平黑線切開
                    y1 = max(0, cy - half_thick)
                    y2 = min(height - 1, cy + half_thick)
                    binary[y1:y2 + 1, :] = 0

        cv2.imwrite(out_path, binary)
    print(f"Step 3 完成，儲存於: {output_dir}")


# ==================== 步驟 4: 拼接 75px 外框 (擴展為 400x400) ====================
def step4_add_border_400(input_dir, output_dir):
    print("\n[Step 4] 正在加上 75px 黑邊框 (250x250 -> 400x400)...")
    rel_files = get_relative_image_paths(input_dir)

    for rel_path in tqdm(rel_files, desc="Step 4 加邊框"):
        in_path = os.path.join(input_dir, rel_path)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with Image.open(in_path) as img:
            target_w = img.width + 2 * BORDER_SIZE
            target_h = img.height + 2 * BORDER_SIZE
            
            # 建立純黑底圖 (400x400)，並將 250x250 居中貼上
            bordered_img = Image.new('L', (target_w, target_h), color=0)
            bordered_img.paste(img, (BORDER_SIZE, BORDER_SIZE))
            bordered_img.save(out_path)

    print(f"Step 4 完成，儲存於: {output_dir}")


# ==================== 步驟 5: 4x4 蛇行拼接成大圖 ====================
def step5_splice_4x4_cross_folders(input_dir, output_dir):
    print("\n[Step 5] 正在跨資料夾彙整所有影像進行 4x4 拼接...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 取得所有子資料夾中的所有圖片（依照自然排序：0, 1, ..., 61）
    all_image_paths = []
    
    # 取得排序後的子資料夾名稱
    subfolders = sorted(
        [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))],
        key=natural_sort_key
    )

    if not subfolders:
        # 若沒有子資料夾，直接讀取根目錄
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_image_paths = sorted(files, key=natural_sort_key)
    else:
        # 依序走訪每個子資料夾，將所有圖片按順序串聯成一個長清單
        for subfolder in subfolders:
            folder_path = os.path.join(input_dir, subfolder)
            files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            files.sort(key=natural_sort_key)
            all_image_paths.extend(files)

    total_images = len(all_image_paths)
    if total_images == 0:
        print(f"⚠️ {input_dir} 中沒有找到可拼接的圖片！")
        return

    # 計算總共需要多少張大圖 (例如 930 張 / 16 = 58.12 -> 59 張)
    num_groups = (total_images + IMAGES_PER_GROUP - 1) // IMAGES_PER_GROUP
    print(f"共載入 {total_images} 張圖片，將依序生成 {num_groups} 張 4x4 拼接圖。")

    # 取得單張圖片尺寸 (400x400)
    with Image.open(all_image_paths[0]) as sample_img:
        img_w, img_h = sample_img.size

    # 計算畫布與邊界尺寸
    collage_w = GRID_SIZE[1] * img_w
    collage_h = GRID_SIZE[0] * img_h
    total_w = collage_w + 2 * SPLICE_PADDING
    total_h = collage_h + 2 * SPLICE_PADDING

    # 2. 進行分組與蛇行拼接
    for g_idx in tqdm(range(num_groups), desc="拼接 4x4 大圖"):
        start_idx = g_idx * IMAGES_PER_GROUP
        end_idx = min(start_idx + IMAGES_PER_GROUP, total_images)
        group = all_image_paths[start_idx:end_idx]

        # 建立純黑畫布
        collage = Image.new('L', (total_w, total_h), color=0)

        for idx, img_path in enumerate(group):
            with Image.open(img_path) as img:
                row = idx // GRID_SIZE[1]
                
                # S 型（蛇行）排版邏輯
                if row % 2 == 0:
                    col = idx % GRID_SIZE[1]                     # 偶數列由左至右
                else:
                    col = (GRID_SIZE[1] - 1) - (idx % GRID_SIZE[1])   # 奇數列由右至左

                x = SPLICE_PADDING + col * img_w
                y = SPLICE_PADDING + row * img_h
                collage.paste(img, (x, y))

        # 輸出命名為 plate_001.png ~ plate_059.png
        out_filename = f"plate_{g_idx + 1:03d}.png"
        collage.save(os.path.join(output_dir, out_filename))

    print(f"Step 5 完成！共輸出 {num_groups} 張拼接圖至：{os.path.abspath(output_dir)}")

# ==================== 主執行程式 ====================
if __name__ == '__main__':
    # 依序執行 5 個步驟（若只想單獨跑某步驟或手動修圖後接續跑，直接將其餘步驟註解即可）：
    
    # # 步驟 1: 放大至 250x250
    step1_resize_to_250(INPUT_DIR, DIR_STEP1)

    # 步驟 2: 邊緣抗鋸齒處理
    step2_vector_smooth(DIR_STEP1, DIR_STEP2)

    # 步驟 3: 內部孔洞破洞切線（預設寬度 5px，可在 250x250 上切出清晰開口）
    step3_process_holes(DIR_STEP2, DIR_STEP3, cut_line_thickness=5)

    # # 步驟 4: 加上 75px 黑色外框，擴大為 400x400
    step4_add_border_400(DIR_STEP3, DIR_STEP4)

    # # 步驟 5: 4x4 (16張) 蛇行拼接成大圖
    step5_splice_4x4_cross_folders(DIR_STEP4, DIR_STEP5)

    print("\n全流程執行完畢！")


    """ Codex Explanation: 
    



    Step 3：偵測封閉孔洞，畫水平黑色切口

    [`step3_process_holes()` (line 176)](C:/Research/TerahertzCode/Terahertz/process_emnist_pipeline.py:176) 讀取 Step 2 的結果：
    1. 把所有黑色像素做 8 鄰接連通區域分析，斜向相接也算連通。
    2. 逐一檢查黑色區域是否接觸圖片四邊。接觸邊界的視為外部背景；完全不接觸邊界的視為封閉孔洞，例如 0、8、B 中間的黑區。
    3. 計算每個孔洞所有像素的平均 y 座標 cy，在該高度畫水平黑帶。
    4. 輸出至 step3_holes_processed。
    目前主程式設定 cut_line_thickness=5，因此會清除 cy-2 到 cy+2，共 5 列像素。

    這裡最需要注意的是切口範圍：
    binary[y1:y2 + 1, :] = 0
    冒號 : 代表整張圖片的寬度。所以它會在孔洞中心高度，橫向切斷該列所有白色筆畫，而不是只在孔洞某一側開一個小缺口。例如 0 通常會同時切開左右兩側；8 若偵測到上下兩個洞，通常會產生兩條黑帶。

    另外，孔洞標籤是在切割前一次算好的，切開第一個洞後不會重新偵測；後續仍依照原本的孔洞清單繼續切割。
    """