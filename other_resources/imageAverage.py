import os
import cv2
import numpy as np
import argparse
from datetime import datetime

# ---------- Parse ----------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["average", "one_burst"],
    default="average",
    help="average: 根據檔名分析多個 burst；one_burst: 所有圖片視為同一次連拍"
)
args = parser.parse_args()

input_dir = "other_data/NVLab260612/65"
output_dir = "other_data/NVLab260612_averaged"
os.makedirs(output_dir, exist_ok=True)

exts = (".png", ".bmp", ".jpg", ".jpeg")

# ============================================================
# Mode 1 : one_burst
# ============================================================
if args.mode == "one_burst":

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(exts)
    ])

    imgs = []

    for fname in image_files:
        img = cv2.imread(
            os.path.join(input_dir, fname),
            cv2.IMREAD_UNCHANGED
        )

        if img is None:
            continue

        imgs.append(img.astype(np.float32))

    if len(imgs) == 0:
        raise RuntimeError("No valid images found.")

    stack = np.stack(imgs, axis=0)
    avg_img = stack.mean(axis=0)

    if avg_img.max() > 255:
        avg_img = np.clip(avg_img, 0, 65535).astype(np.uint16)
    else:
        avg_img = np.clip(avg_img, 0, 255).astype(np.uint8)

    out_name = image_files[0]
    out_path = os.path.join(output_dir, out_name)

    cv2.imwrite(out_path, avg_img)

    print(f"Averaged {len(imgs)} images -> {out_name}")

# ============================================================
# Mode 2 : average (原本流程)
# ============================================================
else:

    records = []

    # ---------- 解析檔名 ----------
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(exts):
            continue

        name, _ = os.path.splitext(fname)
        parts = name.split("_")

        # MultiSnap_YYYY-MM-DD_HH-MM-SS_msec_index
        time_str = parts[1] + "_" + parts[2]
        t = datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S")

        burst_idx = int(parts[-1])

        records.append({
            "time": t,
            "burst_idx": burst_idx,
            "path": os.path.join(input_dir, fname)
        })

    # ---------- 依時間排序 ----------
    records.sort(key=lambda x: x["time"])

    # ---------- 重建每一次連拍 ----------
    bursts = []
    current_burst = []

    for r in records:
        if r["burst_idx"] == 0:
            if len(current_burst) > 0:
                bursts.append(current_burst)
            current_burst = [r]
        else:
            current_burst.append(r)

    if len(current_burst) > 0:
        bursts.append(current_burst)

    # ---------- 對每一次連拍做 average ----------
    for i, burst in enumerate(bursts):

        imgs = []

        for r in burst:
            img = cv2.imread(r["path"], cv2.IMREAD_UNCHANGED)

            if img is None:
                continue

            imgs.append(img.astype(np.float32))

        if len(imgs) == 0:
            continue

        stack = np.stack(imgs, axis=0)
        avg_img = stack.mean(axis=0)

        if avg_img.max() > 255:
            avg_img = np.clip(avg_img, 0, 65535).astype(np.uint16)
        else:
            avg_img = np.clip(avg_img, 0, 255).astype(np.uint8)

        ref = min(burst, key=lambda r: r["time"])

        ref_name = os.path.basename(ref["path"])
        name, ext = os.path.splitext(ref_name)

        base = "_".join(name.split("_")[:-1])
        out_name = f"{base}_average{ext}"

        out_path = os.path.join(output_dir, out_name)

        cv2.imwrite(out_path, avg_img)

        print(f"Burst {i}: averaged {len(imgs)} images -> {out_name}")