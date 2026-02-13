#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_dead_pixels.py (學長寫的)
依據提供的壞點座標，使用 8 鄰域平均修復；若鄰居也壞，會自動擴半徑找有效像素。

用法範例：
  # 單張影像 + 直接在命令列指定壞點 (y,x)
  python fix_dead_pixels.py input.png --coords 140,188 144,190 --out repaired.png

  # 單張影像 + 從檔案讀壞點 (每行: y,x)
  python fix_dead_pixels.py input.png --coord-file bad_points.csv --out repaired.png

  # 批次資料夾（只抓 *.png），座標同一份
  python fix_dead_pixels.py ./imgs --glob "*.png" --coord-file bad_points.txt --outdir out

檔案格式說明：
  coord-file：純文字或 CSV，每行一組「y,x」，例如：
      142,187
      143,189
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Set, Tuple, Optional

import numpy as np
from PIL import Image

Coord = Tuple[int, int]  # (y, x)

# -------------------- 座標讀寫 --------------------
def parse_coord_string_list(items: Iterable[str]) -> List[Coord]:
    coords: List[Coord] = []
    for it in items:
        it = it.strip().replace("(", "").replace(")", "")
        if not it:
            continue
        if "," in it:
            y, x = it.split(",", 1)
        else:
            # 若用空白分隔
            y, x = it.split()
        coords.append((int(y), int(x)))
    return coords

def read_coords_from_file(path: Path) -> List[Coord]:
    coords: List[Coord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace("(", "").replace(")", "")
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) >= 2:
                y, x = int(parts[0]), int(parts[1])
                coords.append((y, x))
    return coords

# -------------------- 修復核心 --------------------
def in_bounds(y: int, x: int, H: int, W: int) -> bool:
    return 0 <= y < H and 0 <= x < W

def ring_neighbors(y: int, x: int, r: int) -> List[Coord]:
    """
    回傳半徑 r 的「外圈」鄰居座標（不包含中心與小於 r 的內圈），
    以 8-connected 外圈近似：上下左右 + 四個對角 + 其間點（形成正方形外框）。
    """
    coords: List[Coord] = []
    y0, y1 = y - r, y + r
    x0, x1 = x - r, x + r
    # 上下邊
    for xx in range(x0, x1 + 1):
        coords.append((y0, xx))
        coords.append((y1, xx))
    # 左右邊（去掉角落避免重複）
    for yy in range(y0 + 1, y1):
        coords.append((yy, x0))
        coords.append((yy, x1))
    return coords

def repair_one_pixel(arrf: np.ndarray,
                     y: int, x: int,
                     bad_set: Set[Coord],
                     max_radius: int = 5,
                     prefer_median: bool = True) -> np.ndarray:
    """
    修復單一像素：
    - 先嘗試 r=1 的 8 鄰域；若鄰居也壞或越界，擴到 r=2, 3, ...，只用「外圈」有效像素平均。
    - 若直到 max_radius 都找不到有效像素：回退為該區域的中位數/平均。
    回傳 shape=(C,) 的像素值（浮點）。
    """
    H, W = arrf.shape[:2]
    C = 1 if arrf.ndim == 2 else arrf.shape[2]

    # 逐步擴半徑找有效外圈
    for r in range(1, max_radius + 1):
        cand = ring_neighbors(y, x, r)
        valid_vals = []
        for yy, xx in cand:
            if not in_bounds(yy, xx, H, W):
                continue
            if (yy, xx) in bad_set:
                continue
            if C == 1:
                valid_vals.append(arrf[yy, xx])
            else:
                valid_vals.append(arrf[yy, xx, :])
        if len(valid_vals) > 0:
            vals = np.stack(valid_vals, axis=0).astype(np.float32)
            return vals.mean(axis=0)

    # 仍然沒有：用周邊方窗（最後一個半徑）做 robust 填值
    r = max_radius
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    patch = arrf[y0:y1, x0:x1]
    # 排除已知壞點
    mask = np.ones(patch.shape[:2], dtype=bool)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if (yy, xx) in bad_set:
                mask[yy - y0, xx - x0] = False

    if mask.sum() == 0:
        # 最糟：整塊都壞；直接用原值或全圖均值
        if C == 1:
            return np.array(arrf[y, x], dtype=np.float32)
        else:
            return np.array(arrf[y, x, :], dtype=np.float32)

    if C == 1:
        vals = patch[mask]
    else:
        vals = patch[mask, :]

    if prefer_median:
        return np.median(vals, axis=0)
    else:
        return np.mean(vals, axis=0)

def fix_dead_pixels(arr_u8: np.ndarray,
                    bad_coords: List[Coord],
                    max_radius: int = 5,
                    prefer_median: bool = True) -> np.ndarray:
    """
    主函式：依 bad_coords 修復影像。
    arr_u8: (H,W) 灰階或 (H,W,3) RGB, dtype=uint8
    """
    arrf = arr_u8.astype(np.float32).copy()
    H, W = arrf.shape[:2]
    C = 1 if arrf.ndim == 2 else arrf.shape[2]

    bad_set: Set[Coord] = set([(int(y), int(x)) for y, x in bad_coords if in_bounds(int(y), int(x), H, W)])
    if not bad_set:
        return arr_u8.copy()

    for (y, x) in bad_set:
        new_val = repair_one_pixel(arrf, y, x, bad_set, max_radius=max_radius, prefer_median=prefer_median)
        if C == 1:
            arrf[y, x] = new_val
        else:
            arrf[y, x, :] = new_val

    out = np.clip(arrf, 0, 255).astype(np.uint8)
    return out

# -------------------- I/O 與 CLI --------------------
def load_image_any(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode == "L":
        return np.array(img, dtype=np.uint8)
    return np.array(img.convert("RGB"), dtype=np.uint8)

def save_png(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path, format="PNG")
    else:
        Image.fromarray(arr, mode="RGB").save(path, format="PNG")

def list_inputs(inputs: List[str], glob: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    for p in inputs:
        P = Path(p)
        if P.is_dir():
            if glob:
                paths.extend(sorted(P.glob(glob)))
            else:
                paths.extend(sorted([x for x in P.iterdir() if x.suffix.lower() in (".png",".bmp",".jpg",".jpeg",".tif",".tiff")]))
        else:
            paths.append(P)
    return [x for x in paths if x.suffix.lower() in (".png",".bmp",".jpg",".jpeg",".tif",".tiff")]

def main():
    ap = argparse.ArgumentParser(description="用 8 鄰域平均（含自動擴半徑）修復指定壞點")
    ap.add_argument("inputs", nargs="+", help="影像檔或資料夾")
    ap.add_argument("--glob", default="*.png", help="當輸入是資料夾時的檔案過濾 (預設: *.png)")
    ap.add_argument("--coords", nargs="*", default=[],
                    help="直接指定壞點，如： --coords 142,187 143,189（y,x）")
    ap.add_argument("--coord-file", type=str, default=None, help="壞點座標檔路徑（每行 y,x）")
    ap.add_argument("--out", type=str, default=None, help="單檔輸出路徑（輸入為單一檔時可用）")
    ap.add_argument("--outdir", type=str, default="repaired", help="批次輸出資料夾")
    ap.add_argument("--max-radius", type=int, default=5, help="最大擴張半徑（找有效像素）")
    ap.add_argument("--median", action="store_true", help="無有效像素時偏好使用中位數（預設 True）")
    args = ap.parse_args()

    # 組合座標
    bad_coords: List[Coord] = []
    if args.coord_file:
        bad_coords.extend(read_coords_from_file(Path(args.coord_file)))
    if args.coords:
        bad_coords.extend(parse_coord_string_list(args.coords))
    if not bad_coords:
        raise SystemExit("請提供壞點座標：--coords y,x ... 或 --coord-file path")

    inputs = list_inputs(args.inputs, args.glob)
    if not inputs:
        raise SystemExit("找不到輸入影像")

    if len(inputs) == 1 and args.out:
        # 單檔 + 指定輸出檔
        img = load_image_any(inputs[0])
        fixed = fix_dead_pixels(img, bad_coords, max_radius=args.max_radius, prefer_median=True if args.median else True)
        out_path = Path(args.out)
        if out_path.suffix.lower() == "":
            out_path = out_path.with_suffix(".png")
        save_png(fixed, out_path)
        print(f"Saved -> {out_path}")
    else:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for p in inputs:
            img = load_image_any(p)
            fixed = fix_dead_pixels(img, bad_coords, max_radius=args.max_radius, prefer_median=True if args.median else True)
            out_path = outdir / (p.stem + ".png")
            save_png(fixed, out_path)
            print(f"[{p.name}] -> {out_path}")

if __name__ == "__main__":
    main()
