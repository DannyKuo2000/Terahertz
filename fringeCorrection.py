"""Remove a fixed fringe grid by sampling the centre of each grid cell.

Edit INPUT_PATH, OUTPUT_DIR, VERTICAL_FRINGES and HORIZONTAL_FRINGES below,
then run this file. Fringe positions are pixel coordinates: x for vertical
fringes and y for horizontal fringes.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import RectBivariateSpline


# ===================== User settings =====================
# INPUT_PATH = Path("other_data/NVLab260909/NVLab260909_4_fixed/MultiSnap_2026-09-09_17-23-57_0823_average.png")
INPUT_PATH = Path("other_data/NVLab260909/NVLab260909_4_fixed/MultiSnap_2026-09-09_17-23-52_0461_average.png")
OUTPUT_DIR = Path("other_data/NVLab260909/NVLab260909_4_fixed")

# Enter the manually measured fringe locations in pixels.
# Example: VERTICAL_FRINGES = [42, 86, 130]
VERTICAL_FRINGES = [23-1, 61-1, 115-1, 165-1, 216-1, 264-1, 312-1, 359-1]
HORIZONTAL_FRINGES = [24-1, 72-1, 120-1, 168-1, 216-1, 264-1]

# Average this square around each cell centre. Use an odd number.
CENTER_PATCH_SIZE = 3
# ==========================================================


def cell_centres(fringe_positions, image_length):
    """Return cell centres, with the outermost centres on the image edges."""
    positions = sorted({int(round(p)) for p in fringe_positions
                        if 0 < int(round(p)) < image_length - 1})
    # The first and last samples are deliberately attached to the image edges.
    # Interior samples remain at the midpoint between adjacent fringe lines.
    if not positions:
        return np.array([0.0, float(image_length - 1)])
    centres = [0.0]
    centres.extend((left + right) / 2.0
                    for left, right in zip(positions[:-1], positions[1:]))
    centres.append(float(image_length - 1))
    return np.asarray(centres)


def sample_centres(image, x_centres, y_centres, patch_size):
    radius = max(0, int(patch_size) // 2)
    values = []
    points = []
    for y in y_centres:
        for x in x_centres:
            cx, cy = int(round(x)), int(round(y))
            x0, x1 = max(0, cx - radius), min(image.shape[1], cx + radius + 1)
            y0, y1 = max(0, cy - radius), min(image.shape[0], cy + radius + 1)
            points.append((x, y))
            values.append(float(image[y0:y1, x0:x1].mean()))
    return np.asarray(points), np.asarray(values)


def save_fringe_overlay(image, vertical, horizontal, path):
    rgb = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for x in vertical:
        draw.line((x, 0, x, image.shape[0] - 1), fill=(255, 0, 0), width=1)
    for y in horizontal:
        draw.line((0, y, image.shape[1] - 1, y), fill=(255, 0, 0), width=1)
    rgb.save(path)


def save_centres_overlay(image, points, path):
    rgb = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 255, 0))
    rgb.save(path)


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"找不到輸入影像: {INPUT_PATH}")
    if not VERTICAL_FRINGES or not HORIZONTAL_FRINGES:
        raise ValueError("請在程式上方填入 VERTICAL_FRINGES 和 HORIZONTAL_FRINGES")

    image = np.asarray(Image.open(INPUT_PATH).convert("L"), dtype=np.float32)
    height, width = image.shape
    x_centres = cell_centres(VERTICAL_FRINGES, width)
    y_centres = cell_centres(HORIZONTAL_FRINGES, height)
    points, values = sample_centres(image, x_centres, y_centres, CENTER_PATCH_SIZE)

    value_grid = values.reshape(len(y_centres), len(x_centres))
    spline = RectBivariateSpline(
        y_centres,
        x_centres,
        value_grid,
        kx=min(3, len(y_centres) - 1),
        ky=min(3, len(x_centres) - 1),
        s=0,
    )
    reconstructed = spline(np.arange(height), np.arange(width))
    reconstructed = np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(OUTPUT_DIR / "original.png")
    save_fringe_overlay(image.astype(np.uint8), VERTICAL_FRINGES,
                        HORIZONTAL_FRINGES, OUTPUT_DIR / "original_with_fringes.png")
    save_centres_overlay(image.astype(np.uint8), points,
                            OUTPUT_DIR / "centre_points.png")
    Image.fromarray(reconstructed).save(OUTPUT_DIR / "reconstructed.png")
    print(f"找到 {len(x_centres)} x {len(y_centres)} = {len(points)} 個中心點")
    print(f"輸出已儲存至: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
