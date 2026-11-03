# This script is designed to be one-time used. Therefore, it can be deleted or modified.
"""Copy PNGs from GroundTruthNew2 Plate folders to the Desktop.

Each source folder must be named ``Plate_XX``.  Its PNG files are copied to
``~/Desktop/GroundTruthNew2_imaging`` and renamed to ``XX_<original name>``.
For example, ``Plate_01/all_final_outputs.png`` becomes
``01_all_final_outputs.png``.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent / "results" / "GroundTruthNew2"
DESTINATION_DIR = Path.home() / "Desktop" / "GroundTruthNew2_imaging"
PLATE_NAME = re.compile(r"Plate_(.+)")


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"找不到來源資料夾：{SOURCE_DIR}")

    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0

    for plate_dir in sorted(SOURCE_DIR.iterdir()):
        if not plate_dir.is_dir():
            continue

        match = PLATE_NAME.fullmatch(plate_dir.name)
        if match is None:
            print(f"略過非 Plate_XX 資料夾：{plate_dir.name}")
            continue

        plate_number = match.group(1)
        png_files = sorted(
            path for path in plate_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )

        if len(png_files) != 2:
            print(f"警告：{plate_dir.name} 找到 {len(png_files)} 個 PNG（預期為 2 個）")

        for source_file in png_files:
            destination_file = DESTINATION_DIR / f"{plate_number}_{source_file.name}"
            shutil.copy2(source_file, destination_file)
            print(f"已複製：{source_file.name} -> {destination_file.name}")
            copied += 1

    print(f"完成：共複製 {copied} 個 PNG 到 {DESTINATION_DIR}")


if __name__ == "__main__":
    main()
