"""Generate simple grayscale object images.

The first object is a 400 x 400 black image with a centered 200 x 200
white square.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def generate_center_square(size: int = 400, square_size: int = 200) -> Image.Image:
    """Return a grayscale image with a centered white square.

    Pixels inside the square are 255; all remaining pixels are 0.
    """
    if square_size > size:
        raise ValueError("square_size cannot be larger than size")

    pixels = np.zeros((size, size), dtype=np.uint8)
    start = (size - square_size) // 2
    end = start + square_size
    pixels[start:end, start:end] = 255
    return Image.fromarray(pixels, mode="L")


def main() -> None:
    output_path = Path("other_data/GeneratedObjects/center_square.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_center_square().save(output_path)
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
