from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize_like_camera(img, gamma=0.6, clip_percent=0.01):
    """
    Simulate a simple auto-contrast + gamma workflow.
    This function is kept for optional use, but is not used in main().
    """
    img = img.astype(np.float32)

    low = np.percentile(img, clip_percent * 100)
    high = np.percentile(img, (1 - clip_percent) * 100)
    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-8)
    img = img ** gamma
    img = (img * 255).astype(np.uint8)
    return img


def load_raw_image(raw_path, width, height, dtype):
    data = np.fromfile(raw_path, dtype=dtype)
    expected_size = width * height

    if data.size > expected_size:
        print(
            f"[Warning] {raw_path} has {data.size} values; "
            f"truncating extra {data.size - expected_size} values."
        )
        data = data[:expected_size]
    elif data.size < expected_size:
        raise ValueError(
            f"{raw_path} has only {data.size} values; expected {expected_size}. "
            "Please check width, height, or dtype."
        )

    return data.reshape((height, width))


def to_uint8(image):
    image = image.astype(np.float32)
    min_val = float(image.min())
    max_val = float(image.max())

    if max_val - min_val < 1e-8:
        return np.zeros(image.shape, dtype=np.uint8)

    image = (image - min_val) / (max_val - min_val)
    return (image * 255).astype(np.uint8)


def resolve_output_path(input_path, output_path=None, suffix=".bmp"):
    input_path = Path(input_path)

    if output_path is None:
        return input_path.with_suffix(suffix)

    output_path = Path(output_path)

    if output_path.suffix:
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{input_path.stem}{suffix}"


def convert_one_file(raw_path, width, height, dtype, output_path=None, show=False):
    raw_path = Path(raw_path)
    image = load_raw_image(raw_path, width, height, dtype)
    image_uint8 = to_uint8(image)

    save_path = resolve_output_path(raw_path, output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(image_uint8)
    if img.mode != "L":
        img = img.convert("L")
    img.save(save_path)

    if show:
        plt.imshow(image, cmap="gray")
        plt.title(raw_path.name)
        plt.axis("off")
        plt.show()

    print(f"Saved: {save_path}")
    return save_path


def convert_path(input_path, width, height, dtype, output_name=None, show=False):
    input_path = Path(input_path)

    if input_path.is_file():
        return [convert_one_file(input_path, width, height, dtype, output_name, show)]

    if input_path.is_dir():
        raw_files = sorted(input_path.glob("*.raw"))
        if not raw_files:
            raise FileNotFoundError(f"No .raw files found in folder: {input_path}")

        output_dir = None if output_name is None else Path(output_name)
        return [
            convert_one_file(raw_file, width, height, dtype, output_dir, show)
            for raw_file in raw_files
        ]

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def parse_dtype(dtype_name):
    try:
        return np.dtype(dtype_name)
    except TypeError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def main():
    parser = argparse.ArgumentParser(description="Convert RAW image(s) to BMP.")
    parser.add_argument("input_path", help="Path to a .raw file or a folder of .raw files.")
    parser.add_argument("--width", type=int, default=384, help="Image width.")
    parser.add_argument("--height", type=int, default=288, help="Image height.")
    parser.add_argument(
        "--dtype",
        default="float32",
        help="NumPy dtype of the RAW file, e.g. float32, uint16.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Output file path for single-file mode, or output folder for batch mode. "
            "If None, keep the original filename and only change extension to .bmp."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the original image after conversion.",
    )

    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)

    convert_path(
        input_path=args.input_path,
        width=args.width,
        height=args.height,
        dtype=dtype,
        output_name=args.output_name,
        show=args.show,
    )


if __name__ == "__main__":
    main()
