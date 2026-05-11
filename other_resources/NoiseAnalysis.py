from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


MULTISNAP_PATTERN = re.compile(
    r"^MultiSnap_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<time>\d{2}-\d{2}-\d{2})_"
    r"(?P<msec>\d{4})_"
    r"(?P<frame_idx>\d+)$"
)


@dataclass
class FrameRecord:
    path: Path
    capture_time: datetime
    frame_idx: int
    milliseconds: int


COLOR_NOISE_MODELS = {
    "white": 0.0,
    "pink": 1.0,
    "brown": 2.0,
    "blue": -1.0,
    "violet": -2.0,
}


def parse_multisnap_filename(path: Path) -> FrameRecord | None:
    match = MULTISNAP_PATTERN.match(path.stem)
    if match is None:
        return None

    base_time = datetime.strptime(
        f"{match.group('date')}_{match.group('time')}",
        "%Y-%m-%d_%H-%M-%S",
    )
    milliseconds = int(match.group("msec"))
    capture_time = base_time + timedelta(milliseconds=milliseconds)

    return FrameRecord(
        path=path,
        capture_time=capture_time,
        frame_idx=int(match.group("frame_idx")),
        milliseconds=milliseconds,
    )


def load_image_as_gray(path: Path) -> np.ndarray:
    img = Image.open(path)
    array = np.asarray(img, dtype=np.float64)
    if array.ndim == 3:
        array = np.mean(array, axis=2)
    return array


def collect_frame_records(input_dir: Path, recursive: bool = False) -> list[FrameRecord]:
    pattern = "**/*.bmp" if recursive else "*.bmp"
    records = []

    for bmp_path in sorted(input_dir.glob(pattern)):
        record = parse_multisnap_filename(bmp_path)
        if record is None:
            print(f"[Skip] Filename does not match MultiSnap pattern: {bmp_path.name}")
            continue
        records.append(record)

    records.sort(key=lambda item: (item.capture_time, item.frame_idx, item.path.name))
    return records


def split_into_bursts(records: list[FrameRecord]) -> list[list[FrameRecord]]:
    bursts = []
    current_burst = []

    for record in records:
        if record.frame_idx == 0 and current_burst:
            bursts.append(current_burst)
            current_burst = [record]
        else:
            current_burst.append(record)

    if current_burst:
        bursts.append(current_burst)

    return bursts


def radial_profile(power_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = power_2d.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    y, x = np.indices((h, w))
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    radius_int = radius.astype(np.int32)

    radial_sum = np.bincount(radius_int.ravel(), weights=power_2d.ravel())
    radial_count = np.bincount(radius_int.ravel())
    valid = radial_count > 0

    freqs = np.arange(len(radial_sum), dtype=np.float64)[valid]
    radial_mean = radial_sum[valid] / radial_count[valid]
    return freqs, radial_mean


def radial_profile_with_fixed_length(power_2d: np.ndarray, target_length: int) -> np.ndarray:
    freqs, radial_mean = radial_profile(power_2d)
    profile = np.full(target_length, np.nan, dtype=np.float64)
    usable = min(len(freqs), target_length)
    if usable > 0:
        profile[:usable] = radial_mean[:usable]
    return profile


def fit_color_noise(radial_freq: np.ndarray, radial_psd: np.ndarray) -> dict:
    valid = (
        (radial_freq > 1)
        & np.isfinite(radial_freq)
        & np.isfinite(radial_psd)
        & (radial_psd > 0)
    )
    if np.count_nonzero(valid) < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "alpha": np.nan,
            "noise_color": "unknown",
        }

    x = np.log10(radial_freq[valid])
    y = np.log10(radial_psd[valid])
    slope, intercept = np.polyfit(x, y, deg=1)
    alpha = -slope

    noise_color = min(
        COLOR_NOISE_MODELS,
        key=lambda name: abs(alpha - COLOR_NOISE_MODELS[name]),
    )

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "alpha": float(alpha),
        "noise_color": noise_color,
    }


def build_radial_lookup(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    y, x = np.indices((h, w))
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return radius.astype(np.int32)


def estimate_white_floor(radial_psd: np.ndarray) -> float:
    if radial_psd.size == 0:
        return 0.0
    start = int(0.75 * radial_psd.size)
    high_band = radial_psd[start:] if start < radial_psd.size else radial_psd
    if high_band.size == 0:
        high_band = radial_psd
    return float(np.median(high_band))


def expand_radial_profile_to_2d(radial_profile_1d: np.ndarray, radius_lookup: np.ndarray) -> np.ndarray:
    clipped_radius = np.clip(radius_lookup, 0, len(radial_profile_1d) - 1)
    return radial_profile_1d[clipped_radius]


def split_white_and_colored_noise(
    diff_stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    fft_stack = np.fft.fftshift(np.fft.fft2(diff_stack, axes=(-2, -1)), axes=(-2, -1))
    power_stack = np.abs(fft_stack) ** 2
    mean_power_2d = np.mean(power_stack, axis=0)

    radial_freq, radial_psd = radial_profile(mean_power_2d)
    fit_result = fit_color_noise(radial_freq, radial_psd)
    white_floor = estimate_white_floor(radial_psd)

    white_radial = np.full_like(radial_psd, fill_value=white_floor, dtype=np.float64)
    colored_radial = np.clip(radial_psd - white_radial, a_min=0.0, a_max=None)

    radius_lookup = build_radial_lookup(mean_power_2d.shape)
    total_2d = expand_radial_profile_to_2d(np.maximum(radial_psd, 1e-12), radius_lookup)
    white_2d = expand_radial_profile_to_2d(np.maximum(white_radial, 0.0), radius_lookup)
    white_weight = np.clip(white_2d / total_2d, 0.0, 1.0)

    white_fft = fft_stack * white_weight[None, :, :]
    colored_fft = fft_stack - white_fft

    white_stack = np.fft.ifft2(np.fft.ifftshift(white_fft, axes=(-2, -1)), axes=(-2, -1)).real
    colored_stack = np.fft.ifft2(np.fft.ifftshift(colored_fft, axes=(-2, -1)), axes=(-2, -1)).real

    split_summary = {
        **fit_result,
        "white_floor": white_floor,
        "radial_freq": radial_freq,
        "radial_psd": radial_psd,
        "white_radial": white_radial,
        "colored_radial": colored_radial,
        "white_fraction": float(np.mean(white_weight)),
    }
    return white_stack, colored_stack, split_summary


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val - min_val < 1e-12:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)


def save_image(image: np.ndarray, output_path: Path, cmap: str = "gray") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(image, cmap=cmap)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def get_every_tenth_frame_indices(burst: list[FrameRecord]) -> list[int]:
    selected = {0}
    selected.update(record.frame_idx for record in burst if (record.frame_idx + 1) % 10 == 0)
    burst_indices = {record.frame_idx for record in burst}
    return sorted(frame_idx for frame_idx in selected if frame_idx in burst_indices)


def get_ten_frame_groups(burst: list[FrameRecord]) -> list[list[FrameRecord]]:
    sorted_burst = sorted(burst, key=lambda record: record.frame_idx)
    groups = []
    current_group = []

    for record in sorted_burst:
        current_group.append(record)
        if len(current_group) == 10:
            groups.append(current_group)
            current_group = []

    if current_group:
        groups.append(current_group)

    return groups


def compute_average_image(records: list[FrameRecord]) -> np.ndarray:
    images = [load_image_as_gray(record.path) for record in records]
    if not images:
        raise ValueError("No images found in group.")
    first_shape = images[0].shape
    for image, record in zip(images, records):
        if image.shape != first_shape:
            raise ValueError(f"Image shape mismatch in averaged group at {record.path.name}.")
    return np.mean(np.stack(images, axis=0), axis=0)


def compute_spatial_display_limits(
    bursts: list[list[FrameRecord]],
    original_percentiles: tuple[float, float] = (1.0, 99.0),
    non_dc_percentile: float = 99.0,
) -> dict:
    original_pixels = []
    non_dc_pixels = []

    for burst in bursts:
        frame_map = {record.frame_idx: record for record in burst}
        for frame_idx in get_every_tenth_frame_indices(burst):
            if frame_idx not in frame_map:
                continue

            image = load_image_as_gray(frame_map[frame_idx].path)
            dc_value = float(np.mean(image))
            non_dc_image = image - dc_value

            original_pixels.append(image.reshape(-1))
            non_dc_pixels.append(np.abs(non_dc_image).reshape(-1))

    if not original_pixels:
        return {
            "original_vmin": 0.0,
            "original_vmax": 1.0,
            "non_dc_absmax": 1.0,
        }

    original_concat = np.concatenate(original_pixels)
    non_dc_concat = np.concatenate(non_dc_pixels) if non_dc_pixels else np.array([1.0], dtype=np.float64)

    original_vmin = float(np.percentile(original_concat, original_percentiles[0]))
    original_vmax = float(np.percentile(original_concat, original_percentiles[1]))
    non_dc_absmax = float(np.percentile(non_dc_concat, non_dc_percentile))

    if original_vmax <= original_vmin:
        original_vmax = original_vmin + 1.0
    if non_dc_absmax < 1e-12:
        non_dc_absmax = 1.0

    return {
        "original_vmin": original_vmin,
        "original_vmax": original_vmax,
        "non_dc_absmax": non_dc_absmax,
    }


def compute_averaged_noise_display_limit(
    bursts: list[list[FrameRecord]],
    percentile: float = 99.5,
) -> float:
    averaged_non_dc_pixels = []

    for burst in bursts:
        for group in get_ten_frame_groups(burst):
            avg_image = compute_average_image(group)
            avg_non_dc = avg_image - float(np.mean(avg_image))
            averaged_non_dc_pixels.append(np.abs(avg_non_dc).reshape(-1))

    if not averaged_non_dc_pixels:
        return 1.0

    non_dc_concat = np.concatenate(averaged_non_dc_pixels)
    absmax = float(np.percentile(non_dc_concat, percentile))
    if absmax < 1e-12:
        absmax = 1.0
    return absmax


def compute_single_frame_noise_display_limit(
    bursts: list[list[FrameRecord]],
    percentile: float = 99.5,
) -> float:
    non_dc_pixels = []

    for burst in bursts:
        for record in burst:
            image = load_image_as_gray(record.path)
            non_dc_image = image - float(np.mean(image))
            non_dc_pixels.append(np.abs(non_dc_image).reshape(-1))

    if not non_dc_pixels:
        return 1.0

    non_dc_concat = np.concatenate(non_dc_pixels)
    absmax = float(np.percentile(non_dc_concat, percentile))
    if absmax < 1e-12:
        absmax = 1.0
    return absmax


def smooth_1d(profile: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1:
        return profile.astype(np.float64)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(profile, kernel, mode="same")


def resample_profiles_to_uniform_time(
    times_ms: np.ndarray,
    profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    if len(times_ms) < 2:
        return None, None, None

    sort_idx = np.argsort(times_ms)
    times_ms = np.asarray(times_ms, dtype=np.float64)[sort_idx]
    profiles = np.asarray(profiles, dtype=np.float64)[sort_idx]

    dt_ms = np.diff(times_ms)
    dt_ms = dt_ms[dt_ms > 0]
    if dt_ms.size == 0:
        return None, None, None

    uniform_dt_ms = float(np.median(dt_ms))
    uniform_times_ms = np.arange(times_ms[0], times_ms[-1] + 0.5 * uniform_dt_ms, uniform_dt_ms)
    resampled = np.empty((uniform_times_ms.size, profiles.shape[1]), dtype=np.float64)

    for idx in range(profiles.shape[1]):
        resampled[:, idx] = np.interp(uniform_times_ms, times_ms, profiles[:, idx])

    return uniform_times_ms, resampled, uniform_dt_ms / 1000.0


def save_time_profile_and_temporal_fft(
    burst: list[FrameRecord],
    output_dir: Path,
    burst_id: int,
) -> None:
    if len(burst) < 2:
        return

    burst_label = burst_id + 1
    ref_record = next((record for record in burst if record.frame_idx == 0), burst[0])

    times_ms = []
    column_profiles = []
    row_profiles = []
    frame_indices = []

    for record in sorted(burst, key=lambda item: item.frame_idx):
        image = load_image_as_gray(record.path)
        non_dc_image = image - float(np.mean(image))

        # Compress 288 -> 384 and 384 -> 288 using mean as requested.
        column_profiles.append(np.mean(non_dc_image, axis=0))
        row_profiles.append(np.mean(non_dc_image, axis=1))
        times_ms.append((record.capture_time - ref_record.capture_time).total_seconds() * 1000.0)
        frame_indices.append(record.frame_idx)

    times_ms = np.asarray(times_ms, dtype=np.float64)
    column_profiles = np.asarray(column_profiles, dtype=np.float64)
    row_profiles = np.asarray(row_profiles, dtype=np.float64)

    def save_profile_heatmap(profiles: np.ndarray, spatial_label: str, output_name: str) -> None:
        vmax = float(np.nanpercentile(np.abs(profiles), 99))
        vmax = max(vmax, 1e-12)

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(
            profiles.T,
            aspect="auto",
            origin="lower",
            extent=[times_ms[0], times_ms[-1], 0, profiles.shape[1] - 1],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel("Relative Time from frame 0000 (ms)")
        ax.set_ylabel(spatial_label)
        ax.set_title(f"Burst {burst_label} {spatial_label} Mean Profile Over Time")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean non-DC intensity")
        fig.tight_layout()
        fig.savefig(output_dir / output_name, dpi=200)
        plt.close(fig)

    save_profile_heatmap(
        column_profiles,
        spatial_label="column index",
        output_name=f"burst_{burst_label:03d}_column_profile_over_time.png",
    )
    save_profile_heatmap(
        row_profiles,
        spatial_label="row index",
        output_name=f"burst_{burst_label:03d}_row_profile_over_time.png",
    )

    def save_temporal_fft_heatmap(
        profiles: np.ndarray,
        spatial_label: str,
        output_name: str,
        csv_name: str,
    ) -> None:
        uniform_times_ms, resampled_profiles, dt_sec = resample_profiles_to_uniform_time(times_ms, profiles)
        if uniform_times_ms is None or resampled_profiles is None or dt_sec is None:
            return

        detrended = resampled_profiles - np.mean(resampled_profiles, axis=0, keepdims=True)
        fft_values = np.fft.rfft(detrended, axis=0)
        power = np.abs(fft_values) ** 2
        freqs_hz = np.fft.rfftfreq(detrended.shape[0], d=dt_sec)

        # Focus visualization on temporal variation; omit DC bin from the FFT map.
        if power.shape[0] > 1:
            power = power[1:, :]
            freqs_hz = freqs_hz[1:]
        if power.size == 0:
            return

        log_power = np.log10(np.maximum(power, 1e-12))
        vmin = float(np.nanpercentile(log_power, 5))
        vmax = float(np.nanpercentile(log_power, 95))
        if vmax <= vmin:
            vmax = vmin + 1.0

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(
            log_power.T,
            aspect="auto",
            origin="lower",
            extent=[freqs_hz[0], freqs_hz[-1], 0, log_power.shape[1] - 1],
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Temporal frequency (Hz)")
        ax.set_ylabel(spatial_label)
        ax.set_title(f"Burst {burst_label} {spatial_label} Temporal FFT")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log10(power)")
        fig.tight_layout()
        fig.savefig(output_dir / output_name, dpi=200)
        plt.close(fig)

        with (output_dir / csv_name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["temporal_frequency_hz"] + [f"{spatial_label}_{i}" for i in range(log_power.shape[1])])
            for freq_hz, row in zip(freqs_hz, log_power):
                writer.writerow([freq_hz, *row.tolist()])

        mean_power = np.mean(power, axis=1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(freqs_hz, mean_power, linewidth=1.8)
        ax.set_xlabel("Temporal frequency (Hz)")
        ax.set_ylabel("Mean temporal FFT power")
        ax.set_title(f"Burst {burst_label} Mean {spatial_label} Temporal FFT")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"burst_{burst_label:03d}_{spatial_label.replace(' ', '_')}_temporal_fft_mean.png",
            dpi=200,
        )
        plt.close(fig)

    save_temporal_fft_heatmap(
        column_profiles,
        spatial_label="column index",
        output_name=f"burst_{burst_label:03d}_column_temporal_fft.png",
        csv_name=f"burst_{burst_label:03d}_column_temporal_fft.csv",
    )
    save_temporal_fft_heatmap(
        row_profiles,
        spatial_label="row index",
        output_name=f"burst_{burst_label:03d}_row_temporal_fft.png",
        csv_name=f"burst_{burst_label:03d}_row_temporal_fft.csv",
    )

def analyze_stripe_motion_for_burst(
    burst: list[FrameRecord],
    output_dir: Path,
    burst_id: int,
    profile_smooth_window: int = 9,
) -> None:
    burst_label = burst_id + 1
    rows = []
    vertical_profiles = []
    horizontal_profiles = []

    ref_record = next((record for record in burst if record.frame_idx == 0), burst[0])

    for record in sorted(burst, key=lambda item: item.frame_idx):
        image = load_image_as_gray(record.path)
        non_dc_image = image - float(np.mean(image))

        vertical_profile = smooth_1d(np.mean(non_dc_image, axis=0), window=profile_smooth_window)
        horizontal_profile = smooth_1d(np.mean(non_dc_image, axis=1), window=profile_smooth_window)

        vertical_profiles.append(vertical_profile)
        horizontal_profiles.append(horizontal_profile)

        rel_time_ms = (record.capture_time - ref_record.capture_time).total_seconds() * 1000.0
        vertical_peak_x = int(np.argmax(vertical_profile))
        vertical_valley_x = int(np.argmin(vertical_profile))
        horizontal_peak_y = int(np.argmax(horizontal_profile))
        horizontal_valley_y = int(np.argmin(horizontal_profile))

        rows.append(
            {
                "burst_id": burst_label,
                "frame_idx": record.frame_idx,
                "relative_time_ms": rel_time_ms,
                "vertical_peak_x": vertical_peak_x,
                "vertical_peak_value": float(vertical_profile[vertical_peak_x]),
                "vertical_valley_x": vertical_valley_x,
                "vertical_valley_value": float(vertical_profile[vertical_valley_x]),
                "horizontal_peak_y": horizontal_peak_y,
                "horizontal_peak_value": float(horizontal_profile[horizontal_peak_y]),
                "horizontal_valley_y": horizontal_valley_y,
                "horizontal_valley_value": float(horizontal_profile[horizontal_valley_y]),
            }
        )

    csv_path = output_dir / f"burst_{burst_label:03d}_stripe_motion.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    times_ms = [row["relative_time_ms"] for row in rows]
    vertical_positions = np.array([row["vertical_peak_x"] for row in rows], dtype=np.float64)
    horizontal_positions = np.array([row["horizontal_peak_y"] for row in rows], dtype=np.float64)
    vertical_peak_values = np.array([row["vertical_peak_value"] for row in rows], dtype=np.float64)
    horizontal_peak_values = np.array([row["horizontal_peak_value"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].plot(times_ms, vertical_positions, marker="o", linewidth=1.5)
    axes[0, 0].set_title("Vertical Stripe Peak Position")
    axes[0, 0].set_xlabel("Relative Time (ms)")
    axes[0, 0].set_ylabel("x position")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)

    axes[0, 1].plot(times_ms, horizontal_positions, marker="o", linewidth=1.5)
    axes[0, 1].set_title("Horizontal Stripe Peak Position")
    axes[0, 1].set_xlabel("Relative Time (ms)")
    axes[0, 1].set_ylabel("y position")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    axes[1, 0].plot(times_ms, vertical_peak_values, marker="o", linewidth=1.5)
    axes[1, 0].set_title("Vertical Stripe Peak Strength")
    axes[1, 0].set_xlabel("Relative Time (ms)")
    axes[1, 0].set_ylabel("mean non-DC intensity")
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    axes[1, 1].plot(times_ms, horizontal_peak_values, marker="o", linewidth=1.5)
    axes[1, 1].set_title("Horizontal Stripe Peak Strength")
    axes[1, 1].set_xlabel("Relative Time (ms)")
    axes[1, 1].set_ylabel("mean non-DC intensity")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Burst {burst_label} Stripe Motion Tracking")
    fig.tight_layout()
    fig.savefig(output_dir / f"burst_{burst_label:03d}_stripe_motion_summary.png", dpi=200)
    plt.close(fig)

    vertical_profiles = np.asarray(vertical_profiles, dtype=np.float64)
    horizontal_profiles = np.asarray(horizontal_profiles, dtype=np.float64)

    vertical_vmax = float(np.nanpercentile(np.abs(vertical_profiles), 99))
    horizontal_vmax = float(np.nanpercentile(np.abs(horizontal_profiles), 99))
    vertical_vmax = max(vertical_vmax, 1e-12)
    horizontal_vmax = max(horizontal_vmax, 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    im0 = axes[0].imshow(
        vertical_profiles.T,
        aspect="auto",
        origin="lower",
        extent=[times_ms[0], times_ms[-1], 0, vertical_profiles.shape[1] - 1],
        cmap="seismic",
        vmin=-vertical_vmax,
        vmax=vertical_vmax,
    )
    axes[0].set_title("Vertical Stripe Profile Over Time")
    axes[0].set_xlabel("Relative Time (ms)")
    axes[0].set_ylabel("x position")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="mean non-DC intensity")

    im1 = axes[1].imshow(
        horizontal_profiles.T,
        aspect="auto",
        origin="lower",
        extent=[times_ms[0], times_ms[-1], 0, horizontal_profiles.shape[1] - 1],
        cmap="seismic",
        vmin=-horizontal_vmax,
        vmax=horizontal_vmax,
    )
    axes[1].set_title("Horizontal Stripe Profile Over Time")
    axes[1].set_xlabel("Relative Time (ms)")
    axes[1].set_ylabel("y position")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="mean non-DC intensity")

    fig.tight_layout()
    fig.savefig(output_dir / f"burst_{burst_label:03d}_stripe_profile_over_time.png", dpi=200)
    plt.close(fig)


def save_selected_frame_spectra(
    burst: list[FrameRecord],
    output_dir: Path,
    burst_id: int,
) -> None:
    burst_label = burst_id + 1
    groups = get_ten_frame_groups(burst)

    for group in groups:
        start_idx = group[0].frame_idx
        end_idx = group[-1].frame_idx
        image = compute_average_image(group)

        fft_2d = np.fft.fftshift(np.fft.fft2(image))
        power_2d = np.abs(fft_2d) ** 2
        radial_freq, radial_psd = radial_profile(power_2d)
        fit_result = fit_color_noise(radial_freq, radial_psd)
        radial_psd = np.maximum(radial_psd, 1e-12)
        valid = (radial_freq > 0) & np.isfinite(radial_psd) & (radial_psd > 0)

        fig, ax = plt.subplots(figsize=(8, 5))
        if np.any(valid):
            ax.loglog(radial_freq[valid], radial_psd[valid], linewidth=2)
        ax.set_xlabel("Radial frequency (pixel index)")
        ax.set_ylabel("Power spectral density")
        ax.set_title(
            f"Burst {burst_label} Avg {start_idx:04d}~{end_idx:04d} Spectrum | "
            f"{fit_result['noise_color']} noise (alpha={fit_result['alpha']:.3f})"
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"burst_{burst_label:03d}_frames_{start_idx:04d}_{end_idx:04d}_avg_spectrum.png",
            dpi=200,
        )
        plt.close(fig)


def save_combined_selected_frame_spectra(
    burst: list[FrameRecord],
    output_dir: Path,
    burst_id: int,
) -> None:
    burst_label = burst_id + 1
    groups = get_ten_frame_groups(burst)

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    num_lines = max(len(groups), 1)

    for line_idx, group in enumerate(groups):
        start_idx = group[0].frame_idx
        end_idx = group[-1].frame_idx
        image = compute_average_image(group)

        fft_2d = np.fft.fftshift(np.fft.fft2(image))
        power_2d = np.abs(fft_2d) ** 2
        radial_freq, radial_psd = radial_profile(power_2d)
        radial_psd = np.maximum(radial_psd, 1e-12)
        valid = (radial_freq > 0) & np.isfinite(radial_psd) & (radial_psd > 0)
        color = cmap(line_idx / max(num_lines - 1, 1))

        if np.any(valid):
            ax.loglog(
                radial_freq[valid],
                radial_psd[valid],
                linewidth=1.8,
                color=color,
                label=f"{start_idx:04d}~{end_idx:04d}",
            )

    ax.set_title(f"Burst {burst_label} Averaged Spectra Overlay")
    ax.set_xlabel("Radial frequency")
    ax.set_ylabel("PSD")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"burst_{burst_label:03d}_averaged_spectra_overlay.png", dpi=200)
    plt.close(fig)


def save_selected_frame_spatial_decomposition(
    burst: list[FrameRecord],
    output_dir: Path,
    burst_id: int,
    display_limits: dict,
) -> None:
    frame_map = {record.frame_idx: record for record in burst}
    burst_label = burst_id + 1
    selected_indices = get_every_tenth_frame_indices(burst)

    for frame_idx in selected_indices:
        if frame_idx not in frame_map:
            continue

        record = frame_map[frame_idx]
        image = load_image_as_gray(record.path)
        dc_value = float(np.mean(image))
        dc_image = np.full_like(image, dc_value, dtype=np.float64)
        non_dc_image = image - dc_image

        orig_vmin = display_limits["original_vmin"]
        orig_vmax = display_limits["original_vmax"]
        non_dc_absmax = display_limits["non_dc_absmax"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im0 = axes[0].imshow(image, cmap="gray", vmin=orig_vmin, vmax=orig_vmax)
        axes[0].set_title(f"Original\nmean={dc_value:.3f}")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(dc_image, cmap="gray", vmin=orig_vmin, vmax=orig_vmax)
        axes[1].set_title("DC Part")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(
            non_dc_image,
            cmap="seismic",
            vmin=-non_dc_absmax,
            vmax=non_dc_absmax,
        )
        axes[2].set_title("Non-DC Part")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"Burst {burst_label} Frame {frame_idx:04d} Spatial Decomposition")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"burst_{burst_label:03d}_frame_{frame_idx:04d}_spatial_decomposition.png",
            dpi=200,
        )
        plt.close(fig)


def save_averaged_noise_spatial_maps(
    burst: list[FrameRecord],
    output_dir: Path,
    burst_id: int,
    noise_absmax: float,
) -> None:
    burst_label = burst_id + 1
    groups = get_ten_frame_groups(burst)

    for group in groups:
        start_idx = group[0].frame_idx
        end_idx = group[-1].frame_idx
        avg_image = compute_average_image(group)
        avg_non_dc = avg_image - float(np.mean(avg_image))

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            avg_non_dc,
            cmap="turbo",
            vmin=-noise_absmax,
            vmax=noise_absmax,
        )
        ax.set_title(f"Burst {burst_label} Avg Noise Map {start_idx:04d}~{end_idx:04d}")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensity (non-DC)")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"burst_{burst_label:03d}_frames_{start_idx:04d}_{end_idx:04d}_avg_noise_map.png",
            dpi=200,
        )
        plt.close(fig)


def save_single_frame_noise_spatial_maps(
    burst: list[FrameRecord],
    output_dir: Path,
    burst_id: int,
    noise_absmax: float,
) -> None:
    burst_label = burst_id + 1

    for record in sorted(burst, key=lambda item: item.frame_idx):
        image = load_image_as_gray(record.path)
        non_dc_image = image - float(np.mean(image))

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            non_dc_image,
            cmap="turbo",
            vmin=-noise_absmax,
            vmax=noise_absmax,
        )
        ax.set_title(f"Burst {burst_label} Frame {record.frame_idx:04d} Noise Map")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensity (non-DC)")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"burst_{burst_label:03d}_frame_{record.frame_idx:04d}_noise_map.png",
            dpi=200,
        )
        plt.close(fig)


def save_spectrum_over_time(
    burst: list[FrameRecord],
    output_dir: Path,
    burst_id: int,
) -> None:
    if not burst:
        return

    burst_label = burst_id + 1
    frame_map = {record.frame_idx: record for record in burst}
    max_shape = max(load_image_as_gray(record.path).shape[0] for record in burst)
    max_shape = max(max_shape, max(load_image_as_gray(record.path).shape[1] for record in burst))

    ref_record = next((record for record in burst if record.frame_idx == 0), burst[0])
    times_ms = []
    spectra = []
    frame_indices = []

    for record in sorted(burst, key=lambda item: item.frame_idx):
        image = load_image_as_gray(record.path)
        fft_2d = np.fft.fftshift(np.fft.fft2(image))
        power_2d = np.abs(fft_2d) ** 2
        spectrum = radial_profile_with_fixed_length(power_2d, target_length=max_shape)

        rel_time_ms = (record.capture_time - ref_record.capture_time).total_seconds() * 1000.0
        times_ms.append(rel_time_ms)
        spectra.append(spectrum)
        frame_indices.append(record.frame_idx)

    spectra_array = np.asarray(spectra, dtype=np.float64)
    spectra_array = np.maximum(spectra_array, 1e-12)
    log_spectra = np.log10(spectra_array)

    valid_freq_mask = np.any(np.isfinite(log_spectra), axis=0)
    if np.count_nonzero(valid_freq_mask) == 0:
        return

    freq_axis = np.arange(log_spectra.shape[1], dtype=np.float64)
    freq_axis = freq_axis[valid_freq_mask]
    log_spectra = log_spectra[:, valid_freq_mask]

    vmin = float(np.nanpercentile(log_spectra, 5))
    vmax = float(np.nanpercentile(log_spectra, 95))
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.imshow(
        log_spectra.T,
        aspect="auto",
        origin="lower",
        extent=[times_ms[0], times_ms[-1], freq_axis[0], freq_axis[-1]],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Relative Time from frame 0000 (ms)")
    ax.set_ylabel("Radial frequency (pixel index)")
    ax.set_title(f"Burst {burst_label} Spectrum Evolution")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(PSD)")
    fig.tight_layout()
    fig.savefig(output_dir / f"burst_{burst_label:03d}_spectrum_over_time.png", dpi=200)
    plt.close(fig)

    csv_path = output_dir / f"burst_{burst_label:03d}_spectrum_over_time.csv"
    header = ["frame_idx", "relative_time_ms"] + [f"freq_{int(freq)}" for freq in freq_axis]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame_idx, rel_time_ms, spectrum_row in zip(frame_indices, times_ms, log_spectra):
            writer.writerow([frame_idx, rel_time_ms, *spectrum_row.tolist()])


def analyze_burst(burst: list[FrameRecord], burst_id: int) -> tuple[list[dict], dict]:
    if not burst:
        return [], {}

    ref_record = next((record for record in burst if record.frame_idx == 0), burst[0])
    ref_image = load_image_as_gray(ref_record.path)
    stack = []
    diff_stack = []
    rows = []

    for record in burst:
        image = load_image_as_gray(record.path)
        if image.shape != ref_image.shape:
            raise ValueError(
                f"Image shape mismatch in burst {burst_id}: "
                f"{record.path.name} has shape {image.shape}, expected {ref_image.shape}."
            )

        rel_time_ms = (record.capture_time - ref_record.capture_time).total_seconds() * 1000.0
        diff = image - ref_image
        stack.append(image)
        diff_stack.append(diff)
        rows.append(
            {
                "burst_id": burst_id,
                "filename": record.path.name,
                "frame_idx": record.frame_idx,
                "capture_time": record.capture_time.isoformat(timespec="milliseconds"),
                "relative_time_ms": rel_time_ms,
                "mean": float(np.mean(image)),
                "std": float(np.std(image)),
                "variance": float(np.var(image)),
                "min": float(np.min(image)),
                "max": float(np.max(image)),
                "median": float(np.median(image)),
                "mean_abs_diff_from_0000": float(np.mean(np.abs(diff))),
                "rmse_from_0000": float(np.sqrt(np.mean(diff ** 2))),
                "drift_mean_from_0000": float(np.mean(diff)),
            }
        )

    stack_array = np.stack(stack, axis=0)
    diff_stack_array = np.stack(diff_stack, axis=0)
    temporal_std_map = np.std(stack_array, axis=0)
    white_stack, colored_stack, split_summary = split_white_and_colored_noise(diff_stack_array)
    burst_summary = {
        "burst_id": burst_id,
        "reference_file": ref_record.path.name,
        "num_frames": len(burst),
        "reference_time": ref_record.capture_time.isoformat(timespec="milliseconds"),
        "duration_ms": float(rows[-1]["relative_time_ms"] if rows else 0.0),
        "mean_std": float(np.mean([row["std"] for row in rows])),
        "max_std": float(np.max([row["std"] for row in rows])),
        "mean_rmse_from_0000": float(np.mean([row["rmse_from_0000"] for row in rows])),
        "temporal_std_mean": float(np.mean(temporal_std_map)),
        "temporal_std_max": float(np.max(temporal_std_map)),
        "noise_color": split_summary["noise_color"],
        "psd_slope": split_summary["slope"],
        "psd_alpha": split_summary["alpha"],
        "white_fraction": split_summary["white_fraction"],
        "temporal_std_map": temporal_std_map,
    }
    return rows, burst_summary


def save_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(summaries: list[dict], csv_path: Path) -> None:
    if not summaries:
        return

    sanitized = []
    for summary in summaries:
        sanitized.append(
            {
                k: v
                for k, v in summary.items()
                if k != "temporal_std_map"
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sanitized[0].keys()))
        writer.writeheader()
        writer.writerows(sanitized)


def save_temporal_std_map(summary: dict, output_dir: Path) -> None:
    temporal_std_map = summary["temporal_std_map"]
    burst_id = summary["burst_id"]

    np.save(output_dir / f"burst_{burst_id:03d}_temporal_std_map.npy", temporal_std_map)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(temporal_std_map, cmap="inferno")
    ax.set_title(f"Burst {burst_id} Temporal Std Map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Std")
    fig.tight_layout()
    fig.savefig(output_dir / f"burst_{burst_id:03d}_temporal_std_map.png", dpi=200)
    plt.close(fig)


def compute_global_and_per_burst_pixel_means(
    bursts: list[list[FrameRecord]],
    output_dir: Path,
) -> None:
    overall_pixel_sum = 0.0
    overall_pixel_count = 0
    burst_rows = []

    for burst_id, burst in enumerate(bursts, start=1):
        burst_pixel_sum = 0.0
        burst_pixel_count = 0
        frame_count = 0

        for record in burst:
            image = load_image_as_gray(record.path)
            burst_pixel_sum += float(np.sum(image))
            burst_pixel_count += int(image.size)
            frame_count += 1

        if burst_pixel_count == 0:
            continue

        burst_mean = burst_pixel_sum / burst_pixel_count
        burst_rows.append(
            {
                "burst_id": burst_id,
                "num_frames": frame_count,
                "num_pixels": burst_pixel_count,
                "pixel_mean": burst_mean,
            }
        )

        overall_pixel_sum += burst_pixel_sum
        overall_pixel_count += burst_pixel_count

    if overall_pixel_count == 0:
        return

    overall_mean = overall_pixel_sum / overall_pixel_count

    summary_lines = [
        f"Global pixel mean across all bursts: {overall_mean:.6f}",
        "",
        "Per-burst pixel means:",
    ]
    for row in burst_rows:
        summary_lines.append(
            f"Burst {row['burst_id']}: mean={row['pixel_mean']:.6f}, "
            f"frames={row['num_frames']}, pixels={row['num_pixels']}"
        )

    summary_path = output_dir / "global_and_per_burst_pixel_means.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    csv_path = output_dir / "per_burst_pixel_means.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(burst_rows[0].keys()))
        writer.writeheader()
        writer.writerows(burst_rows)

    print(f"Global pixel mean across all bursts: {overall_mean:.6f}")
    for row in burst_rows:
        print(
            f"Burst {row['burst_id']} pixel mean: {row['pixel_mean']:.6f} "
            f"(frames={row['num_frames']})"
        )


def save_average_matrix_txt(matrix: np.ndarray, output_path: Path) -> None:
    np.savetxt(output_path, matrix, fmt="%.6f")


def save_average_matrix_3d_plot(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    downsample_step: int = 4,
) -> None:
    step = max(1, int(downsample_step))
    sampled = matrix[::step, ::step]
    y = np.arange(0, matrix.shape[0], step)
    x = np.arange(0, matrix.shape[1], step)
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, sampled, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("mean intensity")
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="mean intensity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def compute_pixelwise_average_maps(
    bursts: list[list[FrameRecord]],
    output_dir: Path,
) -> None:
    overall_sum = None
    overall_count = 0

    for burst_id, burst in enumerate(bursts, start=1):
        burst_sum = None
        burst_count = 0

        for record in burst:
            image = load_image_as_gray(record.path)

            if burst_sum is None:
                burst_sum = np.zeros_like(image, dtype=np.float64)
            if overall_sum is None:
                overall_sum = np.zeros_like(image, dtype=np.float64)

            if image.shape != burst_sum.shape or image.shape != overall_sum.shape:
                raise ValueError(
                    f"Image shape mismatch for pixelwise averaging at {record.path.name}: "
                    f"got {image.shape}, expected {burst_sum.shape}."
                )

            burst_sum += image
            overall_sum += image
            burst_count += 1
            overall_count += 1

        if burst_sum is None or burst_count == 0:
            continue

        burst_avg = burst_sum / burst_count
        save_average_matrix_txt(
            burst_avg,
            output_dir / f"burst_{burst_id:03d}_pixelwise_average.txt",
        )
        save_average_matrix_3d_plot(
            burst_avg,
            output_dir / f"burst_{burst_id:03d}_pixelwise_average_3d.png",
            title=f"Burst {burst_id} Pixelwise Average",
        )

    if overall_sum is None or overall_count == 0:
        return

    overall_avg = overall_sum / overall_count
    save_average_matrix_txt(
        overall_avg,
        output_dir / "all_bursts_pixelwise_average.txt",
    )
    save_average_matrix_3d_plot(
        overall_avg,
        output_dir / "all_bursts_pixelwise_average_3d.png",
        title="All Bursts Pixelwise Average",
    )


def plot_metric_over_time(rows: list[dict], metric: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    burst_ids = sorted({row["burst_id"] for row in rows})
    for burst_id in burst_ids:
        burst_rows = [row for row in rows if row["burst_id"] == burst_id]
        times = [row["relative_time_ms"] for row in burst_rows]
        values = [row[metric] for row in burst_rows]
        ax.plot(times, values, marker="o", linewidth=1.5, markersize=4, label=f"Burst {burst_id + 1}")

    ax.set_xlabel("Relative Time from frame 0000 (ms)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_report(
    input_dir: Path,
    frame_rows: list[dict],
    summaries: list[dict],
    output_dir: Path,
) -> Path:
    report_path = output_dir / "noise_report.txt"
    overall_mean_temporal_std = np.mean([s["temporal_std_mean"] for s in summaries]) if summaries else np.nan
    overall_mean_rmse = np.mean([s["mean_rmse_from_0000"] for s in summaries]) if summaries else np.nan

    lines = [
        f"Input folder: {input_dir}",
        f"Total valid frames: {len(frame_rows)}",
        f"Total bursts: {len(summaries)}",
        f"Overall mean temporal std: {overall_mean_temporal_std:.4f}",
        f"Overall mean RMSE from frame 0000: {overall_mean_rmse:.4f}",
        "",
        "Burst summaries:",
    ]

    for summary in summaries:
        lines.extend(
            [
                (
                    f"Burst {summary['burst_id'] + 1}: "
                    f"{summary['num_frames']} frames, "
                    f"duration={summary['duration_ms']:.3f} ms, "
                    f"noise_color={summary['noise_color']}, "
                    f"alpha={summary['psd_alpha']:.3f}, "
                    f"white_fraction={summary['white_fraction']:.3f}, "
                    f"temporal_std_mean={summary['temporal_std_mean']:.4f}, "
                    f"mean_rmse_from_0000={summary['mean_rmse_from_0000']:.4f}, "
                    f"reference={summary['reference_file']}"
                )
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def analyze_noise(input_dir: Path, output_dir: Path, recursive: bool = False) -> None:
    records = collect_frame_records(input_dir, recursive=recursive)
    if not records:
        raise FileNotFoundError(f"No valid MultiSnap BMP files found in: {input_dir}")

    bursts = split_into_bursts(records)
    display_limits = compute_spatial_display_limits(bursts)
    averaged_noise_absmax = compute_averaged_noise_display_limit(bursts)
    single_frame_noise_absmax = compute_single_frame_noise_display_limit(bursts)
    frame_rows = []
    summaries = []

    for burst_id, burst in enumerate(bursts):
        burst_rows, burst_summary = analyze_burst(burst, burst_id)
        if not burst_rows:
            continue
        frame_rows.extend(burst_rows)
        summaries.append(burst_summary)
        save_selected_frame_spectra(
            burst=burst,
            output_dir=output_dir,
            burst_id=burst_id,
        )
        save_combined_selected_frame_spectra(
            burst=burst,
            output_dir=output_dir,
            burst_id=burst_id,
        )
        save_spectrum_over_time(
            burst=burst,
            output_dir=output_dir,
            burst_id=burst_id,
        )
        save_selected_frame_spatial_decomposition(
            burst=burst,
            output_dir=output_dir,
            burst_id=burst_id,
            display_limits=display_limits,
        )
        save_averaged_noise_spatial_maps(
            burst=burst,
            output_dir=output_dir,
            burst_id=burst_id,
            noise_absmax=averaged_noise_absmax,
        )
        save_single_frame_noise_spatial_maps(
            burst=burst,
            output_dir=output_dir,
            burst_id=burst_id,
            noise_absmax=single_frame_noise_absmax,
        )
        save_time_profile_and_temporal_fft(
            burst=burst,
            output_dir=output_dir,
            burst_id=burst_id,
        )
        analyze_stripe_motion_for_burst(
            burst=burst,
            output_dir=output_dir,
            burst_id=burst_id,
        )

    if not frame_rows:
        raise RuntimeError("No burst could be analyzed.")

    compute_global_and_per_burst_pixel_means(bursts, output_dir)
    compute_pixelwise_average_maps(bursts, output_dir)
    save_csv(frame_rows, output_dir / "noise_metrics_per_frame.csv")
    save_summary_csv(summaries, output_dir / "noise_metrics_per_burst.csv")

    plot_metric_over_time(
        frame_rows,
        metric="std",
        output_path=output_dir / "noise_std_vs_time.png",
        title="Noise Std vs Relative Time",
    )
    plot_metric_over_time(
        frame_rows,
        metric="mean",
        output_path=output_dir / "noise_mean_vs_time.png",
        title="Noise Mean vs Relative Time",
    )

    report_path = write_report(input_dir, frame_rows, summaries, output_dir)

    print(f"Analyzed {len(frame_rows)} frames across {len(summaries)} bursts.")
    print(f"Saved frame metrics to: {output_dir / 'noise_metrics_per_frame.csv'}")
    print(f"Saved burst summary to: {output_dir / 'noise_metrics_per_burst.csv'}")
    print(f"Saved report to: {report_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze noise evolution in MultiSnap BMP files.")
    parser.add_argument(
        "input_dir",
        help="Folder containing MultiSnap BMP noise images.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Default: sibling folder named <input_dir>_noise_analysis",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search BMP files recursively.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else input_dir.parent / f"{input_dir.name}_noise_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze_noise(input_dir=input_dir, output_dir=output_dir, recursive=args.recursive)


if __name__ == "__main__":
    main()
# running example:
# python other_resources/NoiseAnalysis.py other_data/NVLab260417/con50_bri-100
