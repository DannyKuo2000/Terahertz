from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def angular_spectrum_propagate_numpy(field, wavelength, dx, z, include_evanescent=False):
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")

    k = 2.0 * np.pi / wavelength
    kx = 2.0 * np.pi * fx_grid
    ky = 2.0 * np.pi * fy_grid

    argument = k**2 - kx**2 - ky**2
    kz_real = np.sqrt(np.maximum(argument, 0.0))

    if include_evanescent:
        kz_imag = np.sqrt(np.maximum(-argument, 0.0))
        transfer = np.exp(1j * kz_real * z) * np.exp(-kz_imag * abs(z))
    else:
        transfer = np.exp(1j * kz_real * z)

    spectrum = np.fft.fft2(field)
    return np.fft.ifft2(spectrum * transfer)


def make_gaussian_amplitude(grid_size=512, beam_waist=2.8e-3, dx=25e-6):
    coords = (np.arange(grid_size) - grid_size / 2) * dx
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius_sq = xx**2 + yy**2
    amplitude = np.exp(-radius_sq / (beam_waist**2))
    return amplitude, xx, yy


def make_conditioned_phase(xx, yy):
    radius_sq = xx**2 + yy**2
    radial_scale = np.max(radius_sq) + 1e-12

    lens_like = -2.4 * np.pi * (radius_sq / radial_scale)
    astigmatism = 0.9 * np.pi * ((xx**2 - yy**2) / radial_scale)
    # stripes = 0.55 * np.pi * np.sin(0.5 * np.pi * xx / 1.4e-3)
    stripes = np.pi * np.sin(xx / 1.4e-3)
    
    #phase = lens_like + astigmatism + stripes
    phase = stripes
    #return np.angle(np.exp(1j * phase))
    return phase


def normalize_image(img):
    img = np.asarray(img, dtype=np.float64)
    img = img - img.min()
    peak = img.max()
    if peak <= 1e-12:
        return np.zeros_like(img)
    return img / peak


def apply_colormap(img, mode="inferno"):
    img = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)

    if mode == "gray":
        rgb = np.stack([img, img, img], axis=-1)
        return (255.0 * rgb).astype(np.uint8)

    # if mode == "phase":
    #     hue = (img + np.pi) / (2.0 * np.pi)
    #     hue = np.mod(hue, 1.0)
    #     saturation = np.ones_like(hue) * 0.85
    #     value = np.ones_like(hue) * 0.95
    #     return hsv_to_rgb_uint8(hue, saturation, value)
    
    if mode == "phase":
        img = np.asarray(img, dtype=np.float64)
        img = img - img.min()
        img = img / (img.max() + 1e-12)
        anchors = np.array(
            [
                [59, 76, 192],
                [120, 120, 120],
                [180, 4, 38],
            ],
            dtype=np.float64,
        )
        positions = np.linspace(0.0, 1.0, len(anchors))
        flat = img.reshape(-1)
        channels = [
            np.interp(flat, positions, anchors[:, idx])
            for idx in range(3)
        ]
        rgb = np.stack(channels, axis=-1).reshape(img.shape + (3,))
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    if mode == "viridis":
        anchors = np.array(
            [
                [68, 1, 84],
                [59, 82, 139],
                [33, 145, 140],
                [94, 201, 98],
                [253, 231, 37],
            ],
            dtype=np.float64,
        )
    else:
        anchors = np.array(
            [
                [0, 0, 4],
                [66, 10, 104],
                [147, 38, 103],
                [221, 81, 58],
                [252, 255, 164],
            ],
            dtype=np.float64,
        )

    positions = np.linspace(0.0, 1.0, len(anchors))
    flat = img.reshape(-1)
    channels = [np.interp(flat, positions, anchors[:, idx]) for idx in range(3)]
    rgb = np.stack(channels, axis=-1).reshape(img.shape + (3,))
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def hsv_to_rgb_uint8(h, s, v):
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = np.mod(i, 6)

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(255.0 * rgb, 0.0, 255.0).astype(np.uint8)


def to_panel_image(img, mode, panel_size):
    colored = apply_colormap(img, mode=mode)
    panel = Image.fromarray(colored, mode="RGB")
    return panel.resize((panel_size, panel_size), resample=Image.Resampling.BICUBIC)


def render_frame(z_mm, zero_intensity, phase_intensity, initial_phase, extent_mm):
    del extent_mm
    diff_map = normalize_image(np.abs(phase_intensity - zero_intensity))

    panel_size = 290
    margin = 24
    title_h = 58
    footer_h = 42
    gap = 18
    frame_w = margin * 2 + panel_size * 4 + gap * 3
    frame_h = title_h + panel_size + footer_h + margin * 2

    frame = Image.new("RGB", (frame_w, frame_h), color=(248, 248, 245))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()

    title = "Same Gaussian amplitude, different initial phase"
    subtitle = f"Propagation distance z = {z_mm:.1f} mm"
    draw.text((margin, 16), title, fill=(20, 20, 20), font=font)
    draw.text((margin, 34), subtitle, fill=(70, 70, 70), font=font)

    panels = [
        (zero_intensity, "Zero-phase intensity", "inferno"),
        (phase_intensity, "Conditioned-phase intensity", "inferno"),
        (diff_map, "Absolute difference", "viridis"),
        (initial_phase, "Initial conditioned phase", "phase"),
    ]

    y0 = title_h + margin
    for idx, (img, label, mode) in enumerate(panels):
        x0 = margin + idx * (panel_size + gap)
        panel_img = to_panel_image(img, mode=mode, panel_size=panel_size)
        frame.paste(panel_img, (x0, y0))
        draw.rectangle(
            [x0 - 1, y0 - 1, x0 + panel_size, y0 + panel_size],
            outline=(120, 120, 120),
            width=1,
        )
        draw.text((x0, y0 + panel_size + 10), label, fill=(25, 25, 25), font=font)

    return np.asarray(frame)


def generate_phase_importance_animation(
    output_path="other_resources/PhaseImportantDisplayOutput/phase_importance_demo.gif",
    grid_size=1024,  # size
    dx=35e-6,  # pixel pitch
    wavelength=1.5e-3,
    beam_waist=2.8e-3,  # gaussian beam waist
    z_max=0.10,
    frame_count=48,
    fps=1.5,
):
    amplitude, xx, yy = make_gaussian_amplitude(
        grid_size=grid_size,
        beam_waist=beam_waist,
        dx=dx,
    )
    zero_phase = np.zeros_like(amplitude)
    conditioned_phase = make_conditioned_phase(xx, yy)

    zero_field = amplitude * np.exp(1j * zero_phase)
    phase_field = amplitude * np.exp(1j * conditioned_phase)

    z_values = np.linspace(0.0, z_max, frame_count)
    extent_half_mm = 0.5 * grid_size * dx * 1e3
    extent_mm = [-extent_half_mm, extent_half_mm, -extent_half_mm, extent_half_mm]

    frames = []
    for z in z_values:
        zero_prop = angular_spectrum_propagate_numpy(zero_field, wavelength, dx, z)
        phase_prop = angular_spectrum_propagate_numpy(phase_field, wavelength, dx, z)

        zero_intensity = normalize_image(np.abs(zero_prop) ** 2)
        phase_intensity = normalize_image(np.abs(phase_prop) ** 2)
        frame = render_frame(
            z_mm=z * 1e3,
            zero_intensity=zero_intensity,
            phase_intensity=phase_intensity,
            initial_phase=conditioned_phase,
            extent_mm=extent_mm,
        )
        frames.append(frame)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(round(1000 / max(fps, 1)))
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )

    preview_path = output_path.with_name(f"{output_path.stem}_first_frame.png")
    pil_frames[0].save(preview_path)
    return output_path, preview_path


if __name__ == "__main__":
    movie_path, preview_path = generate_phase_importance_animation()
    print(f"Animation saved to: {movie_path}")
    print(f"Preview saved to: {preview_path}")
