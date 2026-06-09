import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
import math
from PIL import Image
from config import ENCODER_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Experiments Relative parameters:
    Refractive index: 1.7
    Absorption coefficient: 1e-5
    Sub THz: 0.2004e12


lens frame: Newport M-LH-2A: https://www.newport.com/p/M-LH-2A
laser lens: 1.55 µm BCX Lens
"""
class SourceLayer(nn.Module):
    """
    Simulate real source pattern
    """
    def __init__(self, use_input=True, input=None, mode="white", size_source=(128, 128),
                sigma=0.3, amplitude=1.0, center=(0.0, 0.0),
                rotation=0.0, aspect_ratio=1.0,
                crop_size_source=None, resize_size_source=None, pad_size_source=None,
                source_is_intensity=True, new_size_source=None):
        
        super().__init__()
        self.use_input = use_input
        self.input = input
        self.mode = mode
        self.size_source = size_source
        self.source_is_intensity = source_is_intensity
        if pad_size_source is None and new_size_source is not None:
            pad_size_source = new_size_source

        # Gaussian beam parameter
        self.sigma = sigma
        self.amplitude = amplitude
        self.center = center
        self.rotation = rotation
        self.aspect_ratio = aspect_ratio

        # Resize/pad layer
        self.resize_pad = CropResizeDisplacePadLayer(
            crop_size=crop_size_source,
            resize_size=resize_size_source,
            pad_size=pad_size_source,
        )

    def forward(self, x):
        device = x.device  
        if self.use_input:
            if self.input is None:
                raise ValueError("Need source image")
            
            img = Image.open(self.input).convert("L") 
            transform = T.ToTensor()
            source_background = transform(img).unsqueeze(0)  # (1, 1, H, W)
            if self.source_is_intensity:
                source_background = torch.sqrt(torch.clamp(source_background, min=0.0))

            input_resized = self.resize_pad(source_background.to(device=device, dtype=x.dtype))
            return x * input_resized

        else:
            H, W = self.size_source

            if self.mode == "white":
                src = torch.ones((1, 1, H, W), dtype=x.dtype, device=device)

            elif self.mode == "gaussian":
                yy, xx = torch.meshgrid(
                    torch.linspace(-1, 1, H, device=device, dtype=x.dtype),
                    torch.linspace(-1, 1, W, device=device, dtype=x.dtype),
                    indexing="ij"
                )

                xx = xx - self.center[0]
                yy = yy - self.center[1]

                if self.rotation != 0.0:
                    cos_t = math.cos(self.rotation)
                    sin_t = math.sin(self.rotation)
                    x_rot = cos_t * xx - sin_t * yy
                    y_rot = sin_t * xx + cos_t * yy
                    xx, yy = x_rot, y_rot

                # Gaussian
                xx = xx / self.aspect_ratio
                r2 = xx**2 + yy**2
                src = self.amplitude * torch.exp(-r2 / (2 * self.sigma**2))
                src = src.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            else:
                raise ValueError(f"Unknown source mode: {self.mode}")

            src_resized = self.resize_pad(src)
            return x * src_resized.to(device=device, dtype=x.dtype)

class CropResizeDisplacePadLayer(nn.Module):
    """
    Simulation of crop, resize, displace, pad
    """
    def __init__(self, crop_size=None, resize_size=None, displace=None, pad_size=None, mode='bilinear'):
        """
        resize_size: tuple (H_resize, W_resize) or None
        new_size: tuple (H_new, W_new) or None
        mode: resize interpolation mode ('bilinear', 'nearest', etc.)
        """
        super().__init__()
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.displace = displace if displace is not None else (0, 0)
        self.pad_size = pad_size
        self.mode = mode

    def forward(self, x):
        """
        x: tensor (B, C, H, W), could be real
        """
        B, C, H, W = x.shape
        # -------------------------
        # Crop
        # ------------------------- 
        if self.crop_size is not None: #and (H != self.crop_size[0] or W != self.crop_size[1]):
            cur_H, cur_W = x.shape[-2:]
            target_H, target_W = self.crop_size
            start_h = (cur_H - target_H) // 2 if cur_H > target_H else 0
            start_w = (cur_W - target_W) // 2 if cur_W > target_W else 0
            
            if torch.is_complex(x):
                real = x.real[..., start_h:start_h + target_H, start_w:start_w + target_W]
                imag = x.imag[..., start_h:start_h + target_H, start_w:start_w + target_W]
                x = torch.complex(real, imag)
            else:
                x = x[..., start_h:start_h + target_H, start_w:start_w + target_W]
            
            print(x.shape)

        # -------------------------
        # Resize
        # -------------------------
        if self.resize_size is not None: #and (H != self.resize_size[0] or W != self.resize_size[1]):
            if torch.is_complex(x):
                real = F.interpolate(x.real, size=self.resize_size, mode=self.mode, align_corners=False)
                imag = F.interpolate(x.imag, size=self.resize_size, mode=self.mode, align_corners=False)
                x = torch.complex(real, imag)
            else:
                x = F.interpolate(x, size=self.resize_size, mode=self.mode, align_corners=False)
            print(x.shape)
        # -------------------------
        # Displacement / Padding or Cropping ??pad_size
        # -------------------------
        if self.pad_size is not None:
            cur_H, cur_W = x.shape[-2:]
            target_H, target_W = self.pad_size


            pad_h = target_H - cur_H
            pad_w = target_W - cur_W

            if pad_h >= 0 and pad_w >= 0:
                # zero padding
                pad_top = pad_h // 2 + self.displace[0]
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2 + self.displace[1]
                pad_right = pad_w - pad_left
                if torch.is_complex(x):
                    real = F.pad(x.real, (pad_left, pad_right, pad_top, pad_bottom))
                    imag = F.pad(x.imag, (pad_left, pad_right, pad_top, pad_bottom))
                    x = torch.complex(real, imag)
                else:
                    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
            else:
                # center crop
                start_h = (cur_H - target_H) // 2 if cur_H > target_H else 0
                start_w = (cur_W - target_W) // 2 if cur_W > target_W else 0
                x = x[..., start_h:start_h + target_H, start_w:start_w + target_W]
            print(x.shape)
        
        return x

# ====== Air Diffraction Calculation ======


class DiffractiveLayer(nn.Module):
    """
    Using angular spectrum method to simulate free space propagation
    """
    def __init__(self, dx=0.00075, num_size=128, frequency=0.2004e12, z=0.1, refractive_index=1, 
                pad_factor=1, window=None, mask_evanescent=False, reverse_z=False):
        super().__init__()
        self.dx = dx  # resolution (m)
        self.size = num_size  # number of optical neurons in one dimension
        self.wl = 2.998e8 / frequency  # wavelength = light speed / frequency (m)
        self.z = z  # distance between two layers (m)
        self.n = refractive_index
        self.pad_factor = pad_factor  # zero-padding
        self.window = window
        self.mask_evanescent = mask_evanescent  # calculation evanescent
        self.reverse_z = reverse_z  # backward propagation: (-z)

        print(num_size)
        # ==============================================
        # Step 1. 
        # ==============================================
        fx = np.fft.fftshift(np.fft.fftfreq(self.size * self.pad_factor, d=self.dx))
        fxx, fyy = np.meshgrid(fx, fx)

        # ==============================================
        # Step 2. wave number
        # ==============================================
        kx = 2 * np.pi * fxx
        ky = 2 * np.pi * fyy
        k = 2 * np.pi * self.n / self.wl 

        # ==============================================
        # Step 3. kz
        # ==============================================
        argument = k**2 - kx**2 - ky**2
        tmp = np.sqrt(np.abs(argument))
        kz = np.where(argument >= 0, tmp, 1j * tmp)

        # ==============================================
        # Step 4. propagation
        # ==============================================
        if self.reverse_z:
            H = np.exp(-1j * kz * self.z)
            H[argument < 0] = 0.0  # prevent evanescent error
        else:
            H = np.exp(1j * kz * self.z)
            if self.mask_evanescent:
                H[argument < 0] = 0.0

        self.H = torch.from_numpy(H.astype(np.complex64)).to(device)


        # ==============================================
        # Step 5. aliasing check
        # ==============================================
        if self.dx > self.wl / 2:
            print(f"Warning: dx={self.dx*1e3:.3f} mm > λ/2={self.wl/2*1e3:.3f} mm, could aliasing")

    def forward(self, E):
        # make sure it is tensor
        if isinstance(E, np.ndarray):
            E = torch.from_numpy(E).to(device)
        E = E.to(torch.complex64)

        # Step A. Padding
        if self.pad_factor > 1:
            pad = (self.size * (self.pad_factor - 1)) // 2
            if torch.is_complex(E):
                real = torch.nn.functional.pad(E.real, (pad, pad, pad, pad))
                imag = torch.nn.functional.pad(E.imag, (pad, pad, pad, pad))
                E = torch.complex(real, imag)
            else:
                E = torch.nn.functional.pad(E, (pad, pad, pad, pad), value=0.0)

        # Step B. 
        F = torch.fft.fftshift(torch.fft.fft2(E))
        print(F.size())
        print(self.H.size())
        propagated = torch.fft.ifft2(torch.fft.ifftshift(F * self.H))

        # Step C. 
        if self.pad_factor > 1:
            if self.window == "hann":
                window = torch.hann_window(propagated.shape[-1], device=propagated.device)
                window2d = window[:, None] * window[None, :]
                propagated = propagated * window2d
            start = pad
            end = start + self.size
            propagated = propagated[..., start:end, start:end]

        return propagated

'''class DiffractiveLayer(nn.Module): 
    """
    Adding attenuation in angular spectrum method 
    """
    def __init__(self, dx=0.00075, num_size=128, frequency=0.2004e12, z=0.1, refractive_index=1, 
                pad_factor=4, keep_pad=False, mask_evanescent=False, reverse_z=False, multi_step=1, eps=1e-3,
                alpha_global=0.0, beta_freq=0.0, use_geom_atten=False):
        """
        1. alpha_global: attenuation in air (m^-1): exp(-alpha_global * z)
            usually use 1~10(m**-1)
        2. beta_freq: high frequency attenuation (m^-1): exp(-beta_freq * (kx^2+ky^2) * z)
            usually use 1e-7
        3. use_geom_atten: geometric 1/z attenuation
        """
        super().__init__()
        self.dx = dx
        self.size = num_size
        self.wl = 2.998e8 / frequency
        self.z = z
        self.n = refractive_index
        self.pad_factor = pad_factor
        self.keep_pad = keep_pad
        self.mask_evanescent = mask_evanescent
        self.reverse_z = reverse_z
        self.multi_step = multi_step   
        self.eps = eps  # evanescent epsilon

        # attenuation parameter
        self.alpha_global = alpha_global
        self.beta_freq = beta_freq
        self.use_geom_atten = use_geom_atten

        # ======================================================
        # Step 1. 
        # ======================================================
        fx = np.fft.fftshift(np.fft.fftfreq(self.size * self.pad_factor, d=self.dx))  
        fxx, fyy = np.meshgrid(fx, fx)

        # Step 2. kx, ky
        self.kx = 2 * np.pi * fxx
        self.ky = 2 * np.pi * fyy

        # Step 3. k
        self.k = 2 * np.pi * self.n / self.wl

        # Step 4. kz
        argument = self.k**2 - self.kx**2 - self.ky**2

        tmp = np.sqrt(np.abs(argument)).astype(np.float32)

        kz = np.where(argument >= 0, tmp, 1j * tmp).astype(np.complex64)

        is_evan = (argument < 0)
        alpha = np.where(is_evan, np.real(kz), 0.0).astype(np.float32)
        freq_decay = (self.kx**2 + self.ky**2).astype(np.float32)

        self.kz_torch = torch.from_numpy(kz).to(device=device, dtype=torch.complex64)
        self.alpha_torch = torch.from_numpy(alpha).to(device=device, dtype=torch.float32)
        self.is_evan_torch = torch.from_numpy(is_evan.astype(np.float32)).to(device=device, dtype=torch.float32)
        self.freq_decay_torch = torch.from_numpy(freq_decay).to(device=device, dtype=torch.float32)

        # dx check
        if self.dx > self.wl / 2:
            print(f"Warning: dx={self.dx*1e3:.3f} mm > λ/2={self.wl/2*1e3:.3f} mm, could aliasing")

    def forward(self, E):
        if isinstance(E, np.ndarray):
            E = torch.from_numpy(E).to(device)

        if self.pad_factor > 1:
            pad = (self.size * (self.pad_factor - 1)) // 2
            if torch.is_complex(E):
                real = torch.nn.functional.pad(E.real, (pad, pad, pad, pad))
                imag = torch.nn.functional.pad(E.imag, (pad, pad, pad, pad))
                E = torch.complex(real, imag)
            else:
                E = torch.nn.functional.pad(E, (pad, pad, pad, pad), mode='constant', value=0)

        dz = self.z / self.multi_step
        z_done = 0.0
        for step in range(self.multi_step):
            z_rem = self.z - z_done - dz

            # ------------------------------
            # Step A. 
            # ------------------------------
            if self.reverse_z:
                H = torch.exp(-1j * self.kz_torch * dz)
                H = H * (1 - self.is_evan_torch)
            else:
                H = torch.exp(1j * self.kz_torch * dz)

                if self.mask_evanescent:
                    atten = torch.exp(-self.alpha_torch * z_rem)
                    keep = ((1 - self.is_evan_torch) > 0) | (atten >= self.eps)
                    H = H * keep
                
                """
                if self.alpha_global > 0:
                    H = H * torch.exp(torch.tensor(-self.alpha_global * dz, dtype=H.dtype, device=H.device))

                # high frequency attenuation
                if self.beta_freq > 0:
                    H = H * torch.exp(-self.beta_freq * self.freq_decay_torch * dz)"""

            # ------------------------------
            # Step B. 
            # ------------------------------
            c_fft = torch.fft.fftshift(torch.fft.fft2(E, dim=(-2, -1)))  
            angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c_fft * H))

            E = angular_spectrum
            z_done += dz

            if self.use_geom_atten and z_done > 0: 
                # MODIFIED: use 1/sqrt(z) (less aggressive than 1/z), prevent tiny z causing huge factors
                #           and ensure tensor dtype/device matching E, clamp maximum factor
                eps_z = 1e-9  # floor to avoid division by zero
                z_safe = max(z_done, eps_z)  # python float safe lower bound
                z_tensor = torch.tensor(z_safe, dtype=E.dtype, device=E.device)
                geom_factor = 1.0 / torch.sqrt(z_tensor)  # amplitude ~ 1/sqrt(z)
                # limit the maximum amplification to avoid pathological case
                geom_factor = torch.clamp(geom_factor, max=1e3)  # MODIFIED: guard against huge factor
                E = E * geom_factor  # MODIFIED

            # ------------------------------
            # Step C. 
            # ------------------------------
            E = self._crop_and_pad(E, energy_frac=0.99)

        if self.pad_factor > 1 and self.keep_pad == False:
            start = pad
            end = start + self.size
            E = E[..., start:end, start:end]
        return E

    
    def _crop_and_pad(self, E, energy_frac=0.99, target_shape=None, margin_pix=10):
        """
        - E: torch tensor (complex or real) shape (H,W)
        - target_shape: tuple (H_target, W_target). 
        """
        H_target, W_target = (E.shape[-2], E.shape[-1]) if target_shape is None else tuple(target_shape)

        I_np = (E.abs()**2).detach().cpu().numpy()
        B, C, H, W = I_np.shape
        cy, cx = H // 2, W // 2

        y = np.arange(H) - cy # y = [-cy, -cy+1, -cy+2, ...]
        x = np.arange(W) - cx
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2

        idx = np.argsort(R2.ravel())
        cumulative = np.cumsum(I_np.ravel()[idx])
        total = cumulative[-1] # total energy
        target = energy_frac * total
        cut_idx = np.searchsorted(cumulative, target)
        cut_idx = min(cut_idx, len(idx)-1)

        r2_target = R2.ravel()[idx[cut_idx]]
        r = int(np.ceil(np.sqrt(r2_target))) + margin_pix 

        r = min(r, cy, cx, H - cy - 1, W - cx - 1)

        y0, y1 = cy - r, cy + r
        x0, x1 = cx - r, cx + r

        if y1 <= y0 or x1 <= x0:
            return E
        
        E_crop = E[..., y0:y1, x0:x1]
        newH, newW = E_crop.shape[-2], E_crop.shape[-1]

        pad_top = (H_target - newH) // 2
        pad_bottom = H_target - newH - pad_top
        pad_left = (W_target - newW) // 2
        pad_right = W_target - newW - pad_left

        if pad_top < 0 or pad_bottom < 0 or pad_left < 0 or pad_right < 0:
            return E

        if torch.is_complex(E_crop):
            real = torch.nn.functional.pad(E_crop.real, (pad_left, pad_right, pad_top, pad_bottom))
            imag = torch.nn.functional.pad(E_crop.imag, (pad_left, pad_right, pad_top, pad_bottom))
            E_new = torch.complex(real, imag)
        else:
            E_new = torch.nn.functional.pad(E_crop, (pad_left, pad_right, pad_top, pad_bottom))

        assert E_new.shape[-2] == H_target and E_new.shape[-1] == W_target, 
            f"pad failed: got {E_new.shape[-2:]} expected {(H_target, W_target)}"

        return E_new.to(E.device)'''

class LensLayer(nn.Module):
    """
    Simulate lens layer
    """
    def __init__(self, focal_length, dx, num_size, wavelength,
                pupil_type=None, pupil_radius=None, pupil_width=None,
                phase_model="exact", mode="forward", outside="one",
                frame=False, frame_inner=0.02375, frame_outer=0.0254):
        super().__init__()
        self.f = float(focal_length)
        self.dx = float(dx)
        self.N  = int(num_size)
        self.wl = float(wavelength)
        self.frame_inner = frame_inner
        self.frame_outer = frame_outer

        k = 2*np.pi / self.wl
        x = (np.arange(self.N) - self.N/2) * self.dx
        X, Y = np.meshgrid(x, x)

        if phase_model == "exact":
            # Spherical lens phase
            if mode == "backward":
                H_lens = np.exp(1j * k * (np.sqrt(X**2 + Y**2 + self.f**2) - self.f))
            else:
                H_lens = np.exp(-1j * k * (np.sqrt(X**2 + Y**2 + self.f**2) - self.f))
        else:
            # Paraxial lens phase
            if mode == "backward":
                H_lens = np.exp(1j * k * (X**2 + Y**2) / (2*self.f))
            else: 
                H_lens = np.exp(-1j * k * (X**2 + Y**2) / (2*self.f))

        # pupil shape
        if pupil_type == "circular":
            assert pupil_radius is not None, "set pupil_radius"
            P = ((X**2 + Y**2) <= (pupil_radius**2)).astype(np.float32)
        elif pupil_type == "square":
            assert pupil_width is not None, "set pupil_width"
            half = pupil_width/2
            P = ((np.abs(X) <= half) & (np.abs(Y) <= half)).astype(np.float32)
        else:
            P = np.ones_like(X, dtype=np.float32)

        # outside the lens frame
        if outside == "one":
            self.H = H_lens * P + (1 - P)   # set outside = 1
        else:
            self.H = H_lens * P             # set outside = 0

        if frame == True:
            frame_mask = (((X**2 + Y**2) >= (frame_inner**2)) & ((X**2 + Y**2) <= (frame_outer**2))).astype(np.float32)
            self.H = self.H * (1 - frame_mask)
        self.H = torch.from_numpy(self.H).to(torch.complex64)
    def forward(self, E):
        H = self.H.to(E.device)
        return E * H

# ======= Interface Interaction Calculation ====== 
class FresnelInterface(nn.Module):
    """
    Polariztion simulation
    """
    def __init__(self, dx=0.00075, num_size=128, frequency=0.2e12, keep_reflection=False, complex_index=False, n1=1, n2=1.7):
        super().__init__()
        self.dx = dx
        self.size = num_size
        self.n1 = n1 if complex_index else complex(n1, 0.0)
        self.n2 = n2 if complex_index else complex(n2, 0.0)
        self.keep_reflection = keep_reflection
        self.wl = 2.998e8 / frequency  # 
        self.k0 = 2 * np.pi / self.wl  # 

        fx = np.fft.fftshift(np.fft.fftfreq(self.size, d=self.dx))
        fxx, fyy = np.meshgrid(fx, fx)
        kx = 2 * np.pi * fxx
        ky = 2 * np.pi * fyy
        k_perp = np.sqrt(kx**2 + ky**2)

        sin_theta_i = k_perp / (self.k0 * abs(self.n1))
        sin_theta_i = np.clip(sin_theta_i, 0, 1)

        # cos(theta_i), sin(theta_t), cos(theta_t)
        cos_theta_i = np.sqrt(1 - sin_theta_i**2 + 0j)
        sin_theta_t = (self.n1 / self.n2) * sin_theta_i
        cos_theta_t = np.sqrt(1 - sin_theta_t**2 + 0j)  

        # Fresnel TE (s) & TM (p)
        rs = (self.n1 * cos_theta_i - self.n2 * cos_theta_t) / (self.n1 * cos_theta_i + self.n2 * cos_theta_t)
        ts = (2 * self.n1 * cos_theta_i) / (self.n1 * cos_theta_i + self.n2 * cos_theta_t)

        rp = (self.n2 * cos_theta_i - self.n1 * cos_theta_t) / (self.n2 * cos_theta_i + self.n1 * cos_theta_t)
        tp = (2 * self.n1 * cos_theta_i) / (self.n2 * cos_theta_i + self.n1 * cos_theta_t)

        # average intensity from rs, rp, ts, tp
        R = 0.5 * (np.abs(rs)**2 + np.abs(rp)**2)
        T = 0.5 * (np.abs(ts)**2 + np.abs(tp)**2)

        self.R = torch.from_numpy(R).to(torch.float32)  
        self.T = torch.from_numpy(T).to(torch.float32)

        self.rs = torch.from_numpy(rs).to(torch.complex64)
        self.rp = torch.from_numpy(rp).to(torch.complex64)
        self.ts = torch.from_numpy(ts).to(torch.complex64)
        self.tp = torch.from_numpy(tp).to(torch.complex64)

    def forward(self, E):
        E_f = torch.fft.fftshift(torch.fft.fft2(E))

        t_avg = 0.5 * (self.ts + self.tp).to(E.device)
        r_avg = 0.5 * (self.rs + self.rp).to(E.device)

        E_f_transmitted = E_f * t_avg
        E_f_reflected = E_f * r_avg

        E_out = torch.fft.ifft2(torch.fft.ifftshift(E_f_transmitted))

        if self.keep_reflection:
            E_ref = torch.fft.ifft2(torch.fft.ifftshift(E_f_reflected))
            return E_out, E_ref
        else:
            return E_out

class RadialAttenuationLayer(nn.Module):
    def __init__(self, R0_ratio=0.8, exponent=2, min_factor=0):
        super().__init__()
        self.R0_ratio = R0_ratio
        self.exponent = exponent
        self.min_factor = min_factor 
        
    def forward(self, E):
        H, W = E.shape[-2:]
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=E.device) - cy
        x = torch.arange(W, device=E.device) - cx
        X, Y = torch.meshgrid(x, y, indexing='xy')
        R = torch.sqrt(X**2 + Y**2) 
        R0 = self.R0_ratio * R.max()

        attenuation = torch.ones_like(R)
        mask = R >= R0
        attenuation[mask] = torch.exp(- ((R[mask]-R0)/(max(H,W)-R0))**self.exponent)
        attenuation = torch.clamp(attenuation, self.min_factor, 1.0)

        return E * attenuation

class SensorLayer(nn.Module):
    """
    Simulation of sensor end, crop and flip
    """
    def __init__(self, crop_size=(288, 384), displacement=(0, 0), bin_size=1, flip=False):
        """
        crop_size: crop size
        bin_size: pixel binning size
        flip: flip image
        """
        super().__init__()
        self.crop_size = crop_size
        self.displacement = displacement
        self.bin_size = bin_size
        self.flip = flip

    def forward(self, E):
        B, C, H, W = E.shape
        crop_h, crop_w = self.crop_size
        hh, hw = crop_h // 2, crop_w // 2
        center_h, center_w = H // 2, W // 2
        disp_h, disp_w = self.displacement

        E_crop = E[..., center_h - hh + disp_h:center_h + hh + disp_h,
                    center_w - hw + disp_w:center_w + hw + disp_w]

        I_crop = torch.abs(E_crop) ** 2
        if self.bin_size > 1:
            bin_size = self.bin_size
            out_h = I_crop.shape[-2] // bin_size
            out_w = I_crop.shape[-1] // bin_size
            I_crop = I_crop[..., :out_h * bin_size, :out_w * bin_size]
            I_crop = I_crop.reshape(B, C, out_h, bin_size, out_w, bin_size).mean(dim=(3, 5))

        if self.flip:
            I_crop = torch.flip(I_crop, dims=[-2, -1])

        print(torch.min(I_crop))
        print(torch.max(I_crop))
        return I_crop.to(torch.float32)
    
# ====== Sensor Noise Simulation ======
class SensorNoiseLayer(nn.Module):
    """
    Simple sensor noise simulation
    """
    def __init__(self, blur_kernel_size=15, blur_sigma=5, gray_mean=0.6, gray_sigma=0.02, gray_ratio=0.55, noise_std=10/255.):
        """
        config:
            - blur_kernel_size
            - blur_sigma
            - gray_mean
            - gray_sigma
            - gray_ratio
            - noise_std
        """
        super().__init__()
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.gray_mean = gray_mean
        self.gray_sigma = gray_sigma
        self.gray_ratio = gray_ratio
        self.noise_std = noise_std

        # differentiable Gaussian kernel
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())

    def forward(self, x):
        # Gaussian blur
        x = self._gaussian_blur(x)

        # Gray background
        gray_bg = torch.randn_like(x) * self.gray_sigma + self.gray_mean  
        x = (1 - self.gray_ratio) * x + self.gray_ratio * gray_bg

        # Add Gaussian noise
        noise = torch.randn_like(x) * self.noise_std
        x = torch.clamp(x + noise, 0.0, 1.0)  

        return x

    def _create_gaussian_kernel(self):
        k = self.blur_kernel_size
        sigma = self.blur_sigma
        coords = torch.arange(k) - k // 2
        grid = coords.repeat(k).view(k, k)
        x = grid
        y = grid.t()
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k, k)  
        kernel = kernel.repeat(3, 1, 1, 1) 
        return kernel

    def _gaussian_blur(self, x):
        return F.conv2d(x, self.gaussian_kernel, 
                        padding=self.blur_kernel_size // 2, groups=3)


# ====== Material Phase Control ======
class MaterialLayer(nn.Module):
    """
    ONN layer simulation
    """
    def __init__(self, num_size=128, block_size=(1, 1), return_phases=True):
        super().__init__()
        self.block_size = block_size
        self.return_phases = return_phases
        h_small = math.ceil(num_size / block_size[0])
        w_small = math.ceil(num_size / block_size[1])

        init_phase = 2 * np.pi * np.random.rand(h_small, w_small).astype(np.float32)
        self.phase = nn.Parameter(torch.from_numpy(init_phase))

    def forward(self, x):
        """
        x: (B, C, H, W) tensor, could be real or complex
        block_size: (block_h, block_w), multi-phase to one pixel
        """
        B, C, H, W = x.shape
        block_h, block_w = self.block_size

        phase_full = self.phase.repeat_interleave(block_h, dim=0).repeat_interleave(block_w, dim=1) # 複製
        phase_full = phase_full[:H, :W] 

        phase_mask = torch.exp(1j * phase_full).to(x.device)

        if self.return_phases == True:
            #print(f"Max: {torch.max(self.phase)}, Min: {torch.min(self.phase)}")
            return x * phase_mask, self.phase
        else:
            return x * phase_mask

# ====== ONN ensemblance ======
class ONN(nn.Module):
    """
    Simple ONN structure sample
    """
    def __init__(self, config=ENCODER_CONFIG):
        super().__init__()
        self.layers = nn.ModuleList()  # Module list 
        self.layer_names = []  # Mudule name
        
        # SourceLayer
        use_input           = config["use_input"]
        input               = config["input"]
        mode_source         = config["mode_source"]
        size_source         = config["size_source"]
        sigma               = config["sigma"]
        amplitude           = config["amplitude"]
        center              = config["center"]
        rotaion             = config["rotation"]
        aspect_ratio        = config["aspect_ratio"]
        resize_size_source  = config["resize_size_source"]
        new_size_source     = config["new_size_source"]

        # ResizePadLayer
        resize_size = config["resize_size"]
        pad_size    = config["pad_size"]

        # DiffractiveLayer 
        num_layers      = config["num_layers"]
        dx              = config["dx"]
        num_size        = config["num_size"]
        frequency       = config["frequency"]
        z_values        = config["z"]  # e.g. [0.2, 0.3]
        n               = config["refractive_index"]
        pad_factor      = config["pad_factor"]
        window          = config["window"]
        #keep_pad        = config["keep_pad"]
        mask_evanescent = config["mask_evanescent"]
        reverse_z       = config["reverse_z"]
        #multi_step      = config["multi_step"]
        #eps             = config["eps"]
        #alpha_global    = config["alpha_global"]
        #beta_freq       = config["beta_freq"]
        #use_geom_atten  = config["use_geom_atten"]

        # LensLayer 
        focal_length = config["focal_length"]
        dx           = config["dx"]
        num_size     = config["num_size"]
        wavelength   = config["wavelength"]
        pupil_type   = config["pupil_type"]
        pupil_radius = config["pupil_radius"]
        pupil_width  = config["pupil_width"]
        phase_model  = config["phase_model"]
        mode_lens    = config["mode_lens"]
        outside      = config["outside"]
        frame        = config["frame"]
        frame_inner  = config["frame_inner"]
        frame_outer  = config["frame_outer"]

        # SensorLayer
        active_sensor   = config["active_sensor"]
        crop_size       = config["crop_size"]
        bin_size        = config["bin_size"]
        flip            = config["flip"]

        # SensorNoiseLayer
        active_sensor_noise = config["active_sensor_noise"]
        blur_kernel_size    = config["blur_kernel_size"]
        blur_sigma          = config["blur_sigma"]
        gray_mean           = config["gray_mean"]
        gray_sigma          = config["gray_sigma"]
        gray_ratio          = config["gray_ratio"]
        noise_std           = config["noise_std"]

        # MaterialLayer
        num_size_material   = config["num_size"]
        block_size          = config["block_size"]
        return_phases       = config["return_phases"]
        self.return_phases       = config["return_phases"]
        

        # -------------------------------
        # Construct layers
        # -------------------------------
        total_index = 1
        resize_pad_layer_index = 1
        diffractive_layer_index = 1
        material_layer_index = 1
        
        self.layers.append(CropResizeDisplacePadLayer(resize_size=(160, 160), pad_size=(160, 160)))
        self.layer_names.append(f"{total_index}_ResizePadLayer{resize_pad_layer_index}")
        resize_pad_layer_index += 1
        total_index += 1

        self.layers.append(SourceLayer(use_input=use_input, input=input, mode=mode_source, size_source=size_source, sigma=sigma, amplitude=amplitude, 
                                    center=center, rotation=rotaion, aspect_ratio=aspect_ratio, resize_size_source=resize_size_source, new_size_source=new_size_source))
        self.layer_names.append(f"{total_index}_SourceLayer")
        total_index += 1

        self.layers.append(CropResizeDisplacePadLayer(resize_size=resize_size, pad_size=pad_size))
        self.layer_names.append(f"{total_index}_ResizePadLayer{resize_pad_layer_index}")
        resize_pad_layer_index += 1
        total_index += 1

        z_values_index = 0
        for z_values_index in range(num_layers):
            self.layers.append(
                DiffractiveLayer(dx=dx, num_size=num_size, frequency=frequency, z=z_values[z_values_index], refractive_index=n,
                                pad_factor=pad_factor, window=window, mask_evanescent=mask_evanescent, reverse_z=reverse_z)
            )
            self.layer_names.append(f"{total_index}_DiffractiveLayer{diffractive_layer_index}")
            diffractive_layer_index += 1
            total_index += 1

            self.layers.append(MaterialLayer(num_size=num_size_material, block_size=block_size, return_phases=return_phases))
            self.layer_names.append(f"{total_index}_MaterialLayer{material_layer_index}")
            material_layer_index += 1
            total_index += 1

        self.layers.append(DiffractiveLayer(dx=dx, num_size=num_size, frequency=frequency, z=z_values[z_values_index], refractive_index=n,
                                            pad_factor=pad_factor, window=window, mask_evanescent=mask_evanescent, reverse_z=reverse_z))
        self.layer_names.append(f"{total_index}_DiffractiveLayer{diffractive_layer_index}")
        diffractive_layer_index += 1
        total_index += 1

        # self.layers.append(LensLayer(focal_length=focal_length, dx=dx, num_size=num_size, wavelength=wavelength, pupil_type=pupil_type,
        #                             pupil_radius=pupil_radius, pupil_width=pupil_width, phase_model=phase_model, mode=mode_lens, outside=outside, frame=frame,
        #                             frame_inner=frame_inner, frame_outer=frame_outer))
        # self.layer_names.append(f"{total_index}_LensLayer")
        # total_index += 1
        
        # self.layers.append(DiffractiveLayer(dx=dx, num_size=num_size, frequency=frequency, z=z_values[z_values_index], refractive_index=n,
        #                                     pad_factor=pad_factor, window=window, mask_evanescent=mask_evanescent, reverse_z=reverse_z))
        # self.layer_names.append(f"{total_index}_DiffractiveLayer{diffractive_layer_index}")
        # diffractive_layer_index += 1
        # total_index += 1

        # Sensor / Noise
        if active_sensor:
            self.layers.append(SensorLayer(crop_size=crop_size, bin_size=bin_size, flip=flip))
            self.layer_names.append(f"{total_index}_SensorLayer")
            total_index += 1
        if active_sensor_noise:
            self.layers.append(SensorNoiseLayer(blur_kernel_size=blur_kernel_size, blur_sigma=blur_sigma,
                                                gray_mean=gray_mean, gray_sigma=gray_sigma,
                                                gray_ratio=gray_ratio, noise_std=noise_std))
            self.layer_names.append(f"{total_index}_SensorNoiseLayer")
            total_index += 1
        
        self.layers.append(CropResizeDisplacePadLayer(resize_size=(128, 128), pad_size=(128, 128)))
        self.layer_names.append(f"{total_index}_ResizePadLayer{resize_pad_layer_index}")
        resize_pad_layer_index += 1
        total_index += 1

    def forward(self, x):
        # ======        
        # return_phases=True, output phase of every material layer
        # ======
        phase_list = [] 

        for layer in self.layers:
            if self.return_phases and isinstance(layer, MaterialLayer):
                x, phase = layer(x)
                phase_list.append(phase)
            else:
                x = layer(x)

        if self.return_phases:
            return x, phase_list
        else:
            return x
