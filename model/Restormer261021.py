## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Modified in local version in 261021

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # bias here
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

#! [HF-RECOVERY] Begin: High-Frequency Recovery Branch
class HighFrequencyChannelAttention(nn.Module):
    #! [HF-RECOVERY] Added: Channel attention for detail feature selection
    def __init__(self, channels, reduction=4):
        super(HighFrequencyChannelAttention, self).__init__()
        hidden_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.body(self.pool(x))


class HighFrequencyResidualBlock(nn.Module):
    def __init__(self, channels=32):
        super(HighFrequencyResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.channel_attention = HighFrequencyChannelAttention(channels)

    def forward(self, x):
        residual = self.conv2(self.activation(self.conv1(x)))
        return x + self.channel_attention(residual)


class HighFrequencyRecoveryBlock(nn.Module):
    def __init__(self, in_channels, feature_channels=32, num_blocks=4):
        super(HighFrequencyRecoveryBlock, self).__init__()
        #! [HF-RECOVERY] Added: Learnable Laplacian-initialized high-pass path
        self.high_pass = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        with torch.no_grad():
            laplacian = torch.tensor(
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                dtype=self.high_pass.weight.dtype,
            )
            self.high_pass.weight.copy_(laplacian.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))

        #! [HF-RECOVERY] Modified: Fuse raw and high-pass features while preserving shallow feature shape
        self.shallow_features = nn.Conv2d(
            in_channels, feature_channels, kernel_size=3, padding=1
        )
        self.high_pass_features = nn.Conv2d(
            in_channels, feature_channels, kernel_size=3, padding=1
        )
        self.residual_blocks = nn.Sequential(*[
            HighFrequencyResidualBlock(feature_channels)
            for _ in range(num_blocks)  # Use the same number of refinement blocks as in the main Restormer model
        ])

    def forward(self, x):
        high_pass_features = self.high_pass(x)
        features = self.shallow_features(x) + self.high_pass_features(high_pass_features)
        return self.residual_blocks(features)
#! [HF-RECOVERY] End: High-Frequency Recovery Branch

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, config):
        super(Restormer, self).__init__()

        # Core runtime settings
        self.padding_factor = config["padding_factor"]
        self.use_input_padding = config.get("use_input_padding", True)
        self.output_crop_size = config.get("output_crop_size", None)
        self.dual_pixel_task = config.get("dual_pixel_task", False)

        # Model hyperparameters
        self.inp_channels = config["inp_channels"]
        self.out_channels = config["out_channels"]
        self.dim = config["dim"]
        self.num_blocks = config["num_blocks"]
        self.num_refinement_blocks = config.get("num_refinement_blocks", 4)
        self.heads = config["heads"]
        self.ffn_expansion_factor = config.get("ffn_expansion_factor", 2.66)
        self.bias = config.get("bias", False)
        self.LayerNorm_type = config.get("LayerNorm_type", "WithBias")

        self.patch_embed = OverlapPatchEmbed(self.inp_channels, self.dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=self.dim,
                num_heads=self.heads[0],
                ffn_expansion_factor=self.ffn_expansion_factor,
                bias=self.bias,
                LayerNorm_type=self.LayerNorm_type,
            )
            for _ in range(self.num_blocks[0])
        ])

        self.down1_2 = Downsample(self.dim)  # From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(self.dim * 2**1),
                num_heads=self.heads[1],
                ffn_expansion_factor=self.ffn_expansion_factor,
                bias=self.bias,
                LayerNorm_type=self.LayerNorm_type,
            )
            for _ in range(self.num_blocks[1])
        ])

        self.down2_3 = Downsample(int(self.dim * 2**1))  # From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(self.dim * 2**2),
                num_heads=self.heads[2],
                ffn_expansion_factor=self.ffn_expansion_factor,
                bias=self.bias,
                LayerNorm_type=self.LayerNorm_type,
            )
            for _ in range(self.num_blocks[2])
        ])

        self.down3_4 = Downsample(int(self.dim * 2**2))  # From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=int(self.dim * 2**3),
                num_heads=self.heads[3],
                ffn_expansion_factor=self.ffn_expansion_factor,
                bias=self.bias,
                LayerNorm_type=self.LayerNorm_type,
            )
            for _ in range(self.num_blocks[3])
        ])

        self.up4_3 = Upsample(int(self.dim * 2**3))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(self.dim * 2**3), int(self.dim * 2**2), kernel_size=1, bias=self.bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(self.dim * 2**2),
                num_heads=self.heads[2],
                ffn_expansion_factor=self.ffn_expansion_factor,
                bias=self.bias,
                LayerNorm_type=self.LayerNorm_type,
            )
            for _ in range(self.num_blocks[2])
        ])

        self.up3_2 = Upsample(int(self.dim * 2**2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(self.dim * 2**2), int(self.dim * 2**1), kernel_size=1, bias=self.bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(self.dim * 2**1),
                num_heads=self.heads[1],
                ffn_expansion_factor=self.ffn_expansion_factor,
                bias=self.bias,
                LayerNorm_type=self.LayerNorm_type,
            )
            for _ in range(self.num_blocks[1])
        ])

        self.up2_1 = Upsample(int(self.dim * 2**1))  # From Level 2 to Level 1

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=int(self.dim * 2**1),
                num_heads=self.heads[0],
                ffn_expansion_factor=self.ffn_expansion_factor,
                bias=self.bias,
                LayerNorm_type=self.LayerNorm_type,
            )
            for _ in range(self.num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=int(self.dim * 2**1),
                num_heads=self.heads[0],
                ffn_expansion_factor=self.ffn_expansion_factor,
                bias=self.bias,
                LayerNorm_type=self.LayerNorm_type,
            )
            for _ in range(self.num_refinement_blocks)
        ])

        #! [HF-RECOVERY] Begin: Refinement feature injection modules
        self.high_frequency_recovery = HighFrequencyRecoveryBlock(self.inp_channels)
        self.refinement_recovery_projections = nn.ModuleList([
            nn.Conv2d(32, int(self.dim * 2**1), kernel_size=1, bias=self.bias)
            for _ in range(self.num_refinement_blocks)
        ])
        self.refinement_recovery_gammas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1))
            for _ in range(self.num_refinement_blocks)
        ])
        #! [HF-RECOVERY] Added: Multi-scale decoder feature projections
        self.decoder_recovery_projections = nn.ModuleDict({
            "level3": nn.Conv2d(32, int(self.dim * 2**2), kernel_size=1, bias=self.bias),
            "level2": nn.Conv2d(32, int(self.dim * 2**1), kernel_size=1, bias=self.bias),
            "level1": nn.Conv2d(32, int(self.dim * 2**1), kernel_size=1, bias=self.bias),
        })
        self.decoder_recovery_gammas = nn.ParameterDict({
            "level3": nn.Parameter(torch.tensor(0.1)),
            "level2": nn.Parameter(torch.tensor(0.1)),
            "level1": nn.Parameter(torch.tensor(0.1)),
        })
        #! [HF-RECOVERY] End: Refinement feature injection modules

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(self.dim, int(self.dim * 2**1), kernel_size=1, bias=self.bias)
        ###########################

        self.output = nn.Conv2d(int(self.dim * 2**1), self.out_channels, kernel_size=3, stride=1, padding=1, bias=self.bias)


    # padding to ensure downsampling
    def _pad_input(self, x):
        """
        Pad input so H and W are divisible by 8.
        Returns:
            x_pad
            pad_info = (pad_left, pad_right, pad_top, pad_bottom, H, W)
        """
        H, W = x.shape[-2:]

        factor = self.padding_factor
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

        return x, (pad_left, pad_right, pad_top, pad_bottom, H, W)
    
    # cropping to ensure loss calculation
    def _crop_output(self, x, pad_info=None, crop_size=None):
        """
        First remove padding, then optionally center-crop to a target size.
        """
        if pad_info is not None:
            pad_left, pad_right, pad_top, pad_bottom, H, W = pad_info
            if not (pad_top == pad_bottom == pad_left == pad_right == 0):
                x = x[:, :, pad_top:pad_top + H, pad_left:pad_left + W]

        if crop_size is None:
            return x

        if isinstance(crop_size, int):
            target_h = target_w = crop_size
        else:
            target_h, target_w = crop_size

        _, _, H, W = x.shape
        if target_h > H or target_w > W:
            raise ValueError(
                f"crop_size={crop_size} is larger than current tensor size {(H, W)}"
            )

        crop_top = (H - target_h) // 2
        crop_left = (W - target_w) // 2
        return x[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]

    #! [HF-RECOVERY] Begin: Backward-compatible checkpoint loading
    def load_state_dict(self, state_dict, strict=True, **kwargs):
        incompatible_keys = super(Restormer, self).load_state_dict(
            state_dict, strict=False, **kwargs
        )
        recovery_prefixes = (
            "high_frequency_recovery.",
            "refinement_recovery_projections.",
            "refinement_recovery_gammas.",
            "decoder_recovery_projections.",
            "decoder_recovery_gammas.",
        )
        missing_keys = [
            key for key in incompatible_keys.missing_keys
            if not key.startswith(recovery_prefixes)
        ]
        unexpected_keys = incompatible_keys.unexpected_keys
        if strict and (missing_keys or unexpected_keys):
            error_messages = []
            if missing_keys:
                error_messages.append(
                    "Missing key(s) in state_dict: {}.".format(
                        ", ".join('"{}"'.format(key) for key in missing_keys)
                    )
                )
            if unexpected_keys:
                error_messages.append(
                    "Unexpected key(s) in state_dict: {}.".format(
                        ", ".join('"{}"'.format(key) for key in unexpected_keys)
                    )
                )
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_messages)
                )
            )
        return type(incompatible_keys)(missing_keys, unexpected_keys)
    #! [HF-RECOVERY] End: Backward-compatible checkpoint loading

    def forward(self, inp_img):

        # padding
        if self.use_input_padding:
            inp_img, pad_info = self._pad_input(inp_img)
        else:
            pad_info = None
            inp_img = inp_img

        #! [HF-RECOVERY] Begin: Shared recovery feature extraction
        recovery_features = self.high_frequency_recovery(inp_img)
        #! [HF-RECOVERY] End: Shared recovery feature extraction

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        #! [HF-RECOVERY] Added: Inject quarter-resolution recovery features
        recovery_level3 = F.avg_pool2d(recovery_features, kernel_size=4, stride=4)
        recovery_level3 = self.decoder_recovery_projections["level3"](recovery_level3)
        if recovery_level3.shape[-2:] != inp_dec_level3.shape[-2:]:
            recovery_level3 = F.interpolate(
                recovery_level3, size=inp_dec_level3.shape[-2:],
                mode="bilinear", align_corners=False
            )
        inp_dec_level3 = inp_dec_level3 + self.decoder_recovery_gammas["level3"] * recovery_level3
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        #! [HF-RECOVERY] Added: Inject half-resolution recovery features
        recovery_level2 = F.avg_pool2d(recovery_features, kernel_size=2, stride=2)
        recovery_level2 = self.decoder_recovery_projections["level2"](recovery_level2)
        if recovery_level2.shape[-2:] != inp_dec_level2.shape[-2:]:
            recovery_level2 = F.interpolate(
                recovery_level2, size=inp_dec_level2.shape[-2:],
                mode="bilinear", align_corners=False
            )
        inp_dec_level2 = inp_dec_level2 + self.decoder_recovery_gammas["level2"] * recovery_level2
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        #! [HF-RECOVERY] Added: Inject full-resolution recovery features before decoder level 1
        recovery_level1 = self.decoder_recovery_projections["level1"](recovery_features)
        if recovery_level1.shape[-2:] != inp_dec_level1.shape[-2:]:
            recovery_level1 = F.interpolate(
                recovery_level1, size=inp_dec_level1.shape[-2:],
                mode="bilinear", align_corners=False
            )
        inp_dec_level1 = inp_dec_level1 + self.decoder_recovery_gammas["level1"] * recovery_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        #! [HF-RECOVERY] Begin: Per-refinement recovery feature injection
        for refinement_block, projection, gamma in zip(
            self.refinement,
            self.refinement_recovery_projections,
            self.refinement_recovery_gammas,
        ):
            projected_recovery = projection(recovery_features)
            if projected_recovery.shape[-2:] != out_dec_level1.shape[-2:]:
                projected_recovery = F.interpolate(
                    projected_recovery,
                    size=out_dec_level1.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            out_dec_level1 = refinement_block(
                out_dec_level1 + gamma * projected_recovery
            )
        #! [HF-RECOVERY] End: Per-refinement recovery feature injection

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1)

        out_dec_level1 = out_dec_level1 + inp_img

        out = self._crop_output(out_dec_level1, pad_info, self.output_crop_size)
        return out
