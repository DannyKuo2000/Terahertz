import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder=None, config=None):
        """
        Autoencoder with optional sensor + sensor noise
        Args:
            encoder: 模擬光學前端（opticalSimulation.py）
            decoder: Restormer 解碼器
            config: dict, 來自 config.py (AUTOENCODER_CONFIG)
        """
        super().__init__()
        self.encoder = encoder

        # 根據 config 決定是否啟用 sensor / sensor_noise
        if config is not None:
            self.decoder = decoder if config.get("use_decoder", True) else None
        else:
            self.decoder = None
        
        

    def forward(self, x):
        # Encoding
        latent = self.encoder(x)

        # Decoding
        if self.decoder is not None:
            recon = self.decoder(latent)
        else:
            recon = latent
        return recon
