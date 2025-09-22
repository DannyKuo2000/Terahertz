import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder=None, sensor=None, sensor_noise=None, config=None):
        """
        Autoencoder with optional sensor + sensor noise
        Args:
            encoder: 模擬光學前端（opticalSimulation.py）
            decoder: Restormer 解碼器
            sensor:  Sensor 模組（可選）
            sensor_noise: SensorNoise 模組（可選）
            config: dict, 來自 config.py (AUTOENCODER_CONFIG)
        """
        super().__init__()
        self.encoder = encoder

        # 根據 config 決定是否啟用 sensor / sensor_noise
        if config is not None:
            self.sensor = sensor if config.get("use_sensor", True) else None
            self.sensor_noise = sensor_noise if config.get("use_sensor_noise", True) else None
            self.decoder = decoder if config.get("use_decoder", True) else None
        else:
            self.sensor = sensor
            self.sensor_noise = sensor_noise
            self.decoder = None
        
        

    def forward(self, x):
        # Encoding
        latent = self.encoder(x)

        # Plug-and-play sensor
        if self.sensor is not None:
            latent = self.sensor(latent)

        # Plug-and-play sensor noise
        if self.sensor_noise is not None:
            latent = self.sensor_noise(latent)

        # Decoding
        if self.decoder is not None:
            recon = self.decoder(latent)
        else:
            recon = latent
        return recon
