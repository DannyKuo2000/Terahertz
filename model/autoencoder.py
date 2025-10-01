import torch
import torch.nn as nn
from config import AUTOENCODER_CONFIG 

class Autoencoder(nn.Module):
    def __init__(self, encoder=None, decoder=None, config=AUTOENCODER_CONFIG):
        super().__init__()
        self.use_encoder = config.get("use_encoder", True)
        self.use_decoder = config.get("use_decoder", True)

        self.encoder = encoder if self.use_encoder else None
        self.decoder = decoder if self.use_decoder else None

        # ModuleList 只是用來展示，不參與 forward
        self.moduleList = nn.ModuleList(
            [m for m in [self.encoder, self.decoder] if m is not None]
        )

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        if self.decoder is not None:
            x = self.decoder(x)
        return x
