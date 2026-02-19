import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=128):
        super().__init__()

        # --- Encoder ---
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=2, padding=1),  # 128 x 64 x 64
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),             # 256 x 32 x 32
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),            # 512 x 16 x 16
            nn.GroupNorm(64, 512),
            nn.LeakyReLU(0.2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(512, latent_channels, kernel_size=3, stride=2, padding=1), # latent_channels x 8 x 8
            nn.GroupNorm(16, latent_channels),
            nn.Tanh()
        )

        # --- Decoder ---
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 512 x 16 x 16
            nn.GroupNorm(64, 512),
            nn.LeakyReLU(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256 x 32 x 32
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),    # 128 x 64 x 64
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(128, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 3 x 128 x 128
            nn.Tanh()
        )

    def preprocess(self, x):
        # Adjusted to scale input to [-1, 1] and center around 0
        return (x - 0.5) * 2

    def encode(self, x):
        x = self.preprocess(x)
        z1 = self.encoder1(x)
        z2 = self.encoder2(z1)
        z3 = self.encoder3(z2)
        z = self.encoder4(z3)
        return z

    def decode(self, z):
        z = self.decoder1(z)
        z = self.decoder2(z)
        z = self.decoder3(z)
        x_recon = self.decoder4(z)
        return (x_recon + 1) / 2  # Rescale to [0, 1]

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
