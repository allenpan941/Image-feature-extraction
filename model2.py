import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=128):
        super().__init__()

        # --- Residual Block ---
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.norm1 = nn.GroupNorm(16, channels)
                self.norm2 = nn.GroupNorm(16, channels)

            def forward(self, x):
                residual = x
                x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
                x = self.norm2(self.conv2(x))
                return F.leaky_relu(x + residual, 0.2)

        # --- Encoder ---
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(64, 512),
            nn.LeakyReLU(0.2),
            ResidualBlock(512)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(512, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, latent_channels),
            nn.Tanh()
        )

        # --- Decoder ---
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(64, 512),
            nn.LeakyReLU(0.2),
            ResidualBlock(512)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(128, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

        # Learnable Preprocessing Scale
        self.input_scale = nn.Parameter(torch.tensor(1.0))

    def preprocess(self, x):
        # Learnable scaling factor
        return (x - 0.5) * 2 * self.input_scale

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
