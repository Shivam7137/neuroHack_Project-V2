"""Torch autoencoder for canonical EEG windows."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by optional dependency checks
    torch = None
    nn = None


class EEGAutoencoder(nn.Module if nn is not None else object):
    """Compact 1D convolutional autoencoder for [batch, channels, samples] EEG."""

    def __init__(self, channels: int = 8, latent_channels: int = 32) -> None:
        if nn is None:
            raise ImportError("torch is required to use EEGAutoencoder.")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, latent_channels, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, channels, kernel_size=7, padding=3),
        )

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, latents):
        return self.decoder(latents)

    def forward(self, inputs):
        latents = self.encode(inputs)
        recon = self.decode(latents)
        return {"reconstruction": recon, "latents": latents}
