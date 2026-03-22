"""Conditional VAE over canonical EEG windows."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by optional dependency checks
    torch = None
    nn = None


class EEGConditionalVAE(nn.Module if nn is not None else object):
    """Simple CVAE that conditions on concentration/stress scalars."""

    def __init__(
        self,
        channels: int = 8,
        samples: int = 500,
        condition_dim: int = 2,
        latent_dim: int = 64,
    ) -> None:
        if nn is None:
            raise ImportError("torch is required to use EEGConditionalVAE.")
        super().__init__()
        self.channels = channels
        self.samples = samples
        input_dim = channels * samples
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, input_dim),
        )

    def encode(self, inputs, conditions):
        flattened = inputs.reshape(inputs.shape[0], -1)
        hidden = self.encoder(torch.cat([flattened, conditions], dim=1))
        return self.mu_head(hidden), self.logvar_head(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latents, conditions):
        decoded = self.decoder(torch.cat([latents, conditions], dim=1))
        return decoded.reshape(-1, self.channels, self.samples)

    def forward(self, inputs, conditions):
        mu, logvar = self.encode(inputs, conditions)
        latents = self.reparameterize(mu, logvar)
        reconstruction = self.decode(latents, conditions)
        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "latents": latents,
        }
