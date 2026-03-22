"""Generator neural network models."""

from src.generator.models.autoencoder import EEGAutoencoder
from src.generator.models.cvae import EEGConditionalVAE

__all__ = ["EEGAutoencoder", "EEGConditionalVAE"]
