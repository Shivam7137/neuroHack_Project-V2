"""Loss functions for synthetic EEG training."""

from src.generator.losses.diversity import diversity_loss
from src.generator.losses.spectral import bandpower_loss, covariance_loss, multiresolution_stft_loss, reconstruction_loss
from src.generator.losses.teacher import teacher_feature_loss, teacher_output_loss

__all__ = [
    "bandpower_loss",
    "covariance_loss",
    "diversity_loss",
    "multiresolution_stft_loss",
    "reconstruction_loss",
    "teacher_feature_loss",
    "teacher_output_loss",
]
