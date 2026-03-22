"""Spectral and structural losses for EEG generation."""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - exercised by optional dependency checks
    torch = None


def _require_torch():
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for generator losses.")


def reconstruction_loss(prediction, target):
    _require_torch()
    return torch.mean(torch.abs(prediction - target))


def multiresolution_stft_loss(prediction, target, n_ffts: tuple[int, ...] = (32, 64, 128)):
    _require_torch()
    total = prediction.new_tensor(0.0)
    for n_fft in n_ffts:
        hop = max(n_fft // 4, 1)
        pred_spec = torch.stft(
            prediction.reshape(-1, prediction.shape[-1]),
            n_fft=n_fft,
            hop_length=hop,
            return_complex=True,
        ).abs()
        target_spec = torch.stft(
            target.reshape(-1, target.shape[-1]),
            n_fft=n_fft,
            hop_length=hop,
            return_complex=True,
        ).abs()
        total = total + torch.mean(torch.abs(pred_spec - target_spec))
    return total / len(n_ffts)


def bandpower_loss(prediction, target):
    _require_torch()
    pred_power = torch.mean(torch.square(torch.fft.rfft(prediction, dim=-1).abs()), dim=-1)
    target_power = torch.mean(torch.square(torch.fft.rfft(target, dim=-1).abs()), dim=-1)
    return torch.mean(torch.abs(pred_power - target_power))


def covariance_loss(prediction, target):
    _require_torch()

    def _covariance(batch):
        centered = batch - batch.mean(dim=-1, keepdim=True)
        denom = max(batch.shape[-1] - 1, 1)
        return centered @ centered.transpose(-1, -2) / denom

    return torch.mean(torch.abs(_covariance(prediction) - _covariance(target)))
