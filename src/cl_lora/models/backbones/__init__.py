"""Backbone model wrappers for SpectralGPT and SoftCon."""

from .softcon import load_softcon
from .spectral_gpt import load_mae_encoder, load_spectral_gpt

__all__ = [
    "load_spectral_gpt",
    "load_mae_encoder",
    "load_softcon",
]
