"""GPU Video Tools - Windows-first, GPU/CPU-aware video processing toolkit."""

__version__ = "0.1.0"
__author__ = "AutoCut Project"

from .exceptions import (
    GPUVideoToolsError,
    DeviceMissingError,
    DeviceNotFoundError,
    EncoderNotAvailableError,
)

__all__ = [
    "GPUVideoToolsError",
    "DeviceMissingError",
    "DeviceNotFoundError",
    "EncoderNotAvailableError",
]
