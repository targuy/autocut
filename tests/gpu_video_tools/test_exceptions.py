"""Tests for exception classes."""

import pytest
from gpu_video_tools.exceptions import (
    GPUVideoToolsError,
    DeviceMissingError,
    DeviceNotFoundError,
    EncoderNotAvailableError,
)


def test_base_exception():
    """Test base exception class."""
    exc = GPUVideoToolsError("Test error")
    assert str(exc) == "Test error"
    assert isinstance(exc, Exception)


def test_device_missing_error():
    """Test DeviceMissingError."""
    exc = DeviceMissingError("transcode")
    
    assert exc.tool_name == "transcode"
    assert "No device specified" in str(exc)
    assert "transcode" in str(exc)
    assert "gpu-tools analyze --write-defaults" in str(exc)
    assert "CLI argument" in str(exc)
    assert "Batch CSV" in str(exc)
    assert "Config file" in str(exc)


def test_device_not_found_error():
    """Test DeviceNotFoundError."""
    exc = DeviceNotFoundError("nvidia:5")
    
    assert exc.device_spec == "nvidia:5"
    assert "nvidia:5" in str(exc)
    assert "not found" in str(exc)


def test_device_not_found_error_with_available():
    """Test DeviceNotFoundError with available devices."""
    available = ["nvidia:0", "cpu"]
    exc = DeviceNotFoundError("nvidia:5", available)
    
    assert exc.device_spec == "nvidia:5"
    assert exc.available_devices == available
    assert "nvidia:0" in str(exc)
    assert "cpu" in str(exc)
    assert "Available devices" in str(exc)


def test_encoder_not_available_error():
    """Test EncoderNotAvailableError."""
    exc = EncoderNotAvailableError("h264_nvenc", "nvidia:0")
    
    assert exc.encoder == "h264_nvenc"
    assert exc.device == "nvidia:0"
    assert "h264_nvenc" in str(exc)
    assert "nvidia:0" in str(exc)
    assert "not available" in str(exc)


def test_encoder_not_available_error_with_alternatives():
    """Test EncoderNotAvailableError with available encoders."""
    available = ["libx264", "libx265"]
    exc = EncoderNotAvailableError("h264_nvenc", "nvidia:0", available)
    
    assert exc.available_encoders == available
    assert "libx264" in str(exc)
    assert "libx265" in str(exc)
    assert "Available encoders" in str(exc)


def test_exceptions_are_catchable():
    """Test that exceptions can be caught properly."""
    # DeviceMissingError is a GPUVideoToolsError
    try:
        raise DeviceMissingError("test")
    except GPUVideoToolsError:
        pass  # Should catch
    
    # DeviceNotFoundError is a GPUVideoToolsError
    try:
        raise DeviceNotFoundError("test")
    except GPUVideoToolsError:
        pass  # Should catch
    
    # EncoderNotAvailableError is a GPUVideoToolsError
    try:
        raise EncoderNotAvailableError("enc", "dev")
    except GPUVideoToolsError:
        pass  # Should catch
