"""Tests for device management and resolution."""

import pytest
from gpu_video_tools.gpu import (
    parse_device_spec,
    enumerate_devices,
    resolve_device,
    ResolvedDevice,
)
from gpu_video_tools.exceptions import DeviceNotFoundError


def test_parse_device_spec_cpu():
    """Test parsing CPU device spec."""
    vendor, index = parse_device_spec('cpu')
    assert vendor == 'cpu'
    assert index is None


def test_parse_device_spec_auto():
    """Test parsing auto device spec."""
    vendor, index = parse_device_spec('auto')
    assert vendor == 'auto'
    assert index is None


def test_parse_device_spec_nvidia():
    """Test parsing NVIDIA device spec."""
    vendor, index = parse_device_spec('nvidia:0')
    assert vendor == 'nvidia'
    assert index == 0
    
    vendor, index = parse_device_spec('nvidia:1')
    assert vendor == 'nvidia'
    assert index == 1


def test_parse_device_spec_amd():
    """Test parsing AMD device spec."""
    vendor, index = parse_device_spec('amd:0')
    assert vendor == 'amd'
    assert index == 0


def test_parse_device_spec_invalid():
    """Test parsing invalid device specs."""
    with pytest.raises(ValueError):
        parse_device_spec('invalid')
    
    with pytest.raises(ValueError):
        parse_device_spec('nvidia:abc')
    
    with pytest.raises(ValueError):
        parse_device_spec('unknown:0')


def test_enumerate_devices():
    """Test device enumeration (CPU should always be available)."""
    devices = enumerate_devices()
    
    # CPU should always be available
    assert len(devices) >= 1
    
    cpu_devices = [d for d in devices if d.vendor == 'cpu']
    assert len(cpu_devices) == 1
    
    cpu_device = cpu_devices[0]
    assert cpu_device.device_spec == 'cpu'
    assert cpu_device.index is None
    assert cpu_device.display_name == 'CPU'


def test_resolve_device_cpu():
    """Test resolving CPU device."""
    device = resolve_device('cpu')
    
    assert device.vendor == 'cpu'
    assert device.index is None
    assert device.device_spec == 'cpu'
    assert 'CPUExecutionProvider' in device.onnx_providers


def test_resolve_device_auto():
    """Test auto device resolution."""
    device = resolve_device('auto', policy='balance')
    
    # Should resolve to some device
    assert device is not None
    assert device.device_spec in ['cpu', 'nvidia:0', 'amd:0']  # Could be any available device


def test_resolve_device_not_found():
    """Test resolving non-existent device."""
    # Try to resolve a GPU that likely doesn't exist
    with pytest.raises(DeviceNotFoundError):
        resolve_device('nvidia:99')


def test_resolved_device_attributes():
    """Test ResolvedDevice attributes."""
    device = resolve_device('cpu')
    
    # Check required attributes
    assert hasattr(device, 'vendor')
    assert hasattr(device, 'index')
    assert hasattr(device, 'display_name')
    assert hasattr(device, 'device_spec')
    assert hasattr(device, 'ffmpeg_decode_args')
    assert hasattr(device, 'ffmpeg_encode_suffix')
    assert hasattr(device, 'ffmpeg_filter_hw')
    assert hasattr(device, 'onnx_providers')
    
    # CPU should have specific values
    assert device.ffmpeg_encode_suffix == ''
    assert device.ffmpeg_filter_hw is None


def test_device_string_representation():
    """Test device string representation."""
    device = resolve_device('cpu')
    device_str = str(device)
    
    assert 'CPU' in device_str
    assert 'cpu' in device_str
