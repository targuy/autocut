"""Device enumeration, resolution, and FFmpeg/ONNX configuration."""

import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .exceptions import DeviceNotFoundError


@dataclass
class ResolvedDevice:
    """Represents a resolved device with all necessary configuration."""
    
    vendor: str  # 'nvidia', 'amd', or 'cpu'
    index: Optional[int]  # None for CPU
    display_name: str  # e.g., "NVIDIA GeForce RTX 3080"
    device_spec: str  # Original spec string, e.g., "nvidia:0"
    
    # FFmpeg configuration
    ffmpeg_decode_args: List[str]
    ffmpeg_encode_suffix: str  # e.g., "_nvenc", "_amf", or ""
    ffmpeg_filter_hw: Optional[str]  # e.g., "scale_cuda" or None
    
    # ONNX Runtime providers (in priority order)
    onnx_providers: List[str]
    
    def __str__(self):
        return f"{self.display_name} ({self.device_spec})"


def enumerate_devices() -> List[ResolvedDevice]:
    """Enumerate all available devices (GPUs + CPU)."""
    devices = []
    
    # Enumerate NVIDIA devices
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            devices.append(ResolvedDevice(
                vendor='nvidia',
                index=i,
                display_name=f"NVIDIA {name}",
                device_spec=f"nvidia:{i}",
                ffmpeg_decode_args=['-hwaccel', 'cuda', '-hwaccel_device', str(i)],
                ffmpeg_encode_suffix='_nvenc',
                ffmpeg_filter_hw='scale_cuda',
                onnx_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            ))
        pynvml.nvmlShutdown()
    except (ImportError, Exception):
        # pynvml not available or no NVIDIA devices
        pass
    
    # Enumerate AMD devices (Windows-specific via WMI)
    if platform.system() == 'Windows':
        try:
            import wmi
            c = wmi.WMI()
            amd_devices = []
            for i, gpu in enumerate(c.Win32_VideoController()):
                if 'AMD' in gpu.Name or 'Radeon' in gpu.Name:
                    amd_devices.append((i, gpu.Name))
            
            for idx, (adapter_idx, name) in enumerate(amd_devices):
                devices.append(ResolvedDevice(
                    vendor='amd',
                    index=idx,
                    display_name=f"AMD {name}",
                    device_spec=f"amd:{idx}",
                    ffmpeg_decode_args=[
                        '-init_hw_device', f'd3d11va=ad{idx},adapter={adapter_idx}',
                        '-filter_hw_device', f'ad{idx}',
                        '-hwaccel', 'd3d11va'
                    ],
                    ffmpeg_encode_suffix='_amf',
                    ffmpeg_filter_hw=None,  # AMD typically uses software scaling
                    onnx_providers=['DmlExecutionProvider', 'CPUExecutionProvider'],
                ))
        except (ImportError, Exception):
            # wmi not available or no AMD devices
            pass
    
    # Always add CPU device
    devices.append(ResolvedDevice(
        vendor='cpu',
        index=None,
        display_name="CPU",
        device_spec="cpu",
        ffmpeg_decode_args=[],
        ffmpeg_encode_suffix='',
        ffmpeg_filter_hw=None,
        onnx_providers=['CPUExecutionProvider'],
    ))
    
    return devices


def parse_device_spec(spec: str) -> Tuple[str, Optional[int]]:
    """Parse device spec string into (vendor, index).
    
    Args:
        spec: Device specification, e.g., 'nvidia:0', 'amd:1', 'cpu', 'auto'
    
    Returns:
        Tuple of (vendor, index). Index is None for CPU.
    
    Raises:
        ValueError: If spec format is invalid.
    """
    spec = spec.lower().strip()
    
    if spec == 'cpu':
        return ('cpu', None)
    elif spec == 'auto':
        return ('auto', None)
    elif ':' in spec:
        vendor, idx_str = spec.split(':', 1)
        if vendor not in ('nvidia', 'amd'):
            raise ValueError(f"Unknown vendor: {vendor}")
        try:
            index = int(idx_str)
        except ValueError:
            raise ValueError(f"Invalid device index: {idx_str}")
        return (vendor, index)
    else:
        raise ValueError(f"Invalid device spec: {spec}")


def resolve_device(spec: str, policy: str = 'balance') -> ResolvedDevice:
    """Resolve a device spec string to a ResolvedDevice.
    
    Args:
        spec: Device specification string
        policy: Policy for 'auto' resolution - 'balance', 'prefer-nvidia', 'prefer-amd'
    
    Returns:
        ResolvedDevice instance
    
    Raises:
        DeviceNotFoundError: If device cannot be found
    """
    devices = enumerate_devices()
    
    vendor, index = parse_device_spec(spec)
    
    if vendor == 'auto':
        # Apply policy to choose a device
        if policy == 'prefer-nvidia':
            nvidia_devs = [d for d in devices if d.vendor == 'nvidia']
            if nvidia_devs:
                return nvidia_devs[0]
        elif policy == 'prefer-amd':
            amd_devs = [d for d in devices if d.vendor == 'amd']
            if amd_devs:
                return amd_devs[0]
        
        # Default balance: prefer any GPU over CPU
        gpu_devs = [d for d in devices if d.vendor != 'cpu']
        if gpu_devs:
            return gpu_devs[0]
        
        # Fall back to CPU
        cpu_devs = [d for d in devices if d.vendor == 'cpu']
        if cpu_devs:
            return cpu_devs[0]
        
        raise DeviceNotFoundError('auto', [d.device_spec for d in devices])
    
    # Find matching device
    for device in devices:
        if device.vendor == vendor and device.index == index:
            return device
    
    # Device not found
    available = [d.device_spec for d in devices]
    raise DeviceNotFoundError(spec, available)


def get_available_encoders() -> dict:
    """Query FFmpeg for available encoders.
    
    Returns:
        Dict mapping encoder names to descriptions.
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        encoders = {}
        in_encoder_section = False
        for line in result.stdout.splitlines():
            if '------' in line:
                in_encoder_section = True
                continue
            if in_encoder_section and line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    # Format: " V..... libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)"
                    encoder_name = parts[1]
                    desc = ' '.join(parts[2:]) if len(parts) > 2 else ''
                    encoders[encoder_name] = desc
        
        return encoders
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # FFmpeg not available or error
        return {}


def check_encoder_available(encoder_name: str, device: ResolvedDevice) -> bool:
    """Check if a specific encoder is available for the given device.
    
    Args:
        encoder_name: Base encoder name (e.g., 'h264', 'hevc', 'av1')
        device: Resolved device
    
    Returns:
        True if encoder is available, False otherwise.
    """
    available_encoders = get_available_encoders()
    
    if device.vendor == 'cpu':
        # Check for software encoders
        cpu_encoders = {
            'h264': 'libx264',
            'hevc': 'libx265',
            'av1': 'libaom-av1',
        }
        encoder = cpu_encoders.get(encoder_name)
        return encoder in available_encoders if encoder else False
    else:
        # Check for hardware encoders
        full_encoder = f"{encoder_name}{device.ffmpeg_encode_suffix}"
        return full_encoder in available_encoders
