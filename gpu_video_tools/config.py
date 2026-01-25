"""Configuration management with device precedence: CLI > batch > config."""

import os
import platform
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python 3.10

import tomli_w

from .exceptions import DeviceMissingError
from .gpu import resolve_device, ResolvedDevice


def get_default_config_path() -> Path:
    """Get the default configuration file path."""
    if platform.system() == 'Windows':
        base = Path(os.environ.get('USERPROFILE', Path.home()))
    else:
        base = Path.home()
    
    return base / '.gpu_video_tools' / 'config.toml'


class Config:
    """Configuration manager with device precedence."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize config manager.
        
        Args:
            config_path: Path to config file. If None, uses default location.
        """
        self.config_path = config_path or get_default_config_path()
        self._data: Dict[str, Any] = {}
        
        if self.config_path.exists():
            self.load()
    
    def load(self):
        """Load configuration from TOML file."""
        with open(self.config_path, 'rb') as f:
            self._data = tomllib.load(f)
    
    def save(self, data: Optional[Dict[str, Any]] = None):
        """Save configuration to TOML file.
        
        Args:
            data: Configuration data to save. If None, saves current data.
        """
        if data is not None:
            self._data = data
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'wb') as f:
            tomli_w.dump(self._data, f)
    
    def get_device_for_tool(
        self,
        tool_name: str,
        cli_device: Optional[str] = None,
        batch_device: Optional[str] = None,
        policy: str = 'balance'
    ) -> ResolvedDevice:
        """Resolve device for a tool using precedence: CLI > batch > config.
        
        Args:
            tool_name: Name of the tool (e.g., 'transcode', 'scenes')
            cli_device: Device specified via CLI (highest priority)
            batch_device: Device from batch CSV row
            policy: Policy for 'auto' device resolution
        
        Returns:
            ResolvedDevice instance
        
        Raises:
            DeviceMissingError: If no device is specified in any source
        """
        # Precedence: CLI > batch > config
        device_spec = None
        
        if cli_device:
            device_spec = cli_device
        elif batch_device:
            device_spec = batch_device
        else:
            # Check config file
            defaults = self._data.get('defaults', {})
            device_spec = defaults.get(tool_name)
        
        if not device_spec:
            raise DeviceMissingError(tool_name)
        
        return resolve_device(device_spec, policy=policy)
    
    def get_device_limits(self) -> Dict[str, int]:
        """Get per-device concurrency limits from config.
        
        Returns:
            Dict mapping device specs to max concurrent jobs.
        """
        return self._data.get('device_limits', {})
    
    def get_codec_preferences(self) -> Dict[str, list]:
        """Get codec-to-device preferences from config.
        
        Returns:
            Dict mapping codec names to ordered list of preferred devices.
        """
        return self._data.get('codecs', {})
    
    def set_defaults(self, defaults: Dict[str, str]):
        """Set default devices for tools.
        
        Args:
            defaults: Dict mapping tool names to device specs
        """
        if 'defaults' not in self._data:
            self._data['defaults'] = {}
        self._data['defaults'].update(defaults)
    
    def set_device_limits(self, limits: Dict[str, int]):
        """Set per-device concurrency limits.
        
        Args:
            limits: Dict mapping device specs to max concurrent jobs
        """
        self._data['device_limits'] = limits
    
    def set_codec_preferences(self, preferences: Dict[str, list]):
        """Set codec-to-device preferences.
        
        Args:
            preferences: Dict mapping codec names to ordered device lists
        """
        self._data['codecs'] = preferences


def create_default_config(devices: list, output_path: Optional[Path] = None) -> Config:
    """Create a default configuration based on available devices.
    
    Args:
        devices: List of ResolvedDevice instances
        output_path: Path to save config. If None, uses default location.
    
    Returns:
        Config instance with defaults set
    """
    config = Config(config_path=output_path)
    
    # Determine best devices for each tool
    gpu_devices = [d for d in devices if d.vendor != 'cpu']
    cpu_device = next((d for d in devices if d.vendor == 'cpu'), None)
    
    # Prefer GPU for compute-heavy tasks, CPU for light tasks
    best_gpu = gpu_devices[0].device_spec if gpu_devices else 'cpu'
    
    defaults = {
        'probe': 'cpu',  # Lightweight, no benefit from GPU
        'scenes': 'cpu',  # PySceneDetect is CPU-based
        'faces': best_gpu,  # ONNX inference benefits from GPU
        'extract': best_gpu,  # Frame decoding benefits from GPU
        'transcode': best_gpu,  # Encoding is GPU-accelerated
        'cutlist': 'cpu',  # Concat operation, minimal benefit from GPU
    }
    
    config.set_defaults(defaults)
    
    # Set conservative concurrency limits
    limits = {}
    for device in devices:
        if device.vendor == 'cpu':
            limits['cpu'] = 2
        else:
            limits[device.device_spec] = 1  # One job per GPU by default
    
    config.set_device_limits(limits)
    
    # Set codec preferences (prefer all devices, GPU first)
    codec_prefs = {}
    for codec in ['h264', 'hevc', 'av1']:
        prefs = [d.device_spec for d in gpu_devices]
        if cpu_device:
            prefs.append(cpu_device.device_spec)
        codec_prefs[codec] = prefs
    
    config.set_codec_preferences(codec_prefs)
    
    return config
