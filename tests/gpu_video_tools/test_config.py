"""Tests for configuration management and device precedence."""

import tempfile
from pathlib import Path

import pytest

from gpu_video_tools.config import Config, create_default_config, get_default_config_path
from gpu_video_tools.gpu import enumerate_devices, resolve_device
from gpu_video_tools.exceptions import DeviceMissingError


def test_get_default_config_path():
    """Test getting default config path."""
    path = get_default_config_path()
    
    assert isinstance(path, Path)
    assert '.gpu_video_tools' in str(path)
    assert path.name == 'config.toml'


def test_config_initialization():
    """Test config initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        assert config.config_path == config_path
        assert isinstance(config._data, dict)


def test_config_save_and_load():
    """Test saving and loading config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        # Set some defaults
        config.set_defaults({
            'transcode': 'nvidia:0',
            'scenes': 'cpu',
        })
        
        config.save()
        
        # Load in new instance
        config2 = Config(config_path)
        config2.load()
        
        defaults = config2._data.get('defaults', {})
        assert defaults['transcode'] == 'nvidia:0'
        assert defaults['scenes'] == 'cpu'


def test_config_device_precedence_cli():
    """Test CLI device takes precedence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        config.set_defaults({'transcode': 'nvidia:0'})
        config.save()
        
        # CLI device should override config
        device = config.get_device_for_tool(
            'transcode',
            cli_device='cpu',
            batch_device=None
        )
        
        assert device.device_spec == 'cpu'


def test_config_device_precedence_batch():
    """Test batch device takes precedence over config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        config.set_defaults({'transcode': 'nvidia:0'})
        config.save()
        
        # Batch device should override config
        device = config.get_device_for_tool(
            'transcode',
            cli_device=None,
            batch_device='cpu'
        )
        
        assert device.device_spec == 'cpu'


def test_config_device_precedence_config():
    """Test config device is used when CLI and batch are None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        config.set_defaults({'transcode': 'cpu'})
        config.save()
        
        # Config device should be used
        device = config.get_device_for_tool(
            'transcode',
            cli_device=None,
            batch_device=None
        )
        
        assert device.device_spec == 'cpu'


def test_config_device_missing_error():
    """Test DeviceMissingError when no device specified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        # No defaults set, all sources are None
        with pytest.raises(DeviceMissingError) as exc_info:
            config.get_device_for_tool(
                'transcode',
                cli_device=None,
                batch_device=None
            )
        
        assert 'transcode' in str(exc_info.value)
        assert 'No device specified' in str(exc_info.value)


def test_config_device_limits():
    """Test device limits configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        limits = {
            'nvidia:0': 2,
            'cpu': 4,
        }
        config.set_device_limits(limits)
        config.save()
        
        retrieved_limits = config.get_device_limits()
        assert retrieved_limits == limits


def test_config_codec_preferences():
    """Test codec preferences configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        prefs = {
            'h264': ['nvidia:0', 'cpu'],
            'hevc': ['nvidia:0', 'amd:0'],
        }
        config.set_codec_preferences(prefs)
        config.save()
        
        retrieved_prefs = config.get_codec_preferences()
        assert retrieved_prefs == prefs


def test_create_default_config():
    """Test creating default config from devices."""
    devices = enumerate_devices()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        
        config = create_default_config(devices, config_path)
        config.save()
        
        # Verify config was created
        assert config_path.exists()
        
        # Verify defaults are set
        defaults = config._data.get('defaults', {})
        assert 'transcode' in defaults
        assert 'scenes' in defaults
        assert 'faces' in defaults
        
        # Verify device limits are set
        limits = config._data.get('device_limits', {})
        assert 'cpu' in limits
        
        # Verify codec preferences are set
        codecs = config._data.get('codecs', {})
        assert 'h264' in codecs


def test_config_auto_creates_directory():
    """Test that saving config creates parent directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'nested' / 'dir' / 'config.toml'
        config = Config(config_path)
        
        config.set_defaults({'transcode': 'cpu'})
        config.save()
        
        assert config_path.exists()
        assert config_path.parent.exists()
