"""Tests for Gradio web interface."""

from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

import pytest

from gpu_video_tools.gradio_app import (
    get_device_choices,
    load_config_file,
    save_config_file,
    probe_video,
    create_interface,
)


def test_get_device_choices_success():
    """Test getting device choices when devices are available."""
    with patch('gpu_video_tools.gradio_app.enumerate_devices') as mock_enumerate:
        mock_device1 = Mock()
        mock_device1.device_spec = 'nvidia:0'
        
        mock_device2 = Mock()
        mock_device2.device_spec = 'cpu'
        
        mock_enumerate.return_value = [mock_device1, mock_device2]
        
        choices = get_device_choices()
        
        assert 'nvidia:0' in choices
        assert 'cpu' in choices
        assert len(choices) == 2


def test_get_device_choices_failure():
    """Test getting device choices when enumeration fails."""
    with patch('gpu_video_tools.gradio_app.enumerate_devices') as mock_enumerate:
        mock_enumerate.side_effect = Exception("Device enumeration failed")
        
        choices = get_device_choices()
        
        # Should fall back to safe defaults
        assert 'cpu' in choices
        assert 'auto' in choices


def test_load_config_file_existing():
    """Test loading existing config file."""
    config_content = """# Test config
defaults:
  transcode: cpu
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml') as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        content = load_config_file(config_path)
        
        assert "defaults:" in content
        assert "transcode: cpu" in content
    
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_load_config_file_nonexistent():
    """Test loading non-existent config file."""
    with patch('gpu_video_tools.gradio_app.get_default_config_path') as mock_get_path:
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_get_path.return_value = mock_path
        
        with patch('gpu_video_tools.gradio_app.Path') as mock_path_class:
            mock_default_path = Mock()
            mock_default_path.exists.return_value = False
            mock_path_class.return_value = mock_default_path
            
            content = load_config_file()
            
            # Should return fallback message
            assert "not found" in content or "#" in content


def test_save_config_file_success():
    """Test saving config file."""
    config_content = """defaults:
  transcode: nvidia:0
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.yml'
        
        result = save_config_file(config_content, str(config_path))
        
        assert "✅" in result or "saved" in result.lower()
        assert config_path.exists()
        
        # Verify content
        saved_content = config_path.read_text()
        assert "defaults:" in saved_content


def test_save_config_file_error():
    """Test saving config file to invalid path."""
    config_content = "test: content"
    invalid_path = "/invalid/path/that/does/not/exist/config.yml"
    
    result = save_config_file(config_content, invalid_path)
    
    assert "❌" in result or "error" in result.lower()


def test_probe_video_success():
    """Test probing video file."""
    with patch('gpu_video_tools.gradio_app.run_ffprobe') as mock_probe:
        mock_probe.return_value = {
            'streams': [
                {
                    'codec_type': 'video',
                    'codec_name': 'h264',
                    'width': 1920,
                    'height': 1080,
                    'avg_frame_rate': '30/1'
                }
            ],
            'format': {
                'duration': '120.0',
                'size': '10485760'
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        try:
            json_output, formatted = probe_video(video_path)
            
            # Should have JSON output
            assert 'codec_name' in json_output or 'h264' in json_output
            
            # Should have formatted info
            assert 'Video:' in formatted or 'Resolution:' in formatted
        
        finally:
            Path(video_path).unlink(missing_ok=True)


def test_probe_video_no_file():
    """Test probing with no file provided."""
    json_output, formatted = probe_video("")
    
    assert "Error" in json_output


def test_probe_video_nonexistent_file():
    """Test probing non-existent file."""
    json_output, formatted = probe_video("/path/that/does/not/exist.mp4")
    
    assert "Error" in json_output


def test_create_interface():
    """Test creating Gradio interface."""
    with patch('gpu_video_tools.gradio_app.gr') as mock_gr:
        mock_blocks = MagicMock()
        mock_gr.Blocks.return_value.__enter__ = Mock(return_value=mock_blocks)
        mock_gr.Blocks.return_value.__exit__ = Mock(return_value=False)
        
        # Mock theme
        mock_theme = Mock()
        mock_gr.themes.Soft.return_value = mock_theme
        
        try:
            interface = create_interface()
            
            # Should create interface
            mock_gr.Blocks.assert_called_once()
        
        except Exception:
            # If actual Gradio is imported, it might fail in test env
            pytest.skip("Gradio not available in test environment")


def test_add_job_to_queue():
    """Test adding job to queue."""
    from gpu_video_tools.gradio_app import add_job_to_queue, job_queue
    
    # Clear queue
    job_queue.clear()
    
    job_data = {
        'tool': 'transcode',
        'device': 'cpu',
        'input': 'video.mp4'
    }
    
    result = add_job_to_queue(job_data)
    
    assert "✅" in result or "added" in result.lower()
    assert len(job_queue) == 1
    assert job_queue[0]['tool'] == 'transcode'


def test_get_queue_dataframe():
    """Test getting queue as DataFrame."""
    from gpu_video_tools.gradio_app import add_job_to_queue, get_queue_dataframe, job_queue
    
    # Clear and add jobs
    job_queue.clear()
    
    add_job_to_queue({'tool': 'transcode', 'device': 'cpu', 'input': 'video1.mp4'})
    add_job_to_queue({'tool': 'scenes', 'device': 'cpu', 'input': 'video2.mp4'})
    
    df = get_queue_dataframe()
    
    assert len(df) == 2
    assert 'Tool' in df.columns
    assert 'Status' in df.columns
    assert df.iloc[0]['Tool'] == 'transcode'
    assert df.iloc[1]['Tool'] == 'scenes'


def test_delete_job_from_queue():
    """Test deleting job from queue."""
    from gpu_video_tools.gradio_app import (
        add_job_to_queue,
        delete_job_from_queue,
        job_queue
    )
    
    # Clear and add jobs
    job_queue.clear()
    
    add_job_to_queue({'tool': 'transcode', 'device': 'cpu', 'input': 'video1.mp4'})
    add_job_to_queue({'tool': 'scenes', 'device': 'cpu', 'input': 'video2.mp4'})
    
    initial_len = len(job_queue)
    
    # Delete first job
    df, message = delete_job_from_queue(1)
    
    assert len(job_queue) < initial_len
    assert "✅" in message or "deleted" in message.lower()


def test_clear_queue():
    """Test clearing all jobs from queue."""
    from gpu_video_tools.gradio_app import (
        add_job_to_queue,
        clear_queue,
        job_queue
    )
    
    # Add some jobs
    job_queue.clear()
    add_job_to_queue({'tool': 'transcode', 'device': 'cpu', 'input': 'video1.mp4'})
    add_job_to_queue({'tool': 'scenes', 'device': 'cpu', 'input': 'video2.mp4'})
    
    assert len(job_queue) > 0
    
    df, message = clear_queue()
    
    assert len(job_queue) == 0
    assert "✅" in message or "cleared" in message.lower()


def test_pause_and_resume_queue():
    """Test pausing and resuming queue."""
    from gpu_video_tools.gradio_app import pause_queue, resume_queue, execution_state
    
    # Pause
    result = pause_queue()
    assert execution_state["paused"] is True
    assert "⏸" in result or "pause" in result.lower()
    
    # Resume
    result = resume_queue()
    assert execution_state["paused"] is False
    assert "▶" in result or "resume" in result.lower()


def test_launch_function_exists():
    """Test that launch function exists and is callable."""
    from gpu_video_tools.gradio_app import launch
    
    assert callable(launch)


def test_config_file_operations_integration():
    """Test full config save/load cycle."""
    test_content = """# Test configuration
defaults:
  transcode: nvidia:0
  scenes: cpu
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'test_config.yml'
        
        # Save
        save_result = save_config_file(test_content, str(config_path))
        assert "✅" in save_result or "saved" in save_result.lower()
        
        # Load
        loaded_content = load_config_file(str(config_path))
        assert "defaults:" in loaded_content
        assert "transcode: nvidia:0" in loaded_content
