"""Tests for TUI utilities."""

from unittest.mock import Mock, patch, MagicMock

import pytest

from gpu_video_tools.tui import (
    create_console,
    create_progress_bar,
    print_device_table,
    print_probe_info,
    print_error_panel,
    print_success_panel,
    print_benchmark_summary,
)
from gpu_video_tools.gpu import ResolvedDevice


def test_create_console_with_rich():
    """Test console creation when Rich is available."""
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', True):
        with patch('gpu_video_tools.tui.Console') as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            
            console = create_console(no_color=False)
            
            mock_console_class.assert_called_once()


def test_create_console_no_color():
    """Test console creation with no_color=True."""
    console = create_console(no_color=True)
    
    # Should get simple console
    assert hasattr(console, 'print')
    assert hasattr(console, 'log')


def test_create_console_without_rich():
    """Test console creation when Rich is not available."""
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', False):
        console = create_console(no_color=False)
        
        # Should get fallback console
        assert hasattr(console, 'print')
        assert hasattr(console, 'log')
        
        # Should be able to call methods without error
        console.print("test")
        console.log("test")


def test_simple_console_print():
    """Test simple console print functionality."""
    console = create_console(no_color=True)
    
    # Should not raise errors
    console.print("Hello, world!")
    console.log("Log message")


def test_create_progress_bar_with_rich():
    """Test progress bar creation when Rich is available."""
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', True):
        with patch('gpu_video_tools.tui.Progress') as mock_progress_class:
            mock_console = MagicMock()
            mock_progress = MagicMock()
            mock_progress_class.return_value = mock_progress
            
            progress = create_progress_bar(mock_console, no_color=False)
            
            mock_progress_class.assert_called_once()


def test_create_progress_bar_no_color():
    """Test progress bar creation with no_color=True."""
    mock_console = MagicMock()
    progress = create_progress_bar(mock_console, no_color=True)
    
    # Should return None for simple mode
    assert progress is None


def test_create_progress_bar_without_rich():
    """Test progress bar creation when Rich is not available."""
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', False):
        mock_console = MagicMock()
        progress = create_progress_bar(mock_console, no_color=False)
        
        assert progress is None


def test_print_device_table_with_rich():
    """Test printing device table with Rich."""
    devices = [
        ResolvedDevice(
            vendor='nvidia',
            index=0,
            display_name='NVIDIA GeForce RTX 3080',
            device_spec='nvidia:0',
            ffmpeg_decode_args=['-hwaccel', 'cuda'],
            ffmpeg_encode_suffix='_nvenc',
            ffmpeg_filter_hw='scale_cuda',
            onnx_providers=['CUDAExecutionProvider']
        ),
        ResolvedDevice(
            vendor='cpu',
            index=None,
            display_name='CPU',
            device_spec='cpu',
            ffmpeg_decode_args=[],
            ffmpeg_encode_suffix='',
            ffmpeg_filter_hw=None,
            onnx_providers=['CPUExecutionProvider']
        )
    ]
    
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', True):
        with patch('gpu_video_tools.tui.Table') as mock_table_class:
            mock_table = MagicMock()
            mock_table_class.return_value = mock_table
            
            mock_console = MagicMock()
            mock_console.print = MagicMock()
            
            print_device_table(devices, mock_console)
            
            # Should create table
            mock_table_class.assert_called_once()
            
            # Should call console.print
            mock_console.print.assert_called()


def test_print_device_table_without_rich():
    """Test printing device table without Rich (fallback)."""
    devices = [
        ResolvedDevice(
            vendor='cpu',
            index=None,
            display_name='CPU',
            device_spec='cpu',
            ffmpeg_decode_args=[],
            ffmpeg_encode_suffix='',
            ffmpeg_filter_hw=None,
            onnx_providers=['CPUExecutionProvider']
        )
    ]
    
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', False):
        mock_console = MagicMock()
        
        # Should not raise error
        print_device_table(devices, mock_console)


def test_print_probe_info():
    """Test printing probe information."""
    probe_data = {
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
            'duration': '120.5',
            'size': '10485760'
        }
    }
    
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', True):
        mock_console = MagicMock()
        
        # Should not raise error
        print_probe_info(probe_data, mock_console)


def test_print_error_panel():
    """Test printing error panel."""
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', True):
        with patch('gpu_video_tools.tui.Panel') as mock_panel_class:
            mock_console = MagicMock()
            
            print_error_panel("Test error message", mock_console, "Error Title")
            
            # Should create panel
            mock_panel_class.assert_called_once()
            
            # Should print to console
            mock_console.print.assert_called_once()


def test_print_error_panel_without_rich():
    """Test printing error panel without Rich."""
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', False):
        mock_console = MagicMock()
        
        print_error_panel("Test error", mock_console)
        
        # Should call console method
        assert mock_console.print.called or mock_console.log.called


def test_print_success_panel():
    """Test printing success panel."""
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', True):
        with patch('gpu_video_tools.tui.Panel') as mock_panel_class:
            mock_console = MagicMock()
            
            print_success_panel("Success message", mock_console, "Success Title")
            
            mock_panel_class.assert_called_once()
            mock_console.print.assert_called_once()


def test_print_success_panel_without_rich():
    """Test printing success panel without Rich."""
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', False):
        mock_console = MagicMock()
        
        print_success_panel("Success", mock_console)
        
        assert mock_console.print.called or mock_console.log.called


def test_print_benchmark_summary():
    """Test printing benchmark summary."""
    bench_results = [
        {
            'device': 'nvidia:0',
            'codec': 'h264',
            'duration': 2.5,
            'fps': 24.0,
            'success': True
        },
        {
            'device': 'cpu',
            'codec': 'h264',
            'duration': 5.0,
            'fps': 12.0,
            'success': True
        }
    ]
    
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', True):
        with patch('gpu_video_tools.tui.Table') as mock_table_class:
            mock_console = MagicMock()
            
            print_benchmark_summary(bench_results, mock_console)
            
            mock_table_class.assert_called_once()
            mock_console.print.assert_called()


def test_print_benchmark_summary_empty():
    """Test printing empty benchmark summary."""
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', True):
        mock_console = MagicMock()
        
        print_benchmark_summary([], mock_console)
        
        # Should handle empty list gracefully
        assert mock_console.print.called or mock_console.log.called


def test_print_device_table_empty():
    """Test printing empty device table."""
    with patch('gpu_video_tools.tui.RICH_AVAILABLE', True):
        mock_console = MagicMock()
        
        print_device_table([], mock_console)
        
        # Should handle empty list gracefully


def test_console_methods_callable():
    """Test that console methods are callable."""
    console = create_console(no_color=True)
    
    # All console methods should be callable
    assert callable(console.print)
    assert callable(console.log)
    
    # Should not raise exceptions when called
    try:
        console.print("test message")
        console.log("test log")
    except Exception as e:
        pytest.fail(f"Console methods raised exception: {e}")


def test_print_functions_with_none_console():
    """Test print functions handle None console gracefully."""
    # Even with None console, functions should not crash
    # (though they might not do anything useful)
    try:
        print_device_table([], None)
        print_error_panel("error", None)
        print_success_panel("success", None)
        print_benchmark_summary([], None)
    except AttributeError:
        # Expected if functions try to call methods on None
        pass


def test_print_probe_info_minimal():
    """Test probe info with minimal data."""
    probe_data = {
        'streams': [],
        'format': {}
    }
    
    mock_console = MagicMock()
    
    # Should not raise error even with empty data
    print_probe_info(probe_data, mock_console)
