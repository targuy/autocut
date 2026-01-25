"""Tests for benchmark utilities."""

import csv
import time
from pathlib import Path

import pytest

from gpu_video_tools.bench import (
    BenchmarkTimer,
    benchmark_context,
    BenchmarkLogger,
    get_file_size,
)
from gpu_video_tools.gpu import ResolvedDevice


def test_benchmark_timer_start_stop():
    """Test basic timer start/stop."""
    timer = BenchmarkTimer()
    
    assert timer.start_time is None
    assert timer.end_time is None
    
    timer.start()
    assert timer.start_time is not None
    
    time.sleep(0.05)
    
    timer.stop()
    assert timer.end_time is not None
    assert timer.duration >= 0.05


def test_benchmark_timer_duration_while_running():
    """Test getting duration while timer is still running."""
    timer = BenchmarkTimer()
    timer.start()
    
    time.sleep(0.05)
    
    # Should return current duration even without stop()
    duration = timer.duration
    assert duration >= 0.05


def test_benchmark_timer_duration_not_started():
    """Test duration when timer not started."""
    timer = BenchmarkTimer()
    assert timer.duration == 0.0


def test_benchmark_context():
    """Test benchmark context manager."""
    with benchmark_context() as timer:
        time.sleep(0.05)
    
    assert timer.duration >= 0.05
    assert timer.end_time is not None


def test_benchmark_context_with_exception():
    """Test benchmark context manager with exception."""
    with pytest.raises(ValueError):
        with benchmark_context() as timer:
            time.sleep(0.02)
            raise ValueError("Test error")
    
    # Timer should still be stopped
    assert timer.end_time is not None
    assert timer.duration >= 0.02


def test_benchmark_logger_initialization(tmp_path):
    """Test BenchmarkLogger initialization."""
    csv_path = tmp_path / 'bench.csv'
    logger = BenchmarkLogger(str(csv_path))
    
    assert logger.csv_path == csv_path
    assert csv_path.exists()
    
    # Check header
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == BenchmarkLogger.FIELDNAMES


def test_benchmark_logger_creates_directory(tmp_path):
    """Test that logger creates parent directories."""
    csv_path = tmp_path / 'subdir' / 'bench.csv'
    logger = BenchmarkLogger(str(csv_path))
    
    assert csv_path.exists()
    assert csv_path.parent.exists()


def test_benchmark_logger_log_result_cpu(tmp_path):
    """Test logging a result with CPU device."""
    csv_path = tmp_path / 'bench.csv'
    logger = BenchmarkLogger(str(csv_path))
    
    device = ResolvedDevice(
        vendor='cpu',
        index=None,
        device_spec='cpu',
        display_name='CPU',
        ffmpeg_decode_args=[],
        ffmpeg_encode_suffix='',
        ffmpeg_filter_hw=None,
        onnx_providers=['CPUExecutionProvider'],
    )
    
    timer = BenchmarkTimer()
    timer.start()
    time.sleep(0.01)
    timer.stop()
    
    logger.log_result(
        tool='transcode',
        device=device,
        timer=timer,
        success=True,
        cmdline='ffmpeg -i input.mp4 output.mp4',
        input_path='input.mp4',
        output_path='output.mp4',
        frames_in=100,
        frames_out=100,
        bytes_in=1024,
        bytes_out=512,
        notes='Test run',
    )
    
    # Read back and verify
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 1
    row = rows[0]
    assert row['tool'] == 'transcode'
    assert row['vendor'] == 'cpu'
    assert row['success'] == 'true'
    assert float(row['duration_s']) > 0
    assert row['frames_in'] == '100'
    assert row['avg_fps'] != ''  # Should be calculated


def test_benchmark_logger_log_result_nvidia(tmp_path):
    """Test logging a result with NVIDIA device."""
    csv_path = tmp_path / 'bench.csv'
    logger = BenchmarkLogger(str(csv_path))
    
    device = ResolvedDevice(
        vendor='nvidia',
        index=0,
        device_spec='nvidia:0',
        display_name='NVIDIA GeForce RTX 3090',
        ffmpeg_decode_args=['-hwaccel', 'cuda'],
        ffmpeg_encode_suffix='_nvenc',
        ffmpeg_filter_hw='cuda',
        onnx_providers=['CUDAExecutionProvider'],
    )
    
    timer = BenchmarkTimer()
    timer.start()
    time.sleep(0.01)
    timer.stop()
    
    logger.log_result(
        tool='transcode',
        device=device,
        timer=timer,
        success=True,
        frames_out=1000,
    )
    
    # Read back and verify
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    row = rows[0]
    assert row['vendor'] == 'nvidia'
    assert row['gpu_name'] == 'NVIDIA GeForce RTX 3090'
    assert row['gpu_index'] == '0'
    assert row['hw_backend'] == 'cuda'


def test_benchmark_logger_log_result_amd(tmp_path):
    """Test logging a result with AMD device."""
    csv_path = tmp_path / 'bench.csv'
    logger = BenchmarkLogger(str(csv_path))
    
    device = ResolvedDevice(
        vendor='amd',
        index=0,
        device_spec='amd:0',
        display_name='AMD Radeon RX 6900 XT',
        ffmpeg_decode_args=['-hwaccel', 'd3d11va'],
        ffmpeg_encode_suffix='_amf',
        ffmpeg_filter_hw=None,
        onnx_providers=['DmlExecutionProvider'],
    )
    
    timer = BenchmarkTimer()
    timer.start()
    time.sleep(0.01)
    timer.stop()
    
    logger.log_result(
        tool='scenes',
        device=device,
        timer=timer,
        success=True,
    )
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    row = rows[0]
    assert row['vendor'] == 'amd'
    assert row['gpu_name'] == 'AMD Radeon RX 6900 XT'
    assert row['hw_backend'] == 'd3d11va'


def test_benchmark_logger_log_result_failure(tmp_path):
    """Test logging a failed operation."""
    csv_path = tmp_path / 'bench.csv'
    logger = BenchmarkLogger(str(csv_path))
    
    timer = BenchmarkTimer()
    timer.start()
    timer.stop()
    
    logger.log_result(
        tool='transcode',
        device=None,
        timer=timer,
        success=False,
        notes='Device not found',
    )
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    row = rows[0]
    assert row['success'] == 'false'
    assert row['notes'] == 'Device not found'


def test_benchmark_logger_multiple_results(tmp_path):
    """Test logging multiple results."""
    csv_path = tmp_path / 'bench.csv'
    logger = BenchmarkLogger(str(csv_path))
    
    device = ResolvedDevice('cpu', None, 'CPU', 'cpu', [], '', None, [])
    
    for i in range(3):
        timer = BenchmarkTimer()
        timer.start()
        time.sleep(0.01)
        timer.stop()
        
        logger.log_result(
            tool='transcode',
            device=device,
            timer=timer,
            success=True,
        )
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 3


def test_benchmark_logger_avg_fps_calculation(tmp_path):
    """Test average FPS calculation."""
    csv_path = tmp_path / 'bench.csv'
    logger = BenchmarkLogger(str(csv_path))
    
    device = ResolvedDevice('cpu', None, 'CPU', 'cpu', [], '', None, [])
    
    timer = BenchmarkTimer()
    timer.start()
    time.sleep(0.1)
    timer.stop()
    
    logger.log_result(
        tool='transcode',
        device=device,
        timer=timer,
        success=True,
        frames_out=100,
    )
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    row = rows[0]
    avg_fps = float(row['avg_fps'])
    # 100 frames / ~0.1s should be around 1000 fps
    assert avg_fps > 900  # Allow some tolerance


def test_get_file_size(tmp_path):
    """Test getting file size."""
    test_file = tmp_path / 'test.txt'
    test_file.write_text('Hello, World!')
    
    size = get_file_size(str(test_file))
    assert size == 13


def test_get_file_size_not_found():
    """Test getting size of non-existent file."""
    size = get_file_size('/nonexistent/file.txt')
    assert size is None


def test_get_file_size_empty_file(tmp_path):
    """Test getting size of empty file."""
    test_file = tmp_path / 'empty.txt'
    test_file.touch()
    
    size = get_file_size(str(test_file))
    assert size == 0
