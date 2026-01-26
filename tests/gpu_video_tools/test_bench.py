"""Tests for benchmarking utilities."""

import csv
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpu_video_tools.bench import (
    BenchmarkTimer,
    benchmark_context,
    BenchmarkLogger,
    get_file_size,
)
from gpu_video_tools.gpu import ResolvedDevice


def test_benchmark_timer_basic():
    """Test basic timer functionality."""
    timer = BenchmarkTimer()
    
    # Before starting
    assert timer.start_time is None
    assert timer.end_time is None
    assert timer.duration == 0.0
    
    # Start timer
    timer.start()
    assert timer.start_time is not None
    
    # Wait a bit
    time.sleep(0.1)
    
    # Stop timer
    timer.stop()
    assert timer.end_time is not None
    
    # Duration should be positive
    assert timer.duration > 0
    assert timer.duration >= 0.1


def test_benchmark_timer_duration_before_stop():
    """Test that duration can be read before stopping."""
    timer = BenchmarkTimer()
    timer.start()
    
    time.sleep(0.05)
    
    # Should be able to read duration before stopping
    duration1 = timer.duration
    assert duration1 > 0
    
    time.sleep(0.05)
    
    # Duration should increase
    duration2 = timer.duration
    assert duration2 > duration1


def test_benchmark_context():
    """Test benchmark context manager."""
    with benchmark_context() as timer:
        time.sleep(0.1)
    
    # Timer should have been stopped
    assert timer.end_time is not None
    assert timer.duration >= 0.1


def test_benchmark_context_with_exception():
    """Test that timer stops even if exception is raised."""
    try:
        with benchmark_context() as timer:
            time.sleep(0.05)
            raise ValueError("Test error")
    except ValueError:
        pass
    
    # Timer should still have been stopped
    assert timer.end_time is not None
    assert timer.duration >= 0.05


def test_benchmark_logger_initialization():
    """Test BenchmarkLogger initialization."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        csv_path = f.name
    
    try:
        logger = BenchmarkLogger(csv_path)
        
        # CSV file should exist
        assert Path(csv_path).exists()
        
        # Should have header
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            assert 'tool' in header
            assert 'duration_s' in header
            assert 'success' in header
            assert 'vendor' in header
    
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_benchmark_logger_log_result():
    """Test logging benchmark result."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        csv_path = f.name
    
    try:
        logger = BenchmarkLogger(csv_path)
        
        # Create mock device
        device = ResolvedDevice(
            vendor='nvidia',
            index=0,
            display_name='NVIDIA GeForce RTX 3080',
            device_spec='nvidia:0',
            ffmpeg_decode_args=['-hwaccel', 'cuda'],
            ffmpeg_encode_suffix='_nvenc',
            ffmpeg_filter_hw='scale_cuda',
            onnx_providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Create timer
        timer = BenchmarkTimer()
        timer.start()
        time.sleep(0.1)
        timer.stop()
        
        # Log result
        logger.log_result(
            tool='transcode',
            device=device,
            timer=timer,
            success=True,
            cmdline='ffmpeg -i input.mp4 output.mp4',
            input_path='input.mp4',
            output_path='output.mp4',
            bytes_in=1024000,
            bytes_out=512000,
        )
        
        # Read and verify
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        row = rows[0]
        
        assert row['tool'] == 'transcode'
        assert row['vendor'] == 'nvidia'
        assert row['gpu_index'] == '0'
        assert row['success'] == 'True'
        assert float(row['duration_s']) >= 0.1
        assert row['bytes_in'] == '1024000'
        assert row['bytes_out'] == '512000'
    
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_benchmark_logger_append_multiple():
    """Test appending multiple results."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        csv_path = f.name
    
    try:
        logger = BenchmarkLogger(csv_path)
        
        device = ResolvedDevice(
            vendor='cpu',
            index=None,
            display_name='CPU',
            device_spec='cpu',
            ffmpeg_decode_args=[],
            ffmpeg_encode_suffix='',
            ffmpeg_filter_hw=None,
            onnx_providers=['CPUExecutionProvider']
        )
        
        # Log multiple results
        for i in range(3):
            timer = BenchmarkTimer()
            timer.start()
            time.sleep(0.05)
            timer.stop()
            
            logger.log_result(
                tool=f'tool{i}',
                device=device,
                timer=timer,
                success=True,
            )
        
        # Read and verify
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 3
        assert rows[0]['tool'] == 'tool0'
        assert rows[1]['tool'] == 'tool1'
        assert rows[2]['tool'] == 'tool2'
    
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_benchmark_logger_cpu_device():
    """Test logging with CPU device."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        csv_path = f.name
    
    try:
        logger = BenchmarkLogger(csv_path)
        
        device = ResolvedDevice(
            vendor='cpu',
            index=None,
            display_name='CPU',
            device_spec='cpu',
            ffmpeg_decode_args=[],
            ffmpeg_encode_suffix='',
            ffmpeg_filter_hw=None,
            onnx_providers=['CPUExecutionProvider']
        )
        
        timer = BenchmarkTimer()
        timer.start()
        timer.stop()
        
        logger.log_result(
            tool='test',
            device=device,
            timer=timer,
            success=True,
        )
        
        # Read and verify
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        row = rows[0]
        
        assert row['vendor'] == 'cpu'
        assert row['gpu_index'] == ''  # No index for CPU
        assert row['gpu_name'] == 'CPU'
    
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_benchmark_logger_no_device():
    """Test logging without device information."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        csv_path = f.name
    
    try:
        logger = BenchmarkLogger(csv_path)
        
        timer = BenchmarkTimer()
        timer.start()
        timer.stop()
        
        logger.log_result(
            tool='test',
            device=None,
            timer=timer,
            success=False,
        )
        
        # Read and verify
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        row = rows[0]
        
        assert row['tool'] == 'test'
        assert row['success'] == 'False'
        # Device fields should be empty or 'unknown'
    
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_get_file_size_existing():
    """Test getting file size for existing file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        file_path = f.name
    
    try:
        size = get_file_size(file_path)
        assert size > 0
        assert size == len("test content")
    
    finally:
        Path(file_path).unlink(missing_ok=True)


def test_get_file_size_nonexistent():
    """Test getting file size for non-existent file."""
    size = get_file_size('/path/to/nonexistent/file.txt')
    assert size == 0


def test_get_file_size_directory():
    """Test getting file size for directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        size = get_file_size(tmpdir)
        # Should return 0 or handle gracefully
        assert size >= 0


def test_benchmark_logger_timestamps():
    """Test that timestamps are recorded."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        csv_path = f.name
    
    try:
        logger = BenchmarkLogger(csv_path)
        
        timer = BenchmarkTimer()
        timer.start()
        time.sleep(0.05)
        timer.stop()
        
        logger.log_result(
            tool='test',
            device=None,
            timer=timer,
            success=True,
        )
        
        # Read and verify
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        row = rows[0]
        
        # Should have timestamps
        assert row['ts_start']
        assert row['ts_end']
        
        # Timestamps should be parseable
        # (format depends on implementation)
    
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_benchmark_logger_with_notes():
    """Test logging with notes field."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        csv_path = f.name
    
    try:
        logger = BenchmarkLogger(csv_path)
        
        timer = BenchmarkTimer()
        timer.start()
        timer.stop()
        
        logger.log_result(
            tool='test',
            device=None,
            timer=timer,
            success=True,
            notes='Test notes with special characters: áéíóú'
        )
        
        # Read and verify
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        row = rows[0]
        assert 'notes' in row
    
    finally:
        Path(csv_path).unlink(missing_ok=True)
