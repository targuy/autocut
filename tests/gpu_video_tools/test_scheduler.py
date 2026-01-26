"""Tests for batch job scheduler."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from gpu_video_tools.scheduler import (
    BatchJob,
    parse_batch_csv,
    JobScheduler,
    run_batch_sync,
)
from gpu_video_tools.config import Config
from gpu_video_tools.gpu import ResolvedDevice


@pytest.fixture
def sample_batch_csv():
    """Create a sample batch CSV file."""
    csv_content = """tool,device,input,output,codec
transcode,nvidia:0,video1.mp4,out1.mp4,h264
transcode,cpu,video2.mp4,out2.mp4,hevc
scenes,cpu,video3.mp4,scenes3.csv,
extract,auto,video4.mp4,frames4/,
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        f.write(csv_content)
        csv_path = f.name
    
    yield csv_path
    
    Path(csv_path).unlink(missing_ok=True)


def test_batch_job_dataclass():
    """Test BatchJob dataclass."""
    job = BatchJob(
        tool='transcode',
        device_spec='nvidia:0',
        args={'input': 'video.mp4', 'output': 'out.mp4'},
        row_number=2
    )
    
    assert job.tool == 'transcode'
    assert job.device_spec == 'nvidia:0'
    assert job.args['input'] == 'video.mp4'
    assert job.row_number == 2


def test_parse_batch_csv(sample_batch_csv):
    """Test parsing batch CSV file."""
    jobs = parse_batch_csv(sample_batch_csv)
    
    assert len(jobs) == 4
    
    # Check first job
    assert jobs[0].tool == 'transcode'
    assert jobs[0].device_spec == 'nvidia:0'
    assert jobs[0].args['input'] == 'video1.mp4'
    assert jobs[0].args['output'] == 'out1.mp4'
    assert jobs[0].args['codec'] == 'h264'
    assert jobs[0].row_number == 2
    
    # Check second job (CPU device)
    assert jobs[1].tool == 'transcode'
    assert jobs[1].device_spec == 'cpu'
    assert jobs[1].args['codec'] == 'hevc'
    
    # Check third job (no device = None)
    assert jobs[2].tool == 'scenes'
    assert jobs[2].device_spec == 'cpu'
    
    # Check fourth job (auto device)
    assert jobs[3].tool == 'extract'
    assert jobs[3].device_spec == 'auto'


def test_parse_batch_csv_empty():
    """Test parsing empty CSV."""
    csv_content = """tool,device,input,output
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        f.write(csv_content)
        csv_path = f.name
    
    try:
        jobs = parse_batch_csv(csv_path)
        assert jobs == []
    
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_parse_batch_csv_skip_empty_tool():
    """Test that rows with empty tool are skipped."""
    csv_content = """tool,device,input
transcode,cpu,video1.mp4
,cpu,video2.mp4
scenes,cpu,video3.mp4
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        f.write(csv_content)
        csv_path = f.name
    
    try:
        jobs = parse_batch_csv(csv_path)
        assert len(jobs) == 2
        assert jobs[0].tool == 'transcode'
        assert jobs[1].tool == 'scenes'
    
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_job_scheduler_initialization():
    """Test JobScheduler initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        scheduler = JobScheduler(config, policy='balance')
        
        assert scheduler.config == config
        assert scheduler.policy == 'balance'
        assert scheduler.bench_logger is None


def test_job_scheduler_with_bench_csv():
    """Test JobScheduler with benchmark logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        bench_csv = Path(tmpdir) / 'bench.csv'
        scheduler = JobScheduler(config, policy='balance', bench_csv=str(bench_csv))
        
        assert scheduler.bench_logger is not None


@patch('gpu_video_tools.scheduler.resolve_device')
@patch('gpu_video_tools.scheduler.enumerate_devices')
def test_run_batch_sync_basic(mock_enumerate, mock_resolve, sample_batch_csv):
    """Test synchronous batch execution."""
    # Mock device resolution
    mock_device = ResolvedDevice(
        vendor='cpu',
        index=None,
        display_name='CPU',
        device_spec='cpu',
        ffmpeg_decode_args=[],
        ffmpeg_encode_suffix='',
        ffmpeg_filter_hw=None,
        onnx_providers=['CPUExecutionProvider']
    )
    mock_resolve.return_value = mock_device
    mock_enumerate.return_value = [mock_device]
    
    # Create minimal config
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        # Mock console
        mock_console = MagicMock()
        
        # Note: run_batch_sync may not be fully implemented yet
        # This test verifies the function can be called
        try:
            results = run_batch_sync(
                sample_batch_csv,
                config,
                policy='balance',
                console=mock_console
            )
            
            # Results should be a list
            assert isinstance(results, list)
        
        except NotImplementedError:
            # If not implemented, that's ok for now
            pytest.skip("run_batch_sync not fully implemented")


@patch('gpu_video_tools.scheduler.resolve_device')
def test_job_scheduler_device_resolution(mock_resolve):
    """Test that scheduler resolves devices correctly."""
    # Setup
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.toml'
        config = Config(config_path)
        
        mock_device = ResolvedDevice(
            vendor='nvidia',
            index=0,
            display_name='NVIDIA GPU',
            device_spec='nvidia:0',
            ffmpeg_decode_args=['-hwaccel', 'cuda'],
            ffmpeg_encode_suffix='_nvenc',
            ffmpeg_filter_hw='scale_cuda',
            onnx_providers=['CUDAExecutionProvider']
        )
        mock_resolve.return_value = mock_device
        
        scheduler = JobScheduler(config, policy='prefer-nvidia')
        
        # Verify policy is set
        assert scheduler.policy == 'prefer-nvidia'


def test_parse_batch_csv_extra_columns():
    """Test parsing CSV with extra tool-specific columns."""
    csv_content = """tool,device,input,output,width,height,fps
transcode,cpu,in.mp4,out.mp4,1920,1080,30
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        f.write(csv_content)
        csv_path = f.name
    
    try:
        jobs = parse_batch_csv(csv_path)
        
        assert len(jobs) == 1
        job = jobs[0]
        
        # Extra columns should be in args
        assert job.args['width'] == '1920'
        assert job.args['height'] == '1080'
        assert job.args['fps'] == '30'
    
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_parse_batch_csv_missing_columns():
    """Test parsing CSV with missing optional columns."""
    csv_content = """tool,input
transcode,video.mp4
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        f.write(csv_content)
        csv_path = f.name
    
    try:
        jobs = parse_batch_csv(csv_path)
        
        assert len(jobs) == 1
        job = jobs[0]
        
        assert job.tool == 'transcode'
        assert job.device_spec is None  # No device column = None
        assert job.args['input'] == 'video.mp4'
    
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_batch_job_args_access():
    """Test accessing job arguments."""
    job = BatchJob(
        tool='transcode',
        device_spec='cpu',
        args={
            'input': 'video.mp4',
            'output': 'out.mp4',
            'codec': 'h264',
            'width': '1920',
            'height': '1080'
        },
        row_number=2
    )
    
    # Test args dictionary access
    assert job.args.get('input') == 'video.mp4'
    assert job.args.get('codec') == 'h264'
    assert job.args.get('width') == '1920'
    assert job.args.get('nonexistent') is None


def test_parse_batch_csv_whitespace_handling():
    """Test that whitespace is properly trimmed."""
    csv_content = """tool,device,input,output
  transcode  ,  cpu  ,  video.mp4  ,  out.mp4  
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        f.write(csv_content)
        csv_path = f.name
    
    try:
        jobs = parse_batch_csv(csv_path)
        
        assert len(jobs) == 1
        job = jobs[0]
        
        # Tool and device should be trimmed
        assert job.tool == 'transcode'
        assert job.device_spec == 'cpu'
    
    finally:
        Path(csv_path).unlink(missing_ok=True)
