"""Tests for scheduler module."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from gpu_video_tools.scheduler import (
    BatchJob,
    parse_batch_csv,
    JobScheduler,
    run_batch_sync,
)
from gpu_video_tools.config import Config
from gpu_video_tools.gpu import ResolvedDevice


def test_batch_job_creation():
    """Test BatchJob creation."""
    job = BatchJob(
        tool='transcode',
        device_spec='nvidia:0',
        args={'input': 'test.mp4', 'output': 'out.mp4'},
        row_number=2,
    )
    
    assert job.tool == 'transcode'
    assert job.device_spec == 'nvidia:0'
    assert job.args['input'] == 'test.mp4'
    assert job.row_number == 2


def test_parse_batch_csv(tmp_path):
    """Test parsing batch CSV file."""
    csv_path = tmp_path / 'batch.csv'
    csv_content = """tool,device,input,output,vcodec
transcode,nvidia:0,in1.mp4,out1.mp4,h264
transcode,amd:0,in2.mp4,out2.mp4,hevc
scenes,cpu,in3.mp4,scenes.csv,
"""
    csv_path.write_text(csv_content)
    
    jobs = parse_batch_csv(str(csv_path))
    
    assert len(jobs) == 3
    assert jobs[0].tool == 'transcode'
    assert jobs[0].device_spec == 'nvidia:0'
    assert jobs[0].args['input'] == 'in1.mp4'
    assert jobs[0].args['vcodec'] == 'h264'
    assert jobs[0].row_number == 2
    
    assert jobs[1].device_spec == 'amd:0'
    assert jobs[2].device_spec == 'cpu'


def test_parse_batch_csv_empty_rows(tmp_path):
    """Test parsing batch CSV with empty rows."""
    csv_path = tmp_path / 'batch.csv'
    csv_content = """tool,device,input
transcode,nvidia:0,in1.mp4
,,,
transcode,cpu,in2.mp4
"""
    csv_path.write_text(csv_content)
    
    jobs = parse_batch_csv(str(csv_path))
    
    # Empty tool rows should be skipped
    assert len(jobs) == 2


def test_job_scheduler_initialization(tmp_path):
    """Test JobScheduler initialization."""
    config_path = tmp_path / 'config.toml'
    config_data = {
        'defaults': {},
        'device_limits': {'nvidia:0': 2, 'cpu': 4}
    }
    config = Config(config_path)
    config.save(config_data)
    config.load()
    
    scheduler = JobScheduler(config, policy='balance')
    
    assert scheduler.config == config
    assert scheduler.policy == 'balance'
    assert 'nvidia:0' in scheduler.device_semaphores
    assert 'cpu' in scheduler.device_semaphores


def test_get_semaphore(tmp_path):
    """Test getting semaphore for device."""
    config_path = tmp_path / 'config.toml'
    config = Config(config_path)
    config.save({'defaults': {}, 'device_limits': {'nvidia:0': 2}})
    config.load()
    
    scheduler = JobScheduler(config)
    
    device = ResolvedDevice(
        vendor='nvidia',
        index=0,
        device_spec='nvidia:0',
        display_name='NVIDIA GPU 0',
        ffmpeg_decode_args=[],
        ffmpeg_encode_suffix='_nvenc',
        ffmpeg_filter_hw='cuda',
        onnx_providers=[],
    )
    
    semaphore = scheduler.get_semaphore(device)
    assert isinstance(semaphore, asyncio.Semaphore)


def test_get_semaphore_default(tmp_path):
    """Test getting default semaphore for unrecognized device."""
    config_path = tmp_path / 'config.toml'
    config = Config(config_path)
    config.save({'defaults': {}, 'device_limits': {'nvidia:0': 2}})
    config.load()
    
    scheduler = JobScheduler(config)
    
    device = ResolvedDevice(
        vendor='amd',
        index=0,
        device_spec='amd:0',
        display_name='AMD GPU 0',
        ffmpeg_decode_args=[],
        ffmpeg_encode_suffix='_amf',
        ffmpeg_filter_hw=None,
        onnx_providers=[],
    )
    
    semaphore = scheduler.get_semaphore(device)
    assert semaphore == scheduler.default_semaphore


@pytest.mark.asyncio
async def test_run_job_success(tmp_path):
    """Test running a successful job."""
    config_path = tmp_path / 'config.toml'
    config = Config(config_path)
    config.save({
        'defaults': {'transcode': 'cpu'},
        'device_limits': {'cpu': 1},
    })
    config.load()
    
    scheduler = JobScheduler(config)
    
    job = BatchJob(
        tool='transcode',
        device_spec='cpu',
        args={'input': 'test.mp4', 'output': 'out.mp4'},
        row_number=2,
    )
    
    console = Mock()
    
    result = await scheduler.run_job(job, console)
    
    assert result['row'] == 2
    assert result['tool'] == 'transcode'
    assert result['success'] is True
    assert result['duration'] > 0


@pytest.mark.asyncio
async def test_run_job_device_error(tmp_path):
    """Test running job with device error."""
    config_path = tmp_path / 'config.toml'
    config = Config(config_path)
    config.save({'defaults': {}, 'device_limits': {}})
    config.load()
    
    scheduler = JobScheduler(config)
    
    job = BatchJob(
        tool='transcode',
        device_spec=None,
        args={'input': 'test.mp4'},
        row_number=2,
    )
    
    console = Mock()
    
    result = await scheduler.run_job(job, console)
    
    assert result['row'] == 2
    assert result['success'] is False
    assert len(result['notes']) > 0


@pytest.mark.asyncio
async def test_run_batch(tmp_path):
    """Test running batch of jobs."""
    config_path = tmp_path / 'config.toml'
    config = Config(config_path)
    config.save({
        'defaults': {'transcode': 'cpu', 'scenes': 'cpu'},
        'device_limits': {'cpu': 2},
    })
    config.load()
    
    scheduler = JobScheduler(config)
    
    jobs = [
        BatchJob('transcode', 'cpu', {'input': 'in1.mp4'}, 2),
        BatchJob('transcode', 'cpu', {'input': 'in2.mp4'}, 3),
        BatchJob('scenes', 'cpu', {'input': 'in3.mp4'}, 4),
    ]
    
    console = Mock()
    
    results = await scheduler.run_batch(jobs, console)
    
    assert len(results) == 3
    assert all(r['success'] for r in results)


def test_run_batch_sync(tmp_path):
    """Test synchronous batch execution."""
    csv_path = tmp_path / 'batch.csv'
    csv_content = """tool,device,input
transcode,cpu,in1.mp4
scenes,cpu,in2.mp4
"""
    csv_path.write_text(csv_content)
    
    config_path = tmp_path / 'config.toml'
    config = Config(config_path)
    config.save({
        'defaults': {'transcode': 'cpu', 'scenes': 'cpu'},
        'device_limits': {'cpu': 2},
    })
    config.load()
    
    console = Mock()
    
    results = run_batch_sync(str(csv_path), config, console=console)
    
    assert len(results) == 2
    assert all(r['success'] for r in results)


def test_run_batch_sync_empty_csv(tmp_path):
    """Test synchronous batch execution with empty CSV."""
    csv_path = tmp_path / 'batch.csv'
    csv_content = """tool,device,input
"""
    csv_path.write_text(csv_content)
    
    config_path = tmp_path / 'config.toml'
    config = Config(config_path)
    config.save({'defaults': {}, 'device_limits': {}})
    config.load()
    
    console = Mock()
    
    results = run_batch_sync(str(csv_path), config, console=console)
    
    assert len(results) == 0


def test_scheduler_with_benchmark_logging(tmp_path):
    """Test scheduler with benchmark CSV logging."""
    bench_csv = tmp_path / 'bench.csv'
    config_path = tmp_path / 'config.toml'
    config = Config(config_path)
    config.save({
        'defaults': {'transcode': 'cpu'},
        'device_limits': {'cpu': 1},
    })
    config.load()
    
    scheduler = JobScheduler(config, bench_csv=str(bench_csv))
    
    assert scheduler.bench_logger is not None
    assert bench_csv.exists()
