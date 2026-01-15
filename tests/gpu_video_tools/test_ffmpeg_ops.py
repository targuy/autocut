"""Tests for FFmpeg operations."""

import pytest
from gpu_video_tools.ffmpeg_ops import (
    build_transcode_command,
    build_extract_frames_command,
    parse_ffmpeg_progress,
    extract_progress_stats,
    create_synthetic_video,
)
from gpu_video_tools.gpu import resolve_device


# Test constants
CODEC_ENCODER_MAPPINGS = [
    ('h264', 'libx264'),
    ('hevc', 'libx265'),
    ('av1', 'libaom-av1'),
]


def test_build_transcode_command_cpu():
    """Test building transcode command for CPU."""
    device = resolve_device('cpu')
    
    cmd = build_transcode_command(
        'input.mp4',
        'output.mp4',
        device,
        codec='h264',
        width=1920,
        height=1080,
        skip_encoder_check=True,
    )
    
    assert 'ffmpeg' in cmd
    assert '-i' in cmd
    assert 'input.mp4' in cmd
    assert 'output.mp4' in cmd
    assert '-c:v' in cmd
    assert 'libx264' in cmd
    assert 'scale=1920:1080' in ' '.join(cmd)


def test_build_transcode_command_with_bitrate():
    """Test transcode command with bitrate."""
    device = resolve_device('cpu')
    
    cmd = build_transcode_command(
        'input.mp4',
        'output.mp4',
        device,
        codec='h264',
        bitrate='8M',
        skip_encoder_check=True,
    )
    
    assert '-b:v' in cmd
    assert '8M' in cmd


def test_build_transcode_command_with_crf():
    """Test transcode command with CRF."""
    device = resolve_device('cpu')
    
    cmd = build_transcode_command(
        'input.mp4',
        'output.mp4',
        device,
        codec='h264',
        crf=23,
        skip_encoder_check=True,
    )
    
    assert '-crf' in cmd
    assert '23' in cmd


def test_build_transcode_command_with_fps():
    """Test transcode command with FPS."""
    device = resolve_device('cpu')
    
    cmd = build_transcode_command(
        'input.mp4',
        'output.mp4',
        device,
        codec='h264',
        fps=30.0,
        skip_encoder_check=True,
    )
    
    cmd_str = ' '.join(cmd)
    assert 'fps=30' in cmd_str


def test_build_transcode_command_with_preset():
    """Test transcode command with preset."""
    device = resolve_device('cpu')
    
    cmd = build_transcode_command(
        'input.mp4',
        'output.mp4',
        device,
        codec='h264',
        preset='fast',
        skip_encoder_check=True,
    )
    
    assert '-preset' in cmd
    assert 'fast' in cmd


def test_build_extract_frames_command():
    """Test building extract frames command."""
    device = resolve_device('cpu')
    
    cmd = build_extract_frames_command(
        'input.mp4',
        'frames/',
        device,
        fps=1.0,
        suffix='jpg',
    )
    
    assert 'ffmpeg' in cmd
    assert '-i' in cmd
    assert 'input.mp4' in cmd
    cmd_str = ' '.join(cmd)
    assert 'fps=1' in cmd_str
    assert 'frame_' in cmd[-1]
    assert '.jpg' in cmd[-1]


def test_build_extract_frames_with_every_n():
    """Test extract frames with every N frames."""
    device = resolve_device('cpu')
    
    cmd = build_extract_frames_command(
        'input.mp4',
        'frames/',
        device,
        every_n=30,
        suffix='png',
    )
    
    cmd_str = ' '.join(cmd)
    assert 'select=' in cmd_str
    assert '.png' in cmd[-1]


def test_build_extract_frames_with_time_range():
    """Test extract frames with time range."""
    device = resolve_device('cpu')
    
    cmd = build_extract_frames_command(
        'input.mp4',
        'frames/',
        device,
        start=10.0,
        end=60.0,
        suffix='jpg',
    )
    
    assert '-ss' in cmd
    assert '10.0' in cmd
    assert '-t' in cmd
    assert '50.0' in cmd  # duration = end - start


def test_parse_ffmpeg_progress():
    """Test parsing FFmpeg progress lines."""
    # Valid progress line
    result = parse_ffmpeg_progress('frame=100')
    assert result == {'frame': '100'}
    
    result = parse_ffmpeg_progress('fps=30.5')
    assert result == {'fps': '30.5'}
    
    # Invalid lines
    assert parse_ffmpeg_progress('') is None
    assert parse_ffmpeg_progress('no equals sign') is None


def test_extract_progress_stats():
    """Test extracting stats from progress data."""
    progress_data = {
        'out_time_us': '5000000',
        'fps': '30.0',
        'speed': '2.5x',
        'total_size': '1048576',
        'frame': '150',
    }
    
    stats = extract_progress_stats(progress_data)
    
    assert stats['time_us'] == 5000000
    assert stats['time_s'] == 5.0
    assert stats['fps'] == 30.0
    assert stats['speed'] == 2.5
    assert stats['total_size'] == 1048576
    assert stats['frame'] == 150


def test_extract_progress_stats_with_time_ms():
    """Test extracting stats with time in milliseconds."""
    progress_data = {
        'out_time_ms': '5000',
    }
    
    stats = extract_progress_stats(progress_data)
    
    assert stats['time_ms'] == 5000
    assert stats['time_s'] == 5.0


def test_create_synthetic_video():
    """Test creating synthetic video command."""
    cmd = create_synthetic_video('output.mp4', duration=2.0, fps=30)
    
    assert 'ffmpeg' in cmd
    assert '-f' in cmd
    assert 'lavfi' in cmd
    assert 'color=' in ' '.join(cmd)
    assert 'sine=' in ' '.join(cmd)
    assert '-t' in cmd
    assert '2.0' in cmd
    assert 'output.mp4' in cmd


def test_build_transcode_different_codecs():
    """Test building transcode commands for different codecs."""
    device = resolve_device('cpu')
    
    for codec, expected_encoder in CODEC_ENCODER_MAPPINGS:
        cmd = build_transcode_command(
            'input.mp4',
            'output.mp4',
            device,
            codec=codec,
            skip_encoder_check=True,
        )
        
        # Command should be built with the expected encoder
        assert '-c:v' in cmd
        assert expected_encoder in cmd
