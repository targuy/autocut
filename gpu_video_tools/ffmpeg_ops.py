"""FFmpeg operations: command building, progress parsing, and probing."""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

from .gpu import ResolvedDevice, check_encoder_available
from .exceptions import EncoderNotAvailableError


def run_ffprobe(input_path: str) -> Dict[str, Any]:
    """Run ffprobe and return JSON metadata.
    
    Args:
        input_path: Path to input video file
    
    Returns:
        Dict containing video metadata
    
    Raises:
        subprocess.CalledProcessError: If ffprobe fails
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        input_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def build_transcode_command(
    input_path: str,
    output_path: str,
    device: ResolvedDevice,
    codec: str = 'h264',
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    bitrate: Optional[str] = None,
    crf: Optional[int] = None,
    preset: Optional[str] = None,
) -> List[str]:
    """Build FFmpeg transcode command for the given device.
    
    Args:
        input_path: Input video file
        output_path: Output video file
        device: Resolved device to use
        codec: Video codec (h264, hevc, av1)
        width: Output width (optional)
        height: Output height (optional)
        fps: Output frame rate (optional)
        bitrate: Target bitrate, e.g., '8M' (optional)
        crf: Constant Rate Factor (optional, alternative to bitrate)
        preset: Encoder preset (optional)
    
    Returns:
        List of command arguments
    
    Raises:
        EncoderNotAvailableError: If encoder is not available
    """
    # Check encoder availability
    if not check_encoder_available(codec, device):
        raise EncoderNotAvailableError(
            encoder=f"{codec}{device.ffmpeg_encode_suffix}",
            device=device.device_spec
        )
    
    cmd = ['ffmpeg', '-y']
    
    # Add hardware decode acceleration
    cmd.extend(device.ffmpeg_decode_args)
    
    # Input
    cmd.extend(['-i', input_path])
    
    # Build filter chain
    filters = []
    
    # Scale filter
    if width or height:
        w = width or -1
        h = height or -1
        
        if device.ffmpeg_filter_hw:
            # Hardware scaling (e.g., scale_cuda)
            filters.append(f"{device.ffmpeg_filter_hw}={w}:{h}")
        else:
            # Software scaling
            filters.append(f"scale={w}:{h}")
    
    # FPS filter
    if fps:
        filters.append(f"fps={fps}")
    
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    
    # Encoder selection
    if device.vendor == 'cpu':
        encoder_map = {
            'h264': 'libx264',
            'hevc': 'libx265',
            'av1': 'libaom-av1',
        }
        encoder = encoder_map[codec]
    else:
        encoder = f"{codec}{device.ffmpeg_encode_suffix}"
    
    cmd.extend(['-c:v', encoder])
    
    # Encoding options
    if bitrate:
        cmd.extend(['-b:v', bitrate])
    elif crf is not None:
        cmd.extend(['-crf', str(crf)])
    
    if preset:
        cmd.extend(['-preset', preset])
    
    # Copy audio stream
    cmd.extend(['-c:a', 'copy'])
    
    # Progress output
    cmd.extend(['-progress', 'pipe:1'])
    
    # Output
    cmd.append(output_path)
    
    return cmd


def build_extract_frames_command(
    input_path: str,
    output_dir: str,
    device: ResolvedDevice,
    fps: Optional[float] = None,
    every_n: Optional[int] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    suffix: str = 'jpg',
) -> List[str]:
    """Build FFmpeg command to extract frames.
    
    Args:
        input_path: Input video file
        output_dir: Output directory for frames
        device: Resolved device
        fps: Extract at this frame rate (optional)
        every_n: Extract every Nth frame (optional)
        start: Start time in seconds (optional)
        end: End time in seconds (optional)
        suffix: Output format (jpg or png)
    
    Returns:
        List of command arguments
    """
    cmd = ['ffmpeg', '-y']
    
    # Add hardware decode acceleration
    cmd.extend(device.ffmpeg_decode_args)
    
    # Start time
    if start is not None:
        cmd.extend(['-ss', str(start)])
    
    # Input
    cmd.extend(['-i', input_path])
    
    # End time
    if end is not None:
        duration = end - (start or 0)
        cmd.extend(['-t', str(duration)])
    
    # Frame rate or select filter
    if fps:
        cmd.extend(['-vf', f'fps={fps}'])
    elif every_n:
        cmd.extend(['-vf', f'select=not(mod(n\\,{every_n}))'])
    
    # Output pattern
    output_pattern = str(Path(output_dir) / f'frame_%06d.{suffix}')
    cmd.append(output_pattern)
    
    return cmd


def build_concat_command(
    input_files: List[str],
    output_path: str,
    timecode_overlay: bool = False,
) -> List[str]:
    """Build FFmpeg command to concatenate video segments.
    
    Args:
        input_files: List of input video files
        output_path: Output video file
        timecode_overlay: Whether to add timecode overlay
    
    Returns:
        List of command arguments
    """
    # Create concat file content
    concat_content = '\n'.join([f"file '{f}'" for f in input_files])
    
    cmd = ['ffmpeg', '-y']
    cmd.extend(['-f', 'concat', '-safe', '0', '-i', 'pipe:0'])
    
    if timecode_overlay:
        cmd.extend(['-vf', "drawtext=text='%{pts\\:hms}':x=10:y=10:fontsize=24:fontcolor=white"])
    else:
        cmd.extend(['-c', 'copy'])
    
    cmd.append(output_path)
    
    return cmd, concat_content


def parse_ffmpeg_progress(line: str) -> Optional[Dict[str, Any]]:
    """Parse a line from FFmpeg's -progress pipe:1 output.
    
    Args:
        line: Single line from FFmpeg progress output
    
    Returns:
        Dict with parsed key-value pairs, or None if not a progress line
    """
    line = line.strip()
    if not line or '=' not in line:
        return None
    
    key, value = line.split('=', 1)
    return {key: value}


def extract_progress_stats(progress_data: Dict[str, str]) -> Dict[str, Any]:
    """Extract useful statistics from FFmpeg progress data.
    
    Args:
        progress_data: Accumulated progress data from FFmpeg
    
    Returns:
        Dict with extracted statistics (fps, speed, time, size, etc.)
    """
    stats = {}
    
    # Time in microseconds
    if 'out_time_us' in progress_data:
        stats['time_us'] = int(progress_data['out_time_us'])
        stats['time_s'] = stats['time_us'] / 1_000_000
    elif 'out_time_ms' in progress_data:
        stats['time_ms'] = int(progress_data['out_time_ms'])
        stats['time_s'] = stats['time_ms'] / 1_000
    
    # FPS
    if 'fps' in progress_data:
        try:
            stats['fps'] = float(progress_data['fps'])
        except ValueError:
            pass
    
    # Speed
    if 'speed' in progress_data:
        speed_str = progress_data['speed'].rstrip('x')
        try:
            stats['speed'] = float(speed_str)
        except ValueError:
            pass
    
    # Total size
    if 'total_size' in progress_data:
        try:
            stats['total_size'] = int(progress_data['total_size'])
        except ValueError:
            pass
    
    # Frame number
    if 'frame' in progress_data:
        try:
            stats['frame'] = int(progress_data['frame'])
        except ValueError:
            pass
    
    return stats


def create_synthetic_video(output_path: str, duration: float = 2.0, fps: int = 30) -> List[str]:
    """Create a synthetic test video using FFmpeg.
    
    Args:
        output_path: Path for output video
        duration: Duration in seconds
        fps: Frame rate
    
    Returns:
        Command that was/should be run
    """
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', f'color=c=blue:s=1280x720:r={fps}',
        '-f', 'lavfi',
        '-i', f'sine=frequency=1000:sample_rate=48000',
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-c:a', 'aac',
        output_path
    ]
    return cmd
