"""Benchmarking utilities with CSV logging."""

import csv
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .gpu import ResolvedDevice


class BenchmarkTimer:
    """Timer for benchmarking operations."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time


@contextmanager
def benchmark_context():
    """Context manager for benchmarking code blocks.
    
    Usage:
        with benchmark_context() as timer:
            # do work
            pass
        print(f"Duration: {timer.duration:.2f}s")
    """
    timer = BenchmarkTimer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


class BenchmarkLogger:
    """CSV logger for benchmark results."""
    
    FIELDNAMES = [
        'ts_start',
        'ts_end',
        'tool',
        'cmdline',
        'input_path',
        'output_path',
        'duration_s',
        'frames_in',
        'frames_out',
        'avg_fps',
        'bytes_in',
        'bytes_out',
        'vendor',
        'gpu_name',
        'gpu_index',
        'driver',
        'hw_backend',
        'success',
        'notes',
    ]
    
    def __init__(self, csv_path: str):
        """Initialize benchmark logger.
        
        Args:
            csv_path: Path to CSV file for appending results
        """
        self.csv_path = Path(csv_path)
        self._ensure_header()
    
    def _ensure_header(self):
        """Ensure CSV file exists with header."""
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
    
    def log_result(
        self,
        tool: str,
        device: Optional[ResolvedDevice],
        timer: BenchmarkTimer,
        success: bool,
        cmdline: str = '',
        input_path: str = '',
        output_path: str = '',
        frames_in: Optional[int] = None,
        frames_out: Optional[int] = None,
        bytes_in: Optional[int] = None,
        bytes_out: Optional[int] = None,
        notes: str = '',
    ):
        """Log a benchmark result to CSV.
        
        Args:
            tool: Tool name (e.g., 'transcode', 'scenes')
            device: Resolved device used
            timer: BenchmarkTimer with timing information
            success: Whether operation succeeded
            cmdline: Command line executed
            input_path: Input file path
            output_path: Output file path
            frames_in: Number of input frames
            frames_out: Number of output frames
            bytes_in: Input file size in bytes
            bytes_out: Output file size in bytes
            notes: Additional notes
        """
        ts_start = datetime.fromtimestamp(timer.start_time).isoformat() if timer.start_time else ''
        ts_end = datetime.fromtimestamp(timer.end_time).isoformat() if timer.end_time else ''
        duration = timer.duration
        
        # Calculate average FPS
        avg_fps = None
        if frames_out and duration > 0:
            avg_fps = frames_out / duration
        
        # Extract device info
        vendor = device.vendor if device else ''
        gpu_name = device.display_name if device else ''
        gpu_index = device.index if device and device.index is not None else ''
        hw_backend = ''
        if device:
            if device.vendor == 'nvidia':
                hw_backend = 'cuda'
            elif device.vendor == 'amd':
                hw_backend = 'd3d11va'
        
        row = {
            'ts_start': ts_start,
            'ts_end': ts_end,
            'tool': tool,
            'cmdline': cmdline,
            'input_path': input_path,
            'output_path': output_path,
            'duration_s': f'{duration:.3f}',
            'frames_in': frames_in or '',
            'frames_out': frames_out or '',
            'avg_fps': f'{avg_fps:.2f}' if avg_fps else '',
            'bytes_in': bytes_in or '',
            'bytes_out': bytes_out or '',
            'vendor': vendor,
            'gpu_name': gpu_name,
            'gpu_index': gpu_index,
            'driver': '',  # Could be populated from pynvml/wmi
            'hw_backend': hw_backend,
            'success': 'true' if success else 'false',
            'notes': notes,
        }
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)


def get_file_size(path: str) -> Optional[int]:
    """Get file size in bytes.
    
    Args:
        path: File path
    
    Returns:
        File size in bytes, or None if file doesn't exist
    """
    try:
        return Path(path).stat().st_size
    except (FileNotFoundError, OSError):
        return None
