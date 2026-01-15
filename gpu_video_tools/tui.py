"""TUI utilities using Rich for colorful output and progress bars."""

from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Progress = None


def create_console(no_color: bool = False) -> Any:
    """Create a Rich console instance.
    
    Args:
        no_color: Disable color output
    
    Returns:
        Console instance, or simple print wrapper if Rich not available
    """
    if RICH_AVAILABLE and not no_color:
        return Console()
    else:
        # Fallback to simple print
        class SimpleConsole:
            def print(self, *args, **kwargs):
                print(*args)
            
            def log(self, *args, **kwargs):
                print(*args)
        
        return SimpleConsole()


def create_progress_bar(console: Any, no_color: bool = False):
    """Create a Rich progress bar.
    
    Args:
        console: Console instance
        no_color: Disable color output
    
    Returns:
        Progress instance or None if Rich not available
    """
    if RICH_AVAILABLE and not no_color:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
    return None


def print_device_table(devices: List[Any], console: Any):
    """Print a table of available devices.
    
    Args:
        devices: List of ResolvedDevice instances
        console: Console instance
    """
    if RICH_AVAILABLE and hasattr(console, 'print'):
        table = Table(title="Available Devices", box=box.ROUNDED)
        table.add_column("Device Spec", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Vendor", style="yellow")
        table.add_column("FFmpeg Backend", style="blue")
        table.add_column("ONNX Provider", style="magenta")
        
        for device in devices:
            backend = "CPU"
            if device.vendor == 'nvidia':
                backend = "CUDA/NVENC"
            elif device.vendor == 'amd':
                backend = "D3D11VA/AMF"
            
            onnx_provider = device.onnx_providers[0] if device.onnx_providers else "N/A"
            
            table.add_row(
                device.device_spec,
                device.display_name,
                device.vendor.upper(),
                backend,
                onnx_provider,
            )
        
        console.print(table)
    else:
        # Fallback to simple text
        print("\n=== Available Devices ===")
        for device in devices:
            print(f"  {device.device_spec}: {device.display_name} ({device.vendor})")
        print()


def print_probe_info(probe_data: Dict[str, Any], console: Any):
    """Print video probe information in a formatted table.
    
    Args:
        probe_data: ffprobe JSON output
        console: Console instance
    """
    if not RICH_AVAILABLE or not hasattr(console, 'print'):
        # Simple fallback
        import json
        print(json.dumps(probe_data, indent=2))
        return
    
    # Format info
    format_info = probe_data.get('format', {})
    
    table = Table(title="Video Information", box=box.ROUNDED)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    table.add_row("Format", format_info.get('format_long_name', 'N/A'))
    table.add_row("Duration", f"{float(format_info.get('duration', 0)):.2f}s")
    table.add_row("Size", f"{int(format_info.get('size', 0)) / 1024 / 1024:.2f} MB")
    table.add_row("Bitrate", f"{int(format_info.get('bit_rate', 0)) / 1000:.0f} kbps")
    
    console.print(table)
    
    # Stream info
    streams = probe_data.get('streams', [])
    for i, stream in enumerate(streams):
        stream_table = Table(title=f"Stream #{i} ({stream.get('codec_type', 'unknown')})", box=box.SIMPLE)
        stream_table.add_column("Property", style="yellow")
        stream_table.add_column("Value", style="white")
        
        if stream.get('codec_type') == 'video':
            stream_table.add_row("Codec", stream.get('codec_name', 'N/A'))
            stream_table.add_row("Resolution", f"{stream.get('width', 0)}x{stream.get('height', 0)}")
            stream_table.add_row("FPS", f"{eval(stream.get('r_frame_rate', '0/1')):.2f}")
            stream_table.add_row("Pixel Format", stream.get('pix_fmt', 'N/A'))
        elif stream.get('codec_type') == 'audio':
            stream_table.add_row("Codec", stream.get('codec_name', 'N/A'))
            stream_table.add_row("Sample Rate", f"{stream.get('sample_rate', 'N/A')} Hz")
            stream_table.add_row("Channels", str(stream.get('channels', 'N/A')))
        
        console.print(stream_table)


def print_error_panel(message: str, console: Any, title: str = "Error"):
    """Print an error message in a panel.
    
    Args:
        message: Error message
        console: Console instance
        title: Panel title
    """
    if RICH_AVAILABLE and hasattr(console, 'print'):
        panel = Panel(message, title=title, border_style="red", box=box.HEAVY)
        console.print(panel)
    else:
        print(f"\n!!! {title} !!!")
        print(message)
        print()


def print_success_panel(message: str, console: Any, title: str = "Success"):
    """Print a success message in a panel.
    
    Args:
        message: Success message
        console: Console instance
        title: Panel title
    """
    if RICH_AVAILABLE and hasattr(console, 'print'):
        panel = Panel(message, title=title, border_style="green", box=box.DOUBLE)
        console.print(panel)
    else:
        print(f"\n### {title} ###")
        print(message)
        print()


def print_benchmark_summary(results: List[Dict[str, Any]], console: Any):
    """Print a summary table of benchmark results.
    
    Args:
        results: List of benchmark result dicts
        console: Console instance
    """
    if not results:
        return
    
    if RICH_AVAILABLE and hasattr(console, 'print'):
        table = Table(title="Benchmark Results", box=box.ROUNDED)
        table.add_column("Device", style="cyan")
        table.add_column("Codec", style="yellow")
        table.add_column("Duration", style="green")
        table.add_column("FPS", style="blue")
        table.add_column("Status", style="magenta")
        
        for result in results:
            table.add_row(
                result.get('device', 'N/A'),
                result.get('codec', 'N/A'),
                f"{result.get('duration', 0):.2f}s",
                f"{result.get('fps', 0):.1f}",
                "✓" if result.get('success') else "✗",
            )
        
        console.print(table)
    else:
        print("\n=== Benchmark Results ===")
        for result in results:
            status = "✓" if result.get('success') else "✗"
            print(f"  {status} {result.get('device', 'N/A')}/{result.get('codec', 'N/A')}: "
                  f"{result.get('duration', 0):.2f}s @ {result.get('fps', 0):.1f} fps")
        print()
