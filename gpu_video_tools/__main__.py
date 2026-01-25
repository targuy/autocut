"""Main CLI entry point for GPU Video Tools."""

import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    print("Error: click is required. Install with: pip install click")
    sys.exit(1)

from . import __version__
from .config import Config, create_default_config, get_default_config_path
from .gpu import enumerate_devices, resolve_device
from .ffmpeg_ops import (
    run_ffprobe,
    build_transcode_command,
    build_extract_frames_command,
    create_synthetic_video,
    parse_ffmpeg_progress,
    extract_progress_stats,
)
from .scenes import detect_scenes, save_scenes_csv
from .faces import detect_faces_in_video, detect_faces_in_frames
from .scheduler import run_batch_sync
from .bench import BenchmarkTimer, BenchmarkLogger, benchmark_context, get_file_size
from .tui import (
    create_console,
    print_device_table,
    print_probe_info,
    print_error_panel,
    print_success_panel,
    print_benchmark_summary,
)
from .exceptions import GPUVideoToolsError, DeviceMissingError


# Global options
@click.group()
@click.version_option(version=__version__)
@click.option('--device', type=str, help='Device to use (nvidia:0, amd:0, cpu, auto)')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--bench-csv', type=click.Path(), help='Benchmark CSV output path')
@click.option('--max-procs-per-device', type=int, default=1, help='Max concurrent jobs per device')
@click.option('--no-color', is_flag=True, help='Disable colored output')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), default='INFO')
@click.option('--policy', type=click.Choice(['balance', 'prefer-nvidia', 'prefer-amd']), default='balance',
              help='Device selection policy for auto')
@click.pass_context
def main(ctx, device, config, bench_csv, max_procs_per_device, no_color, log_level, policy):
    """GPU Video Tools - Windows-first, GPU/CPU-aware video processing."""
    ctx.ensure_object(dict)
    ctx.obj['device'] = device
    ctx.obj['config_path'] = Path(config) if config else None
    ctx.obj['bench_csv'] = bench_csv
    ctx.obj['max_procs'] = max_procs_per_device
    ctx.obj['no_color'] = no_color
    ctx.obj['log_level'] = log_level
    ctx.obj['policy'] = policy
    ctx.obj['console'] = create_console(no_color)


@main.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def probe(ctx, input, output_json):
    """Probe video file and display metadata."""
    console = ctx.obj['console']
    
    try:
        probe_data = run_ffprobe(input)
        
        if output_json:
            print(json.dumps(probe_data, indent=2))
        else:
            print_probe_info(probe_data, console)
    
    except Exception as e:
        print_error_panel(str(e), console, "Probe Failed")
        sys.exit(1)


@main.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('-o', '--output', required=True, type=click.Path(), help='Output video file')
@click.option('--width', type=int, help='Output width')
@click.option('--height', type=int, help='Output height')
@click.option('--fps', type=float, help='Output frame rate')
@click.option('--vcodec', type=click.Choice(['h264', 'hevc', 'av1']), default='h264', help='Video codec')
@click.option('--bitrate', type=str, help='Target bitrate (e.g., 8M)')
@click.option('--crf', type=int, help='Constant Rate Factor (alternative to bitrate)')
@click.option('--preset', type=str, help='Encoder preset')
@click.pass_context
def transcode(ctx, input, output, width, height, fps, vcodec, bitrate, crf, preset):
    """Transcode video with GPU/CPU acceleration."""
    console = ctx.obj['console']
    bench_csv = ctx.obj['bench_csv']
    
    try:
        # Load config and resolve device
        config = Config(ctx.obj['config_path'])
        device = config.get_device_for_tool(
            'transcode',
            cli_device=ctx.obj['device'],
            policy=ctx.obj['policy']
        )
        
        console.log(f"Using device: {device}")
        
        # Build command
        cmd = build_transcode_command(
            input, output, device, vcodec,
            width, height, fps, bitrate, crf, preset
        )
        
        # Execute with timing
        with benchmark_context() as timer:
            console.log(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
        
        if success:
            print_success_panel(f"Transcoded to {output} in {timer.duration:.2f}s", console)
        else:
            print_error_panel(f"Transcode failed:\n{result.stderr}", console)
            sys.exit(1)
        
        # Log benchmark
        if bench_csv:
            logger = BenchmarkLogger(bench_csv)
            logger.log_result(
                tool='transcode',
                device=device,
                timer=timer,
                success=success,
                cmdline=' '.join(cmd),
                input_path=input,
                output_path=output,
                bytes_in=get_file_size(input),
                bytes_out=get_file_size(output),
            )
    
    except GPUVideoToolsError as e:
        print_error_panel(str(e), console)
        sys.exit(1)


@main.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('-o', '--output', required=True, type=click.Path(), help='Output CSV file')
@click.option('--threshold', type=float, default=27.0, help='Detection threshold')
@click.option('--min-scene-len', type=float, default=1.0, help='Minimum scene length (seconds)')
@click.pass_context
def scenes(ctx, input, output, threshold, min_scene_len):
    """Detect scene boundaries and save to CSV."""
    console = ctx.obj['console']
    
    try:
        # Load config (scenes typically use CPU)
        config = Config(ctx.obj['config_path'])
        device = config.get_device_for_tool(
            'scenes',
            cli_device=ctx.obj['device'],
            policy=ctx.obj['policy']
        )
        
        console.log(f"Detecting scenes in {input}...")
        
        with benchmark_context() as timer:
            scene_list = detect_scenes(input, threshold, min_scene_len)
            save_scenes_csv(scene_list, output)
        
        print_success_panel(
            f"Detected {len(scene_list)} scenes in {timer.duration:.2f}s\nSaved to {output}",
            console
        )
    
    except GPUVideoToolsError as e:
        print_error_panel(str(e), console)
        sys.exit(1)
    except Exception as e:
        print_error_panel(f"Scene detection failed: {e}", console)
        sys.exit(1)


@main.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('-o', '--output', required=True, type=click.Path(), help='Output directory')
@click.option('--fps', type=float, help='Extract at this frame rate')
@click.option('--every', type=int, help='Extract every Nth frame')
@click.option('--start', type=float, help='Start time (seconds)')
@click.option('--end', type=float, help='End time (seconds)')
@click.option('--suffix', type=click.Choice(['jpg', 'png']), default='jpg', help='Output format')
@click.pass_context
def extract_frames(ctx, input, output, fps, every, start, end, suffix):
    """Extract frames from video."""
    console = ctx.obj['console']
    
    try:
        # Load config and resolve device
        config = Config(ctx.obj['config_path'])
        device = config.get_device_for_tool(
            'extract',
            cli_device=ctx.obj['device'],
            policy=ctx.obj['policy']
        )
        
        console.log(f"Using device: {device}")
        
        # Create output directory
        Path(output).mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = build_extract_frames_command(
            input, output, device, fps, every, start, end, suffix
        )
        
        # Execute
        with benchmark_context() as timer:
            console.log(f"Extracting frames to {output}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
        
        if success:
            # Count extracted frames
            frame_count = len(list(Path(output).glob(f'*.{suffix}')))
            print_success_panel(
                f"Extracted {frame_count} frames in {timer.duration:.2f}s",
                console
            )
        else:
            print_error_panel(f"Frame extraction failed:\n{result.stderr}", console)
            sys.exit(1)
    
    except GPUVideoToolsError as e:
        print_error_panel(str(e), console)
        sys.exit(1)


@main.command()
@click.option('--video', type=click.Path(exists=True), help='Video file path')
@click.option('--frames-dir', type=click.Path(exists=True), help='Directory of frame images')
@click.option('-o', '--output', required=True, type=click.Path(), help='Output CSV file')
@click.option('--draw', type=click.Path(), help='Directory to save frames with drawn detections')
@click.option('--skip', type=int, default=30, help='Process every Nth frame (for video)')
@click.option('--model', type=click.Path(exists=True), help='Path to YuNet ONNX model')
@click.pass_context
def faces(ctx, video, frames_dir, output, draw, skip, model):
    """Detect faces in video or frame directory."""
    console = ctx.obj['console']
    
    if not video and not frames_dir:
        print_error_panel("Must specify either --video or --frames-dir", console)
        sys.exit(1)
    
    try:
        # Load config and resolve device
        config = Config(ctx.obj['config_path'])
        device = config.get_device_for_tool(
            'faces',
            cli_device=ctx.obj['device'],
            policy=ctx.obj['policy']
        )
        
        console.log(f"Using device: {device}")
        
        # Default model path
        if not model:
            model = Path.home() / '.gpu_video_tools' / 'models' / 'face_detection_yunet_2023mar.onnx'
            if not model.exists():
                print_error_panel(
                    f"Model not found at {model}\n"
                    "Download with:\n"
                    "  mkdir -p ~/.gpu_video_tools/models\n"
                    "  wget https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx -O {model}",
                    console
                )
                sys.exit(1)
        
        # Run detection
        with benchmark_context() as timer:
            if video:
                console.log(f"Detecting faces in video {video}...")
                detect_faces_in_video(
                    video, str(model), output,
                    providers=device.onnx_providers,
                    skip_frames=skip,
                    draw_output_dir=draw
                )
            else:
                console.log(f"Detecting faces in frames {frames_dir}...")
                detect_faces_in_frames(
                    frames_dir, str(model), output,
                    providers=device.onnx_providers,
                    draw_output_dir=draw
                )
        
        print_success_panel(f"Face detection completed in {timer.duration:.2f}s\nSaved to {output}", console)
    
    except GPUVideoToolsError as e:
        print_error_panel(str(e), console)
        sys.exit(1)
    except Exception as e:
        print_error_panel(f"Face detection failed: {e}", console)
        sys.exit(1)


@main.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('-o', '--output', required=True, type=click.Path(), help='Output video file')
@click.option('--cuts', required=True, type=click.Path(exists=True), help='Cuts CSV file')
@click.option('--timecode-overlay', is_flag=True, help='Add timecode overlay')
@click.pass_context
def cutlist(ctx, input, output, cuts, timecode_overlay):
    """Apply cutlist to video (concatenate segments)."""
    console = ctx.obj['console']
    
    console.log("[yellow]cutlist command is a placeholder - full implementation pending[/yellow]")
    print_error_panel("Not yet implemented", console)
    sys.exit(1)


@main.command()
@click.option('--write-defaults', type=click.Path(), is_flag=False, flag_value='',
              help='Write optimized defaults to config file')
@click.option('--quick', is_flag=True, help='Quick analysis (skip benchmarks)')
@click.option('--full', is_flag=True, help='Full analysis with benchmarks')
@click.pass_context
def analyze(ctx, write_defaults, quick, full):
    """Analyze hardware and create optimized default configuration."""
    console = ctx.obj['console']
    
    try:
        console.log("Enumerating devices...")
        devices = enumerate_devices()
        
        if not devices:
            print_error_panel("No devices found!", console)
            sys.exit(1)
        
        print_device_table(devices, console)
        
        # Run benchmarks if not quick mode
        bench_results = []
        if not quick:
            console.log("\nRunning micro-benchmarks...")
            
            # Create synthetic test video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
                tmp_video_path = tmp_video.name
            
            try:
                cmd = create_synthetic_video(tmp_video_path, duration=2.0, fps=30)
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    console.log("[yellow]Could not create synthetic video, skipping benchmarks[/yellow]")
                else:
                    # Benchmark each device with h264 encoding
                    for device in devices:
                        if device.vendor == 'cpu':
                            codec = 'h264'
                            # Skip CPU benchmarks in quick mode as they're slow
                            if quick:
                                continue
                        else:
                            codec = 'h264'
                        
                        try:
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
                                tmp_out_path = tmp_out.name
                            
                            with benchmark_context() as timer:
                                cmd = build_transcode_command(
                                    tmp_video_path, tmp_out_path, device, codec,
                                    width=1280, height=720, crf=23
                                )
                                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                                success = result.returncode == 0
                            
                            fps = 60 / timer.duration if timer.duration > 0 else 0
                            
                            bench_results.append({
                                'device': device.device_spec,
                                'codec': codec,
                                'duration': timer.duration,
                                'fps': fps,
                                'success': success,
                            })
                            
                            console.log(f"  {device.device_spec}/{codec}: {timer.duration:.2f}s @ {fps:.1f} fps")
                            
                            # Cleanup
                            Path(tmp_out_path).unlink(missing_ok=True)
                        
                        except Exception as e:
                            console.log(f"  [red]{device.device_spec}/{codec}: Failed - {e}[/red]")
                            bench_results.append({
                                'device': device.device_spec,
                                'codec': codec,
                                'duration': 0,
                                'fps': 0,
                                'success': False,
                            })
            
            finally:
                # Cleanup synthetic video
                Path(tmp_video_path).unlink(missing_ok=True)
            
            if bench_results:
                print_benchmark_summary(bench_results, console)
        
        # Create default config
        if write_defaults is not None:
            output_path = None
            if write_defaults:
                output_path = Path(write_defaults)
            
            config = create_default_config(devices, output_path)
            config.save()
            
            config_path = config.config_path
            print_success_panel(
                f"Optimized defaults written to:\n  {config_path}\n\n"
                f"You can now run tools without specifying --device.",
                console,
                "Configuration Created"
            )
    
    except Exception as e:
        print_error_panel(f"Analysis failed: {e}", console)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option('--csv', required=True, type=click.Path(exists=True), help='Batch jobs CSV file')
@click.pass_context
def run_batch(ctx, csv):
    """Run batch jobs from CSV file."""
    console = ctx.obj['console']
    
    try:
        # Load config
        config = Config(ctx.obj['config_path'])
        
        console.log(f"Loading batch jobs from {csv}...")
        
        # Run batch
        results = run_batch_sync(
            csv,
            config,
            policy=ctx.obj['policy'],
            bench_csv=ctx.obj['bench_csv'],
            console=console,
        )
        
        # Print summary
        success_count = sum(1 for r in results if r['success'])
        total_duration = sum(r['duration'] for r in results)
        
        print_success_panel(
            f"Batch completed: {success_count}/{len(results)} jobs succeeded\n"
            f"Total time: {total_duration:.2f}s",
            console,
            "Batch Complete"
        )
    
    except Exception as e:
        print_error_panel(f"Batch execution failed: {e}", console)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
