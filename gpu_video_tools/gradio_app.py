"""Gradio web interface for GPU Video Tools."""

import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from datetime import datetime
import threading
import queue

# Optional import of gradio
GRADIO_AVAILABLE = True
try:
    import gradio as gr
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

# Optional import of pandas
PANDAS_AVAILABLE = True
try:
    import pandas as pd
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Optional import of plotly
PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

from .config import Config, get_default_config_path
from .gpu import enumerate_devices, resolve_device
from .ffmpeg_ops import (
    run_ffprobe,
    build_transcode_command,
    build_extract_frames_command,
)
from .scenes import detect_scenes, save_scenes_csv
from .faces import detect_faces_in_video, detect_faces_in_frames
from .scheduler import parse_batch_csv, run_batch_sync
from .bench import BenchmarkLogger, benchmark_context, get_file_size
from .exceptions import GPUVideoToolsError

# Type hints for gradio when available
if TYPE_CHECKING:
    if GRADIO_AVAILABLE:
        GradioComponent = Any
    else:
        GradioComponent = Any


# Check dependencies at module level but don't exit
def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    if not GRADIO_AVAILABLE:
        missing.append("gradio")
    if not PANDAS_AVAILABLE:
        missing.append("pandas")
    if not PLOTLY_AVAILABLE:
        missing.append("plotly")
    return missing


# Global state for queue management
job_queue = []
queue_lock = threading.Lock()
execution_state = {"running": False, "paused": False}


def get_device_choices() -> List[str]:
    """Get list of available devices for dropdown."""
    try:
        devices = enumerate_devices()
        return [dev.device_spec for dev in devices]
    except Exception as e:
        print(f"Warning: Could not enumerate devices: {e}")
        return ["cpu", "auto"]


def load_config_file(config_path: Optional[str] = None) -> str:
    """Load configuration file content."""
    try:
        if config_path:
            path = Path(config_path)
        else:
            path = get_default_config_path()
        
        if path.exists():
            return path.read_text()
        else:
            # Return default template
            default_path = Path(__file__).parent.parent / "tools_preferences.yml"
            if default_path.exists():
                return default_path.read_text()
            return "# Configuration file not found"
    except Exception as e:
        return f"# Error loading config: {e}"


def save_config_file(content: str, config_path: Optional[str] = None) -> str:
    """Save configuration file content."""
    try:
        if config_path:
            path = Path(config_path)
        else:
            path = get_default_config_path()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"✅ Configuration saved to {path}"
    except Exception as e:
        return f"❌ Error saving config: {e}"


def probe_video(video_path: str) -> Tuple[str, str]:
    """Probe video file and return metadata as JSON and formatted text."""
    try:
        if not video_path or not Path(video_path).exists():
            return "Error: No valid video file provided", ""
        
        probe_data = run_ffprobe(video_path)
        json_output = json.dumps(probe_data, indent=2)
        
        # Format key information
        info = []
        if 'streams' in probe_data:
            for stream in probe_data['streams']:
                if stream.get('codec_type') == 'video':
                    info.append(f"Video: {stream.get('codec_name', 'unknown')}")
                    info.append(f"Resolution: {stream.get('width', '?')}x{stream.get('height', '?')}")
                    info.append(f"FPS: {stream.get('avg_frame_rate', '?')}")
                elif stream.get('codec_type') == 'audio':
                    info.append(f"Audio: {stream.get('codec_name', 'unknown')}")
        
        if 'format' in probe_data:
            duration = probe_data['format'].get('duration', 'unknown')
            info.append(f"Duration: {duration}s")
            size = probe_data['format'].get('size', 'unknown')
            if size != 'unknown':
                size_mb = int(size) / (1024 * 1024)
                info.append(f"Size: {size_mb:.2f} MB")
        
        formatted = "\n".join(info)
        return json_output, formatted
    
    except Exception as e:
        return f"Error: {e}", ""


def transcode_video(
    input_path: str,
    device: str,
    codec: str,
    width: Optional[int],
    height: Optional[int],
    fps: Optional[float],
    bitrate: Optional[str],
    crf: Optional[int],
    preset: Optional[str],
    output_name: Optional[str] = None
) -> Tuple[str, str]:
    """Transcode video with specified parameters."""
    try:
        if not input_path or not Path(input_path).exists():
            return "Error: No valid input video", ""
        
        # Generate output path
        if not output_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"transcoded_{timestamp}.mp4"
        
        output_path = Path(tempfile.gettempdir()) / output_name
        
        # Build command
        config = Config()
        resolved_device = resolve_device(device, enumerate_devices(), policy='balance')
        
        cmd = build_transcode_command(
            str(input_path),
            str(output_path),
            resolved_device,
            codec,
            width,
            height,
            fps,
            bitrate,
            crf,
            preset
        )
        
        # Execute transcode
        # Security Note: cmd is a list built by build_transcode_command() from validated
        # parameters - no shell injection risk. subprocess.run default shell=False is safe.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return f"✅ Transcoded successfully to {output_path}", str(output_path)
        else:
            return f"❌ Transcode failed:\n{result.stderr}", ""
    
    except subprocess.TimeoutExpired:
        return "❌ Transcode timed out (5 min limit)", ""
    except Exception as e:
        return f"❌ Error: {e}", ""


def detect_scenes_ui(
    input_path: str,
    threshold: float,
    min_scene_len: float,
    output_name: Optional[str] = None
) -> Tuple[str, str]:
    """Detect scenes and return CSV path."""
    try:
        if not input_path or not Path(input_path).exists():
            return "Error: No valid input video", ""
        
        # Generate output path
        if not output_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"scenes_{timestamp}.csv"
        
        output_path = Path(tempfile.gettempdir()) / output_name
        
        # Detect scenes
        scene_list = detect_scenes(str(input_path), threshold, min_scene_len)
        save_scenes_csv(scene_list, str(output_path))
        
        return f"✅ Detected {len(scene_list)} scenes, saved to {output_path}", str(output_path)
    
    except Exception as e:
        return f"❌ Error: {e}", ""


def extract_frames_ui(
    input_path: str,
    device: str,
    fps: Optional[float],
    every: Optional[int],
    start: Optional[float],
    end: Optional[float],
    suffix: str
) -> Tuple[str, str]:
    """Extract frames from video."""
    try:
        if not input_path or not Path(input_path).exists():
            return "Error: No valid input video", ""
        
        # Generate output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(tempfile.gettempdir()) / f"frames_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        config = Config()
        resolved_device = resolve_device(device, enumerate_devices(), policy='balance')
        
        cmd = build_extract_frames_command(
            str(input_path),
            str(output_dir),
            resolved_device,
            fps,
            every,
            start,
            end,
            suffix
        )
        
        # Execute frame extraction
        # Security Note: cmd list built by build_extract_frames_command() - no shell injection risk
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            frame_count = len(list(output_dir.glob(f'*.{suffix}')))
            return f"✅ Extracted {frame_count} frames to {output_dir}", str(output_dir)
        else:
            return f"❌ Extraction failed:\n{result.stderr}", ""
    
    except subprocess.TimeoutExpired:
        return "❌ Extraction timed out (3 min limit)", ""
    except Exception as e:
        return f"❌ Error: {e}", ""


def detect_faces_ui(
    video_path: Optional[str],
    frames_dir: Optional[str],
    device: str,
    skip_frames: int,
    model_path: Optional[str] = None
) -> Tuple[str, str]:
    """Detect faces in video or frames."""
    try:
        if not video_path and not frames_dir:
            return "Error: Must provide either video or frames directory", ""
        
        if video_path and not Path(video_path).exists():
            return "Error: Video file not found", ""
        
        if frames_dir and not Path(frames_dir).exists():
            return "Error: Frames directory not found", ""
        
        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(tempfile.gettempdir()) / f"faces_{timestamp}.csv"
        
        # Default model path
        if not model_path:
            model_path = str(Path.home() / '.gpu_video_tools' / 'models' / 'face_detection_yunet_2023mar.onnx')
            if not Path(model_path).exists():
                return (
                    f"❌ Model not found at {model_path}\n"
                    "Download from: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
                    ""
                )
        
        # Resolve device
        config = Config()
        resolved_device = resolve_device(device, enumerate_devices(), policy='balance')
        
        # Run detection
        if video_path:
            detect_faces_in_video(
                video_path,
                model_path,
                str(output_path),
                providers=resolved_device.onnx_providers,
                skip_frames=skip_frames
            )
        else:
            detect_faces_in_frames(
                frames_dir,
                model_path,
                str(output_path),
                providers=resolved_device.onnx_providers
            )
        
        return f"✅ Face detection completed, saved to {output_path}", str(output_path)
    
    except Exception as e:
        return f"❌ Error: {e}", ""


def add_job_to_queue(job_data: Dict[str, Any]) -> str:
    """Add a job to the batch queue."""
    try:
        with queue_lock:
            job_data['id'] = len(job_queue) + 1
            job_data['status'] = 'queued'
            job_data['added_at'] = datetime.now().isoformat()
            job_queue.append(job_data)
        
        return f"✅ Job {job_data['id']} added to queue"
    except Exception as e:
        return f"❌ Error adding job: {e}"


def get_queue_dataframe():
    """Get current queue as pandas DataFrame."""
    if not PANDAS_AVAILABLE:
        # Return empty dict if pandas not available
        return {}
    
    with queue_lock:
        if not job_queue:
            return pd.DataFrame(columns=['ID', 'Tool', 'Status', 'Device', 'Input'])
        
        data = []
        for job in job_queue:
            data.append({
                'ID': job.get('id', '?'),
                'Tool': job.get('tool', '?'),
                'Status': job.get('status', 'queued'),
                'Device': job.get('device', 'auto'),
                'Input': Path(job.get('input', '')).name if job.get('input') else '?'
            })
        
        return pd.DataFrame(data)


def delete_job_from_queue(job_id: int):
    """Delete a job from the queue."""
    try:
        with queue_lock:
            global job_queue
            # Create new list atomically
            new_queue = [job for job in job_queue if job.get('id') != job_id]
            job_queue = new_queue
        
        return get_queue_dataframe(), f"✅ Job {job_id} deleted"
    except Exception as e:
        return get_queue_dataframe(), f"❌ Error: {e}"


def clear_queue():
    """Clear all jobs from the queue."""
    try:
        with queue_lock:
            global job_queue
            count = len(job_queue)
            # Atomic replacement
            job_queue = []
        
        return get_queue_dataframe(), f"✅ Cleared {count} jobs from queue"
    except Exception as e:
        return get_queue_dataframe(), f"❌ Error: {e}"


def run_queue() -> str:
    """Execute all jobs in the queue.
    
    NOTE: This is a simplified implementation for demonstration.
    Full implementation would integrate with actual tool execution.
    """
    try:
        with queue_lock:
            if not job_queue:
                return "❌ Queue is empty"
            
            if execution_state["running"]:
                return "❌ Queue is already running"
        
        execution_state["running"] = True
        execution_state["paused"] = False
        
        results = []
        for job in job_queue:
            if execution_state["paused"]:
                break
            
            job['status'] = 'running'
            # Execute job based on tool type
            # NOTE: Full implementation would call actual tool functions here
            tool = job.get('tool', '')
            
            try:
                # Placeholder for actual tool execution
                # TODO: Integrate with transcode, scenes, extract, faces functions
                if tool == 'transcode':
                    # Would call transcode_video() here
                    pass
                elif tool == 'scenes':
                    # Would call detect_scenes_ui() here
                    pass
                elif tool == 'extract':
                    # Would call extract_frames_ui() here
                    pass
                elif tool == 'faces':
                    # Would call detect_faces_ui() here
                    pass
                else:
                    raise ValueError(f"Unknown tool: {tool}")
                
                job['status'] = 'completed'
                results.append(f"✅ Job {job['id']} completed")
            except Exception as e:
                job['status'] = 'failed'
                results.append(f"❌ Job {job['id']} failed: {e}")
        
        execution_state["running"] = False
        
        return "\n".join(results) if results else "⚠️ No jobs executed"
    
    except Exception as e:
        execution_state["running"] = False
        return f"❌ Queue execution failed: {e}"


def pause_queue() -> str:
    """Pause queue execution."""
    execution_state["paused"] = True
    return "⏸️ Queue paused"


def resume_queue() -> str:
    """Resume queue execution."""
    execution_state["paused"] = False
    return "▶️ Queue resumed"


def create_video_tools_tab() -> Any:
    """Create Tab 1: Video Tools Interface."""
    with gr.Tab("Video Tools") as tab:
        gr.Markdown("## Individual Video Processing Tools")
        
        with gr.Tabs():
            # Probe Tab
            with gr.Tab("Probe"):
                with gr.Row():
                    with gr.Column():
                        probe_input = gr.File(label="Input Video", file_types=["video"])
                        probe_btn = gr.Button("Probe Video", variant="primary")
                    
                    with gr.Column():
                        probe_output_json = gr.Textbox(label="Metadata (JSON)", lines=10)
                        probe_output_info = gr.Textbox(label="Quick Info", lines=5)
                
                probe_btn.click(
                    probe_video,
                    inputs=[probe_input],
                    outputs=[probe_output_json, probe_output_info]
                )
            
            # Transcode Tab
            with gr.Tab("Transcode"):
                with gr.Row():
                    with gr.Column():
                        trans_input = gr.File(label="Input Video", file_types=["video"])
                        trans_device = gr.Dropdown(
                            choices=get_device_choices(),
                            label="Device",
                            value="auto"
                        )
                        trans_codec = gr.Radio(
                            choices=["h264", "hevc", "av1"],
                            label="Codec",
                            value="h264"
                        )
                        
                        with gr.Row():
                            trans_width = gr.Number(label="Width (optional)", precision=0)
                            trans_height = gr.Number(label="Height (optional)", precision=0)
                        
                        trans_fps = gr.Number(label="FPS (optional)")
                        trans_bitrate = gr.Textbox(label="Bitrate (e.g., 8M, optional)")
                        trans_crf = gr.Number(label="CRF (optional)", precision=0)
                        trans_preset = gr.Textbox(label="Preset (optional)")
                        trans_output_name = gr.Textbox(label="Output Filename (optional)")
                        
                        trans_btn = gr.Button("Transcode", variant="primary")
                    
                    with gr.Column():
                        trans_output_status = gr.Textbox(label="Status", lines=5)
                        trans_output_path = gr.Textbox(label="Output Path")
                        trans_download = gr.File(label="Download Output")
                
                trans_btn.click(
                    transcode_video,
                    inputs=[
                        trans_input, trans_device, trans_codec,
                        trans_width, trans_height, trans_fps,
                        trans_bitrate, trans_crf, trans_preset, trans_output_name
                    ],
                    outputs=[trans_output_status, trans_output_path]
                )
            
            # Scene Detection Tab
            with gr.Tab("Scenes"):
                with gr.Row():
                    with gr.Column():
                        scenes_input = gr.File(label="Input Video", file_types=["video"])
                        scenes_threshold = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=27,
                            label="Detection Threshold"
                        )
                        scenes_min_len = gr.Slider(
                            minimum=0.5,
                            maximum=10,
                            value=1.0,
                            label="Minimum Scene Length (seconds)"
                        )
                        scenes_output_name = gr.Textbox(label="Output Filename (optional)")
                        scenes_btn = gr.Button("Detect Scenes", variant="primary")
                    
                    with gr.Column():
                        scenes_output_status = gr.Textbox(label="Status", lines=5)
                        scenes_output_path = gr.Textbox(label="CSV Path")
                        scenes_download = gr.File(label="Download CSV")
                
                scenes_btn.click(
                    detect_scenes_ui,
                    inputs=[scenes_input, scenes_threshold, scenes_min_len, scenes_output_name],
                    outputs=[scenes_output_status, scenes_output_path]
                )
            
            # Extract Frames Tab
            with gr.Tab("Extract Frames"):
                with gr.Row():
                    with gr.Column():
                        extract_input = gr.File(label="Input Video", file_types=["video"])
                        extract_device = gr.Dropdown(
                            choices=get_device_choices(),
                            label="Device",
                            value="auto"
                        )
                        extract_fps = gr.Number(label="FPS (optional)")
                        extract_every = gr.Number(label="Every Nth frame (optional)", precision=0)
                        extract_start = gr.Number(label="Start time (seconds, optional)")
                        extract_end = gr.Number(label="End time (seconds, optional)")
                        extract_suffix = gr.Radio(
                            choices=["jpg", "png"],
                            label="Output Format",
                            value="jpg"
                        )
                        extract_btn = gr.Button("Extract Frames", variant="primary")
                    
                    with gr.Column():
                        extract_output_status = gr.Textbox(label="Status", lines=5)
                        extract_output_path = gr.Textbox(label="Output Directory")
                
                extract_btn.click(
                    extract_frames_ui,
                    inputs=[
                        extract_input, extract_device, extract_fps,
                        extract_every, extract_start, extract_end, extract_suffix
                    ],
                    outputs=[extract_output_status, extract_output_path]
                )
            
            # Face Detection Tab
            with gr.Tab("Faces"):
                with gr.Row():
                    with gr.Column():
                        faces_video = gr.File(label="Input Video (optional)", file_types=["video"])
                        faces_frames = gr.Textbox(label="OR Frames Directory Path (optional)")
                        faces_device = gr.Dropdown(
                            choices=get_device_choices(),
                            label="Device",
                            value="auto"
                        )
                        faces_skip = gr.Number(
                            label="Skip Frames (process every Nth)",
                            value=30,
                            precision=0
                        )
                        faces_model = gr.Textbox(label="Model Path (optional)")
                        faces_btn = gr.Button("Detect Faces", variant="primary")
                    
                    with gr.Column():
                        faces_output_status = gr.Textbox(label="Status", lines=5)
                        faces_output_path = gr.Textbox(label="CSV Path")
                        faces_download = gr.File(label="Download CSV")
                
                faces_btn.click(
                    detect_faces_ui,
                    inputs=[faces_video, faces_frames, faces_device, faces_skip, faces_model],
                    outputs=[faces_output_status, faces_output_path]
                )
    
    return tab


def create_batch_queue_tab() -> Any:
    """Create Tab 2: Batch Queue Manager."""
    with gr.Tab("Batch Queue") as tab:
        gr.Markdown("## Batch Job Queue Management")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Add Jobs")
                
                batch_csv_upload = gr.File(label="Upload Batch CSV", file_types=[".csv"])
                gr.Markdown("OR manually add job:")
                
                manual_tool = gr.Dropdown(
                    choices=["transcode", "scenes", "extract", "faces"],
                    label="Tool",
                    value="transcode"
                )
                manual_input = gr.File(label="Input File")
                manual_device = gr.Dropdown(
                    choices=get_device_choices(),
                    label="Device",
                    value="auto"
                )
                manual_add_btn = gr.Button("Add Job", variant="secondary")
                
                add_status = gr.Textbox(label="Status", lines=2)
            
            with gr.Column(scale=2):
                gr.Markdown("### Queue Status")
                
                queue_table = gr.Dataframe(
                    value=get_queue_dataframe(),
                    label="Job Queue",
                    interactive=False
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", variant="secondary")
                    run_btn = gr.Button("Run Queue", variant="primary")
                    pause_btn = gr.Button("Pause", variant="secondary")
                    resume_btn = gr.Button("Resume", variant="secondary")
                    clear_btn = gr.Button("Clear All", variant="stop")
                
                queue_status = gr.Textbox(label="Execution Status", lines=3)
                
                with gr.Row():
                    delete_job_id = gr.Number(label="Job ID to Delete", precision=0)
                    delete_btn = gr.Button("Delete Job", variant="stop")
        
        # Wire up callbacks
        refresh_btn.click(
            lambda: get_queue_dataframe(),
            outputs=[queue_table]
        )
        
        run_btn.click(
            run_queue,
            outputs=[queue_status]
        )
        
        pause_btn.click(
            pause_queue,
            outputs=[queue_status]
        )
        
        resume_btn.click(
            resume_queue,
            outputs=[queue_status]
        )
        
        clear_btn.click(
            clear_queue,
            outputs=[queue_table, queue_status]
        )
        
        delete_btn.click(
            delete_job_from_queue,
            inputs=[delete_job_id],
            outputs=[queue_table, queue_status]
        )
    
    return tab


def create_config_tab() -> Any:
    """Create Tab 3: Configuration Editor."""
    with gr.Tab("Configuration") as tab:
        gr.Markdown("## Configuration Editor")
        gr.Markdown("Edit `tools_preferences.yml` configuration")
        
        with gr.Row():
            with gr.Column():
                config_path_input = gr.Textbox(
                    label="Config File Path (leave empty for default)",
                    placeholder="~/.gpu_video_tools/config.toml"
                )
                
                load_btn = gr.Button("Load Config", variant="secondary")
                
                config_editor = gr.Code(
                    language="yaml",
                    label="Configuration Content",
                    lines=30,
                    value=load_config_file()
                )
                
                with gr.Row():
                    save_btn = gr.Button("Save Config", variant="primary")
                    reset_btn = gr.Button("Reset to Default", variant="secondary")
                
                config_status = gr.Textbox(label="Status", lines=2)
        
        # Wire up callbacks
        load_btn.click(
            load_config_file,
            inputs=[config_path_input],
            outputs=[config_editor]
        )
        
        save_btn.click(
            save_config_file,
            inputs=[config_editor, config_path_input],
            outputs=[config_status]
        )
        
        reset_btn.click(
            lambda: load_config_file(),
            outputs=[config_editor]
        )
    
    return tab


def create_monitoring_tab() -> Any:
    """Create Tab 4: Monitoring Dashboard."""
    with gr.Tab("Monitoring") as tab:
        gr.Markdown("## System Monitoring & Logs")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Device Information")
                
                device_refresh_btn = gr.Button("Refresh Devices", variant="secondary")
                device_info = gr.Dataframe(label="Available Devices")
                
                def get_device_info():
                    try:
                        devices = enumerate_devices()
                        data = []
                        for dev in devices:
                            data.append({
                                'Vendor': dev.vendor,
                                'Index': dev.index if dev.index is not None else '-',
                                'Name': dev.display_name,
                                'Spec': dev.device_spec
                            })
                        return pd.DataFrame(data)
                    except Exception as e:
                        return pd.DataFrame([{'Error': str(e)}])
                
                device_refresh_btn.click(
                    get_device_info,
                    outputs=[device_info]
                )
                
                # Initial load
                device_info.value = get_device_info()
            
            with gr.Column():
                gr.Markdown("### Recent Activity")
                
                activity_log = gr.Textbox(
                    label="Activity Log",
                    lines=20,
                    value="No recent activity",
                    interactive=False
                )
                
                log_refresh_btn = gr.Button("Refresh Logs", variant="secondary")
                
                def get_activity_log():
                    # Placeholder - would read from actual log file
                    return "Activity log functionality coming soon..."
                
                log_refresh_btn.click(
                    get_activity_log,
                    outputs=[activity_log]
                )
        
        with gr.Row():
            gr.Markdown("### Benchmark History")
            benchmark_table = gr.Dataframe(
                label="Recent Benchmarks",
                value=pd.DataFrame(columns=['Tool', 'Device', 'Duration', 'FPS', 'Success'])
            )
    
    return tab


def create_interface() -> Any:
    """Create the complete Gradio interface."""
    with gr.Blocks(
        title="GPU Video Tools",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """
    ) as interface:
        gr.Markdown(
            """
            # 🎬 GPU Video Tools
            ### Hardware-accelerated video processing toolkit
            
            Process videos with GPU/CPU acceleration, detect scenes and faces, manage batch jobs, and monitor performance.
            """
        )
        
        # Create all tabs
        create_video_tools_tab()
        create_batch_queue_tab()
        create_config_tab()
        create_monitoring_tab()
        
        gr.Markdown(
            """
            ---
            **GPU Video Tools** | Windows-first, GPU/CPU-aware video processing
            """
        )
    
    return interface


def launch(
    server_name: str = "localhost",
    server_port: int = 7860,
    share: bool = False,
    auth: Optional[Tuple[str, str]] = None
) -> None:
    """Launch the Gradio web interface.
    
    Raises:
        ImportError: If required dependencies (gradio, pandas, plotly) are not installed
        Exception: If interface launch fails
    """
    # Check dependencies
    missing = check_dependencies()
    if missing:
        error_msg = f"Missing required dependencies: {', '.join(missing)}\nInstall with: pip install {' '.join(missing)}"
        raise ImportError(error_msg)
    
    try:
        # Load config for settings
        config_path = get_default_config_path()
        if config_path.exists():
            config = Config(config_path)
            # Could load Gradio settings from config here
        
        interface = create_interface()
        
        print(f"🚀 Launching GPU Video Tools web interface...")
        print(f"   URL: http://{server_name}:{server_port}")
        
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            auth=auth,
            show_error=True,
            quiet=False
        )
    
    except ImportError:
        raise  # Re-raise ImportError
    except Exception as e:
        raise RuntimeError(f"Failed to launch interface: {e}") from e


if __name__ == "__main__":
    try:
        launch()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
