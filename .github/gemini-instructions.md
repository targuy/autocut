# Gemini AI Instructions for AutoCutVideo

## Project Overview

**AutoCutVideo** is a dual-purpose video processing platform:

1. **AutoCut Pipeline**: Automated video clip generation based on person detection, face visibility, and skin segmentation
2. **GPU Video Tools**: Professional video processing toolkit with hardware acceleration and web interface

**Key Characteristics:**
- Multi-platform support (Windows, Linux, macOS)
- CPU-first design with optional GPU acceleration
- Flexible configuration system
- Both CLI and web UI interfaces
- Comprehensive test coverage
- Production-ready error handling

## Technology Ecosystem

### Language & Environment
- **Python 3.10+** with type hints
- **Poetry** for dependency management (primary)
- **Conda** environment support (alternative)
- **Virtual environments** for isolation

### Core Technologies

**Video Processing:**
- OpenCV for computer vision
- FFmpeg for encoding/decoding
- PySceneDetect for scene analysis
- Ultralytics YOLOv8 for object detection

**Deep Learning:**
- PyTorch for neural networks
- ONNX Runtime for inference
- Hugging Face Transformers for models
- MediaPipe for pose estimation

**Web & CLI:**
- Gradio for web interface
- Click for command-line interface
- Rich for terminal UI
- Plotly for visualizations

**Testing & Quality:**
- pytest for unit/integration tests
- flake8 for linting
- black for code formatting
- mock for test isolation

## Code Quality Standards

### Python Style Guide

```python
# ✅ GOOD: Type hints, docstrings, clear naming
from typing import List, Dict, Any, Optional
from pathlib import Path

def process_video_frames(
    video_path: Path,
    device: str = "cpu",
    skip_frames: int = 1
) -> List[Dict[str, Any]]:
    """Process video frames with device acceleration.
    
    Args:
        video_path: Path to input video file
        device: Processing device (cpu, nvidia:0, amd:0)
        skip_frames: Process every Nth frame
        
    Returns:
        List of frame analysis results
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        DeviceNotFoundError: If specified device unavailable
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    # Implementation...

# ❌ BAD: No types, no docs, unclear
def process(v, d="cpu", s=1):
    if not v.exists(): raise Exception("not found")
    # Implementation...
```

### Configuration Pattern

```python
# ✅ GOOD: Dataclass with validation
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    
    device: str = "cpu"
    max_concurrent: int = 1
    timeout: int = 300
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if self.timeout < 1:
            raise ValueError("timeout must be >= 1")

# ❌ BAD: Dict with no validation
config = {
    "device": "cpu",
    "max_concurrent": 1,
    "timeout": 300
}
```

### Error Handling Pattern

```python
# ✅ GOOD: Specific exceptions with context
class VideoProcessingError(Exception):
    """Base exception for video processing."""
    pass

class InvalidDeviceError(VideoProcessingError):
    """Raised when device specification is invalid."""
    
    def __init__(self, device: str, available: List[str]):
        self.device = device
        self.available = available
        message = (
            f"Device '{device}' is not available.\n"
            f"Available devices: {', '.join(available)}\n"
            f"Use --device cpu or --device auto for automatic selection."
        )
        super().__init__(message)

# ❌ BAD: Generic exception
raise Exception("Device not found")
```

## Architecture Patterns

### Device Management Architecture

```
┌─────────────────────────────────────┐
│     Device Specification Layer      │
│  (nvidia:0, amd:0, cpu, auto)      │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│    Device Resolution & Validation   │
│  - enumerate_devices()              │
│  - resolve_device(spec, policy)     │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│      Hardware Abstraction Layer     │
│  - FFmpeg args for each device      │
│  - ONNX providers for each device   │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│        Tool Execution Layer         │
│  - transcode, probe, scenes, etc.   │
└─────────────────────────────────────┘
```

### Configuration Precedence

```
┌──────────────┐
│  CLI Args    │  Highest priority
└──────┬───────┘
       │
       ↓
┌──────────────┐
│  Batch CSV   │  Medium priority
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Config File  │  Lowest priority
└──────────────┘

Example:
  gpu-tools --device nvidia:0 transcode input.mp4 -o output.mp4
            ^^^^^^^^^^^^^^^^^
            Overrides batch CSV and config file
```

### Web Interface Architecture (Gradio)

```
┌─────────────────────────────────────────────┐
│          Gradio Web Interface               │
├─────────────────────────────────────────────┤
│                                             │
│  Tab 1: Video Tools                         │
│  ├── Probe       (metadata inspection)      │
│  ├── Transcode   (video encoding)           │
│  ├── Scenes      (boundary detection)       │
│  ├── Extract     (frame extraction)         │
│  └── Faces       (face detection)           │
│                                             │
│  Tab 2: Batch Queue Manager                 │
│  ├── Job Creation (CSV/manual)              │
│  ├── Queue Management (CRUD ops)            │
│  └── Execution Control (start/pause/stop)   │
│                                             │
│  Tab 3: Configuration Editor                │
│  ├── YAML Editor (live editing)             │
│  ├── Device Discovery                       │
│  └── Profile Management                     │
│                                             │
│  Tab 4: Monitoring Dashboard                │
│  ├── Device Information                     │
│  ├── Activity Logs                          │
│  └── Benchmark History                      │
│                                             │
└─────────────────────────────────────────────┘
```

## Development Environment Setup

### Local Development

```bash
# Clone repository
git clone https://github.com/targuy/autocut.git
cd autocut

# Option 1: Poetry (Recommended)
poetry install
poetry install -E all  # All optional features
poetry shell

# Option 2: Conda
conda env create -f environment.yml
conda activate autocut
pip install -e .

# Option 3: venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -e .

# Verify installation
gpu-tools --version
gpu-tools analyze --quick
```

### GitHub Codespaces

The project is pre-configured for GitHub Codespaces with `.devcontainer/devcontainer.json`:

- Automatic environment setup
- Pre-installed dependencies
- Configured VS Code extensions
- Port forwarding for web interface
- Git integration

**To use:**
1. Open repository in GitHub
2. Click "Code" → "Codespaces" → "Create codespace"
3. Wait for environment setup (2-3 minutes)
4. Start developing immediately

## Testing Philosophy

### Test Structure

```
tests/
├── gpu_video_tools/           # GPU Tools tests
│   ├── test_gpu.py           # Device management
│   ├── test_config.py        # Configuration
│   ├── test_ffmpeg_ops.py    # FFmpeg operations
│   ├── test_scenes.py        # Scene detection
│   ├── test_faces.py         # Face detection
│   ├── test_bench.py         # Benchmarking
│   ├── test_scheduler.py     # Batch processing
│   ├── test_tui.py           # Terminal UI
│   └── test_gradio_app.py    # Web interface
└── (other test modules)
```

### Testing Best Practices

**1. Mock External Dependencies**
```python
import pytest
from unittest.mock import Mock, patch, MagicMock

@patch('gpu_video_tools.gpu.pynvml')
def test_nvidia_detection(mock_pynvml):
    """Test NVIDIA GPU detection without actual hardware."""
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    # Test logic...
```

**2. Parametrize Test Cases**
```python
@pytest.mark.parametrize("device,expected", [
    ("cpu", "cpu"),
    ("nvidia:0", "nvidia:0"),
    ("amd:0", "amd:0"),
    ("auto", "cpu"),  # When no GPU available
])
def test_device_resolution(device, expected):
    """Test device resolution for various specs."""
    result = resolve_device(device, policy='balance')
    assert result.device_spec == expected
```

**3. Test Error Conditions**
```python
def test_invalid_device_raises_error():
    """Test that invalid device spec raises appropriate error."""
    with pytest.raises(DeviceNotFoundError) as exc_info:
        resolve_device("invalid:999", [])
    
    assert "invalid:999" in str(exc_info.value)
    assert "Available devices" in str(exc_info.value)
```

**4. Fixture for Common Setup**
```python
@pytest.fixture
def temp_video_file():
    """Create temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
    
    yield video_path
    
    # Cleanup
    Path(video_path).unlink(missing_ok=True)
```

## Security Guidelines

### Input Validation

```python
# ✅ GOOD: Validate all inputs
def process_file(file_path: str) -> None:
    path = Path(file_path).resolve()
    
    # Check file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Check file type
    if path.suffix.lower() not in ['.mp4', '.avi', '.mkv', '.mov']:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    # Check file size (e.g., max 10GB)
    if path.stat().st_size > 10 * 1024 * 1024 * 1024:
        raise ValueError(f"File too large: {path.stat().st_size / 1e9:.2f} GB")

# ❌ BAD: No validation
def process_file(file_path):
    # Direct use without checks
    pass
```

### Subprocess Safety

```python
# ✅ GOOD: List arguments, no shell injection
cmd = [
    'ffmpeg',
    '-i', str(input_path),
    '-c:v', codec,
    '-b:v', bitrate,
    str(output_path)
]
subprocess.run(cmd, capture_output=True, timeout=300)

# ❌ BAD: String concatenation with shell=True
cmd = f"ffmpeg -i {input_path} -c:v {codec} {output_path}"
subprocess.run(cmd, shell=True)  # VULNERABLE TO INJECTION
```

### Secrets Management

```python
# ✅ GOOD: Environment variables
import os

api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API_KEY environment variable not set")

# ❌ BAD: Hardcoded secrets
api_key = "sk-1234567890abcdef"  # NEVER DO THIS
```

## Performance Optimization

### GPU Acceleration Strategy

```python
# Priority order for video transcoding
1. NVIDIA NVENC (if available)
   - Fastest encoding
   - Good quality
   - Low CPU usage

2. AMD AMF (if available, Windows only)
   - Fast encoding
   - Good quality
   - Medium CPU usage

3. CPU software encoder
   - Slowest but always available
   - Best quality control
   - High CPU usage
```

### Memory Management

```python
# ✅ GOOD: Process in chunks
def process_large_video(video_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            process_frame(frame)
            
            # Release frame memory
            del frame
    finally:
        cap.release()

# ❌ BAD: Load everything in memory
def process_large_video(video_path):
    frames = load_all_frames(video_path)  # Out of memory!
    for frame in frames:
        process_frame(frame)
```

### Concurrent Processing

```python
# ✅ GOOD: Respect device limits
from asyncio import Semaphore

device_limits = {
    'nvidia:0': Semaphore(1),  # 1 job at a time
    'cpu': Semaphore(2),        # 2 jobs at a time
}

async def process_job(job, device):
    async with device_limits[device]:
        # Process job exclusively
        await execute_job(job, device)
```

## Documentation Standards

### Module Documentation

```python
"""Video transcoding operations with hardware acceleration.

This module provides functions for transcoding videos using various
hardware accelerators (NVIDIA NVENC, AMD AMF) or CPU software encoders.

Example:
    >>> from gpu_video_tools.ffmpeg_ops import build_transcode_command
    >>> from gpu_video_tools.gpu import resolve_device
    >>> 
    >>> device = resolve_device('nvidia:0', policy='balance')
    >>> cmd = build_transcode_command(
    ...     'input.mp4',
    ...     'output.mp4',
    ...     device,
    ...     'h264',
    ...     width=1920,
    ...     height=1080
    ... )
    >>> # Execute command with subprocess.run(cmd)

Supported encoders:
    - NVIDIA: h264_nvenc, hevc_nvenc, av1_nvenc
    - AMD: h264_amf, hevc_amf, av1_amf
    - CPU: libx264, libx265, libaom-av1
"""
```

### Function Documentation

```python
def detect_faces_in_video(
    video_path: str,
    model_path: str,
    output_csv: str,
    providers: List[str] = None,
    skip_frames: int = 1,
    draw_output_dir: Optional[str] = None
) -> None:
    """Detect faces in video and save results to CSV.
    
    Args:
        video_path: Path to input video file
        model_path: Path to YuNet ONNX model
        output_csv: Path for output CSV file
        providers: ONNX Runtime providers (default: ['CPUExecutionProvider'])
        skip_frames: Process every Nth frame (default: 1)
        draw_output_dir: Optional directory for visualization output
        
    Returns:
        None. Results are written to output_csv.
        
    Raises:
        FileNotFoundError: If video or model file not found
        ImportError: If opencv-python or onnxruntime not installed
        
    Example:
        >>> detect_faces_in_video(
        ...     'video.mp4',
        ...     'yunet_model.onnx',
        ...     'faces.csv',
        ...     providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        ...     skip_frames=30
        ... )
        
    Note:
        Requires YuNet model from OpenCV Zoo:
        https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
    """
```

## Gemini-Specific Instructions

When assisting with this codebase:

### Analysis Phase
1. **Understand context** - Read relevant code before suggesting changes
2. **Check patterns** - Follow existing architectural patterns
3. **Review tests** - Understand test coverage and patterns
4. **Consider impact** - Think about backward compatibility

### Implementation Phase
1. **Write tests first** - Test-driven development preferred
2. **Make minimal changes** - Smallest change to solve problem
3. **Follow conventions** - Match existing code style
4. **Add documentation** - Update docs with code changes

### Validation Phase
1. **Run tests** - Ensure all tests pass
2. **Check linting** - flake8 compliance
3. **Format code** - black formatting
4. **Update docs** - Keep documentation in sync

### Communication Phase
1. **Clear explanations** - Explain rationale for changes
2. **Provide examples** - Show usage of new features
3. **Document tradeoffs** - Explain design decisions
4. **Link references** - Reference relevant documentation

## Common Development Tasks

### Adding a New CLI Command

```python
# 1. Add command to __main__.py
@main.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('-o', '--output', required=True)
@click.pass_context
def my_new_command(ctx, input, output):
    """My new command description."""
    console = ctx.obj['console']
    
    try:
        # Implementation
        result = do_something(input, output)
        print_success_panel(f"Success: {result}", console)
    except Exception as e:
        print_error_panel(str(e), console)
        sys.exit(1)

# 2. Add implementation function
def do_something(input_path: str, output_path: str) -> str:
    """Implement the actual logic."""
    pass

# 3. Add tests
def test_my_new_command():
    """Test the new command."""
    pass

# 4. Update documentation
# Add to README.md and USAGE.md
```

### Adding Web Interface Feature

```python
# 1. Add UI function
def my_feature_ui(input_path: str, param: int) -> Tuple[str, str]:
    """UI wrapper for my feature."""
    try:
        if not input_path:
            return "Error: No input provided", ""
        
        result = my_feature(input_path, param)
        return f"✅ Success: {result}", str(result)
    except Exception as e:
        return f"❌ Error: {e}", ""

# 2. Add to Gradio interface
with gr.Tab("My Feature"):
    with gr.Row():
        with gr.Column():
            input_file = gr.File(label="Input")
            param_slider = gr.Slider(0, 100, value=50, label="Parameter")
            run_btn = gr.Button("Run", variant="primary")
        
        with gr.Column():
            output_status = gr.Textbox(label="Status")
            output_result = gr.Textbox(label="Result")
    
    run_btn.click(
        my_feature_ui,
        inputs=[input_file, param_slider],
        outputs=[output_status, output_result]
    )
```

## Troubleshooting Common Issues

### Import Errors
```bash
# Check Python version
python --version  # Must be 3.10+

# Check installed packages
pip list | grep -E "(opencv|torch|gradio)"

# Reinstall dependencies
poetry install --no-cache
```

### GPU Not Detected
```bash
# Check NVIDIA
nvidia-smi
python -c "import pynvml; pynvml.nvmlInit(); print('NVIDIA OK')"

# Check AMD (Windows)
python -c "import wmi; print('WMI OK')"

# Fallback to CPU
gpu-tools --device cpu analyze
```

### Test Failures
```bash
# Run specific test
pytest tests/gpu_video_tools/test_gpu.py::test_parse_device_spec -v

# Run with output
pytest tests/ -v -s

# Run without optional dependencies
pytest tests/ -v --ignore=tests/integration/
```

## Version Control Best Practices

### Commit Messages
```
feat: Add face detection to web interface
fix: Handle missing FFmpeg gracefully
docs: Update USAGE.md with batch processing examples
test: Add tests for device resolution
refactor: Simplify configuration loading
perf: Optimize frame extraction performance
```

### Branch Naming
```
feature/add-audio-processing
bugfix/fix-memory-leak
docs/update-api-reference
refactor/simplify-config
```

## Resources

- **Documentation**: README.md, USAGE.md, gpu_video_tools/README.md
- **Configuration**: pyproject.toml, tools_preferences.yml, config.yml
- **Tests**: tests/ directory for examples
- **Issues**: GitHub Issues for bugs and features
- **Discussions**: GitHub Discussions for questions

---

Last Updated: 2026-01-26
Version: 1.0.0
