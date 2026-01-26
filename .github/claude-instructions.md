# Claude AI Instructions for AutoCutVideo

## Project Context

AutoCutVideo is a comprehensive video processing toolkit with two main components:
1. **AutoCut Pipeline**: Person detection and video clip generation with YOLOv8
2. **GPU Video Tools**: Hardware-accelerated video processing with CLI and web interface

## Core Philosophy

- **Flexibility**: Support CPU-only and GPU-accelerated processing
- **Security**: No hardcoded values, all configuration externalized
- **User-Friendly**: Clear error messages, comprehensive documentation
- **No Regression**: All changes must maintain backward compatibility

## Technology Stack

### Python Environment
- **Version**: Python 3.10+
- **Package Manager**: Poetry (primary) with pip fallback
- **Virtual Environment**: Supports venv, conda, Poetry environments

### Core Dependencies
- **Video Processing**: OpenCV, FFmpeg, scenedetect
- **Deep Learning**: PyTorch, Ultralytics YOLOv8, Transformers
- **GPU Acceleration**: CUDA (NVIDIA), D3D11VA/AMF (AMD), CPU fallback
- **Web Interface**: Gradio, Plotly, Pandas
- **CLI**: Click, Rich (colorful terminal output)
- **Testing**: pytest, mock-based for CI environments

## Coding Standards

### Code Style
- **PEP 8** compliance with 120 character line limit
- **Type hints** for all function signatures
- **Docstrings** for all public APIs (Google style)
- **Error handling** with informative messages
- **Language**: Code in English, user messages in French (AutoCut) or English (GPU Tools)

### Best Practices
```python
# Good: Type hints, clear naming, docstring
def detect_faces(
    image_path: str,
    conf_threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """Detect faces in image with confidence filtering.
    
    Args:
        image_path: Path to input image
        conf_threshold: Minimum confidence score
        
    Returns:
        List of detection dicts with bbox, landmarks, confidence
    """
    pass

# Bad: No types, unclear naming, no docs
def proc(p, t=0.6):
    pass
```

### Configuration Management
- All settings in YAML/TOML files (config.yml, tools_preferences.yml)
- Environment-specific configs (dev, prod, test)
- No secrets in code - use environment variables
- Validation on load with clear error messages

## Project Structure

```
autocut/
├── .github/
│   ├── copilot-instructions.md    # GitHub Copilot directives
│   ├── claude-instructions.md     # This file
│   └── workflows/                 # CI/CD pipelines
├── .vscode/
│   ├── settings.json              # Editor settings
│   ├── launch.json                # Debug configurations
│   └── extensions.json            # Recommended extensions
├── .devcontainer/
│   └── devcontainer.json          # GitHub Codespaces config
├── cli/                           # AutoCut CLI scripts
├── pipeline/                      # AutoCut processing pipeline
├── detectors/                     # Detection modules
├── segmenters/                    # Segmentation modules
├── classifiers/                   # Classification modules
├── gpu_video_tools/              # GPU Video Tools package
│   ├── __main__.py               # CLI entry point
│   ├── gradio_app.py             # Web interface
│   ├── gpu.py                    # Device management
│   ├── config.py                 # Configuration
│   ├── README.md                 # Documentation
│   └── USAGE.md                  # Usage guide
├── tests/                        # Test suite
├── config.yml                    # AutoCut configuration
├── tools_preferences.yml         # GPU Tools configuration
├── pyproject.toml               # Poetry configuration
├── environment.yml              # Conda environment
└── README.md                    # Main documentation
```

## Development Workflow

### Setting Up Environment

**Option 1: Poetry (Recommended)**
```bash
poetry install
poetry install -E all  # All optional dependencies
poetry shell
```

**Option 2: Conda**
```bash
conda env create -f environment.yml
conda activate autocut
pip install -e .
```

**Option 3: GitHub Codespaces**
- Automatically configured via .devcontainer/devcontainer.json
- Pre-installed dependencies and extensions
- Ready-to-use development environment

### Running Tests
```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/gpu_video_tools/ -v

# With coverage
pytest --cov=gpu_video_tools --cov-report=html
```

### Code Quality
```bash
# Linting
flake8 . --max-line-length=120

# Formatting
black . --line-length=120

# Type checking (optional)
mypy gpu_video_tools/ --ignore-missing-imports
```

## GPU Video Tools Architecture

### Device Management System
```python
# Device specifications
- "nvidia:0", "nvidia:1"  # NVIDIA GPU by index
- "amd:0", "amd:1"        # AMD GPU by index
- "cpu"                    # CPU processing
- "auto"                   # Automatic selection

# Device resolution precedence
1. CLI flag: --device nvidia:0
2. Batch CSV: device column
3. Config file: defaults.transcode
4. Error if no device specified
```

### Web Interface (Gradio)

**Four Main Tabs:**
1. **Video Tools**: Individual tool interfaces (probe, transcode, scenes, extract, faces)
2. **Batch Queue**: Job management with CSV import/export
3. **Configuration Editor**: Live YAML editing
4. **Monitoring Dashboard**: Device info, logs, benchmarks

**Key Features:**
- Real-time progress tracking
- Device selection dropdown
- Output preview and download
- Queue state persistence
- Thread-safe operations

### Security Considerations

**subprocess Usage:**
```python
# Good: List args, no shell injection risk
cmd = ['ffmpeg', '-i', input_path, '-c:v', codec, output_path]
subprocess.run(cmd, capture_output=True, timeout=300)

# Bad: String with shell=True
subprocess.run(f"ffmpeg -i {input_path}", shell=True)  # NEVER DO THIS
```

**Input Validation:**
```python
# Validate file paths
if not Path(input_path).exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

# Validate device specs
if device not in enumerate_devices():
    raise DeviceNotFoundError(f"Device '{device}' not available")
```

## Common Tasks

### Adding a New Tool

1. **Create tool function** in appropriate module
2. **Add CLI command** in `__main__.py`
3. **Add web interface** in `gradio_app.py` (if applicable)
4. **Write tests** with proper mocking
5. **Update documentation** in README and USAGE.md
6. **Add configuration** defaults in tools_preferences.yml

### Handling Optional Dependencies

```python
# Pattern for optional imports
GRADIO_AVAILABLE = True
try:
    import gradio as gr
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

def launch():
    if not GRADIO_AVAILABLE:
        raise ImportError("gradio is required. Install with: pip install gradio")
    # ... rest of code
```

### Adding Configuration Options

1. **Add to config file** (config.yml or tools_preferences.yml)
2. **Update Config dataclass** in config.py
3. **Add validation** in __post_init__
4. **Document** in README and USAGE.md
5. **Add tests** for new config options

## Error Handling Best Practices

### Custom Exceptions
```python
class GPUVideoToolsError(Exception):
    """Base exception for GPU Video Tools."""
    pass

class DeviceNotFoundError(GPUVideoToolsError):
    """Raised when specified device is not available."""
    
    def __init__(self, device: str, available: List[str]):
        msg = f"Device '{device}' not found\nAvailable: {', '.join(available)}"
        super().__init__(msg)
```

### User-Friendly Messages
```python
# Good: Actionable error message
raise ValueError(
    "Model file not found at ~/.gpu_video_tools/models/yunet.onnx\n"
    "Download with:\n"
    "  wget https://example.com/yunet.onnx -O ~/.gpu_video_tools/models/yunet.onnx"
)

# Bad: Cryptic error
raise ValueError("Model not found")
```

## Testing Strategy

### Mock External Dependencies
```python
# Mock hardware detection
@patch('gpu_video_tools.gpu.pynvml')
@patch('gpu_video_tools.gpu.wmi')
def test_enumerate_devices(mock_wmi, mock_pynvml):
    # Test without actual hardware
    pass

# Mock subprocess calls
@patch('subprocess.run')
def test_transcode(mock_run):
    mock_run.return_value.returncode = 0
    # Test without running FFmpeg
    pass
```

### Test Coverage Goals
- **Unit tests**: 80%+ coverage for core logic
- **Integration tests**: Key workflows end-to-end
- **CI/CD**: All tests pass before merge
- **Mock-based**: Run without GPU/FFmpeg/models

## Documentation Requirements

### Code Documentation
- **Module docstrings**: Purpose, key classes/functions
- **Class docstrings**: Responsibility, attributes, usage example
- **Function docstrings**: Args, returns, raises, examples
- **Inline comments**: Complex logic only

### User Documentation
- **README.md**: Overview, installation, quick start
- **USAGE.md**: Detailed examples, workflows, troubleshooting
- **API docs**: Auto-generated from docstrings
- **Configuration reference**: All options with defaults

## Versioning and Releases

### Semantic Versioning
- **MAJOR.MINOR.PATCH** (e.g., 0.1.0)
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes

### Release Process
1. Update version in pyproject.toml
2. Update CHANGELOG.md
3. Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions builds and publishes

## Performance Optimization

### GPU Acceleration
- Use hardware decode: `-hwaccel cuda` (NVIDIA), `-hwaccel d3d11va` (AMD)
- Use hardware encode: `h264_nvenc`, `hevc_nvenc`, `h264_amf`, `hevc_amf`
- Batch processing with per-device concurrency limits

### Memory Management
- Process videos in chunks for large files
- Release resources promptly (close file handles, clear GPU memory)
- Use generators for large datasets

## Troubleshooting Guide

### Common Issues

**Import Errors**
- Check virtual environment is activated
- Verify dependencies installed: `pip list` or `poetry show`
- Install missing packages: `poetry install -E all`

**GPU Not Detected**
- NVIDIA: Check CUDA drivers, install pynvml
- AMD: Check drivers (Windows only), install wmi
- Fallback to CPU: Use `--device cpu`

**FFmpeg Issues**
- Verify FFmpeg in PATH: `ffmpeg -version`
- Check encoder support: `ffmpeg -encoders | grep nvenc`
- Install FFmpeg with hardware support if missing

## Claude-Specific Instructions

When assisting with this project:

1. **Always check** existing code patterns before suggesting changes
2. **Maintain** backward compatibility - no breaking changes
3. **Follow** established coding style and conventions
4. **Add tests** for any new functionality
5. **Update docs** when adding features
6. **Use** type hints and docstrings consistently
7. **Handle errors** gracefully with clear messages
8. **Mock** external dependencies in tests
9. **Validate** configuration on load
10. **Keep security** in mind - no hardcoded secrets

### When Making Changes

✅ **Do:**
- Make minimal, focused changes
- Add comprehensive tests
- Update relevant documentation
- Use existing patterns and conventions
- Handle optional dependencies gracefully
- Provide clear commit messages

❌ **Don't:**
- Break existing functionality
- Hardcode values that should be configurable
- Skip error handling
- Ignore test failures
- Leave TODOs without issues
- Make assumptions about user environment

## References

- **Main README**: Overview and installation
- **USAGE.md**: Detailed usage examples
- **pyproject.toml**: Dependencies and configuration
- **tests/**: Example test patterns
- **GitHub Issues**: Bug reports and feature requests

---

Last Updated: 2026-01-26
Version: 1.0.0
