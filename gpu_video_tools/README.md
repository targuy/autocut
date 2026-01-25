# GPU Video Tools

Windows-first, GPU/CPU-aware Python CLI toolset for video processing with explicit device control and smart defaults.

## Features

- **Explicit Device Control**: Pin each job to a specific device (NVIDIA GPU, AMD GPU, or CPU)
- **Hardware Analysis**: Auto-propose optimized defaults by analyzing local hardware
- **Device Precedence**: CLI → Batch CSV → Config file (with required device specification)
- **Batch Processing**: Run multiple jobs concurrently with per-device limits
- **Comprehensive Tools**: Probe, transcode, scene detection, frame extraction, face detection, and more
- **Colorful Progress**: Rich TUI with progress bars and statistics
- **Benchmarking**: CSV logging of performance metrics

## Installation

### Prerequisites

- Python 3.10+ (3.11 recommended)
- FFmpeg installed and in PATH
- For GPU support:
  - NVIDIA: CUDA drivers and toolkit
  - AMD: Latest drivers with AMF support

### Using Poetry (Recommended)

```bash
# Install dependencies
poetry install

# Activate environment
poetry shell

# Verify installation
gpu-tools --help
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### 1. Analyze Your Hardware

```bash
# Enumerate devices and create optimized config
gpu-tools analyze --write-defaults

# Quick analysis (no benchmarks)
gpu-tools analyze --quick --write-defaults

# Full analysis with benchmarks
gpu-tools analyze --full --write-defaults
```

This creates `~/.gpu_video_tools/config.toml` (or `%USERPROFILE%\.gpu_video_tools\config.toml` on Windows) with optimized defaults.

### 2. Use the Tools

All tools require a device specification via one of:
1. CLI argument: `--device <spec>`
2. Batch CSV column
3. Config file `[defaults]` section

If no device is specified in any source, the tool will **error with instructions**.

## Device Specifications

Device strings follow these formats:

- `nvidia:0`, `nvidia:1` - NVIDIA GPU by index
- `amd:0`, `amd:1` - AMD GPU by index  
- `cpu` - CPU encoding/processing
- `auto` - Automatic selection based on policy

### Device Selection Policy

When using `auto`, the `--policy` flag controls selection:

- `balance` (default): Prefer any GPU over CPU
- `prefer-nvidia`: Prefer NVIDIA GPUs
- `prefer-amd`: Prefer AMD GPUs

## Tools

### probe

Display video metadata:

```bash
gpu-tools probe input.mp4
gpu-tools probe input.mp4 --json
```

### transcode

Transcode video with hardware acceleration:

```bash
# Using device from config
gpu-tools transcode input.mp4 -o output.mp4 --vcodec h264 --width 1920 --height 1080

# Explicit device
gpu-tools --device nvidia:0 transcode input.mp4 -o output.mp4 --vcodec hevc --crf 23

# CPU encoding
gpu-tools --device cpu transcode input.mp4 -o output.mp4 --vcodec h264 --bitrate 8M
```

Supported codecs:
- `h264` - H.264/AVC (NVENC/AMF/libx264)
- `hevc` - H.265/HEVC (NVENC/AMF/libx265)
- `av1` - AV1 (NVENC/AMF/libaom-av1)

### scenes

Detect scene boundaries:

```bash
gpu-tools scenes input.mp4 -o scenes.csv --threshold 27 --min-scene-len 1.0
```

### extract-frames

Extract frames from video:

```bash
# Extract at specific FPS
gpu-tools --device nvidia:0 extract-frames input.mp4 -o frames/ --fps 1

# Extract every Nth frame
gpu-tools extract-frames input.mp4 -o frames/ --every 30

# Extract with time range
gpu-tools extract-frames input.mp4 -o frames/ --start 10 --end 60 --suffix png
```

### faces

Detect faces using YuNet ONNX model:

```bash
# From video
gpu-tools --device nvidia:0 faces --video input.mp4 -o detections.csv --skip 30

# From frames directory
gpu-tools faces --frames-dir frames/ -o detections.csv

# With drawn detections
gpu-tools faces --video input.mp4 -o detections.csv --draw output_frames/
```

**Note**: Download the YuNet model first:

```bash
mkdir -p ~/.gpu_video_tools/models
wget https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -O ~/.gpu_video_tools/models/face_detection_yunet_2023mar.onnx
```

### analyze

Analyze hardware and create configuration:

```bash
# Quick analysis
gpu-tools analyze --quick --write-defaults

# Full benchmarking
gpu-tools analyze --full --write-defaults

# Custom config location
gpu-tools analyze --write-defaults /path/to/config.toml
```

### run-batch

Process batch jobs from CSV:

```bash
gpu-tools run-batch --csv jobs.csv --bench-csv benchmarks.csv
```

Batch CSV format:

```csv
tool,device,input,output,width,height,vcodec
transcode,nvidia:0,input1.mp4,output1.mp4,1920,1080,h264
transcode,amd:0,input2.mp4,output2.mp4,1280,720,hevc
scenes,cpu,input3.mp4,scenes.csv,,
```

## Configuration File

Default location: `~/.gpu_video_tools/config.toml` (or `%USERPROFILE%\.gpu_video_tools\config.toml`)

Example structure:

```toml
[defaults]
probe = "cpu"
scenes = "cpu"
faces = "nvidia:0"
extract = "nvidia:0"
transcode = "nvidia:0"
cutlist = "cpu"

[device_limits]
"nvidia:0" = 1
"amd:0" = 1
"cpu" = 2

[codecs]
h264 = ["nvidia:0", "amd:0", "cpu"]
hevc = ["nvidia:0", "amd:0", "cpu"]
av1 = ["nvidia:0", "cpu"]
```

### Configuration Precedence

Device resolution follows strict precedence:

1. **CLI**: `--device` flag (highest priority)
2. **Batch CSV**: `device` column
3. **Config**: `[defaults].<tool>` setting (lowest priority)

If no device is specified in any source, the tool **exits with an error** and helpful message.

## Global Options

All commands support these global options:

```bash
gpu-tools [OPTIONS] COMMAND [ARGS]

Options:
  --device TEXT                   Device to use (nvidia:0, amd:0, cpu, auto)
  --config PATH                   Config file path
  --bench-csv PATH                Benchmark CSV output path
  --max-procs-per-device INTEGER  Max concurrent jobs per device (default: 1)
  --no-color                      Disable colored output
  --log-level [DEBUG|INFO|WARNING|ERROR]
  --policy [balance|prefer-nvidia|prefer-amd]
  --version                       Show version
  --help                          Show help
```

## Benchmarking

Use `--bench-csv` to log performance metrics:

```bash
gpu-tools --bench-csv bench.csv transcode input.mp4 -o output.mp4
```

CSV schema:
```
ts_start, ts_end, tool, cmdline, input_path, output_path, duration_s,
frames_in, frames_out, avg_fps, bytes_in, bytes_out, vendor, gpu_name,
gpu_index, driver, hw_backend, success, notes
```

## FFmpeg Backends

### NVIDIA (CUDA/NVENC)

- Decode: `-hwaccel cuda -hwaccel_device IDX`
- Encode: `h264_nvenc`, `hevc_nvenc`, `av1_nvenc`
- Filters: `scale_cuda`

### AMD (D3D11VA/AMF) - Windows

- Decode: `-init_hw_device d3d11va=adX,adapter=IDX -hwaccel d3d11va`
- Encode: `h264_amf`, `hevc_amf`, `av1_amf`
- Filters: Software scaling

### CPU

- Encode: `libx264`, `libx265`, `libaom-av1`
- No hardware acceleration

## Troubleshooting

### "No device specified" Error

Run one of:
```bash
gpu-tools analyze --write-defaults
gpu-tools <command> --device cpu ...
```

### NVENC Not Available

Check FFmpeg build:
```bash
ffmpeg -hide_banner -encoders | grep nvenc
```

Rebuild FFmpeg with `--enable-nvenc` if missing.

### AMD AMF Not Available

- Ensure latest AMD drivers installed
- Check FFmpeg build includes AMF support
- Windows only (D3D11VA required)

### YuNet Model Not Found

Download the model:
```bash
mkdir -p ~/.gpu_video_tools/models
wget https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -O ~/.gpu_video_tools/models/face_detection_yunet_2023mar.onnx
```

### ONNX Runtime Provider Errors

Install appropriate ONNX Runtime:

```bash
# For NVIDIA CUDA support
pip install onnxruntime-gpu

# For DirectML (Windows AMD/Intel)
pip install onnxruntime-directml

# CPU only
pip install onnxruntime
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
black gpu_video_tools/
flake8 gpu_video_tools/
```

## Examples

### Single Job with Explicit Device

```bash
gpu-tools --device nvidia:0 transcode input.mp4 -o output.mp4 \
  --vcodec hevc --width 1920 --height 1080 --crf 23
```

### Batch Processing

Create `jobs.csv`:
```csv
tool,device,input,output,vcodec,width,height
transcode,nvidia:0,video1.mp4,out1.mp4,h264,1920,1080
transcode,nvidia:0,video2.mp4,out2.mp4,h264,1920,1080
transcode,amd:0,video3.mp4,out3.mp4,hevc,1280,720
scenes,cpu,video4.mp4,scenes.csv,,
```

Run:
```bash
gpu-tools run-batch --csv jobs.csv --bench-csv results.csv
```

### Pipeline: Extract → Detect Faces

```bash
# Extract frames
gpu-tools --device nvidia:0 extract-frames input.mp4 -o frames/ --fps 1

# Detect faces
gpu-tools --device nvidia:0 faces --frames-dir frames/ -o detections.csv --draw annotated/
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: https://github.com/targuy/autocut/issues
- Documentation: See this README and inline help (`gpu-tools <command> --help`)
