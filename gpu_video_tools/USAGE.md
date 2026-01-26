# GPU Video Tools - Complete Usage Guide

This guide provides detailed examples and tutorials for using GPU Video Tools, both via command-line interface (CLI) and web interface (Gradio UI).

## Table of Contents

1. [Getting Started](#getting-started)
2. [Command-Line Interface (CLI)](#command-line-interface-cli)
3. [Web Interface (Gradio)](#web-interface-gradio)
4. [Configuration](#configuration)
5. [Common Workflows](#common-workflows)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First-Time Setup

After installation, run hardware analysis to create optimized configuration:

```bash
# Analyze hardware and create config file
gpu-tools analyze --write-defaults

# Output:
# Enumerating devices...
# ┌─────────┬───────┬──────────────────────────┬────────────┐
# │ Vendor  │ Index │ Name                     │ Spec       │
# ├─────────┼───────┼──────────────────────────┼────────────┤
# │ nvidia  │ 0     │ NVIDIA GeForce RTX 3080  │ nvidia:0   │
# │ cpu     │ -     │ CPU                      │ cpu        │
# └─────────┴───────┴──────────────────────────┴────────────┘
# 
# Configuration created: ~/.gpu_video_tools/config.toml
```

This creates a configuration file at:
- **Linux/macOS**: `~/.gpu_video_tools/config.toml`
- **Windows**: `%USERPROFILE%\.gpu_video_tools\config.toml`

### Quick Test

Test your installation with a simple probe:

```bash
# Probe a video file (no encoding, CPU-only)
gpu-tools probe path/to/video.mp4
```

---

## Command-Line Interface (CLI)

### 1. Video Probing

Get detailed video metadata:

```bash
# Basic probe
gpu-tools probe video.mp4

# JSON output for scripting
gpu-tools probe video.mp4 --json > metadata.json
```

**Output:**
```
Video: h264
Resolution: 1920x1080
FPS: 30/1
Audio: aac
Duration: 120.5s
Size: 50.23 MB
```

### 2. Video Transcoding

Transcode videos with hardware acceleration:

**Basic transcoding:**
```bash
# Using configured default device
gpu-tools transcode input.mp4 -o output.mp4

# Specify device explicitly
gpu-tools --device nvidia:0 transcode input.mp4 -o output.mp4

# CPU encoding
gpu-tools --device cpu transcode input.mp4 -o output.mp4
```

**Advanced options:**
```bash
# Change codec and resolution
gpu-tools transcode input.mp4 -o output.mp4 \
  --vcodec hevc \
  --width 1280 \
  --height 720

# Constant Rate Factor (CRF) for quality control
gpu-tools transcode input.mp4 -o output.mp4 \
  --vcodec h264 \
  --crf 23

# Bitrate control
gpu-tools transcode input.mp4 -o output.mp4 \
  --bitrate 8M

# Frame rate conversion
gpu-tools transcode input.mp4 -o output.mp4 \
  --fps 60
```

**Encoder presets:**
```bash
# NVIDIA NVENC presets (p1-p7)
gpu-tools --device nvidia:0 transcode input.mp4 -o output.mp4 \
  --preset p4

# CPU x264 presets
gpu-tools --device cpu transcode input.mp4 -o output.mp4 \
  --preset medium
```

### 3. Scene Detection

Detect scene boundaries:

```bash
# Basic scene detection
gpu-tools scenes input.mp4 -o scenes.csv

# Custom threshold (default: 27.0)
gpu-tools scenes input.mp4 -o scenes.csv \
  --threshold 30.0

# Minimum scene length (default: 1.0 seconds)
gpu-tools scenes input.mp4 -o scenes.csv \
  --min-scene-len 2.0
```

**Output CSV format:**
```csv
scene_number,start_time,end_time,duration
1,0.000,5.500,5.500
2,5.500,12.300,6.800
3,12.300,20.000,7.700
```

### 4. Frame Extraction

Extract frames from video:

```bash
# Extract at 1 fps
gpu-tools extract-frames input.mp4 -o frames/ --fps 1

# Extract every 30th frame
gpu-tools extract-frames input.mp4 -o frames/ --every 30

# Extract specific time range
gpu-tools extract-frames input.mp4 -o frames/ \
  --start 10.0 \
  --end 30.0 \
  --fps 2

# PNG format instead of JPEG
gpu-tools extract-frames input.mp4 -o frames/ \
  --fps 1 \
  --suffix png
```

### 5. Face Detection

Detect faces in video or frames:

**From video:**
```bash
# Process every 30th frame
gpu-tools faces --video input.mp4 -o faces.csv --skip 30

# With visualization output
gpu-tools faces --video input.mp4 -o faces.csv \
  --skip 30 \
  --draw annotated_frames/
```

**From extracted frames:**
```bash
# Process all frames in directory
gpu-tools faces --frames-dir frames/ -o faces.csv

# With visualization
gpu-tools faces --frames-dir frames/ -o faces.csv \
  --draw annotated/
```

**Custom model:**
```bash
# Specify YuNet model path
gpu-tools faces --video input.mp4 -o faces.csv \
  --model path/to/face_detection_yunet.onnx
```

**Output CSV format:**
```csv
frame_number,face_id,x,y,width,height,confidence,yaw,pitch,roll
0,1,320,240,150,180,0.95,0.0,0.0,0.0
0,2,800,200,140,170,0.92,5.2,0.0,0.0
```

### 6. Batch Processing

Process multiple videos with a CSV file:

**Create batch CSV:**
```csv
tool,device,input,output,vcodec,width,height,crf
transcode,nvidia:0,video1.mp4,out1.mp4,h264,1920,1080,23
transcode,nvidia:0,video2.mp4,out2.mp4,h264,1920,1080,23
transcode,cpu,video3.mp4,out3.mp4,h264,1280,720,25
scenes,cpu,video4.mp4,scenes4.csv,,,,
```

**Run batch:**
```bash
# Process batch with benchmarking
gpu-tools run-batch --csv jobs.csv --bench-csv results.csv

# Custom max concurrent jobs per device
gpu-tools --max-procs-per-device 2 run-batch --csv jobs.csv
```

### 7. Benchmarking

Log performance metrics for analysis:

```bash
# Enable benchmarking for any command
gpu-tools --bench-csv bench.csv transcode input.mp4 -o output.mp4

# Benchmark multiple operations
gpu-tools --bench-csv bench.csv scenes video1.mp4 -o scenes1.csv
gpu-tools --bench-csv bench.csv transcode video2.mp4 -o out2.mp4
```

**Benchmark CSV columns:**
- `ts_start`, `ts_end`: Timestamps
- `tool`: Command executed
- `duration_s`: Execution time
- `avg_fps`: Processing speed
- `bytes_in`, `bytes_out`: File sizes
- `vendor`, `gpu_name`, `gpu_index`: Device info
- `driver`: Driver version
- `success`: True/False

---

## Web Interface (Gradio)

### Launching the Web UI

**Simple launch:**
```bash
gpu-tools-ui
```

This starts the web interface at `http://localhost:7860`

**Custom configuration:**
```bash
# Different port
GRADIO_SERVER_PORT=8080 gpu-tools-ui

# Listen on all interfaces (accessible from network)
GRADIO_SERVER_NAME=0.0.0.0 gpu-tools-ui
```

**Python launch with authentication:**
```python
from gpu_video_tools.gradio_app import launch

launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,  # Set True for public Gradio link
    auth=("admin", "secure_password")
)
```

### Tab 1: Video Tools

Individual tool interfaces with visual controls.

#### Using Probe
1. Click **"Probe"** tab
2. Upload or drag-drop video file
3. Click **"Probe Video"**
4. View metadata in JSON and formatted output
5. Copy/export results as needed

#### Using Transcode
1. Click **"Transcode"** tab
2. Upload input video
3. Select device: `nvidia:0`, `amd:0`, `cpu`, or `auto`
4. Choose codec: `h264`, `hevc`, or `av1`
5. Set optional parameters:
   - Width/Height for resolution
   - FPS for frame rate
   - Bitrate (e.g., `8M`) or CRF (e.g., `23`)
   - Preset for encoder speed/quality trade-off
6. Click **"Transcode"**
7. Monitor progress and download output

#### Using Scene Detection
1. Click **"Scenes"** tab
2. Upload video
3. Adjust detection threshold (10-50, default: 27)
4. Set minimum scene length (0.5-10 seconds, default: 1.0)
5. Click **"Detect Scenes"**
6. Download CSV with scene boundaries

#### Using Extract Frames
1. Click **"Extract Frames"** tab
2. Upload video
3. Select device
4. Set extraction parameters:
   - FPS: Extract at specific frame rate
   - Every Nth frame: Extract every N frames
   - Start/End time: Extract from time range
   - Format: JPG or PNG
5. Click **"Extract Frames"**
6. Download extracted frames (as zip or directory)

#### Using Face Detection
1. Click **"Faces"** tab
2. Upload video OR specify frames directory path
3. Select device
4. Set skip frames (process every Nth frame for video)
5. Optional: Specify custom YuNet model path
6. Click **"Detect Faces"**
7. Download CSV with face detections

### Tab 2: Batch Queue Manager

Manage multiple jobs with queue system.

#### Creating Batch Jobs

**Method 1: Upload CSV**
1. Create batch CSV file (see format above)
2. Click **"Upload Batch CSV"**
3. Select your CSV file
4. Jobs are added to queue

**Method 2: Manual Entry**
1. Select tool from dropdown
2. Upload input file
3. Select device
4. Click **"Add Job"**

#### Managing Queue

**View Queue:**
- See all jobs with status: `queued`, `running`, `completed`, `failed`
- View ID, Tool, Status, Device, Input file

**Delete Job:**
1. Enter Job ID
2. Click **"Delete Job"**

**Clear All:**
- Click **"Clear All"** to remove all jobs from queue

**Refresh:**
- Click **"Refresh"** to update queue display

#### Running Queue

**Execute:**
1. Click **"Run Queue"** to start processing
2. Monitor individual job progress
3. Jobs execute respecting per-device concurrency limits

**Control:**
- **Pause**: Stop processing new jobs (current job continues)
- **Resume**: Continue processing queue
- **Stop**: Cancel all remaining jobs

#### Queue Persistence
Queue state is saved between sessions. Restart the web interface and your queue will be restored.

### Tab 3: Configuration Editor

Edit configuration file directly from web interface.

#### Editing Configuration
1. Configuration loads automatically from `~/.gpu_video_tools/config.toml`
2. Edit YAML content in the editor
3. Syntax highlighting shows structure
4. Click **"Save Config"** to apply changes

#### Reset to Defaults
Click **"Reset to Default"** to restore original configuration.

#### Device Discovery
Use the device discovery tool to:
1. Enumerate available devices
2. Auto-generate optimized defaults
3. Update configuration with detected hardware

### Tab 4: Monitoring Dashboard

Real-time monitoring and analytics.

#### Device Information
- View all detected devices
- See vendor, index, name, and specification
- Refresh to update device list

#### Activity Log
- View recent operations
- Filter by log level
- Monitor system activity in real-time

#### Benchmark History
- View performance metrics from past jobs
- Sort and filter results
- Export data for analysis

---

## Configuration

### Configuration File Location

**Default path:**
- Linux/macOS: `~/.gpu_video_tools/config.toml`
- Windows: `%USERPROFILE%\.gpu_video_tools\config.toml`

**Custom path:**
```bash
gpu-tools --config /path/to/custom/config.toml probe video.mp4
```

### Configuration Structure

**Device defaults:**
```yaml
defaults:
  probe: cpu
  transcode: auto
  scenes: cpu
  extract: auto
  faces: auto
  cutlist: cpu
```

**Device limits:**
```yaml
device_limits:
  cpu: 2
  nvidia:0: 1
  amd:0: 1
```

**Codec preferences:**
```yaml
codecs:
  h264:
    - nvidia:0
    - amd:0
    - cpu
  hevc:
    - nvidia:0
    - amd:0
    - cpu
  av1:
    - nvidia:0
    - cpu
```

**Gradio settings:**
```yaml
gradio:
  server_port: 7860
  server_name: "localhost"
  enable_queue: true
  max_queue_size: 100
  share: false
  auth: null
```

### Device Precedence

Configuration sources in priority order:
1. **CLI flag**: `--device nvidia:0` (highest priority)
2. **Batch CSV**: `device` column
3. **Config file**: `defaults.<tool>` (lowest priority)

If no device is specified, the tool exits with an error and helpful instructions.

---

## Common Workflows

### Workflow 1: Video Conversion Pipeline

Convert multiple videos to a standard format:

```bash
# 1. Create batch CSV
cat > batch.csv << EOF
tool,device,input,output,vcodec,width,height,crf
transcode,nvidia:0,raw1.mov,processed1.mp4,h264,1920,1080,23
transcode,nvidia:0,raw2.mov,processed2.mp4,h264,1920,1080,23
transcode,nvidia:0,raw3.mov,processed3.mp4,h264,1920,1080,23
EOF

# 2. Run batch with benchmarking
gpu-tools run-batch --csv batch.csv --bench-csv results.csv

# 3. Analyze performance
cat results.csv
```

### Workflow 2: Scene-Based Video Analysis

Extract and analyze specific scenes:

```bash
# 1. Detect scenes
gpu-tools scenes movie.mp4 -o scenes.csv

# 2. Review scenes CSV
cat scenes.csv

# 3. Extract frames from interesting scenes
gpu-tools extract-frames movie.mp4 -o frames_scene1/ \
  --start 5.5 --end 12.3 --fps 1

gpu-tools extract-frames movie.mp4 -o frames_scene2/ \
  --start 12.3 --end 20.0 --fps 1

# 4. Detect faces in extracted frames
gpu-tools faces --frames-dir frames_scene1/ -o faces1.csv
gpu-tools faces --frames-dir frames_scene2/ -o faces2.csv
```

### Workflow 3: Face Detection Pipeline

Process videos for face detection and analysis:

```bash
# 1. Extract frames at 1 fps
gpu-tools --device nvidia:0 extract-frames input.mp4 -o frames/ --fps 1

# 2. Detect faces with visualization
gpu-tools --device nvidia:0 faces \
  --frames-dir frames/ \
  -o detections.csv \
  --draw annotated/

# 3. Process face detection results
# (Use Python/pandas to analyze detections.csv)
```

### Workflow 4: Quality Comparison

Compare different encoding settings:

```bash
# Create test batch
cat > quality_test.csv << EOF
tool,device,input,output,vcodec,crf
transcode,nvidia:0,source.mp4,out_crf18.mp4,h264,18
transcode,nvidia:0,source.mp4,out_crf23.mp4,h264,23
transcode,nvidia:0,source.mp4,out_crf28.mp4,h264,28
EOF

# Run with benchmarking
gpu-tools run-batch --csv quality_test.csv --bench-csv quality_bench.csv

# Compare file sizes and encoding times
ls -lh out_crf*.mp4
cat quality_bench.csv
```

### Workflow 5: Web UI Batch Processing

Using the Gradio interface for batch workflows:

1. **Launch Web UI:**
   ```bash
   gpu-tools-ui
   ```

2. **Prepare batch CSV** with all your jobs

3. **In Browser:**
   - Go to **"Batch Queue"** tab
   - Upload your batch CSV
   - Review queue
   - Click **"Run Queue"**
   - Monitor progress in real-time
   - Download results when complete

4. **Check monitoring:**
   - Go to **"Monitoring"** tab
   - View device utilization
   - Check benchmark history
   - Export metrics

---

## Troubleshooting

### Device Not Found

**Error:**
```
DeviceNotFoundError: Device 'nvidia:0' not found
Available devices: cpu
```

**Solution:**
```bash
# Check available devices
gpu-tools analyze

# Use CPU instead
gpu-tools --device cpu transcode input.mp4 -o output.mp4

# Or update config
gpu-tools analyze --write-defaults
```

### NVENC Not Available

**Error:**
```
EncoderNotAvailableError: h264_nvenc not available
```

**Solution:**
```bash
# Check FFmpeg encoders
ffmpeg -hide_banner -encoders | grep nvenc

# If missing, rebuild FFmpeg with NVENC support
# Or use CPU encoding
gpu-tools --device cpu transcode input.mp4 -o output.mp4 --vcodec h264
```

### YuNet Model Not Found

**Error:**
```
Model not found at ~/.gpu_video_tools/models/face_detection_yunet_2023mar.onnx
```

**Solution:**
```bash
# Download model
mkdir -p ~/.gpu_video_tools/models
wget https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -O ~/.gpu_video_tools/models/face_detection_yunet_2023mar.onnx

# Or specify custom path
gpu-tools faces --video input.mp4 -o faces.csv \
  --model /path/to/your/model.onnx
```

### Web Interface Dependencies Missing

**Error:**
```
ImportError: Missing required dependencies: gradio, pandas, plotly
```

**Solution:**
```bash
# Install web interface dependencies
pip install gradio pandas plotly

# Or with poetry
poetry install -E all
```

### Memory Issues

If processing large videos causes memory issues:

```bash
# Extract frames in chunks
gpu-tools extract-frames input.mp4 -o frames1/ --start 0 --end 60
gpu-tools extract-frames input.mp4 -o frames2/ --start 60 --end 120

# Process faces on extracted frames (lower memory)
gpu-tools faces --frames-dir frames1/ -o faces1.csv
gpu-tools faces --frames-dir frames2/ -o faces2.csv

# Or skip more frames
gpu-tools faces --video input.mp4 -o faces.csv --skip 60
```

### Performance Optimization

**For NVIDIA GPUs:**
```bash
# Use NVENC presets (p1=fastest, p7=slowest/best quality)
gpu-tools --device nvidia:0 transcode input.mp4 -o output.mp4 --preset p4

# Reduce resolution for faster processing
gpu-tools --device nvidia:0 transcode input.mp4 -o output.mp4 \
  --width 1280 --height 720
```

**For CPU:**
```bash
# Use faster preset
gpu-tools --device cpu transcode input.mp4 -o output.mp4 --preset veryfast

# Reduce concurrent jobs if CPU overloaded
gpu-tools --max-procs-per-device 1 run-batch --csv jobs.csv
```

---

## Additional Resources

- **Main Documentation**: See `README.md` for installation and overview
- **Configuration Reference**: See `tools_preferences.yml` for all options
- **API Documentation**: See inline docstrings in Python modules
- **Examples**: See `examples/` directory (if available)

For issues and questions:
- GitHub Issues: https://github.com/targuy/autocut/issues
- Documentation: This guide and `README.md`
