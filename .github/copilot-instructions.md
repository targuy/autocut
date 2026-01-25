# Copilot Instructions for AutoCutVideo

## Project Overview

AutoCutVideo is a modular Python pipeline for video analysis and automatic video clip generation. The system:
- Detects and segments persons using YOLOv8
- Detects faces and measures face occlusion
- Segments skin and calculates skin visibility
- Classifies gender using Hugging Face transformers
- Analyzes video frames at configurable sampling rates
- Automatically cuts video clips based on configurable criteria (gender, face visibility, skin exposure, etc.)

The project is designed for video processing tasks requiring person detection and classification with customizable filtering criteria.

## Technology Stack

- **Language**: Python 3.10+
- **Package Management**: Poetry
- **Core Libraries**:
  - OpenCV (`opencv-python`) for video processing
  - Ultralytics YOLOv8 for detection and segmentation
  - PyTorch for deep learning models
  - Transformers & Hugging Face Hub for gender classification
  - NumPy for numerical operations
  - MediaPipe for pose estimation
  - PyYAML for configuration
  - tqdm for progress bars
- **Development Tools**:
  - pytest for testing
  - flake8 for linting
  - black for code formatting

## Coding Standards

### General Guidelines

- Follow PEP 8 style guide
- Use descriptive variable and function names
- Write docstrings for modules, classes, and functions
- Prefer type hints for function parameters and return values
- Use meaningful comments to explain complex logic

### Python Conventions

- **Indentation**: 4 spaces (never tabs)
- **String Quotes**: Single quotes for strings, double quotes for docstrings
- **Line Length**: Maximum 120 characters (relaxed from PEP 8's 79)
- **Imports**: 
  - Group imports: standard library, third-party, local modules
  - Use absolute imports from project root
  - Handle ImportError gracefully with informative messages
- **Error Handling**: Use try-except blocks with specific error messages in French (project language)

### Naming Conventions

- **Variables/Functions**: `snake_case` (e.g., `face_mask_percentage`, `load_config`)
- **Classes**: `PascalCase` (e.g., `FrameAnalyzer`, `VideoAnalyzer`, `SkinSegmenter`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., though not heavily used in this project)
- **Private Methods**: Prefix with single underscore `_method_name`
- **Config Parameters**: Use descriptive names matching config.yml structure

### Code Organization

- Use dataclasses for configuration objects
- Keep functions focused and single-purpose
- Validate inputs early and fail fast with clear error messages
- Use pathlib.Path or os.path for file operations, normalize paths consistently

## Project Structure

```
AutoCutVideo/
├── .github/                     # GitHub configuration
│   └── copilot-instructions.md  # This file
├── cli/                         # Command-line interface scripts
│   ├── process_video.py         # Main CLI entry point
│   ├── test_frame.py           # Test frame analysis
│   ├── assemble.py             # Assemble clips
│   ├── nameti.py               # Title generation single
│   └── nameitmulti.py          # Title generation batch
├── pipeline/                    # Core processing pipeline
│   ├── analyzer.py             # FrameAnalyzer & VideoAnalyzer classes
│   └── title_generator.py     # LLM-based title generation
├── detectors/                   # Detection modules
│   ├── person.py               # Person detection
│   ├── face.py                 # Face detection
│   ├── mask.py                 # Face mask percentage calculation
│   ├── pose.py                 # Head pose estimation
│   └── nsfw.py                 # NSFW detection wrapper
├── segmenters/                  # Segmentation modules
│   └── skin.py                 # Skin segmentation
├── classifiers/                 # Classification modules
│   └── gender.py               # Gender classification
├── config.py                    # Configuration loader (YAML → dataclass)
├── config.yml                   # Default configuration file
├── utiles.py                    # Utility functions
├── main.py                      # Application entry point
├── pyproject.toml              # Poetry configuration
└── README.md                    # Project documentation (French)
```

### Module Responsibilities

- **cli/**: User-facing command-line tools, argument parsing
- **pipeline/**: Core video processing logic, frame-by-frame analysis
- **detectors/**: Detection algorithms (person, face, NSFW, pose)
- **segmenters/**: Segmentation algorithms (skin)
- **classifiers/**: Classification models (gender)
- **config.py**: Centralized configuration management with validation

## Development Guidelines

### Configuration Management

- All parameters are centralized in `config.yml`
- Use `load_config()` to load and validate configuration
- Configuration is parsed into a `Config` dataclass with type safety
- Paths in config are normalized (expanded, resolved) automatically
- Required model weights must exist at specified paths (validation on load)

### Model Integration

- **YOLOv8 Models**: Loaded via Ultralytics library
  - Person segmentation: `person_yolov8*-seg.pt`
  - Face detection: `face_yolov8*.pt`
  - Skin segmentation: `skin_yolov8*-seg.pt`
- **Gender Classification**: Hugging Face transformers model
  - Default: `rizvandwiki/gender-classification-2`
  - Returns "male"/"female" labels
- **Device Management**: Support for CUDA GPU (`cuda:0`) and CPU (`cpu`)

### Video Processing Pattern

1. **Two-phase sampling**:
   - Fast sampling at `sample_rate` fps (e.g., 1 fps) for initial detection
   - Refined sampling at `refine_rate` fps (e.g., 24 fps) at segment boundaries
2. **Frame Analysis**: Each frame analyzed by `FrameAnalyzer`
   - Modular enable/disable flags for each detection type
   - Configurable thresholds for filtering
3. **Clip Generation**: Continuous segments meeting criteria are extracted without re-encoding

### Testing

- Use pytest for unit tests
- Test scripts in `cli/test_*.py` for component testing on single images
- Each test script outputs JSON results for easy validation
- Run tests with: `pytest --maxfail=1 --disable-warnings -q`

### Code Quality

- **Linting**: Run `flake8 .` before committing
- **Formatting**: Use `black .` to auto-format code
- **Type Checking**: While not enforced, type hints are encouraged for clarity

## Language and Communication

- **Primary Language**: French
  - User-facing messages, logs, comments, and README are in French
  - Variable/function names are in English (standard practice)
- **Error Messages**: Provide clear, actionable error messages in French
- **Comments**: Write comments in French to match project style

## Best Practices

### When Adding Features

1. Check if a module-level flag should be added (e.g., `enable_X_detection`)
2. Add corresponding configuration parameters to `config.yml`
3. Update `Config` dataclass in `config.py` with validation
4. Implement feature in appropriate module (detector/segmenter/classifier)
5. Update README.md with usage examples (in French)

### When Modifying Detection Logic

- Respect the modular architecture (one detector per file)
- Return consistent data structures (dicts with standardized keys)
- Handle edge cases (no detections, invalid input)
- Add optional debug visualization using OpenCV

### When Working with Paths

- Always normalize paths using the `norm()` helper function in `config.py`
- Support both absolute and relative paths
- Validate file existence for required model weights
- Create output directories if they don't exist

### Performance Considerations

- Use `num_workers` parameter for parallel processing where applicable
- Load models once and reuse (avoid reloading in loops)
- Use GPU when available (`device='cuda:0'`)
- Consider memory usage when processing high-resolution videos

## Common Patterns

### Loading Models

```python
from ultralytics import YOLO

model = YOLO(weights_path)
results = model(image, conf=confidence_threshold, device=device)
```

### Configuration Access

```python
from config import load_config

cfg = load_config("config.yml")
device = cfg.device
threshold = cfg.max_face_mask_percentage
```

### Error Handling

```python
try:
    # Operation that might fail
    result = process()
except SpecificException as e:
    print(f"[ERROR] Description claire du problème : {e}")
    sys.exit(1)
```

## References

- [README.md](../README.md) - Main project documentation (French)
- [config.yml](../config.yml) - Configuration file with parameter descriptions
- [pyproject.toml](../pyproject.toml) - Poetry dependencies and project metadata

## Additional Notes

- The project uses Poetry for dependency management; prefer `poetry add` over manual pip installs
- CLI entry point is defined as `autocut` in pyproject.toml
- Debug mode can be enabled via config (`debug: true`) for verbose logging
- The system supports batch processing of directories or single video files
- NSFW detection is optional and configurable (`enable_nsfw`, `nsfw_mode`)
