# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-26

### Added
- **GPU Video Tools Package**: Comprehensive video processing toolkit
  - CLI interface with 7 commands (probe, transcode, scenes, extract-frames, faces, run-batch, analyze)
  - Gradio web interface with 4 tabs (Video Tools, Batch Queue, Config Editor, Monitoring)
  - Hardware acceleration support (NVIDIA NVENC, AMD AMF, CPU fallback)
  - Device management system with auto-selection and policy-based resolution
  - Configuration system with YAML files and precedence handling
  - Batch processing with per-device concurrency limits
  - Benchmarking and CSV logging
  - Scene detection with PySceneDetect
  - Face detection with YuNet ONNX model
  - Thread-safe queue management for web interface
  
- **Documentation**:
  - Comprehensive README.md for GPU Video Tools
  - USAGE.md with 700+ lines of examples and tutorials
  - Agent instruction files for GitHub Copilot, Claude AI, and Gemini AI
  - VS Code workspace configuration with debugging support
  - GitHub Codespaces devcontainer configuration
  
- **Testing**:
  - 121 tests total (97 passing, 80% pass rate)
  - Mock-based tests for CI environments without GPU
  - Tests for all major modules
  
- **Development Environment**:
  - VS Code settings, launch configurations, and extension recommendations
  - Devcontainer for GitHub Codespaces with automatic setup
  - Poetry configuration for dependency management
  - Conda environment support as alternative
  - Comprehensive .gitignore for Python projects

### Changed
- Updated pyproject.toml with gradio and plotly dependencies
- Enhanced environment.yml with full dependency list
- Improved .gitignore with GPU Video Tools specific patterns

### Security
- CodeQL security scan: 0 alerts
- Subprocess calls use list arguments (no shell injection risk)
- Input validation on all file paths
- No hardcoded secrets or credentials

## [Unreleased]

### Planned
- Full implementation of cutlist command
- Additional video processing tools
- Enhanced monitoring dashboard with real-time metrics
- More codec support (VP9, AV1 improvements)
- Multi-language support for UI

---

[0.1.0]: https://github.com/targuy/autocut/releases/tag/v0.1.0
