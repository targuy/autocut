# Development Environment Setup Guide

This document provides comprehensive information about setting up and using the AutoCutVideo development environment.

## Quick Start

### Option 1: GitHub Codespaces (Recommended for Quick Start)

1. Click the "Code" button on GitHub
2. Select "Codespaces" tab
3. Click "Create codespace on main"
4. Wait 2-3 minutes for automatic setup
5. Start coding immediately!

**Features:**
- Pre-configured Python 3.11 environment
- All dependencies pre-installed
- FFmpeg included
- VS Code extensions ready
- Port forwarding for web UI (7860)

### Option 2: Local Development with Poetry

```bash
# Clone repository
git clone https://github.com/targuy/autocut.git
cd autocut

# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
poetry install -E all  # For all optional features

# Activate virtual environment
poetry shell

# Verify installation
gpu-tools --version
pytest tests/ -v
```

### Option 3: Local Development with Conda

```bash
# Clone repository
git clone https://github.com/targuy/autocut.git
cd autocut

# Create conda environment
conda env create -f environment.yml
conda activate autocut

# Install package
pip install -e .

# Verify installation
gpu-tools --version
pytest tests/ -v
```

### Option 4: Local Development with venv

```bash
# Clone repository
git clone https://github.com/targuy/autocut.git
cd autocut

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -e .
pip install -e .[all]  # For all optional features

# Verify installation
gpu-tools --version
pytest tests/ -v
```

## VS Code Configuration

The project includes comprehensive VS Code configuration for optimal development experience.

### Included Configurations

**`.vscode/settings.json`** - Project settings
- Python interpreter configuration
- Linting with flake8 (120 char line length)
- Formatting with black
- YAML, JSON, TOML formatting
- Testing with pytest
- File exclusions for __pycache__, etc.

**`.vscode/launch.json`** - Debug configurations
- Python: Current File
- Python: GPU Tools CLI
- Python: Gradio Web UI
- Python: AutoCut Main
- Python: pytest
- Python: pytest (current file)

**`.vscode/extensions.json`** - Recommended extensions
- Python extension pack
- YAML support
- TOML support
- GitHub Copilot
- GitLens
- Error Lens
- Code Spell Checker
- Git History

### Installing Recommended Extensions

When you open the project in VS Code, you'll be prompted to install recommended extensions. Click "Install All" to set up your environment.

Alternatively, install manually:
```
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.black-formatter
code --install-extension redhat.vscode-yaml
code --install-extension GitHub.copilot
# ... (see .vscode/extensions.json for complete list)
```

## AI Agent Instructions

The project includes detailed instruction files for AI coding assistants:

### GitHub Copilot
**Location:** `.github/copilot-instructions.md`

Provides context about:
- Project structure and architecture
- Coding standards and conventions
- Module responsibilities
- Configuration management
- Testing patterns

### Claude AI
**Location:** `.github/claude-instructions.md`

Comprehensive guide including:
- Technology stack details
- Best practices and patterns
- Security guidelines
- Testing philosophy
- Common development tasks

### Gemini AI
**Location:** `.github/gemini-instructions.md`

Detailed instructions covering:
- Architecture patterns
- Performance optimization
- Documentation standards
- Troubleshooting guide
- Version control practices

## Project Structure

```
autocut/
├── .devcontainer/              # GitHub Codespaces configuration
│   ├── devcontainer.json       # Container definition
│   └── setup.sh                # Post-create setup script
├── .github/                    # GitHub configuration
│   ├── copilot-instructions.md # Copilot AI directives
│   ├── claude-instructions.md  # Claude AI directives
│   └── gemini-instructions.md  # Gemini AI directives
├── .vscode/                    # VS Code configuration
│   ├── settings.json           # Editor settings
│   ├── launch.json             # Debug configurations
│   └── extensions.json         # Recommended extensions
├── cli/                        # AutoCut CLI scripts
├── pipeline/                   # AutoCut processing pipeline
├── detectors/                  # Detection modules
├── segmenters/                 # Segmentation modules
├── classifiers/                # Classification modules
├── gpu_video_tools/           # GPU Video Tools package
│   ├── __main__.py            # CLI entry point
│   ├── gradio_app.py          # Web interface
│   ├── README.md              # Package documentation
│   └── USAGE.md               # Usage guide
├── tests/                     # Test suite
├── .gitignore                 # Git ignore patterns
├── CHANGELOG.md               # Version history
├── VERSION                    # Current version
├── pyproject.toml            # Poetry configuration
├── environment.yml           # Conda environment
└── README.md                 # Main documentation
```

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Specific Module
```bash
pytest tests/gpu_video_tools/ -v
```

### With Coverage
```bash
pytest --cov=gpu_video_tools --cov-report=html
# View coverage report: open htmlcov/index.html
```

### In VS Code
1. Open Test Explorer (beaker icon in sidebar)
2. Click "Refresh Tests"
3. Run all or specific tests
4. View results inline in editor

## Code Quality Tools

### Linting
```bash
flake8 . --max-line-length=120
```

### Formatting
```bash
black . --line-length=120
```

### Type Checking (Optional)
```bash
mypy gpu_video_tools/ --ignore-missing-imports
```

### Run All Quality Checks
```bash
# Create a simple check script
./scripts/check-quality.sh  # (if available)
# Or run manually:
flake8 . --max-line-length=120 && \
black . --check --line-length=120 && \
pytest tests/ -v
```

## Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow coding standards (see AI instruction files)
- Write tests for new functionality
- Update documentation

### 3. Run Quality Checks
```bash
# Format code
black . --line-length=120

# Check linting
flake8 . --max-line-length=120

# Run tests
pytest tests/ -v
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: Add your feature description"
```

Follow conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
```

## Debugging

### CLI Tools
Use the "Python: GPU Tools CLI" launch configuration:
1. Set breakpoints in code
2. Press F5 or Run > Start Debugging
3. Select "Python: GPU Tools CLI"
4. Modify args in .vscode/launch.json as needed

### Web Interface
Use the "Python: Gradio Web UI" launch configuration:
1. Set breakpoints in gradio_app.py
2. Press F5
3. Select "Python: Gradio Web UI"
4. Access at http://localhost:7860

### Tests
Use the "Python: pytest (current file)" configuration:
1. Open test file
2. Set breakpoints
3. Press F5
4. Select "Python: pytest (current file)"

## Port Forwarding (Codespaces)

When running the web interface in Codespaces:
- Port 7860: Gradio Web UI (automatically forwarded)
- Port 8080: Alternative port (if needed)

Access via:
- Ports panel in VS Code
- Forwarded URL provided by Codespaces

## Environment Variables

Optional environment variables:
```bash
# Python
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# GPU Tools
export GPU_TOOLS_CONFIG=~/.gpu_video_tools/config.toml
export GPU_TOOLS_DEVICE=nvidia:0

# Gradio
export GRADIO_SERVER_PORT=7860
export GRADIO_SERVER_NAME=0.0.0.0
```

## Troubleshooting

### VS Code Not Finding Python Interpreter
1. Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `.venv` or conda environment

### Tests Not Discovering
1. Check Python interpreter is correct
2. Ensure pytest is installed: `pip install pytest`
3. Reload VS Code window
4. Check Test Explorer settings

### Import Errors
1. Verify virtual environment is activated
2. Check dependencies: `pip list` or `poetry show`
3. Reinstall: `poetry install` or `pip install -e .`

### Port Already in Use (Web UI)
```bash
# Find process using port 7860
lsof -i :7860  # Linux/macOS
netstat -ano | findstr :7860  # Windows

# Kill process or use different port
export GRADIO_SERVER_PORT=8080
```

## Contributing

1. Follow the coding standards in AI instruction files
2. Write tests for new features
3. Update documentation
4. Ensure all tests pass
5. Run code quality checks
6. Create pull request with clear description

## Resources

- **Documentation**: README.md, USAGE.md, gpu_video_tools/README.md
- **AI Instructions**: .github/*-instructions.md
- **Configuration**: pyproject.toml, environment.yml
- **Tests**: tests/ directory
- **Issues**: GitHub Issues

---

For questions or issues, please open a GitHub Issue or Discussion.
