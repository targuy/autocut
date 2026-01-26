#!/bin/bash
# Setup script for devcontainer initialization

set -e

echo "🚀 Setting up AutoCutVideo development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install FFmpeg for video processing
echo "🎬 Installing FFmpeg..."
sudo apt-get install -y ffmpeg

# Install Poetry
echo "📝 Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/vscode/.local/bin:$PATH"
echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> ~/.bashrc

# Configure Poetry
poetry config virtualenvs.in-project true

# Install project dependencies
echo "📚 Installing Python dependencies..."
if [ -f "pyproject.toml" ]; then
    poetry install --no-interaction
    echo "✅ Dependencies installed via Poetry"
else
    pip install --upgrade pip
    pip install -e .
    echo "✅ Dependencies installed via pip"
fi

# Install development tools
echo "🛠️  Installing development tools..."
pip install pytest pytest-cov flake8 black mypy

# Create necessary directories
echo "📁 Creating workspace directories..."
mkdir -p ~/.gpu_video_tools/models
mkdir -p output frames benchmarks

# Download YuNet face detection model (optional)
echo "🤖 Downloading YuNet face detection model..."
YUNET_URL="https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH="$HOME/.gpu_video_tools/models/face_detection_yunet_2023mar.onnx"

if [ ! -f "$YUNET_PATH" ]; then
    curl -L "$YUNET_URL" -o "$YUNET_PATH" && echo "✅ YuNet model downloaded" || echo "⚠️  YuNet model download failed (optional)"
else
    echo "✅ YuNet model already exists"
fi

# Set up git configuration
echo "🔧 Configuring git..."
git config --global --add safe.directory /workspaces/autocut

# Display environment info
echo ""
echo "✨ Development environment ready!"
echo ""
echo "📋 Environment Information:"
echo "   Python: $(python --version)"
echo "   Poetry: $(poetry --version 2>/dev/null || echo 'Not installed')"
echo "   FFmpeg: $(ffmpeg -version 2>/dev/null | head -n1 || echo 'Not installed')"
echo ""
echo "🎯 Quick Start Commands:"
echo "   poetry shell                    # Activate virtual environment"
echo "   gpu-tools --help                # Show CLI help"
echo "   gpu-tools-ui                    # Launch web interface"
echo "   pytest tests/ -v                # Run tests"
echo ""
echo "Happy coding! 🎉"
