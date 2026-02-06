#!/bin/bash
# Setup script for Linux/macOS

set -e

echo "==================================="
echo "AutoCut-Agent Setup"
echo "==================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.10 or higher is required. Found: $python_version"
    exit 1
fi
echo "✓ Python $python_version found"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    echo "✓ Poetry installed"
else
    echo "✓ Poetry found"
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
echo "✓ Virtual environment created"

# Install dependencies
echo "Installing dependencies..."
if [ -f "pyproject.toml" ]; then
    poetry install
    echo "✓ Dependencies installed with Poetry"
else
    pip install -r requirements.txt
    pip install -e .
    echo "✓ Dependencies installed with pip"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs outputs configs
echo "✓ Directories created"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# AutoCut-Agent Environment Variables

# API Configuration
API_SECRET_KEY=change-this-in-production

# Database (optional, defaults to SQLite)
# DATABASE_URL=postgresql://user:pass@localhost:5432/autocut

# Redis (optional)
# REDIS_URL=redis://localhost:6379/0

# LLM API Keys (optional)
# OPENAI_API_KEY=your-key-here
# ANTHROPIC_API_KEY=your-key-here

# SMTP for Email Alerts (optional)
# SMTP_PASSWORD=your-smtp-password
EOL
    echo "✓ .env file created (please update with your values)"
else
    echo "✓ .env file already exists"
fi

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: AutoCut-Agent setup"
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Check if Redis is running (optional)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✓ Redis is running"
    else
        echo "⚠ Redis is not running (optional for single-node deployment)"
    fi
else
    echo "⚠ Redis not installed (optional for single-node deployment)"
fi

echo ""
echo "==================================="
echo "Setup Complete! 🎉"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Update .env file with your configuration"
echo "3. Edit configs/default.yaml for your use case"
echo "4. Run the agent: autocut-agent start --config configs/default.yaml"
echo ""
echo "For development:"
echo "  autocut-agent start --config configs/development.yaml --dev"
echo ""
echo "Documentation: See README.md and docs/"
echo ""
