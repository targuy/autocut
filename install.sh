#!/bin/bash
# AutoCutVideo Installation Script for Linux/macOS
# This script sets up a virtual environment and installs dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}AutoCutVideo Installation Script${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Detect OS
OS_TYPE="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
fi

echo -e "${GREEN}Detected OS:${NC} $OS_TYPE"

# Step 1: Check Python version
echo -e "\n${YELLOW}[1/6] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10 or higher is required.${NC}"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Step 2: Check and install ffmpeg
echo -e "\n${YELLOW}[2/6] Checking FFmpeg installation...${NC}"
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    echo -e "${GREEN}✓ FFmpeg $FFMPEG_VERSION is already installed${NC}"
else
    echo -e "${YELLOW}FFmpeg not found. Attempting to install...${NC}"
    
    if [ "$OS_TYPE" = "linux" ]; then
        if command -v apt-get &> /dev/null; then
            echo "Installing ffmpeg using apt-get..."
            sudo apt-get update
            sudo apt-get install -y ffmpeg
        elif command -v yum &> /dev/null; then
            echo "Installing ffmpeg using yum..."
            sudo yum install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            echo "Installing ffmpeg using dnf..."
            sudo dnf install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            echo "Installing ffmpeg using pacman..."
            sudo pacman -S --noconfirm ffmpeg
        else
            echo -e "${RED}Could not detect package manager.${NC}"
            echo "Please install ffmpeg manually:"
            echo "  - Ubuntu/Debian: sudo apt-get install ffmpeg"
            echo "  - Fedora: sudo dnf install ffmpeg"
            echo "  - Arch: sudo pacman -S ffmpeg"
            exit 1
        fi
    elif [ "$OS_TYPE" = "macos" ]; then
        if command -v brew &> /dev/null; then
            echo "Installing ffmpeg using Homebrew..."
            brew install ffmpeg
        else
            echo -e "${RED}Homebrew not found.${NC}"
            echo "Please install Homebrew first: https://brew.sh"
            echo "Then run: brew install ffmpeg"
            exit 1
        fi
    else
        echo -e "${RED}Unsupported OS type.${NC}"
        echo "Please install ffmpeg manually."
        exit 1
    fi
    
    if command -v ffmpeg &> /dev/null; then
        echo -e "${GREEN}✓ FFmpeg installed successfully${NC}"
    else
        echo -e "${RED}Failed to install ffmpeg.${NC}"
        exit 1
    fi
fi

# Step 3: Create virtual environment
echo -e "\n${YELLOW}[3/6] Creating virtual environment...${NC}"
VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at $VENV_DIR${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    fi
else
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created at $VENV_DIR${NC}"
fi

# Step 4: Activate virtual environment
echo -e "\n${YELLOW}[4/6] Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Step 5: Upgrade pip
echo -e "\n${YELLOW}[5/6] Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"

# Step 6: Install dependencies
echo -e "\n${YELLOW}[6/6] Installing dependencies...${NC}"

# Ask user about GPU support
echo -e "\n${BLUE}Select installation type:${NC}"
echo "  1) CPU only (no GPU acceleration)"
echo "  2) NVIDIA CUDA (for NVIDIA GPUs)"
echo "  3) All features (CPU + attempt GPU detection)"
read -p "Enter choice (1-3): " INSTALL_CHOICE

case $INSTALL_CHOICE in
    1)
        echo -e "${YELLOW}Installing CPU-only version...${NC}"
        pip install -e ".[cpu]"
        ;;
    2)
        echo -e "${YELLOW}Installing CUDA version...${NC}"
        # First install the base package with CPU dependencies
        pip install -e ".[cpu]"
        # Then manually install CUDA versions
        echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        pip uninstall -y onnxruntime
        pip install onnxruntime-gpu
        echo -e "${GREEN}✓ CUDA packages installed${NC}"
        ;;
    3)
        echo -e "${YELLOW}Installing all features...${NC}"
        pip install -e ".[all]"
        # Try to detect CUDA and install appropriate version
        if command -v nvidia-smi &> /dev/null; then
            echo -e "${YELLOW}NVIDIA GPU detected, installing CUDA support...${NC}"
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
            pip uninstall -y onnxruntime
            pip install onnxruntime-gpu
        fi
        ;;
    *)
        echo -e "${RED}Invalid choice. Installing CPU-only version.${NC}"
        pip install -e ".[cpu]"
        ;;
esac

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Final message
echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${GREEN}======================================${NC}\n"

echo -e "${BLUE}To use AutoCutVideo:${NC}"
echo -e "  1. Activate the virtual environment:"
echo -e "     ${YELLOW}source venv/bin/activate${NC}"
echo -e "  2. Run the application:"
echo -e "     ${YELLOW}python main.py --config config.yml${NC}"
echo -e "     or use the convenience script:"
echo -e "     ${YELLOW}./run.sh${NC}"
echo -e "\n${BLUE}To deactivate the virtual environment:${NC}"
echo -e "  ${YELLOW}deactivate${NC}\n"

# Create a convenience run script
cat > run.sh << 'RUNSCRIPT'
#!/bin/bash
# AutoCutVideo run script

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run install.sh first."
    exit 1
fi

# Run the application
python main.py "$@"
RUNSCRIPT

chmod +x run.sh
echo -e "${GREEN}✓ Created run.sh convenience script${NC}\n"
