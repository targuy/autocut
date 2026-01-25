@echo off
REM AutoCutVideo Installation Script for Windows
REM This script sets up a virtual environment and installs dependencies

setlocal enabledelayedexpansion

echo ======================================
echo AutoCutVideo Installation Script
echo ======================================
echo.

REM Step 1: Check Python installation
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher from https://www.python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found
echo.

REM Step 2: Check and install ffmpeg
echo [2/6] Checking FFmpeg installation...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo FFmpeg not found. 
    echo.
    echo FFmpeg is required for video processing.
    echo.
    echo Installation options:
    echo   1. Automatic: Download pre-built binaries via imageio-ffmpeg ^(recommended^)
    echo   2. Manual: Download from https://ffmpeg.org/download.html
    echo.
    echo The automatic installation will download ffmpeg binaries as part of
    echo the Python package installation.
    echo.
    set INSTALL_FFMPEG=1
) else (
    for /f "tokens=3" %%i in ('ffmpeg -version 2^>^&1 ^| findstr "ffmpeg version"') do set FFMPEG_VERSION=%%i
    echo [OK] FFmpeg !FFMPEG_VERSION! is already installed
    set INSTALL_FFMPEG=0
)
echo.

REM Step 3: Create virtual environment
echo [3/6] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists at venv
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        rmdir /s /q venv
        python -m venv venv
        echo [OK] Virtual environment recreated
    )
) else (
    python -m venv venv
    echo [OK] Virtual environment created at venv
)
echo.

REM Step 4: Activate virtual environment
echo [4/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Step 5: Upgrade pip
echo [5/6] Upgrading pip...
python -m pip install --upgrade pip
echo [OK] pip upgraded
echo.

REM Step 6: Install dependencies
echo [6/6] Installing dependencies...
echo.
echo Select installation type:
echo   1^) CPU only ^(no GPU acceleration^)
echo   2^) NVIDIA CUDA ^(for NVIDIA GPUs^)
echo   3^) All features ^(CPU + attempt GPU detection^)
echo.
set /p INSTALL_CHOICE="Enter choice (1-3): "

if "!INSTALL_CHOICE!"=="1" (
    echo Installing CPU-only version...
    pip install -e ".[cpu]"
    if !INSTALL_FFMPEG!==1 (
        pip install imageio-ffmpeg
    )
) else if "!INSTALL_CHOICE!"=="2" (
    echo Installing CUDA version...
    echo.
    echo Installing base packages...
    pip install -e ".[cpu]"
    echo.
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo Installing ONNX Runtime GPU...
    pip uninstall -y onnxruntime
    pip install onnxruntime-gpu
    if !INSTALL_FFMPEG!==1 (
        pip install imageio-ffmpeg
    )
    echo [OK] CUDA packages installed
) else if "!INSTALL_CHOICE!"=="3" (
    echo Installing all features...
    pip install -e ".[all]"
    echo.
    REM Try to detect CUDA
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        echo NVIDIA GPU detected, installing CUDA support...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        pip uninstall -y onnxruntime
        pip install onnxruntime-gpu
        echo [OK] CUDA support enabled
    ) else (
        echo No NVIDIA GPU detected, using CPU-only packages
    )
) else (
    echo Invalid choice. Installing CPU-only version.
    pip install -e ".[cpu]"
    if !INSTALL_FFMPEG!==1 (
        pip install imageio-ffmpeg
    )
)

echo [OK] Dependencies installed
echo.

REM Final message
echo ======================================
echo Installation completed successfully!
echo ======================================
echo.
echo To use AutoCutVideo:
echo   1. Activate the virtual environment:
echo      venv\Scripts\activate.bat
echo   2. Run the application:
echo      python main.py --config config.yml
echo      or use the convenience script:
echo      run.bat
echo.
echo To deactivate the virtual environment:
echo   deactivate
echo.

REM Create a convenience run script
(
echo @echo off
echo REM AutoCutVideo run script
echo.
echo REM Activate virtual environment
echo if exist "venv\Scripts\activate.bat" ^(
echo     call venv\Scripts\activate.bat
echo ^) else ^(
echo     echo Error: Virtual environment not found. Run install.bat first.
echo     pause
echo     exit /b 1
echo ^)
echo.
echo REM Run the application
echo python main.py %%*
) > run.bat

echo [OK] Created run.bat convenience script
echo.

pause
