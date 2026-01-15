#!/usr/bin/env python3
"""
AutoCutVideo Installation Health Check
Verifies that all required dependencies are installed and working correctly.
"""

import sys
import subprocess
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_header(text):
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.NC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")

def check_python_version():
    """Check Python version is >= 3.10"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (need >= 3.10)")
        return False

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            version = version_line.split()[2] if len(version_line.split()) > 2 else 'unknown'
            print_success(f"FFmpeg {version}")
            return True
    except FileNotFoundError:
        print_error("FFmpeg not found in PATH")
        print("  Install: apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)")
        return False
    except Exception as e:
        print_error(f"FFmpeg check failed: {e}")
        return False

def check_module(module_name, display_name=None, optional=False):
    """Check if a Python module can be imported"""
    display_name = display_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print_success(f"{display_name} {version}")
        return True
    except ImportError:
        if optional:
            print_warning(f"{display_name} not installed (optional)")
        else:
            print_error(f"{display_name} not installed")
        return not optional  # Return True if optional

def check_torch():
    """Check PyTorch and CUDA availability"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else 'unknown'
            print_success(f"PyTorch {version} with CUDA")
            print(f"  GPU: {device_name} ({device_count} device(s))")
        else:
            print_success(f"PyTorch {version} (CPU only)")
            print_warning("  CUDA not available - using CPU")
        
        return True
    except ImportError:
        print_error("PyTorch not installed")
        return False

def check_onnxruntime():
    """Check ONNX Runtime and available providers"""
    try:
        import onnxruntime as ort
        version = ort.__version__
        providers = ort.get_available_providers()
        
        has_cuda = 'CUDAExecutionProvider' in providers
        has_tensorrt = 'TensorrtExecutionProvider' in providers
        
        if has_cuda or has_tensorrt:
            print_success(f"ONNX Runtime {version} (GPU)")
            print(f"  Providers: {', '.join(providers[:3])}")
        else:
            print_success(f"ONNX Runtime {version} (CPU)")
            print_warning(f"  Only CPU provider available")
        
        return True
    except ImportError:
        print_error("ONNX Runtime not installed")
        return False

def check_opencv():
    """Check OpenCV"""
    try:
        import cv2
        version = cv2.__version__
        print_success(f"OpenCV {version}")
        
        # Check CUDA support in OpenCV (if applicable)
        cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False
        if cuda_available:
            print("  CUDA support enabled")
        
        return True
    except ImportError:
        print_error("OpenCV not installed")
        return False

def main():
    print_header("AutoCutVideo Installation Health Check")
    
    all_checks = []
    
    # Core requirements
    print(f"{Colors.BLUE}Core Requirements:{Colors.NC}")
    all_checks.append(check_python_version())
    all_checks.append(check_ffmpeg())
    
    # Python dependencies
    print(f"\n{Colors.BLUE}Python Dependencies:{Colors.NC}")
    all_checks.append(check_torch())
    all_checks.append(check_onnxruntime())
    all_checks.append(check_opencv())
    all_checks.append(check_module('ultralytics', 'Ultralytics YOLOv8'))
    all_checks.append(check_module('numpy', 'NumPy'))
    all_checks.append(check_module('yaml', 'PyYAML'))
    all_checks.append(check_module('transformers', 'Transformers'))
    all_checks.append(check_module('mediapipe', 'MediaPipe'))
    
    # Optional dependencies
    print(f"\n{Colors.BLUE}Optional Dependencies:{Colors.NC}")
    check_module('imageio_ffmpeg', 'imageio-ffmpeg', optional=True)
    check_module('scenedetect', 'PySceneDetect', optional=True)
    
    # Summary
    print_header("Summary")
    
    passed = sum(all_checks)
    total = len(all_checks)
    
    if passed == total:
        print_success(f"All checks passed ({passed}/{total})")
        print("\n✓ AutoCutVideo is ready to use!")
        return 0
    else:
        failed = total - passed
        print_error(f"{failed} check(s) failed ({passed}/{total} passed)")
        print("\nPlease install missing dependencies:")
        print("  CPU only: pip install -e \".[cpu]\"")
        print("  With CUDA: ./install.sh (option 2)")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nHealth check interrupted.")
        sys.exit(130)
