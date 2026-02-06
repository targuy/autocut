#!/usr/bin/env python3
"""GPU task example for AutoCut-Agent.

This example demonstrates a task that requires GPU resources.
The agent will ensure exclusive GPU access during execution.
"""

import sys
import json
import time
from datetime import datetime


def detect_gpu():
    """Detect available GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            return {
                "available": True,
                "count": device_count,
                "name": device_name
            }
    except ImportError:
        pass
    
    return {"available": False, "count": 0, "name": None}


def main():
    """Main function."""
    print("GPU Task started at:", datetime.now().isoformat())
    
    # Detect GPU
    gpu_info = detect_gpu()
    print("GPU Info:", json.dumps(gpu_info, indent=2))
    
    if not gpu_info["available"]:
        print("Warning: No GPU detected, running on CPU")
    
    # Simulate GPU-intensive work
    print("Processing on GPU...")
    time.sleep(5)
    
    # Output result
    result = {
        "status": "success",
        "gpu_info": gpu_info,
        "timestamp": datetime.now().isoformat(),
        "processing_time_seconds": 5
    }
    
    print("Result:", json.dumps(result, indent=2))
    print("GPU Task completed at:", datetime.now().isoformat())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
