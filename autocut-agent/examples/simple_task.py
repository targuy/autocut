#!/usr/bin/env python3
"""Simple task example for AutoCut-Agent.

This is a basic example of a task that can be executed by the agent.
It demonstrates input/output handling and logging.
"""

import sys
import json
import time
from datetime import datetime


def main():
    """Main function."""
    print("Task started at:", datetime.now().isoformat())
    
    # Simulate some work
    print("Processing...")
    time.sleep(2)
    
    # Output result as JSON
    result = {
        "status": "success",
        "message": "Task completed successfully",
        "timestamp": datetime.now().isoformat(),
        "processed_items": 42
    }
    
    print("Result:", json.dumps(result, indent=2))
    print("Task completed at:", datetime.now().isoformat())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
