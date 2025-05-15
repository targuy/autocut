#!/usr/bin/env python3
# main.py

import sys
from cli.process_video import main as cli_main

if __name__ == "__main__":
    # Transmet simplement tous les arguments à votre CLI
    sys.exit(cli_main())
