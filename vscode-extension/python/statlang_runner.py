#!/usr/bin/env python3
"""
StatLang Runner for VS Code Extension

This script is used by the VS Code extension to execute .statlang files
using the StatLang interpreter.
"""

import sys
import os
import traceback
from pathlib import Path

# Add the package to the Python path
current_dir = Path(__file__).parent
stat_lang_dir = current_dir.parent.parent / 'stat_lang'
if stat_lang_dir.exists():
    sys.path.insert(0, str(stat_lang_dir.parent))

try:
    from stat_lang import SASInterpreter as StatLangInterpreter
except ImportError:
    print("ERROR: StatLang package not found. Please install it or check the path.")
    sys.exit(1)


def main():
    """Main entry point for the runner."""
    if len(sys.argv) != 2:
        print("Usage: python statlang_runner.py <file.statlang>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found.")
        sys.exit(1)
    
    if not file_path.endswith('.statlang'):
        print(f"WARNING: File '{file_path}' does not have .statlang extension.")
    
    try:
        # Create interpreter and run the file
        interpreter = StatLangInterpreter()
        interpreter.run_file(file_path)
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
