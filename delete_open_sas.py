#!/usr/bin/env python3
"""
Script to delete open-sas package from PyPI using Warehouse API.

Note: PyPI typically requires web interface for deletion, but this script
attempts to use the API if available. You may need to use the web interface
at https://pypi.org/manage/projects/

Usage:
    export PYPI_USERNAME=your_username
    export PYPI_PASSWORD=your_password_or_token
    python3 delete_open_sas.py
"""

import os
import sys
import requests
from requests.auth import HTTPBasicAuth

PYPI_USERNAME = os.environ.get('PYPI_USERNAME')
PYPI_PASSWORD = os.environ.get('PYPI_PASSWORD', '')
PYPI_TOKEN = os.environ.get('PYPI_TOKEN', '')

if not PYPI_USERNAME:
    print("Error: PYPI_USERNAME environment variable not set")
    print("\nTo delete via web interface:")
    print("1. Go to https://pypi.org/manage/projects/")
    print("2. Find 'open-sas' project")
    print("3. Delete each release (0.1.0, 0.1.1, 0.1.2)")
    print("4. Delete the entire project")
    sys.exit(1)

# PyPI Warehouse API endpoint
BASE_URL = "https://pypi.org"

# Note: PyPI's deletion API may not be publicly available
# The following is a demonstration - you'll likely need to use the web interface
print("Warning: PyPI typically requires deletion through the web interface.")
print("This script may not work. Please use:")
print("  https://pypi.org/manage/projects/")
print()

# Try to delete releases (if API supports it)
releases_to_delete = ['0.1.0', '0.1.1', '0.1.2']
package_name = 'open-sas'

for version in releases_to_delete:
    delete_url = f"{BASE_URL}/pypi/{package_name}/{version}/"
    print(f"Attempting to delete {package_name} version {version}...")
    
    # Most PyPI APIs don't support DELETE - this is for reference only
    print(f"  → Manual deletion required at: {BASE_URL}/project/{package_name}/{version}/")
    print(f"    Or visit: https://pypi.org/manage/project/{package_name}/releases/")

print("\nTo delete via web interface:")
print(f"1. Go to https://pypi.org/manage/project/{package_name}/releases/")
print(f"2. Delete each release: {', '.join(releases_to_delete)}")
print(f"3. Then delete the entire project at: https://pypi.org/manage/project/{package_name}/")
print("\nNote: Package deletion requires confirmation on the web interface.")

