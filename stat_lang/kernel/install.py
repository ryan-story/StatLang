"""
Kernel Installation for StatLang

This module provides functions to install and uninstall
the StatLang Jupyter kernel.
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


def install_kernel(user: bool = True, prefix: Optional[str] = None) -> bool:
    """
    Install the StatLang kernel for Jupyter.
    
    Args:
        user: Install for current user only
        prefix: Installation prefix path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Determine kernel directory
        if prefix:
            kernel_dir = Path(prefix) / 'share' / 'jupyter' / 'kernels' / 'statlang'
        elif user:
            kernel_dir = Path.home() / '.local' / 'share' / 'jupyter' / 'kernels' / 'statlang'
        else:
            kernel_dir = Path(sys.prefix) / 'share' / 'jupyter' / 'kernels' / 'statlang'
        
        # Create kernel directory
        kernel_dir.mkdir(parents=True, exist_ok=True)
        
        # Create kernel specification
        kernel_spec = {
            "argv": [
                sys.executable, 
                "-m", 
                "stat_lang.kernel", 
                "-f", 
                "{connection_file}"
            ],
            "display_name": "StatLang",
            "language": "statlang",
            "mimetype": "text/x-statlang",
            "file_extension": ".statlang",
            "codemirror_mode": "sas",
            "pygments_lexer": "sas"
        }
        
        # Write kernel specification
        with open(kernel_dir / 'kernel.json', 'w') as f:
            json.dump(kernel_spec, f, indent=2)
        
        # Install kernel using jupyter
        cmd = ['jupyter', 'kernelspec', 'install', str(kernel_dir)]
        if user:
            cmd.append('--user')
        if prefix:
            cmd.extend(['--prefix', prefix])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ StatLang kernel installed successfully!")
            print(f"   Kernel directory: {kernel_dir}")
            print("   You can now use StatLang in Jupyter notebooks!")
            return True
        else:
            print(f"❌ Failed to install kernel: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing kernel: {e}")
        return False


def uninstall_kernel() -> bool:
    """
    Uninstall the StatLang kernel.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Remove kernel using jupyter
        result = subprocess.run(
            ['jupyter', 'kernelspec', 'remove', 'statlang', '-f'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("✅ StatLang kernel uninstalled successfully!")
            return True
        else:
            print(f"❌ Failed to uninstall kernel: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error uninstalling kernel: {e}")
        return False


def list_kernels() -> None:
    """List all installed Jupyter kernels."""
    try:
        result = subprocess.run(
            ['jupyter', 'kernelspec', 'list'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("Installed Jupyter kernels:")
            print(result.stdout)
        else:
            print(f"❌ Failed to list kernels: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error listing kernels: {e}")


def check_kernel_installed() -> bool:
    """
    Check if the StatLang kernel is installed.
    
    Returns:
        True if installed, False otherwise
    """
    try:
        result = subprocess.run(
            ['jupyter', 'kernelspec', 'list'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            return 'statlang' in result.stdout
        return False
        
    except Exception:
        return False


def main():
    """Main entry point for kernel installation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Install/Uninstall StatLang Jupyter kernel')
    parser.add_argument('action', choices=['install', 'uninstall', 'list', 'check'],
                       help='Action to perform')
    parser.add_argument('--user', action='store_true', default=True,
                       help='Install for current user only')
    parser.add_argument('--system', action='store_true',
                       help='Install system-wide')
    parser.add_argument('--prefix', type=str,
                       help='Installation prefix')
    
    args = parser.parse_args()
    
    if args.action == 'install':
        user = not args.system
        success = install_kernel(user=user, prefix=args.prefix)
        sys.exit(0 if success else 1)
        
    elif args.action == 'uninstall':
        success = uninstall_kernel()
        sys.exit(0 if success else 1)
        
    elif args.action == 'list':
        list_kernels()
        
    elif args.action == 'check':
        if check_kernel_installed():
            print("✅ StatLang kernel is installed")
            sys.exit(0)
        else:
            print("❌ StatLang kernel is not installed")
            sys.exit(1)


if __name__ == '__main__':
    main()
