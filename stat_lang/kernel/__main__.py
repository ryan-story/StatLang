#!/usr/bin/env python3
"""
StatLang Jupyter Kernel Entry Point
"""

import sys
from .statlang_kernel import StatLangKernel
from ipykernel.kernelapp import IPKernelApp

def main():
    """Main entry point for the kernel."""
    IPKernelApp.launch_instance(kernel_class=StatLangKernel)

if __name__ == '__main__':
    main()
