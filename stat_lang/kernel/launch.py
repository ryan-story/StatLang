#!/usr/bin/env python3
"""
Standalone kernel launcher for StatLang
"""

import sys
from .statlang_kernel import StatLangKernel
from ipykernel.kernelapp import IPKernelApp

if __name__ == '__main__':
    IPKernelApp.launch_instance(kernel_class=StatLangKernel)
