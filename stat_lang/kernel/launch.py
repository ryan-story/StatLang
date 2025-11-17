#!/usr/bin/env python3
"""
Standalone kernel launcher for StatLang
"""

from ipykernel.kernelapp import IPKernelApp

from .statlang_kernel import StatLangKernel

if __name__ == '__main__':
    IPKernelApp.launch_instance(kernel_class=StatLangKernel)
