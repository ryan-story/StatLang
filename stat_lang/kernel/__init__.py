"""
StatLang Jupyter Kernel

This module provides Jupyter notebook support for StatLang,
allowing interactive execution of StatLang code in notebook environments.
"""

from .statlang_kernel import StatLangKernel
from .install import install_kernel, uninstall_kernel, check_kernel_installed, list_kernels

__all__ = ["StatLangKernel", "install_kernel", "uninstall_kernel", "check_kernel_installed", "list_kernels"]
