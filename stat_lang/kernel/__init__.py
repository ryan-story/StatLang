"""
StatLang Jupyter Kernel

This module provides Jupyter notebook support for StatLang,
allowing interactive execution of StatLang code in notebook environments.
"""

from .install import check_kernel_installed, install_kernel, list_kernels, uninstall_kernel
from .statlang_kernel import StatLangKernel

__all__ = ["StatLangKernel", "install_kernel", "uninstall_kernel", "check_kernel_installed", "list_kernels"]
