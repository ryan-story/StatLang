"""
StatLang: Python-based statistical scripting language

A Python package that provides a concise scripting syntax and functionality
with a Python backend for data analysis and manipulation.
"""

__version__ = "0.1.2"
__author__ = "Ryan Story"
__email__ = "ryan@stryve.com"

from .interpreter import SASInterpreter as StatLangInterpreter
from .interpreter import SASInterpreter

__all__ = ["StatLangInterpreter", "SASInterpreter"]
