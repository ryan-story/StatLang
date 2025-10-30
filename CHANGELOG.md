# Changelog

All notable changes to StatLang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2024-10-23

### Fixed
- **Critical Windows Bug**: Fixed kernel logging path issue - now uses platform-independent temp directory instead of Unix-only `/tmp/` path

## [0.1.1] - 2024-10-23

### Fixed
- **Critical**: Moved `ipykernel` and `jupyter` from optional to required dependencies to fix kernel installation errors
- Added Python 3.12 and 3.13 support to classifiers

### Added
- **PROC REG**: Linear regression analysis with MODEL, OUTPUT, and SCORE statements
- Comprehensive demo notebook (statlang_walkthrough.ipynb)
- NOPRINT option support across all procedures
- Clean, professional output
- Proper column ordering in PROC MEANS OUTPUT statements
- Expression evaluator improvements for underscore variable names

### Fixed (Additional)
- Expression evaluator underscore variable name handling
- PROC SORT ascending/descending order handling
- PROC FREQ TABLES statement parsing with options
- PROC MEANS OUTPUT dataset creation and display
- Automatic data printing issues in kernel
- Debug output cleanup for professional appearance

### Changed
- Major codebase cleanup and artifact removal
- Updated documentation and README
- Improved error handling and logging
- Enhanced language consistency

## [0.1.0] - 2024-10-16

### Added
- Initial release of StatLang
- Core interpreter with Python backend
- DATA step functionality with DATALINES support
- PROC MEANS with CLASS variables and OUTPUT statements
- PROC FREQ with cross-tabulations and options
- PROC SORT with ascending/descending order
- PROC PRINT for data display
- Jupyter notebook kernel support
- Macro variable support (%LET, %PUT)
- Library (LIBNAME) functionality
- Command line interface
- Comprehensive documentation

### Features
- Concise scripting syntax for data manipulation
- Python pandas/numpy backend
- Cross-platform compatibility
- Clean, professional output
- Interactive notebook support
- Dataset visualization
- Error handling and reporting

## [0.0.1] - 2024-10-01

### Added
- Initial project setup
- Basic interpreter structure
- Core parsing framework
- Development environment setup
