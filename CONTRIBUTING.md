# Contributing to StatLang

Thank you for your interest in contributing to StatLang! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to a code of conduct adapted from the Contributor Covenant. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs or request features
- Include a clear description of the issue
- Provide steps to reproduce the problem
- Include system information (OS, Python version, etc.)
- For language compatibility issues, include the expected behavior

### Contributing Code

1. **Fork the repository** and clone your fork
2. **Create a feature branch** from `develop` (or `main` for hotfixes)
3. **Make your changes** following the coding standards
4. **Test your changes** thoroughly
5. **Submit a pull request** to `develop` for review
6. Once approved, merge into `develop`
7. When ready for release, open a PR from `develop` to `main` for final review

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/StatLang.git
cd StatLang

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Coding Standards

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write docstrings for all public functions
- Keep functions focused and small
- Add tests for new functionality

### Testing

- Write tests for new features
- Ensure all existing tests pass
- Verify language behavior where applicable
- Use descriptive test names

### Documentation

- Update README.md for significant changes
- Add docstrings for new functions
- Update API documentation if needed
- Include examples for new features

## Areas for Contribution

### High Priority
- **Additional PROC procedures** (PROC SQL, PROC REG, PROC GLM)
- **Advanced macro functionality** (%MACRO, %MEND, %DO loops)
- **Performance optimizations** for large datasets
- **Error handling improvements**

### Medium Priority
- **VS Code extension features** (IntelliSense, debugging)
- Data connectivity enhancements
- **Additional data formats** (Excel, JSON, XML)
- **Documentation improvements**

### Low Priority
- **UI/UX improvements** for notebooks
- **Additional examples** and tutorials
- **Performance benchmarking** tools
- **Integration with other tools**

## Pull Request Process

1. **Update documentation** for any new features
2. **Add tests** for new functionality
3. **Ensure all tests pass** locally
4. **Update CHANGELOG.md** with your changes
5. **Request review** from maintainers

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Language behavior verified

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the wiki for detailed guides

## Release Process

Releases are automated and triggered when code is merged to `main`:

1. **Development flow**: Open PR → merge into `develop`
2. **Release flow**: Open PR from `develop` → `main` and review manually
3. **After merge to `main`**, the Release workflow automatically:
   - Re-runs all tests (ruff, mypy, pytest)
   - Computes next semantic version from commit messages (conventional commits)
   - Bumps versions in `pyproject.toml`, `stat_lang/__init__.py`, and `vscode-extension/package.json`
   - Creates a Git tag and GitHub Release
   - Publishes the Python package to PyPI
   - Publishes the VS Code extension to the Marketplace

**Commit message format**: Use conventional commits so semantic-release can determine version bumps:
- `feat: ...` → minor version bump
- `fix: ...` → patch version bump
- `feat!: ...` or `BREAKING CHANGE: ...` → major version bump

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation

Thank you for contributing to StatLang!
