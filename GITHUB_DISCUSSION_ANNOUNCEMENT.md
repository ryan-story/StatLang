# 🎉 StatLang is Now Available on VS Code Marketplace and PyPI!

We're excited to announce that **StatLang** is now officially available on both the **VS Code Marketplace** and **PyPI**!

## 📦 Installation Options

### VS Code Extension
Install the StatLang extension directly from VS Code:
1. Open VS Code
2. Go to Extensions (Cmd+Shift+X / Ctrl+Shift+X)
3. Search for "StatLang"
4. Click Install

**Or install via command line:**
```bash
code --install-extension RyanBlakeStory.statlang
```

**VS Code Marketplace:** [StatLang Extension](https://marketplace.visualstudio.com/items?itemName=RyanBlakeStory.statlang)

### PyPI Package
Install StatLang via pip:
```bash
pip install statlang
```

**PyPI:** [statlang on PyPI](https://pypi.org/project/statlang/)

## ✨ What is StatLang?

StatLang is an open-source, Python-based statistical scripting language that provides:

- **📊 Statistical Analysis**: Comprehensive statistical procedures (PROC MEANS, PROC FREQ, PROC CORR, PROC REG, and many more)
- **🤖 AI Integration**: Built-in PROC LANGUAGE with LLM capabilities for intelligent data analysis
- **🧠 Complete ML Pipeline**: From data exploration to model deployment using familiar syntax
- **💾 Modern SQL**: PROC SQL powered by DuckDB for high-performance data querying
- **📓 Jupyter Support**: Full Jupyter notebook integration with StatLang kernel
- **🎨 VS Code Support**: Syntax highlighting, code execution, and notebook support
- **🔧 Language Features**: Macro system, format system, and robust expression evaluation

## 🚀 Getting Started

### Quick Start with VS Code

1. Install the StatLang extension (see above)
2. Create a new `.statlang` file
3. Start coding with full syntax highlighting!

### Quick Start with Python

```python
from stat_lang import StatLangInterpreter

interpreter = StatLangInterpreter()
code = """
data work.employees;
    input name $ department $ salary;
    datalines;
Alice Engineering 75000
Bob Marketing 55000
Carol Engineering 80000
;
run;

proc means data=work.employees;
    var salary;
    class department;
run;
"""

interpreter.run_code(code)
```

### Jupyter Notebook Support

After installing via pip:
```bash
python -m stat_lang.kernel install
```

Then select the "StatLang" kernel in your Jupyter notebook!

## 📚 Documentation

- **GitHub Repository**: [StatLang on GitHub](https://github.com/ryan-story/StatLang)
- **Full Documentation**: See our [README](https://github.com/ryan-story/StatLang#readme)
- **Examples**: Check out the [examples directory](https://github.com/ryan-story/StatLang/tree/main/examples)

## 🎯 Current Release

- **VS Code Extension**: v0.2.4
- **PyPI Package**: v0.1.3

## 🤝 Contributing

We welcome contributions! StatLang is an open-source project, and we'd love your help improving it.

- Report bugs: [GitHub Issues](https://github.com/ryan-story/StatLang/issues)
- Contribute code: See our [Contributing Guide](https://github.com/ryan-story/StatLang/blob/main/CONTRIBUTING.md)

## 📝 License

StatLang is released under the MIT License - see [LICENSE](https://github.com/ryan-story/StatLang/blob/main/LICENSE) for details.

## 🙏 Thank You

Thank you for your interest in StatLang! We're excited to see what you build with it.

Happy coding! 🎉

