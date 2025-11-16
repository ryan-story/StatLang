# StatLang VS Code Extension

<p align="center">
  <img src="https://raw.githubusercontent.com/ryan-story/StatLang/main/media/StatLang.png" alt="StatLang Logo" width="200"/>
</p>

<div align="center">
  <h3>Statistical scripting language syntax highlighting and execution support for VS Code</h3>
</div>

## Features

- **Syntax Highlighting**: Statistical scripting language syntax highlighting for `.statlang` files
- **Code Snippets**: Common statistical analysis patterns and procedures
- **File Execution**: Run StatLang files directly from VS Code
- **Notebook Support**: Interactive statistical notebooks (both `.ipynb` with StatLang kernel and `.stlnb` files)
- **IntelliSense**: Code completion and syntax checking
- **Integrated Terminal**: View results in VS Code's integrated terminal

## Installation

1. Install the extension from the VS Code Marketplace
2. Install the StatLang Python package:
   ```bash
   pip install statlang
   ```
   This will automatically install all dependencies including Jupyter kernel support.
3. Install the Jupyter kernel for notebook support:
   ```bash
   python -m statlang.kernel install
   ```

## Usage

### Basic StatLang Files (.statlang)
1. Create a new file with `.statlang` extension
2. Write your StatLang code
3. Use `Ctrl+Shift+P` → "StatLang: Run File" to execute
4. View results in the integrated terminal

### Interactive Notebooks
**Option 1: Standard Jupyter Notebooks (.ipynb)**
1. Install the StatLang kernel: `python -m statlang.kernel install`
2. Create a new Jupyter notebook (`.ipynb`)
3. Select "statlang" as the kernel
4. Write StatLang code in cells and execute

**Option 2: StatLang Notebooks (.stlnb)**
1. Create a new file with `.stlnb` extension
2. Write StatLang code in cells
3. Execute cells individually or run all
4. View formatted output and datasets

### Code Snippets
Type common statistical analysis patterns and press `Tab` to expand:
- `data` → DATA step template
- `proc` → Statistical procedure template
- `means` → PROC MEANS template
- `freq` → PROC FREQ template
- `reg` → PROC REG template
- `sql` → PROC SQL template
- `macro` → Macro definition template

## Supported Features

- **DATA Steps**: Variable creation, conditional logic, DATALINES
- **PROC MEANS**: Descriptive statistics with CLASS variables and OUTPUT statements
- **PROC FREQ**: Frequency tables and cross-tabulations with options
- **PROC SORT**: Data sorting with ascending/descending order
- **PROC PRINT**: Data display and formatting
- **PROC REG**: Linear regression analysis with MODEL, OUTPUT, and SCORE statements
- **PROC SURVEYSELECT**: Random sampling with SRS method, SAMPRATE/N options, and OUTALL flag
- **PROC UNIVARIATE**: Detailed univariate analysis with distribution diagnostics
- **PROC CORR**: Correlation analysis (Pearson, Spearman)
- **PROC FACTOR**: Principal component analysis and factor analysis
- **PROC CLUSTER**: Clustering methods (k-means, hierarchical)
- **PROC NPAR1WAY**: Nonparametric tests (Mann-Whitney, Kruskal-Wallis)
- **PROC TTEST**: T-tests (independent and paired)
- **PROC LOGIT**: Logistic regression modeling
- **PROC TIMESERIES**: Time series analysis and seasonal decomposition
- **PROC TREE/FOREST/BOOST**: Machine learning (decision trees, random forests, gradient boosting)
- **PROC SQL**: SQL query processing with DuckDB backend
- **PROC LANGUAGE**: Built-in LLM integration for text generation and analysis
- **SAS Macro System**: %MACRO/%MEND, %LET, & substitution, %PUT, %IF/%THEN/%ELSE, %DO/%END
- **SAS Format System**: Built-in date/time, numeric, and currency formats with metadata persistence
- **Macro Variables**: %LET, %PUT statements
- **Libraries**: LIBNAME functionality
- **TITLE Statements**: Title support for output formatting

## Configuration

The extension can be configured through VS Code settings:

- `statlang.pythonPath`: Path to Python executable
- `statlang.runtimePath`: Path to StatLang runner script
- `statlang.showOutputOnRun`: Show output channel when running code

## Requirements

- Python 3.8 or higher
- StatLang Python package
- VS Code 1.60.0 or higher

## Demo

Check out the comprehensive demo in `examples/statlang_walkthrough.ipynb` to see all features in action.

## Contributing

Contributions are welcome! Please see the [main project repository](https://github.com/ryan-story/StatLang) for contribution guidelines.

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Support

- 📖 [Documentation](https://github.com/ryan-story/StatLang/wiki)
- 🐛 [Issue Tracker](https://github.com/ryan-story/StatLang/issues)
- 💬 [Discussions](https://github.com/ryan-story/StatLang/discussions)

## ⚖️ Legal Disclaimer

This extension uses original, independently developed code to implement a statistical scripting language for educational and analytical use.

StatLang is provided as-is, under an open-source license, for research and community contribution purposes.
