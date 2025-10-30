# StatLang

<div align="center">
  <h3>An open-source, Python-based statistical scripting language</h3>
  <p>Write and run statistical scripts with full syntax highlighting and a Python backend.</p>
</div>

## Overview

StatLang provides an open-source environment for statistical analysis by offering:
- **Expressive scripting syntax** for data manipulation and analysis
- **Python backend** for execution and performance
- **Jupyter notebook support** with a StatLang kernel
- **VS Code extension** with syntax highlighting and execution
- **Cross-platform compatibility** (Windows, macOS, Linux)
- **Open source** and free to use

### 🌟 **What Makes StatLang Special?**

- **🤖 AI Integration**: Built-in **PROC LANGUAGE** with LLM capabilities for intelligent data analysis
- **🧠 Complete ML Pipeline**: From data exploration to model deployment using familiar, concise syntax
- **💾 Modern SQL**: **PROC SQL** powered by DuckDB for high-performance data querying
- **🔧 Robust language features**: Macro system, format system, and statistical procedures
- **📊 Rich Visualizations**: Professional output formatting with TITLE statements and structured results

## Features

### Core Interpreter
- Scripting-based DATA step functionality with inline data support
- Statistical procedures (MEANS, FREQ, SORT, PRINT)
- Concise data manipulation and analysis syntax
- Python pandas/numpy backend for performance
- Clean, professional output with familiar formatting

### Jupyter Notebook Support
- StatLang kernel for Jupyter notebooks
- Interactive statistical programming in notebook environment
- Rich output display with formatted tables
- Dataset visualization and exploration

- ### VS Code Extension
- Syntax highlighting for `.statlang` files
- Code snippets for common statistical analysis patterns
- File execution directly from VS Code
- Notebook support for interactive analysis

### Supported Features

#### 📊 **Statistical Procedures**
- **PROC MEANS**: Descriptive statistics with CLASS variables and OUTPUT statements
- **PROC FREQ**: Frequency tables and cross-tabulations with options
- **PROC SORT**: Data sorting with ascending/descending order
- **PROC PRINT**: Data display and formatting
- **PROC REG**: Linear regression analysis with MODEL, OUTPUT, and SCORE statements
- **PROC UNIVARIATE**: Detailed univariate analysis with distribution diagnostics
- **PROC CORR**: Correlation analysis (Pearson, Spearman)
- **PROC FACTOR**: Principal component analysis and factor analysis
- **PROC CLUSTER**: Clustering methods (k-means, hierarchical)
- **PROC NPAR1WAY**: Nonparametric tests (Mann-Whitney, Kruskal-Wallis)
- **PROC TTEST**: T-tests (independent and paired)
- **PROC LOGIT**: Logistic regression modeling
- **PROC TIMESERIES**: Time series analysis and seasonal decomposition
- **PROC SURVEYSELECT**: Random sampling with SRS method, SAMPRATE/N options, and OUTALL flag

#### 🤖 **Machine Learning Procedures**
- **PROC TREE**: Decision trees for classification and regression
- **PROC FOREST**: Random forests for ensemble learning
- **PROC BOOST**: Gradient boosting for advanced modeling

#### 💻 **Advanced Features**
- **PROC SQL**: SQL query processing with DuckDB backend
- **PROC LANGUAGE**: Built-in LLM integration for text generation, Q&A, and data analysis
- **Macro System**: Complete macro facility with %MACRO/%MEND, %LET, & substitution, %PUT, %IF/%THEN/%ELSE, %DO/%END
- **Format System**: Built-in date/time, numeric, and currency formats with metadata persistence
- **TITLE Statements**: Professional output formatting

#### 🔧 **Core Data Processing**
- **DATA Steps**: Variable creation, conditional logic, DATALINES input
- **Macro variables**: %LET, %PUT statements
- **Libraries**: LIBNAME functionality
- **NOPRINT option**: Silent execution for procedures

## Installation

### Python Package
```bash
pip install statlang
```

### Jupyter Kernel Installation
```bash
# Install the StatLang kernel
python -m statlang.kernel install

# List available kernels
jupyter kernelspec list
```

### VS Code Extension
1. Install from VS Code Marketplace: "StatLang" by RyanBlakeStory
2. Or install from source (see Development section)

## 🚀 **Exciting New Features**

### 🤖 **LANGUAGE - AI-Powered Analysis**
```statlang
language prompt="Analyze the correlation between income and spending in our dataset";
run;
```
**Built-in LLM integration** for text generation, Q&A, and intelligent data analysis using Hugging Face transformers!

### 🧠 **Complete Machine Learning Workflow**
Check out our **[ML Project Demo](examples/ML_project_in_statlang.ipynb)** - a comprehensive regression analysis project showcasing:
- **PROC UNIVARIATE** for distribution exploration
- **PROC SURVEYSELECT** for train/test splitting  
- **PROC REG** with MODEL, OUTPUT, and SCORE statements
- **Macro system** for reusable analysis workflows
- **Complete ML pipeline** in pure StatLang syntax

### 💾 **SQL - Modern Data Querying**
```statlang
sql;
  select age, income, spend,
         case when income > 60000 then 'High' else 'Low' end as income_group
  from work.customers
  where age between 25 and 50
  order by income desc;
quit;
```
**DuckDB-powered SQL** processing with full dataset integration!

## Quick Start

### 1. Interactive Python Usage
```python
from statlang import StatLangInterpreter

# Create interpreter
interpreter = StatLangInterpreter()

# Create sample data using StatLang syntax
interpreter.run_code('''
data work.employees;
    input employee_id name $ department $ salary;
    datalines;
1 Alice Engineering 75000
2 Bob Marketing 55000
3 Carol Engineering 80000
4 David Sales 45000
;
run;
''')

# Run statistical analysis
interpreter.run_code('''
proc means data=work.employees;
    class department;
    var salary;
run;
''')
```

### 2. Jupyter Notebook Usage
1. Install the StatLang kernel:
   ```bash
   python -m statlang.kernel install
   ```
2. Create a new Jupyter notebook (`.ipynb`)
3. Select "statlang" as the kernel
4. Write StatLang code in cells and execute

### 3. VS Code Usage
1. Install the StatLang extension from the marketplace
2. Create a new file with `.statlang` extension
3. Write your StatLang code
4. Use `Ctrl+Shift+P` → "StatLang: Run File" to execute

### 4. Command Line Usage
```bash
# Run StatLang code from file
python -m statlang.cli run example.statlang

# Interactive mode
python -m statlang.cli interactive
```

## 📚 **Examples & Demos**

### 🎯 **Complete ML Project**
**[ML Project Demo](examples/ML_project_in_statlang.ipynb)** - A comprehensive machine learning workflow:
- Synthetic dataset creation with 30 observations
- **PROC UNIVARIATE** for distribution analysis
- **PROC SURVEYSELECT** for train/test splitting (70/30)
- **PROC REG** with MODEL, OUTPUT, and SCORE statements
- Macro-based reusable analysis functions
- Complete regression analysis pipeline

### 📊 **Comprehensive Walkthrough**
**[StatLang Walkthrough](examples/statlang_walkthrough.ipynb)** - Complete feature demonstration:
- All statistical procedures with examples
- Macro system demonstrations
- Format system usage
- Advanced data manipulation techniques
- Real-world analysis scenarios

## Project Structure

```
StatLang/
├── stat_lang/                # Core Python package
│   ├── __init__.py
│   ├── interpreter.py        # Main statistical interpreter
│   ├── cli.py               # Command line interface
│   ├── kernel/              # Jupyter kernel implementation
│   │   ├── statlang_kernel.py   # Main kernel
│   │   └── install.py       # Kernel installation
│   ├── parser/              # Syntax parser
│   │   ├── data_step_parser.py
│   │   ├── proc_parser.py
│   │   └── macro_parser.py
│   ├── procs/               # Statistical procedure implementations
│   │   ├── proc_means.py
│   │   ├── proc_freq.py
│   │   ├── proc_sort.py
│   │   └── proc_print.py
│   └── utils/               # Utility functions
│       ├── expression_evaluator.py
│       ├── data_utils.py
│       └── libname_manager.py
├── vscode-extension/         # VS Code extension
├── examples/                # Example files and demo notebook
├── media/                   # Logo and icons
├── setup.py                 # Package setup
└── README.md
```

## Development

### Setup Development Environment
```bash
git clone https://github.com/ryan-story/StatLang.git
cd StatLang
pip install -e .
```

### Running Tests
```bash
# Run basic functionality tests
python -c "from statlang import StatLangInterpreter; print('StatLang loaded successfully')"
```

## Key Features Implemented

### ✅ Completed Features
- [x] Core DATA step implementation with DATALINES
- [x] Statistical procedures with CLASS variables and OUTPUT statements
- [x] Frequency analysis with cross-tabulations and options
- [x] Data sorting with ascending/descending order
- [x] Data display and formatting
- [x] Linear regression analysis with PROC REG
- [x] Random sampling with PROC SURVEYSELECT
- [x] Silent execution options
- [x] Jupyter notebook kernel
- [x] VS Code extension with syntax highlighting
- [x] Clean, professional output
- [x] Concise behavior and syntax

### 🚧 Future Enhancements
- [ ] Additional statistical procedures (SQL queries, advanced regression, etc.)
- [ ] Advanced macro functionality
- [ ] Performance optimizations
- [ ] Enhanced data connectivity options

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- Additional statistical procedures
- Macro functionality enhancements
- Performance optimizations
- VS Code extension features
- Documentation and examples

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- 📖 [Documentation](https://github.com/ryan-story/StatLang/wiki)
- 🐛 [Issue Tracker](https://github.com/ryan-story/StatLang/issues)
- 💬 [Discussions](https://github.com/ryan-story/StatLang/discussions)