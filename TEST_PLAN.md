# StatLang End-to-End Test Plan

## Pre-requisites
- Python 3.8+
- `pip install -e .` from repo root

## Kernel Installation
1. python -m stat_lang.kernel install
2. jupyter kernelspec list (verify `statlang` present)

## Notebook Tests
1. Open `examples/statlang_walkthrough.ipynb`
   - Set kernel to StatLang
   - Run cells top-to-bottom; expect dataset print and MEANS output
2. Open `examples/ML_project_in_statlang.ipynb`
   - Set kernel to StatLang
   - Run cells; expect train split, regression output, printed predictions

## CLI Tests
1. Create `sample.statlang` with DATA and PROC MEANS
2. Run: `statlang sample.statlang` (or `python -m stat_lang.cli run sample.statlang`)
3. Ensure non-zero exit status on syntax error

## VS Code Extension (local)
1. Build extension (npm install; npm run compile)
2. Launch VS Code Extension Host
3. Create `.statlang` file and run `StatLang: Run File`
4. Open notebooks, select StatLang kernel indicator (bottom-right) shows `statlang`

## Regression Checks
- Datasets created/cleared as expected
- No residual references to SAS/Open-SAS/osas in UI or outputs
- Kernel log file is `statlang_kernel_debug.log`

