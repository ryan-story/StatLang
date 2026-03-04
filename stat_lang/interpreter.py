"""
StatLang interpreter core.

This module contains the core interpreter class that parses and executes
StatLang code using Python as the backend.
"""

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .parser.data_step_parser import DataStepParser
from .parser.macro_parser import MacroParser
from .parser.proc_parser import ProcParser
from .procs import (
    ProcANOVA,
    ProcAppend,
    ProcBoost,
    ProcCluster,
    ProcContents,
    ProcCorr,
    ProcCVision,
    ProcDatasets,
    ProcDiscrim,
    ProcDNN,
    ProcExport,
    ProcFactor,
    ProcForest,
    ProcFreq,
    ProcGenmod,
    ProcGLM,
    ProcImport,
    ProcLanguage,
    ProcLifereg,
    ProcLLM,
    ProcLogit,
    ProcMeans,
    ProcMixed,
    ProcNLP,
    ProcNpar1way,
    ProcPhreg,
    ProcPrincomp,
    ProcPrint,
    ProcReg,
    ProcRL,
    ProcRobustreg,
    ProcSort,
    ProcSQL,
    ProcSurveySelect,
    ProcTimeseries,
    ProcTranspose,
    ProcTree,
    ProcTtest,
    ProcUnivariate,
)
from .utils.data_utils import DataUtils
from .utils.error_handler import ErrorHandler
from .utils.expression_evaluator import ExpressionEvaluator
from .utils.expression_parser import ExpressionParser
from .utils.format_informat_parser import FormatInformatParser
from .utils.format_processor import FormatProcessor
from .utils.libname_manager import LibnameManager
from .utils.macro_processor import MacroProcessor
from .utils.model_store import ModelStore
from .utils.statlang_dataset import SasDataset, SasDatasetManager


class SASInterpreter:
    """
    Main interpreter for StatLang code execution.

    This class provides the core functionality to parse and execute
    StatLang code using Python libraries as the backend.
    """

    def __init__(self):
        """Initialize the interpreter."""
        self.data_sets: Dict[str, pd.DataFrame] = {}
        self.libraries: Dict[str, str] = {}
        self.macro_variables: Dict[str, str] = {}
        self.options: Dict[str, Any] = {}

        # Initialize parsers
        self.data_step_parser = DataStepParser()
        self.proc_parser = ProcParser()
        self.macro_parser = MacroParser()
        self.expression_parser = ExpressionParser()
        self.expression_evaluator = ExpressionEvaluator()
        self.data_utils = DataUtils()
        self.libname_manager = LibnameManager()
        self.error_handler = ErrorHandler()

        # Initialize new macro and format systems
        self.macro_processor = MacroProcessor()
        self.format_processor = FormatProcessor()
        self.dataset_manager = SasDatasetManager()
        self.format_informat_parser = FormatInformatParser()

        # Model store for persist/score
        self.model_store = ModelStore()

        # Initialize title tracking
        self.current_title: Optional[str] = None

        # Initialize PROC implementations
        self.proc_implementations: Dict[str, Any] = {
            'MEANS': ProcMeans(),
            'FREQ': ProcFreq(),
            'PRINT': ProcPrint(),
            'SORT': ProcSort(),
            'CONTENTS': ProcContents(),
            'UNIVARIATE': ProcUnivariate(),
            'CORR': ProcCorr(),
            'FACTOR': ProcFactor(),
            'CLUSTER': ProcCluster(),
            'NPAR1WAY': ProcNpar1way(),
            'TTEST': ProcTtest(),
            'LOGIT': ProcLogit(),
            'TIMESERIES': ProcTimeseries(),
            'ARIMA': ProcTimeseries(),
            'TREE': ProcTree(),
            'FOREST': ProcForest(),
            'BOOST': ProcBoost(),
            'LANGUAGE': ProcLanguage(),
            'SQL': ProcSQL(),
            'SURVEYSELECT': ProcSurveySelect(),
            'REG': ProcReg(),
            # New SAS procs
            'GLM': ProcGLM(),
            'ANOVA': ProcANOVA(),
            'DISCRIM': ProcDiscrim(),
            'PRINCOMP': ProcPrincomp(),
            'ROBUSTREG': ProcRobustreg(),
            'LIFEREG': ProcLifereg(),
            'PHREG': ProcPhreg(),
            'GENMOD': ProcGenmod(),
            'MIXED': ProcMixed(),
            'TRANSPOSE': ProcTranspose(),
            'APPEND': ProcAppend(),
            'DATASETS': ProcDatasets(),
            'EXPORT': ProcExport(),
            'IMPORT': ProcImport(),
            # Deep learning procs
            'DNN': ProcDNN(),
            'NLP': ProcNLP(),
            'CVISION': ProcCVision(),
            'RL': ProcRL(),
            'LLM': ProcLLM(),
        }

    def run_file(self, file_path: str) -> None:
        """Execute a .statlang file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r') as f:
            sas_code = f.read()
        self.run_code(sas_code)

    def run_code(self, sas_code: str) -> None:
        """Execute SAS code string."""
        code_lines = sas_code.split('\n')

        # Process macro statements first
        expanded_code_lines = self.macro_processor.process_macro_statements(code_lines)

        processed_code = '\n'.join(expanded_code_lines)
        cleaned_code = self._clean_code(processed_code)
        statements = self._split_statements(cleaned_code)

        for statement in statements:
            if statement.strip():
                self._execute_statement(statement.strip())

    def _clean_code(self, code: str) -> str:
        """Remove comments and clean up SAS code."""
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code

    def _split_statements(self, code: str) -> List[str]:
        """Split SAS code into individual statements."""
        lines = code.split('\n')
        statements: List[str] = []
        current_statement = ""
        in_datalines = False
        in_data_step = False

        for line in lines:
            line = line.strip()
            if not line:
                if current_statement:
                    current_statement += '\n'
                continue

            if line.upper().startswith('DATA '):
                in_data_step = True
                current_statement = line
                continue
            elif line.upper().startswith('PROC '):
                if in_data_step and current_statement.strip():
                    statements.append(current_statement.strip())
                    current_statement = ""
                    in_data_step = False
                    in_datalines = False
                current_statement = line
                continue
            elif line.upper() == 'RUN;':
                if current_statement.strip():
                    current_statement += '\n' + line
                    statements.append(current_statement.strip())
                    current_statement = ""
                    in_data_step = False
                    in_datalines = False
                continue
            elif current_statement.upper().startswith('PROC '):
                current_statement += '\n' + line
                continue

            if line.upper() in ['DATALINES;', 'CARDS;']:
                in_datalines = True
                current_statement += '\n' + line
                continue
            elif line == ';' and in_datalines:
                current_statement += '\n' + line
                in_datalines = False
                continue
            elif in_datalines:
                current_statement += '\n' + line
                continue

            if in_data_step:
                current_statement += '\n' + line
            elif line.endswith(';'):
                line_upper = line.strip().upper()
                if line_upper.startswith(('TABLES', 'VAR', 'BY', 'CLASS', 'MODEL', 'OUTPUT', 'WHERE', 'TITLE')):
                    if current_statement.strip():
                        statements.append(current_statement.strip())
                    current_statement = line.strip()
                    statements.append(current_statement.strip())
                    current_statement = ""
                else:
                    current_statement += ' ' + line
                    statements.append(current_statement.strip())
                    current_statement = ""
            else:
                line_upper = line.strip().upper()
                if line_upper.startswith(('TABLES', 'VAR', 'BY', 'CLASS', 'MODEL', 'OUTPUT', 'WHERE', 'TITLE')):
                    if current_statement.strip():
                        statements.append(current_statement.strip())
                    current_statement = line.strip()
                else:
                    current_statement += ' ' + line

        if current_statement.strip():
            statements.append(current_statement.strip())

        return statements

    def _execute_statement(self, statement: str) -> None:
        """Execute a single SAS statement."""
        statement = statement.strip()
        if not statement:
            return

        if statement.upper().startswith('DATA '):
            self._execute_data_step(statement)
        elif statement.upper().startswith('PROC '):
            self._execute_proc(statement)
        elif statement.upper().startswith('LIBNAME '):
            self._execute_libname(statement)
        elif statement.upper().startswith('%LET '):
            self._execute_let(statement)
        elif statement.upper().startswith('%PUT '):
            self._execute_put(statement)
        elif statement.upper().startswith('TITLE '):
            self._execute_title(statement)
        elif statement.upper().startswith(('%MACRO ', '%MEND', '%IF ', '%DO ', '%END',
                                           '%INCLUDE ')):
            pass  # Handled by macro processor
        elif statement.startswith('%') and '(' in statement:
            pass  # Macro calls handled by macro processor
        elif statement.upper() == 'RUN;':
            pass
        else:
            print(f"Warning: Unsupported statement: {statement}")

    # ------------------------------------------------------------------
    # DATA STEP EXECUTION
    # ------------------------------------------------------------------
    def _execute_data_step(self, statement: str) -> None:
        """Execute a DATA step with full feature support."""
        try:
            if 'datalines' in statement.lower() or 'cards' in statement.lower():
                self._execute_datalines(statement)
                return

            data_info = self.data_step_parser.parse_data_step(statement)

            # --- INFILE / INPUT ---
            if data_info.infile_spec and data_info.infile_spec.path:
                input_data = self._execute_infile(data_info)
            # --- MERGE ---
            elif data_info.merge_datasets:
                input_data = self._execute_merge(data_info)
            # --- SET ---
            elif data_info.set_datasets:
                input_data = self._load_datasets(data_info.set_datasets)
            else:
                input_data = pd.DataFrame()

            if input_data is None:
                return

            # --- Row-by-row processing when needed ---
            needs_row_processing = (
                data_info.retain_vars or data_info.do_blocks
                or data_info.has_lag or data_info.has_dif
                or data_info.array_defs
                or (data_info.by_vars and data_info.set_datasets)
            )

            if needs_row_processing and not input_data.empty:
                input_data = self._execute_row_by_row(input_data, data_info)
            else:
                # Vectorised path (original behaviour)
                for where_condition in data_info.where_conditions:
                    input_data = self.data_utils.apply_where_condition(
                        input_data, where_condition, self.expression_parser
                    )

                for assignment in data_info.variable_assignments:
                    if assignment.lower().startswith('if '):
                        input_data = self.expression_evaluator.evaluate_if_then_else(
                            assignment, input_data
                        )
                    else:
                        input_data = self.expression_evaluator.evaluate_assignment(
                            assignment, input_data
                        )

            # --- FILE / PUT ---
            if data_info.file_spec and data_info.file_spec.path:
                self._execute_file_put(input_data, data_info)

            # --- DROP / KEEP / RENAME ---
            if data_info.drop_vars:
                input_data = self.data_utils.drop_columns(input_data, data_info.drop_vars)
            if data_info.keep_vars:
                input_data = self.data_utils.select_columns(input_data, data_info.keep_vars)
            if data_info.rename_vars:
                input_data = self.data_utils.rename_columns(input_data, data_info.rename_vars)

            # --- Store result ---
            self._store_dataset(data_info.output_dataset, input_data, statement)

        except Exception as e:
            print(f"ERROR in DATA step: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    def _execute_datalines(self, statement: str) -> None:
        """Handle DATA step with DATALINES/CARDS."""
        input_data = self.data_step_parser.parse_datalines(statement)
        if input_data is not None and not input_data.empty:
            lines = statement.split('\n')
            output_dataset = None
            for line in lines:
                if line.strip().upper().startswith('DATA '):
                    m = re.match(r'data\s+([^;]+)', line, re.IGNORECASE)
                    if m:
                        output_dataset = m.group(1).strip()
                        break
            if output_dataset:
                self._store_dataset(output_dataset, input_data, statement)

    def _execute_infile(self, data_info) -> pd.DataFrame:
        """Execute INFILE/INPUT to read external file."""
        spec = data_info.infile_spec
        try:
            with open(spec.path, 'r') as f:
                all_lines = f.readlines()

            start = spec.firstobs - 1
            end = spec.obs if spec.obs else len(all_lines)
            data_lines = all_lines[start:end]

            rows: List[Dict[str, Any]] = []
            var_names = data_info.input_spec if data_info.input_spec else []

            # Detect $ for character vars
            clean_vars: List[str] = []
            char_vars: set = set()
            idx = 0
            while idx < len(var_names):
                if var_names[idx] == '$':
                    if clean_vars:
                        char_vars.add(clean_vars[-1])
                    idx += 1
                    continue
                clean_vars.append(var_names[idx])
                idx += 1

            for line in data_lines:
                line = line.rstrip('\n')
                if spec.dsd:
                    import csv
                    import io
                    reader = csv.reader(io.StringIO(line), delimiter=spec.delimiter)
                    values = next(reader, [])
                else:
                    if spec.delimiter == ' ':
                        values = line.split()
                    else:
                        values = line.split(spec.delimiter)

                row: Dict[str, Any] = {}
                for j, vname in enumerate(clean_vars):
                    if j < len(values):
                        val = values[j].strip()
                        if vname in char_vars:
                            row[vname] = val
                        else:
                            try:
                                row[vname] = float(val)
                            except ValueError:
                                row[vname] = val if spec.missover else None
                    else:
                        row[vname] = None
                rows.append(row)

            return pd.DataFrame(rows)
        except FileNotFoundError:
            print(f"ERROR: INFILE not found: {spec.path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"ERROR reading INFILE: {e}")
            return pd.DataFrame()

    def _execute_merge(self, data_info) -> Optional[pd.DataFrame]:
        """Execute MERGE with BY."""
        dfs: List[pd.DataFrame] = []
        for ds_name in data_info.merge_datasets:
            df = self._resolve_dataset(ds_name)
            if df is None:
                print(f"ERROR: Dataset {ds_name} not found for MERGE")
                return None
            dfs.append(df.copy())

        if not dfs:
            return pd.DataFrame()

        result = dfs[0]
        by_vars = data_info.merge_by or data_info.by_vars
        for df in dfs[1:]:
            if by_vars:
                result = pd.merge(result, df, on=by_vars, how='outer')
            else:
                result = pd.merge(result, df, left_index=True, right_index=True, how='outer')

        return result

    def _execute_row_by_row(self, data: pd.DataFrame, data_info) -> pd.DataFrame:
        """Row-by-row DATA step processing for RETAIN, DO, LAG, arrays, FIRST./LAST."""
        self.expression_evaluator.reset_lag_queues()

        # Register arrays
        arrays: Dict[str, List[str]] = {}
        for arr in data_info.array_defs:
            arrays[arr.name.lower()] = arr.variables
        self.expression_evaluator.register_arrays(data_info.array_defs)

        # RETAIN state
        retain_state: Dict[str, Any] = {}
        for var, init in data_info.retain_vars:
            if init is not None:
                try:
                    retain_state[var] = float(init)
                except ValueError:
                    retain_state[var] = init
            else:
                retain_state[var] = None

        # Sort by BY vars for FIRST./LAST. computation
        by_vars = data_info.by_vars
        if by_vars:
            valid_by = [v for v in by_vars if v in data.columns]
            if valid_by:
                data = data.sort_values(valid_by).reset_index(drop=True)

        result_rows: List[Dict[str, Any]] = []

        for i in range(len(data)):
            row = data.iloc[i].to_dict()

            # Apply RETAIN
            for var in retain_state:
                if var not in row or (isinstance(row.get(var), float) and np.isnan(row[var])):
                    row[var] = retain_state[var]

            # Compute FIRST. / LAST.
            if by_vars:
                for bv in by_vars:
                    if bv in data.columns:
                        first = (i == 0) or (data.iloc[i][bv] != data.iloc[i - 1][bv])
                        last = (i == len(data) - 1) or (data.iloc[i][bv] != data.iloc[i + 1][bv])
                        row[f'first_{bv}'] = 1 if first else 0
                        row[f'last_{bv}'] = 1 if last else 0

            # Process assignments
            for assignment in data_info.variable_assignments:
                if assignment.lower().startswith('if '):
                    self.expression_evaluator.evaluate_row_if(assignment, row, arrays)
                else:
                    self.expression_evaluator.evaluate_row_assignment(assignment, row, arrays)

            # Process DO blocks
            for block in data_info.do_blocks:
                self._execute_do_block(block, row, arrays)

            # Update RETAIN state
            for var in retain_state:
                if var in row:
                    retain_state[var] = row[var]

            result_rows.append(row)

        if not result_rows:
            return data

        return pd.DataFrame(result_rows)

    def _execute_do_block(
        self, block, row: Dict[str, Any],
        arrays: Optional[Dict[str, List[str]]] = None,
        max_iter: int = 10000,
    ) -> None:
        """Execute a DO block on a single row."""
        if block.loop_type == 'iterative' and block.var:
            start = int(self.expression_evaluator._evaluate_scalar_expression(
                block.start, row, arrays))
            end = int(self.expression_evaluator._evaluate_scalar_expression(
                block.end, row, arrays))
            step = int(self.expression_evaluator._evaluate_scalar_expression(
                block.step, row, arrays))
            if step == 0:
                step = 1

            for val in range(start, end + 1, step):
                row[block.var] = val
                for stmt in block.body:
                    if stmt.lower().startswith('if '):
                        self.expression_evaluator.evaluate_row_if(stmt, row, arrays)
                    else:
                        self.expression_evaluator.evaluate_row_assignment(stmt, row, arrays)

        elif block.loop_type == 'while' and block.condition:
            count = 0
            while self.expression_evaluator._evaluate_scalar_condition(
                    block.condition, row) and count < max_iter:
                for stmt in block.body:
                    if stmt.lower().startswith('if '):
                        self.expression_evaluator.evaluate_row_if(stmt, row, arrays)
                    else:
                        self.expression_evaluator.evaluate_row_assignment(stmt, row, arrays)
                count += 1

        elif block.loop_type == 'until' and block.condition:
            count = 0
            while count < max_iter:
                for stmt in block.body:
                    if stmt.lower().startswith('if '):
                        self.expression_evaluator.evaluate_row_if(stmt, row, arrays)
                    else:
                        self.expression_evaluator.evaluate_row_assignment(stmt, row, arrays)
                count += 1
                if self.expression_evaluator._evaluate_scalar_condition(
                        block.condition, row):
                    break

    def _execute_file_put(self, data: pd.DataFrame, data_info) -> None:
        """Execute FILE/PUT to write output."""
        spec = data_info.file_spec
        try:
            with open(spec.path, 'w') as f:
                for _, row in data.iterrows():
                    for put_expr in data_info.put_spec:
                        tokens = put_expr.split()
                        out_parts: List[str] = []
                        for tok in tokens:
                            if tok in row:
                                out_parts.append(str(row[tok]))
                            elif tok.startswith(("'", '"')):
                                out_parts.append(tok.strip("'\""))
                            else:
                                out_parts.append(tok)
                        line = spec.delimiter.join(out_parts) if spec.delimiter != ' ' else ' '.join(out_parts)
                        f.write(line + '\n')
            print(f"FILE written: {spec.path}")
        except Exception as e:
            print(f"ERROR writing FILE: {e}")

    # ------------------------------------------------------------------
    # PROC EXECUTION
    # ------------------------------------------------------------------
    def _execute_proc(self, statement: str) -> None:
        """Execute a PROC procedure."""
        try:
            proc_info = self.proc_parser.parse_proc(statement)

            input_data = None
            if proc_info.data_option:
                input_data = self._resolve_dataset(proc_info.data_option)
                if input_data is None:
                    print(f"ERROR: Dataset {proc_info.data_option} not found")
                    return
            else:
                if self.data_sets:
                    dataset_name = list(self.data_sets.keys())[-1]
                    input_data = self.data_sets[dataset_name]
                else:
                    if proc_info.proc_name in ('LANGUAGE', 'SQL', 'DATASETS', 'IMPORT', 'LLM', 'CVISION'):
                        input_data = pd.DataFrame()
                    else:
                        print("ERROR: No dataset available for PROC")
                        return

            if proc_info.proc_name in self.proc_implementations:
                proc_impl = self.proc_implementations[proc_info.proc_name]

                # Special handling for PROC SQL
                if proc_info.proc_name == 'SQL' and hasattr(proc_impl, 'register_dataset'):
                    for ds_name, ds_df in self.data_sets.items():
                        proc_impl.register_dataset(ds_name, ds_df)
                    for libname in self.libname_manager.libraries:
                        lib_datasets = self.libname_manager.get_library_datasets(libname)
                        for ds_name, ds_df in lib_datasets.items():
                            proc_impl.register_dataset(f"{libname}.{ds_name}", ds_df)

                # Special handling for PROC DATASETS
                if proc_info.proc_name == 'DATASETS':
                    results = proc_impl.execute(
                        input_data, proc_info,
                        dataset_manager=self.dataset_manager,
                        data_sets=self.data_sets,
                    )
                # Special handling for PROC APPEND
                elif proc_info.proc_name == 'APPEND':
                    results = proc_impl.execute(
                        input_data, proc_info,
                        dataset_manager=self.dataset_manager,
                        data_sets=self.data_sets,
                    )
                elif proc_info.proc_name == 'PRINT' and self.current_title:
                    results = proc_impl.execute(
                        input_data, proc_info,
                        dataset_manager=self.dataset_manager,
                        title=self.current_title,
                    )
                    self.current_title = None
                else:
                    # Try with full kwargs; fall back if proc doesn't accept model_store
                    try:
                        results = proc_impl.execute(
                            input_data, proc_info,
                            dataset_manager=self.dataset_manager,
                            model_store=self.model_store,
                        )
                    except TypeError:
                        results = proc_impl.execute(
                            input_data, proc_info,
                            dataset_manager=self.dataset_manager,
                        )

                # Display output
                for line in results.get('output_text', []):
                    print(line)

                if results.get('suppress_dataset_display', False):
                    self._suppress_dataset_display = True

                # Store output data
                if results.get('output_data') is not None:
                    out_data = results['output_data']
                    if isinstance(out_data, pd.DataFrame):
                        if proc_info.output_option:
                            self._store_dataset_df(proc_info.output_option, out_data)
                        elif results.get('output_dataset'):
                            self._store_dataset_df(results['output_dataset'], out_data)
                        elif results.get('overwrite_input', False):
                            if proc_info.data_option:
                                self._store_dataset_df(proc_info.data_option, out_data)

                # Store model if provided
                if results.get('model_object') and results.get('model_name'):
                    self.model_store.save(
                        results['model_name'],
                        results['model_object'],
                        metadata=results.get('model_metadata', {}),
                    )

                # Update SYSLAST
                if proc_info.output_option:
                    self.macro_processor.set_variable('SYSLAST', proc_info.output_option)

            else:
                print(f"ERROR: PROC {proc_info.proc_name} not implemented")

        except Exception as e:
            print(f"ERROR in PROC: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _resolve_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Resolve a dataset by name from memory or library."""
        if name in self.data_sets:
            return self.data_sets[name]
        if '.' in name:
            libname, ds_name = name.split('.', 1)
            loaded = self.libname_manager.load_dataset(libname, ds_name)
            if loaded is not None:
                self.data_sets[name] = loaded
                return loaded
        return None

    def _load_datasets(self, names: List[str]) -> Optional[pd.DataFrame]:
        """Load and optionally concatenate datasets."""
        dfs: List[pd.DataFrame] = []
        for name in names:
            df = self._resolve_dataset(name)
            if df is not None:
                dfs.append(df.copy())
            else:
                print(f"ERROR: Dataset {name} not found")
                return None
        if not dfs:
            return pd.DataFrame()
        if len(dfs) == 1:
            return dfs[0]
        return pd.concat(dfs, ignore_index=True)

    def _store_dataset(self, name: str, df: pd.DataFrame, statement: str) -> None:
        """Store a dataset with format metadata."""
        sas_dataset = SasDataset(name=name, dataframe=df)

        lines = statement.split('\n')
        fmt_stmts = self.format_informat_parser.extract_format_statements(lines)
        self.format_informat_parser.apply_format_statements_to_dataset(sas_dataset, fmt_stmts)
        infmt_stmts = self.format_informat_parser.extract_informat_statements(lines)
        self.format_informat_parser.apply_informat_statements_to_dataset(sas_dataset, infmt_stmts)

        self.data_sets[name] = df
        self.dataset_manager.datasets[name] = sas_dataset
        self.macro_processor.set_variable('SYSLAST', name)

        if '.' in name:
            libname, ds_name = name.split('.', 1)
            if self.libname_manager.save_dataset(libname, ds_name, df):
                print(f"Saved dataset {name} to library {libname}")
        else:
            print(f"Saved dataset {name} to library work")

    def _store_dataset_df(self, name: str, df: pd.DataFrame) -> None:
        """Store a DataFrame without format parsing."""
        self.data_sets[name] = df
        sas_ds = SasDataset(name=name, dataframe=df)
        self.dataset_manager.datasets[name] = sas_ds
        self.macro_processor.set_variable('SYSLAST', name)

    def _execute_libname(self, statement: str) -> None:
        """Execute a LIBNAME statement."""
        try:
            result = self.libname_manager.parse_libname_statement(statement)
            if result:
                libname, path = result
                if self.libname_manager.create_library(libname, path):
                    print(f"Library {libname} created and mapped to {path}")
                else:
                    print(f"ERROR: Could not create library {libname}")
            else:
                print(f"ERROR: Invalid LIBNAME statement: {statement}")
        except Exception as e:
            print(f"ERROR in LIBNAME: {e}")

    def _execute_let(self, statement: str) -> None:
        """Execute a %LET macro statement."""
        try:
            self.macro_processor._parse_let_statement(statement)
        except Exception as e:
            print(f"ERROR in %LET: {e}")

    def _execute_put(self, statement: str) -> None:
        """Execute a %PUT statement."""
        match = re.match(r'%PUT\s+(.*?);?\s*$', statement, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            expanded_text = self.macro_processor._substitute_variables(text)
            print(f"MACRO: {expanded_text}")

    def _execute_title(self, statement: str) -> None:
        """Execute a TITLE statement."""
        match = re.match(r'TITLE\s+(.*?);?\s*$', statement, re.IGNORECASE)
        if match:
            title_text = match.group(1).strip()
            if title_text.startswith('"') and title_text.endswith('"'):
                title_text = title_text[1:-1]
            elif title_text.startswith("'") and title_text.endswith("'"):
                title_text = title_text[1:-1]
            self.current_title = self.macro_processor._substitute_variables(title_text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_data_set(self, name: str) -> Optional[pd.DataFrame]:
        """Get a data set by name."""
        return self.data_sets.get(name)

    def list_data_sets(self) -> List[str]:
        """List all available data sets."""
        return list(self.data_sets.keys())

    def clear_workspace(self) -> None:
        """Clear all data sets and reset the workspace."""
        self.data_sets.clear()
        self.libraries.clear()
        self.macro_variables.clear()
        self.options.clear()
        self.model_store.clear()


# Alias for backward compatibility
StatLangInterpreter = SASInterpreter
