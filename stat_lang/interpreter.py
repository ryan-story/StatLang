"""
StatLang interpreter core.

This module contains the core interpreter class that parses and executes
StatLang code using Python as the backend.
"""

import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .parser.data_step_parser import DataStepParser
from .parser.macro_parser import MacroParser
from .parser.proc_parser import ProcParser
from .procs import (
    ProcBoost,
    ProcCluster,
    ProcContents,
    ProcCorr,
    ProcFactor,
    ProcForest,
    ProcFreq,
    ProcLanguage,
    ProcLogit,
    ProcMeans,
    ProcNpar1way,
    ProcPrint,
    ProcReg,
    ProcSort,
    ProcSQL,
    ProcSurveySelect,
    ProcTimeseries,
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
        
        # Initialize title tracking
        self.current_title: Optional[str] = None
        
        # Initialize PROC implementations
        self.proc_implementations = {
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
            'ARIMA': ProcTimeseries(),  # Alias for TIMESERIES
            'TREE': ProcTree(),
            'FOREST': ProcForest(),
            'BOOST': ProcBoost(),
            'LANGUAGE': ProcLanguage(),
            'SQL': ProcSQL(),
            'SURVEYSELECT': ProcSurveySelect(),
            'REG': ProcReg()
        }
        
    def run_file(self, file_path: str) -> None:
        """
        Execute a .statlang file.
        
        Args:
            file_path: Path to the .statlang file to execute
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r') as f:
            sas_code = f.read()
            
        self.run_code(sas_code)
    
    def run_code(self, sas_code: str) -> None:
        """
        Execute SAS code string.
        
        Args:
            sas_code: SAS code to execute
        """
        # Split code into lines for macro processing
        code_lines = sas_code.split('\n')
        
        # Process macro statements first
        expanded_code_lines = self.macro_processor.process_macro_statements(code_lines)
        
        # Join back into code string
        processed_code = '\n'.join(expanded_code_lines)
        
        # Remove comments and clean up code
        cleaned_code = self._clean_code(processed_code)
        
        # Split into statements
        statements = self._split_statements(cleaned_code)
        
        # Execute each statement
        for statement in statements:
            if statement.strip():
                self._execute_statement(statement.strip())
    
    def _clean_code(self, code: str) -> str:
        """Remove comments and clean up SAS code."""
        # Remove /* */ comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Only remove /* */ comments, not single * comments
        # This prevents arithmetic operations like salary * 0.1 from being treated as comments
        return code
    
    def _split_statements(self, code: str) -> List[str]:
        """Split SAS code into individual statements."""
        # Debug: Code splitting
        lines = code.split('\n')
        statements = []
        current_statement = ""
        in_datalines = False
        in_data_step = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_statement:
                    current_statement += '\n'
                continue
                
            # Check for DATA step start
            if line.upper().startswith('DATA '):
                in_data_step = True
                current_statement = line
                continue
            elif line.upper().startswith('PROC '):
                # End current statement if in DATA step
                if in_data_step and current_statement.strip():
                    statements.append(current_statement.strip())
                    current_statement = ""
                    in_data_step = False
                    in_datalines = False
                # Start new PROC statement
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
                # Add intermediate PROC statements to current statement
                current_statement += '\n' + line
                continue
                
            # Check for DATALINES/CARDS
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
            
            # Regular statement processing
            if in_data_step:
                current_statement += '\n' + line
            elif line.endswith(';'):
                # Check if this line starts with a keyword (even if indented)
                line_upper = line.strip().upper()
                if line_upper.startswith(('TABLES', 'VAR', 'BY', 'CLASS', 'MODEL', 'OUTPUT', 'WHERE', 'TITLE')):
                    # This is a new statement, finish the current one first
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
                # Check if this is a new statement (starts with a keyword, even if indented)
                line_upper = line.strip().upper()
                if line_upper.startswith(('TABLES', 'VAR', 'BY', 'CLASS', 'MODEL', 'OUTPUT', 'WHERE', 'TITLE')):
                    if current_statement.strip():
                        statements.append(current_statement.strip())
                    current_statement = line.strip()
                else:
                    current_statement += ' ' + line
        
        # Add the last statement if it doesn't end with semicolon
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        return statements
    
    def _execute_statement(self, statement: str) -> None:
        """Execute a single SAS statement."""
        statement = statement.strip()
        if not statement:
            return
        
        # Determine statement type and execute
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
        elif statement.upper().startswith('%MACRO '):
            # Macro definitions are handled by the macro processor
            pass
        elif statement.upper().startswith('%MEND'):
            # Macro definitions are handled by the macro processor
            pass
        elif statement.upper().startswith('%IF '):
            # Macro conditional logic is handled by the macro processor
            pass
        elif statement.upper().startswith('%DO '):
            # Macro loops are handled by the macro processor
            pass
        elif statement.upper().startswith('%END'):
            # Macro loops are handled by the macro processor
            pass
        elif statement.upper().startswith('%INCLUDE '):
            # Include statements are handled by the macro processor
            pass
        elif statement.startswith('%') and '(' in statement:
            # Macro calls are handled by the macro processor
            pass
        elif statement.upper() == 'RUN;':
            # RUN statement - currently no-op, but could be used for validation
            pass
        else:
            print(f"Warning: Unsupported statement: {statement}")
    
    def _execute_data_step(self, statement: str) -> None:
        """Execute a DATA step."""
        # Debug: Executing DATA step
        
        try:
            # Check if this is a complete DATA step with DATALINES
            if 'datalines' in statement.lower() or 'cards' in statement.lower():
                # Parse DATALINES directly
                input_data = self.data_step_parser.parse_datalines(statement)
                if input_data is not None and not input_data.empty:
                    # Extract output dataset name
                    lines = statement.split('\n')
                    output_dataset = None
                    for line in lines:
                        if line.strip().upper().startswith('DATA '):
                            match = re.match(r'data\s+([^;]+)', line, re.IGNORECASE)
                            if match:
                                output_dataset = match.group(1).strip()
                                break
                    
                    if output_dataset:
                        # Create SAS dataset with format metadata
                        sas_dataset = SasDataset(name=output_dataset, dataframe=input_data)
                        
                        # Parse and apply FORMAT statements
                        format_statements = self.format_informat_parser.extract_format_statements(lines)
                        self.format_informat_parser.apply_format_statements_to_dataset(sas_dataset, format_statements)
                        
                        # Parse and apply INFORMAT statements
                        informat_statements = self.format_informat_parser.extract_informat_statements(lines)
                        self.format_informat_parser.apply_informat_statements_to_dataset(sas_dataset, informat_statements)
                        
                        # Store the result (both as DataFrame and SAS dataset)
                        self.data_sets[output_dataset] = input_data
                        self.dataset_manager.datasets[output_dataset] = sas_dataset
                        
                        # Save to library if it's a library.dataset format; else note work library
                        if '.' in output_dataset:
                            libname, dataset_name = output_dataset.split('.', 1)
                            if self.libname_manager.save_dataset(libname, dataset_name, input_data):
                                print(f"Saved dataset {output_dataset} to library {libname}")
                        else:
                            print(f"Saved dataset {output_dataset} to library work")
                        return
            else:
                # Parse the DATA step normally
                data_info = self.data_step_parser.parse_data_step(statement)
                
                # Get input data
                input_data = None
                if data_info.set_datasets:
                    # For now, just use the first dataset
                    dataset_name = data_info.set_datasets[0]
                    
                    # Check if it's in memory first
                    if dataset_name in self.data_sets:
                        input_data = self.data_sets[dataset_name].copy()
                    else:
                        # Try to load from library
                        if '.' in dataset_name:
                            libname, lib_dataset_name = dataset_name.split('.', 1)
                            loaded_data = self.libname_manager.load_dataset(libname, lib_dataset_name)
                            if loaded_data is not None:
                                input_data = loaded_data.copy()
                                # Don't store the copy back - keep the original dataset intact
                            else:
                                print(f"ERROR: Dataset {dataset_name} not found in library {libname}")
                                return
                        else:
                            print(f"ERROR: Dataset {dataset_name} not found")
                            return
                else:
                    # Create empty dataset
                    input_data = pd.DataFrame()
                
                if input_data is not None:
                    # Apply WHERE conditions
                    for where_condition in data_info.where_conditions:
                        # Debug: Applying WHERE condition
                        input_data = self.data_utils.apply_where_condition(
                            input_data, where_condition, self.expression_parser
                        )
                    
                    # Apply variable assignments
                    for assignment in data_info.variable_assignments:
                        # Debug: Processing assignment
                        if assignment.lower().startswith('if '):
                            # Handle IF/THEN/ELSE statements
                            input_data = self.expression_evaluator.evaluate_if_then_else(assignment, input_data)
                        else:
                            # Handle regular assignments
                            input_data = self.expression_evaluator.evaluate_assignment(assignment, input_data)
                    
                    # Apply DROP/KEEP
                    if data_info.drop_vars:
                        input_data = self.data_utils.drop_columns(input_data, data_info.drop_vars)
                    
                    if data_info.keep_vars:
                        input_data = self.data_utils.select_columns(input_data, data_info.keep_vars)
                    
                    # Apply RENAME
                    if data_info.rename_vars:
                        input_data = self.data_utils.rename_columns(input_data, data_info.rename_vars)
                    
                    # Create SAS dataset with format metadata
                    sas_dataset = SasDataset(name=data_info.output_dataset, dataframe=input_data)
                    
                    # Parse and apply FORMAT statements
                    format_statements = self.format_informat_parser.extract_format_statements(statement.split('\n'))
                    self.format_informat_parser.apply_format_statements_to_dataset(sas_dataset, format_statements)
                    
                    # Parse and apply INFORMAT statements
                    informat_statements = self.format_informat_parser.extract_informat_statements(statement.split('\n'))
                    self.format_informat_parser.apply_informat_statements_to_dataset(sas_dataset, informat_statements)
                    
                    # Store the result (both as DataFrame and SAS dataset)
                    self.data_sets[data_info.output_dataset] = input_data
                    self.dataset_manager.datasets[data_info.output_dataset] = sas_dataset
                    
                    # Save to library if it's a library.dataset format
                    if '.' in data_info.output_dataset:
                        libname, dataset_name = data_info.output_dataset.split('.', 1)
                        if self.libname_manager.save_dataset(libname, dataset_name, input_data):
                            print(f"Saved dataset {data_info.output_dataset} to library {libname}")
                    
                    # Debug: Created dataset
            
        except Exception as e:
            print(f"ERROR in DATA step: {e}")
            import traceback
            traceback.print_exc()
    
    def _execute_proc(self, statement: str) -> None:
        """Execute a PROC procedure."""
        # Debug: Executing PROC
        
        try:
            # Parse the PROC statement
            proc_info = self.proc_parser.parse_proc(statement)
            
            # Get input data
            input_data = None
            if proc_info.data_option:
                dataset_name = proc_info.data_option
                
                # Check if it's in memory first
                if dataset_name in self.data_sets:
                    input_data = self.data_sets[dataset_name]
                else:
                    # Try to load from library
                    if '.' in dataset_name:
                        libname, lib_dataset_name = dataset_name.split('.', 1)
                        loaded_data = self.libname_manager.load_dataset(libname, lib_dataset_name)
                        if loaded_data is not None:
                            input_data = loaded_data
                            # Also store in memory for future use
                            self.data_sets[dataset_name] = input_data
                        else:
                            print(f"ERROR: Dataset {dataset_name} not found in library {libname}")
                            return
                    else:
                        print(f"ERROR: Dataset {dataset_name} not found")
                        return
            else:
                # Use the most recently created dataset
                if self.data_sets:
                    dataset_name = list(self.data_sets.keys())[-1]
                    input_data = self.data_sets[dataset_name]
                else:
                    # Some PROCs don't require datasets (e.g., PROC LANGUAGE, PROC SQL)
                    if proc_info.proc_name in ['LANGUAGE', 'SQL']:
                        input_data = pd.DataFrame()  # Empty DataFrame for procedures that don't need data
                    else:
                        print("ERROR: No dataset available for PROC")
                        return
            
            # Execute the appropriate PROC
            if proc_info.proc_name in self.proc_implementations:
                proc_impl = self.proc_implementations[proc_info.proc_name]
                # Type ignore: proc_impl is dynamically typed but has execute method
                
                # Special handling for PROC SQL - register all datasets
                if proc_info.proc_name == 'SQL' and hasattr(proc_impl, 'register_dataset'):
                    # Register all datasets from memory
                    for dataset_name, dataset_df in self.data_sets.items():
                        proc_impl.register_dataset(dataset_name, dataset_df)
                    
                    # Register datasets from libraries
                    for libname in self.libname_manager.libraries:
                        lib_datasets = self.libname_manager.get_library_datasets(libname)
                        for dataset_name, dataset_df in lib_datasets.items():
                            full_name = f"{libname}.{dataset_name}"
                            proc_impl.register_dataset(full_name, dataset_df)
                
                # Pass title to PROC PRINT if available
                if proc_info.proc_name == 'PRINT' and self.current_title:
                    results = proc_impl.execute(input_data, proc_info, dataset_manager=self.dataset_manager, title=self.current_title)  # type: ignore[attr-defined]
                    # Clear the title after use
                    self.current_title = None
                else:
                    results = proc_impl.execute(input_data, proc_info, dataset_manager=self.dataset_manager)  # type: ignore[attr-defined]
                
                # Display output
                for line in results.get('output_text', []):
                    print(line)
                
                # Check if dataset display should be suppressed
                if results.get('suppress_dataset_display', False):
                    self._suppress_dataset_display = True
                
                # Store output data if created
                if results.get('output_data') is not None:
                    if proc_info.output_option:
                        # OUT= specified: create new dataset
                        self.data_sets[proc_info.output_option] = results['output_data']
                    elif results.get('output_dataset'):
                        # Output dataset specified in results
                        self.data_sets[results['output_dataset']] = results['output_data']
                    elif results.get('overwrite_input', False):
                        # No OUT= specified: overwrite input dataset (for PROC SORT)
                        if proc_info.data_option:
                            self.data_sets[proc_info.data_option] = results['output_data']
            else:
                print(f"ERROR: PROC {proc_info.proc_name} not implemented")
                
        except Exception as e:
            print(f"ERROR in PROC: {e}")
    
    def _execute_libname(self, statement: str) -> None:
        """Execute a LIBNAME statement."""
        # Debug: Executing LIBNAME
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
        # Debug: Executing %LET
        try:
            # Use the new macro processor
            self.macro_processor._parse_let_statement(statement)
            print("Macro variable set successfully")
        except Exception as e:
            print(f"ERROR in %LET: {e}")
    
    def _execute_put(self, statement: str) -> None:
        """Execute a %PUT statement."""
        # Extract text to print
        match = re.match(r'%PUT\s+(.*?);?\s*$', statement, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            # Substitute macro variables
            expanded_text = self.macro_processor._substitute_variables(text)
            print(f"MACRO: {expanded_text}")
        else:
            print(f"ERROR: Invalid %PUT statement: {statement}")
    
    def _execute_title(self, statement: str) -> None:
        """Execute a TITLE statement."""
        # Extract title text
        match = re.match(r'TITLE\s+(.*?);?\s*$', statement, re.IGNORECASE)
        if match:
            title_text = match.group(1).strip()
            # Remove quotes if present
            if title_text.startswith('"') and title_text.endswith('"'):
                title_text = title_text[1:-1]
            elif title_text.startswith("'") and title_text.endswith("'"):
                title_text = title_text[1:-1]
            
            # Substitute macro variables
            expanded_title = self.macro_processor._substitute_variables(title_text)
            
            # Store the title for the next PROC
            self.current_title = expanded_title
        else:
            print(f"ERROR: Invalid TITLE statement: {statement}")
    
    def get_data_set(self, name: str) -> Optional[pd.DataFrame]:
        """
        Get a data set by name.
        
        Args:
            name: Name of the data set
            
        Returns:
            DataFrame if found, None otherwise
        """
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
