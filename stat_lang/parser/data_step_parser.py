"""
DATA Step Parser for Open-SAS

This module parses SAS DATA step syntax and converts it to executable
Python operations on DataFrames.
"""

import re
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DataStepStatement:
    """Represents a parsed DATA step statement."""
    type: str
    content: str
    line_number: int


@dataclass
class DataStepInfo:
    """Information about a DATA step."""
    output_dataset: str
    statements: List[DataStepStatement]
    set_datasets: List[str]
    where_conditions: List[str]
    variable_assignments: List[str]
    drop_vars: List[str]
    keep_vars: List[str]
    rename_vars: Dict[str, str]
    by_vars: List[str]


class DataStepParser:
    """Parser for SAS DATA step syntax."""
    
    def __init__(self):
        self.current_line = 0
        
    def parse_data_step(self, code: str) -> DataStepInfo:
        """
        Parse a complete DATA step.
        
        Args:
            code: The DATA step code to parse
            
        Returns:
            DataStepInfo object containing parsed information
        """
        lines = code.split('\n')
        statements = []
        
        # Find the DATA statement
        data_statement = None
        output_dataset = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            if line.upper().startswith('DATA '):
                data_statement = line
                # Extract output dataset name
                match = re.match(r'data\s+([^;]+)', line, re.IGNORECASE)
                if match:
                    output_dataset = match.group(1).strip()
                break
        
        if not data_statement:
            raise ValueError("No DATA statement found")
            
        # Parse remaining statements
        set_datasets = []
        where_conditions = []
        variable_assignments = []
        drop_vars = []
        keep_vars = []
        rename_vars = {}
        by_vars = []
        
        # First pass: combine multi-line assignments
        # Simple approach: join all lines and then split by semicolons
        full_text = ' '.join(lines)
        
        # Find the DATA step content (between DATA and RUN)
        data_start = full_text.upper().find('DATA ')
        if data_start == -1:
            raise ValueError("No DATA statement found")
        
        # Find the content after DATA statement
        content_start = full_text.find(';', data_start) + 1
        run_pos = full_text.upper().find('RUN;', content_start)
        if run_pos == -1:
            raise ValueError("No RUN statement found")
        
        data_content = full_text[content_start:run_pos].strip()
        # Split by semicolons to get individual statements
        statements = [stmt.strip() for stmt in data_content.split(';') if stmt.strip()]
        
        # Further split statements that contain multiple assignments
        final_statements = []
        for stmt in statements:
            # Check if statement contains multiple assignments (multiple '=' signs)
            if stmt.count('=') > 1:
                # Split by '=' and reconstruct assignments
                parts = stmt.split('=')
                if len(parts) >= 3:
                    # First assignment
                    first_var = parts[0].strip()
                    first_value = parts[1].strip()
                    final_statements.append(f"{first_var} = {first_value}")
                    
                    # Remaining assignments
                    for i in range(2, len(parts)):
                        if i < len(parts) - 1:
                            var = parts[i].strip()
                            value = parts[i + 1].strip()
                            final_statements.append(f"{var} = {value}")
                else:
                    final_statements.append(stmt)
            else:
                final_statements.append(stmt)
        combined_lines = final_statements
        
        # Second pass: parse the combined lines
        for line in combined_lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse SET statement
            if line.upper().startswith('SET '):
                set_match = re.match(r'set\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if set_match:
                    datasets = [ds.strip() for ds in set_match.group(1).split()]
                    set_datasets.extend(datasets)
                    
            # Parse WHERE statement
            elif line.upper().startswith('WHERE '):
                where_match = re.match(r'where\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if where_match:
                    where_conditions.append(where_match.group(1).strip())
                    
            # Parse IF/THEN/ELSE statements
            elif line.upper().startswith('IF '):
                # For now, treat as variable assignment
                variable_assignments.append(line)
                
            # Parse variable assignments
            elif '=' in line and not line.upper().startswith(('DROP', 'KEEP', 'RENAME', 'BY')):
                variable_assignments.append(line)
                
            # Parse DROP statement
            elif line.upper().startswith('DROP '):
                drop_match = re.match(r'drop\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if drop_match:
                    vars_list = [v.strip() for v in drop_match.group(1).split()]
                    drop_vars.extend(vars_list)
                    
            # Parse KEEP statement
            elif line.upper().startswith('KEEP '):
                keep_match = re.match(r'keep\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if keep_match:
                    vars_list = [v.strip() for v in keep_match.group(1).split()]
                    keep_vars.extend(vars_list)
                    
            # Parse RENAME statement
            elif line.upper().startswith('RENAME '):
                rename_match = re.match(r'rename\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if rename_match:
                    # Parse rename pairs like old=new
                    rename_pairs = rename_match.group(1).split()
                    for pair in rename_pairs:
                        if '=' in pair:
                            old, new = pair.split('=', 1)
                            rename_vars[old.strip()] = new.strip()
                            
            # Parse BY statement
            elif line.upper().startswith('BY '):
                by_match = re.match(r'by\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if by_match:
                    vars_list = [v.strip() for v in by_match.group(1).split()]
                    by_vars.extend(vars_list)
        
        # Debug: Parsed assignments are available in variable_assignments
        
        return DataStepInfo(
            output_dataset=output_dataset,
            statements=statements,
            set_datasets=set_datasets,
            where_conditions=where_conditions,
            variable_assignments=variable_assignments,
            drop_vars=drop_vars,
            keep_vars=keep_vars,
            rename_vars=rename_vars,
            by_vars=by_vars
        )
    
    def parse_datalines(self, code: str) -> pd.DataFrame:
        """
        Parse DATALINES/CARDS section to create a DataFrame.
        
        Args:
            code: The DATALINES section code
            
        Returns:
            DataFrame created from the data
        """
        lines = code.split('\n')
        data_lines = []
        in_datalines = False
        input_statement = None
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('INPUT '):
                # Remove semicolon if present
                if line.endswith(';'):
                    input_statement = line[:-1]
                else:
                    input_statement = line
                continue
            elif line.upper() in ['DATALINES;', 'CARDS;']:
                in_datalines = True
                continue
            elif line == ';' and in_datalines:
                break
            elif in_datalines and line:
                data_lines.append(line)
        
        if not data_lines or not input_statement:
            return pd.DataFrame()
            
        # Parse INPUT statement
        input_parts = input_statement[6:].strip().split()  # Remove 'INPUT '
        var_names = []
        var_types = {}
        
        i = 0
        while i < len(input_parts):
            part = input_parts[i]
            if i + 1 < len(input_parts) and input_parts[i + 1] == '$':
                # Character variable
                var_names.append(part)
                var_types[part] = 'str'
                i += 2  # Skip the $ token
            else:
                # Numeric variable
                var_names.append(part)
                var_types[part] = 'float'
                i += 1
        
        # Parse data lines
        data_rows = []
        for line in data_lines:
            values = line.split()
            if len(values) == len(var_names):
                row = {}
                for i, var_name in enumerate(var_names):
                    if var_types[var_name] == 'str':
                        row[var_name] = values[i]
                    else:
                        try:
                            row[var_name] = float(values[i])
                        except ValueError:
                            row[var_name] = None
                data_rows.append(row)
        
        return pd.DataFrame(data_rows)
