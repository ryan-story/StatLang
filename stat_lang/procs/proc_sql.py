"""
PROC SQL Implementation for Open-SAS

This module implements SAS PROC SQL functionality for SQL query processing
using DuckDB for fast, in-memory SQL operations.
"""

import pandas as pd
import numpy as np
import duckdb
import re
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcSQL:
    """Implementation of SAS PROC SQL procedure."""
    
    def __init__(self):
        self.conn = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize DuckDB connection."""
        try:
            self.conn = duckdb.connect()
        except Exception as e:
            print(f"Warning: Could not initialize DuckDB connection: {e}")
            self.conn = None
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC SQL on the given data.
        
        Args:
            data: Input DataFrame (not used for SQL)
            proc_info: Parsed PROC statement information
            
        Returns:
            Dictionary containing results and output data
        """
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Check if DuckDB is available
        if self.conn is None:
            results['output_text'].append("ERROR: DuckDB not available for PROC SQL.")
            results['output_text'].append("Please install: pip install duckdb")
            return results
        
        # Extract SQL statements from the PROC
        sql_statements = self._extract_sql_statements(proc_info)
        
        if not sql_statements:
            results['output_text'].append("ERROR: No SQL statements found in PROC SQL.")
            return results
        
        results['output_text'].append("PROC SQL - SQL Query Processing")
        results['output_text'].append("=" * 50)
        results['output_text'].append("Backend: DuckDB")
        results['output_text'].append("")
        
        # Execute SQL statements
        try:
            for i, sql_stmt in enumerate(sql_statements):
                results['output_text'].append(f"Statement {i+1}:")
                results['output_text'].append("-" * 20)
                results['output_text'].append(sql_stmt.strip())
                results['output_text'].append("")
                
                # Execute the SQL statement
                if sql_stmt.strip().upper().startswith('SELECT'):
                    # SELECT statement - return results
                    result_df = self.conn.execute(sql_stmt).fetchdf()
                    results['output_text'].append("Query Results:")
                    results['output_text'].append(f"Rows: {len(result_df)}, Columns: {len(result_df.columns)}")
                    results['output_text'].append("")
                    
                    # Format results as table
                    results['output_text'].extend(self._format_dataframe_as_table(result_df))
                    results['output_data'] = result_df
                    
                elif sql_stmt.strip().upper().startswith('CREATE TABLE'):
                    # CREATE TABLE statement - create new dataset
                    output_dataset = self._extract_output_dataset(sql_stmt)
                    if output_dataset:
                        # Execute the CREATE TABLE statement
                        self.conn.execute(sql_stmt)
                        results['output_text'].append(f"Created table: {output_dataset}")
                        
                        # Get the actual data from the created table
                        result_df = self.conn.execute(f"SELECT * FROM {output_dataset}").fetchdf()
                        results['output_text'].append(f"Rows: {len(result_df)}, Columns: {len(result_df.columns)}")
                        results['output_text'].append("")
                        results['output_text'].extend(self._format_dataframe_as_table(result_df))
                        results['output_data'] = result_df
                        results['output_dataset'] = output_dataset
                    else:
                        results['output_text'].append("ERROR: Could not determine output dataset name from CREATE TABLE statement.")
                        
                elif sql_stmt.strip().upper().startswith('INSERT INTO'):
                    # INSERT statement
                    self.conn.execute(sql_stmt)
                    results['output_text'].append("INSERT statement executed successfully.")
                    
                else:
                    # Other SQL statements
                    self.conn.execute(sql_stmt)
                    results['output_text'].append("SQL statement executed successfully.")
                
                results['output_text'].append("")
                
        except Exception as e:
            results['output_text'].append(f"ERROR: SQL execution failed: {str(e)}")
            return results
        
        return results
    
    def _extract_sql_statements(self, proc_info: ProcStatement) -> List[str]:
        """Extract SQL statements from PROC SQL."""
        sql_statements = []
        
        # Get SQL statements from the PROC statements
        for stmt in proc_info.statements:
            stmt = stmt.strip()
            if stmt and not stmt.upper().startswith('PROC SQL') and stmt.upper() != 'RUN;':
                # Remove trailing semicolon if present
                if stmt.endswith(';'):
                    stmt = stmt[:-1]
                sql_statements.append(stmt)
        
        return sql_statements
    
    def _register_datasets(self, proc_info: ProcStatement):
        """Register all available datasets with DuckDB."""
        # This would need access to the library manager
        # For now, we'll implement a basic version
        # In a full implementation, this would iterate through all libraries and datasets
        
        # Register the current dataset if available
        if hasattr(proc_info, 'data_option') and proc_info.data_option:
            dataset_name = proc_info.data_option
            if '.' in dataset_name:
                libname, table_name = dataset_name.split('.', 1)
                # This would need access to the actual DataFrame
                # For now, we'll create a placeholder
                pass
    
    def _extract_output_dataset(self, sql_stmt: str) -> Optional[str]:
        """Extract output dataset name from CREATE TABLE statement."""
        # Look for patterns like "CREATE TABLE work.summary AS" or "CREATE TABLE out=work.summary AS"
        patterns = [
            r'CREATE\s+TABLE\s+([\w.]+)\s+AS',
            r'CREATE\s+TABLE\s+out\s*=\s*([\w.]+)\s+AS',
            r'CREATE\s+TABLE\s+([\w.]+)\s+AS\s+SELECT'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sql_stmt, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _format_dataframe_as_table(self, df: pd.DataFrame, max_rows: int = 20) -> List[str]:
        """Format DataFrame as a table for display."""
        if df.empty:
            return ["No data returned."]
        
        # Limit rows for display
        display_df = df.head(max_rows)
        
        # Convert to string representation
        output = []
        
        # Get column names
        col_names = list(display_df.columns)
        col_widths = [max(len(str(col)), len(str(display_df[col].iloc[i] if i < len(display_df) else ''))) 
                     for i, col in enumerate(col_names)]
        
        # Create header
        header = " | ".join(f"{col:<{width}}" for col, width in zip(col_names, col_widths))
        output.append(header)
        output.append("-" * len(header))
        
        # Add rows
        for _, row in display_df.iterrows():
            row_str = " | ".join(f"{str(val):<{width}}" for val, width in zip(row, col_widths))
            output.append(row_str)
        
        # Add summary if truncated
        if len(df) > max_rows:
            output.append(f"... and {len(df) - max_rows} more rows")
        
        return output
    
    def register_dataset(self, table_name: str, df: pd.DataFrame):
        """Register a dataset with DuckDB."""
        if self.conn is not None:
            try:
                # Convert lib.dataset to lib_dataset format
                if '.' in table_name:
                    libname, dataset_name = table_name.split('.', 1)
                    table_name = f"{libname}_{dataset_name}"
                
                self.conn.register(table_name, df)
            except Exception as e:
                print(f"Warning: Could not register dataset {table_name}: {e}")
    
    def close(self):
        """Close DuckDB connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
