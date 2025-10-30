"""
Notebook-Aware SAS Interpreter

This module provides a specialized SAS interpreter optimized
for notebook environments with rich output and visualization.
"""

import io
import json
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, List, Optional
from .interpreter import SASInterpreter


class NotebookSASInterpreter(SASInterpreter):
    """SAS Interpreter optimized for notebook environments."""
    
    def __init__(self):
        super().__init__()
        self.execution_history = []
        self.output_buffer = io.StringIO()
        self.error_buffer = io.StringIO()
    
    def run_code(self, sas_code: str) -> Dict[str, Any]:
        """
        Run SAS code and return structured results for notebook display.
        
        Args:
            sas_code: SAS code to execute
            
        Returns:
            Dictionary containing execution results
        """
        # Clear buffers
        self.output_buffer = io.StringIO()
        self.error_buffer = io.StringIO()
        
        # Capture output
        with redirect_stdout(self.output_buffer), redirect_stderr(self.error_buffer):
            super().run_code(sas_code)
        
        # Get captured output
        output = self.output_buffer.getvalue()
        errors = self.error_buffer.getvalue()
        
        # Get datasets created/modified in this execution
        datasets = self._get_datasets_info()
        
        # Get PROC results if any
        proc_results = self._get_proc_results()
        
        # Create execution result
        result = {
            'success': len(errors) == 0,
            'output': output,
            'errors': errors,
            'datasets': datasets,
            'proc_results': proc_results,
            'code': sas_code
        }
        
        # Add to execution history
        self.execution_history.append(result)
        
        return result
    
    def _get_datasets_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all datasets in the interpreter."""
        datasets = {}
        
        for name, df in self.data_sets.items():
            datasets[name] = {
                'name': name,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'head': df.head(10).to_dict('records') if not df.empty else [],
                'tail': df.tail(5).to_dict('records') if not df.empty else [],
                'memory_usage': df.memory_usage(deep=True).sum(),
                'null_counts': df.isnull().sum().to_dict(),
                'summary_stats': self._get_summary_stats(df) if not df.empty else {}
            }
        
        return datasets
    
    def _get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for numeric columns."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return {}
        
        return df[numeric_cols].describe().to_dict()
    
    def _get_proc_results(self) -> List[Dict[str, Any]]:
        """Get results from PROC procedures executed in this session."""
        # This would be enhanced to capture specific PROC outputs
        # For now, return empty list
        return []
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get a dataset by name with additional metadata."""
        df = super().get_data_set(name)
        if df is None:
            return None
        
        return df
    
    def display_dataset(self, name: str, max_rows: int = 10) -> Dict[str, Any]:
        """
        Get dataset information formatted for notebook display.
        
        Args:
            name: Dataset name
            max_rows: Maximum number of rows to display
            
        Returns:
            Dictionary with display information
        """
        df = self.get_dataset(name)
        if df is None:
            return {'error': f'Dataset {name} not found'}
        
        return {
            'name': name,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'head': df.head(max_rows).to_dict('records'),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
    
    def create_dataset_summary(self, name: str) -> str:
        """Create a formatted summary of a dataset."""
        info = self.display_dataset(name)
        if 'error' in info:
            return f"Error: {info['error']}"
        
        summary = f"""
Dataset: {info['name']}
Shape: {info['shape'][0]} observations, {info['shape'][1]} variables
Memory Usage: {info['memory_usage']:,} bytes

Columns:
"""
        
        for col, dtype in info['dtypes'].items():
            null_count = info['null_counts'].get(col, 0)
            summary += f"  {col}: {dtype} ({null_count} nulls)\n"
        
        return summary
    
    def export_dataset(self, name: str, format: str = 'json') -> str:
        """
        Export dataset in various formats for notebook display.
        
        Args:
            name: Dataset name
            format: Export format ('json', 'csv', 'html')
            
        Returns:
            Exported data as string
        """
        df = self.get_dataset(name)
        if df is None:
            return f"Dataset {name} not found"
        
        if format == 'json':
            return df.to_json(orient='records', indent=2)
        elif format == 'csv':
            return df.to_csv(index=False)
        elif format == 'html':
            return df.to_html(index=False, classes='table table-striped')
        else:
            return f"Unsupported format: {format}"
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history."""
        return self.execution_history.copy()
    
    def clear_history(self):
        """Clear the execution history."""
        self.execution_history.clear()
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get a summary of the current workspace."""
        return {
            'datasets': list(self.data_sets.keys()),
            'libraries': list(self.libname_manager.list_libraries().keys()),
            'macro_variables': dict(self.macro_parser.macro_variables),
            'execution_count': len(self.execution_history)
        }
