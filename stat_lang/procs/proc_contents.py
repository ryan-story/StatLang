"""
PROC CONTENTS Implementation for Open-SAS

This module implements SAS PROC CONTENTS functionality for displaying
dataset metadata and variable information.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcContents:
    """Implementation of SAS PROC CONTENTS procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC CONTENTS on the given data.
        
        Args:
            data: Input DataFrame
            proc_info: Parsed PROC statement information
            
        Returns:
            Dictionary containing results and output data
        """
        results = {
            'output_text': [],
            'output_data': None
        }
        
        results['output_text'].append("PROC CONTENTS - Dataset Information")
        results['output_text'].append("=" * 50)
        results['output_text'].append("")
        
        # Dataset-level information
        results['output_text'].append("Dataset Information:")
        results['output_text'].append(f"  Observations: {len(data)}")
        results['output_text'].append(f"  Variables: {len(data.columns)}")
        results['output_text'].append("")
        
        # Variable-level information
        results['output_text'].append("Variable Information:")
        results['output_text'].append("-" * 80)
        
        # Create variable information table
        var_info = []
        for i, col in enumerate(data.columns, 1):
            var_type = "Character" if data[col].dtype == 'object' else "Numeric"
            var_length = "Variable" if data[col].dtype == 'object' else "8"
            
            # Calculate some basic statistics for numeric variables
            if data[col].dtype != 'object':
                non_null_count = data[col].count()
                null_count = len(data[col]) - non_null_count
            else:
                non_null_count = data[col].count()
                null_count = len(data[col]) - non_null_count
            
            var_info.append({
                'Variable': col,
                'Type': var_type,
                'Length': var_length,
                'Position': i,
                'Non-Null': non_null_count,
                'Null': null_count
            })
        
        # Format variable information
        lines = []
        lines.append(f"{'#':<3} {'Variable':<20} {'Type':<10} {'Length':<8} {'Non-Null':<10} {'Null':<8}")
        lines.append("-" * 80)
        
        for info in var_info:
            lines.append(f"{info['Position']:<3} {info['Variable']:<20} {info['Type']:<10} {info['Length']:<8} {info['Non-Null']:<10} {info['Null']:<8}")
        
        results['output_text'].extend(lines)
        
        # Create output DataFrame
        output_df = pd.DataFrame(var_info)
        results['output_data'] = output_df
        
        return results
