"""
PROC SORT Implementation for Open-SAS

This module implements SAS PROC SORT functionality for sorting
datasets by specified variables.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcSort:
    """Implementation of SAS PROC SORT procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC SORT on the given data.
        
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
        
        # Get BY variables
        by_vars = proc_info.options.get('by', [])
        if not by_vars:
            results['output_text'].append("ERROR: BY statement required for PROC SORT.")
            return results
        
        # Get ascending/descending information
        by_ascending = proc_info.options.get('by_ascending', [True] * len(by_vars))
        
        # Ensure by_ascending has the same length as by_vars
        if len(by_ascending) != len(by_vars):
            by_ascending = [True] * len(by_vars)
        
        # Filter to only include variables that exist in the data
        valid_by_vars = []
        valid_ascending = []
        for i, var in enumerate(by_vars):
            if var in data.columns:
                valid_by_vars.append(var)
                valid_ascending.append(by_ascending[i])
        
        if not valid_by_vars:
            results['output_text'].append("ERROR: No valid BY variables found in data.")
            return results
        
        # Check for NODUPKEY option
        nodupkey = proc_info.options.get('nodupkey', False)
        
        # Sort the data with proper ascending/descending order
        sorted_data = data.sort_values(by=valid_by_vars, ascending=valid_ascending)
        
        # Handle NODUPKEY option
        if nodupkey:
            # Remove duplicate observations based on BY variables
            sorted_data = sorted_data.drop_duplicates(subset=valid_by_vars)
            results['output_text'].append(f"PROC SORT - Sorted with NODUPKEY option")
        else:
            results['output_text'].append("PROC SORT - Dataset Sorted")
        
        results['output_text'].append("=" * 50)
        
        # Create sort order description
        sort_order_desc = []
        for i, var in enumerate(valid_by_vars):
            order = "DESC" if not valid_ascending[i] else "ASC"
            sort_order_desc.append(f"{var} ({order})")
        
        results['output_text'].append(f"BY Variables: {', '.join(sort_order_desc)}")
        results['output_text'].append(f"Observations: {len(sorted_data)}")
        results['output_text'].append("")
        
        # Handle dataset output behavior
        if proc_info.output_option:
            # OUT= specified: preserve input dataset, create new sorted dataset
            results['output_data'] = sorted_data
            results['output_dataset'] = proc_info.output_option
            results['overwrite_input'] = False
        else:
            # No OUT= specified: overwrite input dataset with sorted data
            results['output_data'] = sorted_data
            results['overwrite_input'] = True
            # The input dataset name will be determined by the interpreter
        
        return results
