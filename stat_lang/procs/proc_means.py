"""
PROC MEANS Implementation for Open-SAS

This module implements SAS PROC MEANS functionality using pandas
for descriptive statistics calculations.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcMeans:
    """Implementation of SAS PROC MEANS procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC MEANS on the given data.
        
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
        
        # Get analysis variables
        var_vars = proc_info.options.get('var', [])
        if not var_vars:
            # If no VAR specified, use all numeric columns
            var_vars = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Get BY variables
        by_vars = proc_info.options.get('by', [])
        
        # Get CLASS variables (treat same as BY for grouping)
        class_vars = proc_info.options.get('class', [])
        
        # Combine BY and CLASS variables for grouping
        group_vars = by_vars + class_vars
        
        # Filter to only include variables that exist in the data
        var_vars = [var for var in var_vars if var in data.columns]
        group_vars = [var for var in group_vars if var in data.columns]
        
        if not var_vars:
            results['output_text'].append("ERROR: No valid analysis variables found.")
            return results
        
        # Check for NOPRINT option
        noprint = proc_info.options.get('noprint', False)
        
        # Calculate statistics
        if group_vars:
            # Grouped analysis
            grouped = data.groupby(group_vars)
            stats_df = grouped[var_vars].agg(['count', 'mean', 'std', 'min', 'max'])
            
            # Flatten column names
            stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
            stats_df = stats_df.reset_index()
            
            # Only add output text if NOPRINT is not specified
            if not noprint:
                results['output_text'].append("PROC MEANS - Grouped Analysis")
                results['output_text'].append("=" * 50)
            
        else:
            # Overall analysis
            stats_dict = {}
            for var in var_vars:
                var_data = data[var].dropna()
                if len(var_data) > 0:
                    stats_dict[var] = {
                        'N': len(var_data),
                        'Mean': var_data.mean(),
                        'Std Dev': var_data.std(),
                        'Minimum': var_data.min(),
                        'Maximum': var_data.max()
                    }
            
            stats_df = pd.DataFrame(stats_dict).T
            stats_df = stats_df.round(6)
            
            # Only add output text if NOPRINT is not specified
            if not noprint:
                results['output_text'].append("PROC MEANS - Descriptive Statistics")
                results['output_text'].append("=" * 50)
        
        # Only add output text if NOPRINT is not specified
        if not noprint:
            # Format output
            results['output_text'].append(f"Analysis Variables: {', '.join(var_vars)}")
            if group_vars:
                results['output_text'].append(f"Grouping Variables: {', '.join(group_vars)}")
            results['output_text'].append("")
            
            # Convert DataFrame to formatted text
            output_lines = self._format_dataframe(stats_df)
            results['output_text'].extend(output_lines)
        
        # Handle OUTPUT statement
        output_spec = proc_info.options.get('output', '')
        if output_spec:
            # Parse OUTPUT statement: out=dataset_name stat=var1 var2 ...
            output_data = self._parse_output_statement(data, group_vars, var_vars, output_spec)
            if output_data is not None:
                results['output_data'] = output_data
                # Extract output dataset name from OUTPUT statement
                out_match = re.search(r'out\s*=\s*([\w.]+)', output_spec, re.IGNORECASE)
                if out_match:
                    results['output_dataset'] = out_match.group(1)
                    # If NOPRINT is specified, don't display the output dataset
                    if noprint:
                        results['suppress_dataset_display'] = True
        
        # Set output data if requested via output_option (for backward compatibility)
        elif proc_info.output_option:
            results['output_data'] = stats_df
            results['output_dataset'] = proc_info.output_option
        
        return results
    
    def _format_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Format DataFrame for text output."""
        lines = []
        
        # Get column widths
        col_widths = {}
        for col in df.columns:
            col_widths[col] = max(len(str(col)), df[col].astype(str).str.len().max())
        
        # Create header
        header = " | ".join(f"{col:<{col_widths[col]}}" for col in df.columns)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Add rows
        for idx, row in df.iterrows():
            row_str = " | ".join(f"{str(val):<{col_widths[col]}}" for col, val in zip(df.columns, row))
            lines.append(row_str)
        
        return lines
    
    def _parse_output_statement(self, data: pd.DataFrame, group_vars: List[str], var_vars: List[str], output_spec: str) -> Optional[pd.DataFrame]:
        """
        Parse OUTPUT statement and create output dataset.
        
        Args:
            data: Input DataFrame
            group_vars: Grouping variables
            var_vars: Analysis variables
            output_spec: OUTPUT statement specification
            
        Returns:
            DataFrame with output statistics or None if parsing fails
        """
        try:
            # Parse statistics specifications: stat=var1 var2 ...
            stat_specs = {}
            
            # Find all stat=variable patterns
            stat_patterns = re.findall(r'(\w+)\s*=\s*([\w\s]+)', output_spec)
            for stat_name, var_list in stat_patterns:
                if stat_name.lower() == 'out':
                    continue  # Skip OUT= specification
                
                # Parse variable names
                vars_for_stat = [v.strip() for v in var_list.split()]
                stat_specs[stat_name.lower()] = vars_for_stat
            
            if not stat_specs:
                return None
            
            # Calculate statistics
            if group_vars:
                # Grouped analysis
                grouped = data.groupby(group_vars)
                output_data = grouped[var_vars].agg(['mean', 'min', 'max'])
                
                # Flatten column names
                output_data.columns = ['_'.join(col).strip() for col in output_data.columns.values]
                output_data = output_data.reset_index()
                
                # Rename columns according to OUTPUT specification
                new_columns = {}
                for stat_name, var_list in stat_specs.items():
                    for i, var in enumerate(var_list):
                        if i < len(var_vars):
                            # Find the corresponding column in the output
                            for col in output_data.columns:
                                if col.endswith(f'_{stat_name}') and var_vars[i] in col:
                                    new_columns[col] = var
                                    break
                
                output_data = output_data.rename(columns=new_columns)
                
                # Reorder columns: grouping variables first, then OUTPUT specification variables
                ordered_columns = []
                
                # Add grouping variables first
                for col in output_data.columns:
                    if col in group_vars:
                        ordered_columns.append(col)
                
                # Add OUTPUT specification variables in order
                for stat_name, var_list in stat_specs.items():
                    for var in var_list:
                        if var in output_data.columns and var not in ordered_columns:
                            ordered_columns.append(var)
                
                # Add any remaining columns
                for col in output_data.columns:
                    if col not in ordered_columns:
                        ordered_columns.append(col)
                
                output_data = output_data[ordered_columns]
                
            else:
                # Overall analysis
                output_data = pd.DataFrame()
                for stat_name, var_list in stat_specs.items():
                    for var in var_list:
                        if var in var_vars:
                            if stat_name == 'mean':
                                output_data[var] = [data[var].mean()]
                            elif stat_name == 'min':
                                output_data[var] = [data[var].min()]
                            elif stat_name == 'max':
                                output_data[var] = [data[var].max()]
                            elif stat_name == 'std':
                                output_data[var] = [data[var].std()]
                            elif stat_name == 'count':
                                output_data[var] = [len(data[var].dropna())]
            
            return output_data
            
        except Exception as e:
            print(f"Error parsing OUTPUT statement: {e}")
            return None
