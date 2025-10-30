"""
PROC CORR Implementation for Open-SAS

This module implements SAS PROC CORR functionality for correlation analysis
including Pearson, Spearman, and Kendall correlation coefficients.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcCorr:
    """Implementation of SAS PROC CORR procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC CORR on the given data.
        
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
        
        # Get VAR variables
        var_vars = proc_info.options.get('var', [])
        if not var_vars:
            # If no VAR specified, use all numeric columns
            var_vars = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to only include variables that exist in the data
        var_vars = [var for var in var_vars if var in data.columns]
        
        if not var_vars:
            results['output_text'].append("ERROR: No valid numeric variables found for correlation analysis.")
            return results
        
        if len(var_vars) < 2:
            results['output_text'].append("ERROR: At least 2 variables required for correlation analysis.")
            return results
        
        # Get correlation method (default: pearson)
        method = proc_info.options.get('method', 'pearson').lower()
        if method not in ['pearson', 'spearman', 'kendall']:
            method = 'pearson'
        
        # Get WITH variables if specified
        with_vars = proc_info.options.get('with', [])
        
        results['output_text'].append("PROC CORR - Correlation Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Method: {method.title()}")
        results['output_text'].append("")
        
        # Select data for analysis
        if with_vars:
            # WITH option: correlate VAR variables with WITH variables
            with_vars = [var for var in with_vars if var in data.columns]
            if not with_vars:
                results['output_text'].append("ERROR: No valid WITH variables found.")
                return results
            
            analysis_data = data[var_vars + with_vars].select_dtypes(include=[np.number])
            corr_matrix = self._compute_correlation_matrix(analysis_data, method)
            
            # Create correlation table between VAR and WITH variables
            var_with_corr = corr_matrix.loc[var_vars, with_vars]
            results['output_text'].extend(self._format_with_correlation(var_with_corr, var_vars, with_vars, method))
            
            # Store correlation matrix
            results['output_data'] = var_with_corr
            
        else:
            # Standard correlation matrix for all VAR variables
            analysis_data = data[var_vars].select_dtypes(include=[np.number])
            corr_matrix = self._compute_correlation_matrix(analysis_data, method)
            
            # Format and display correlation matrix
            results['output_text'].extend(self._format_correlation_matrix(corr_matrix, method))
            
            # Store correlation matrix
            results['output_data'] = corr_matrix
        
        return results
    
    def _compute_correlation_matrix(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Compute correlation matrix using specified method."""
        # Remove rows with any missing values for correlation calculation
        clean_data = data.dropna()
        
        if len(clean_data) < 2:
            # Return NaN matrix if insufficient data
            return pd.DataFrame(np.nan, index=data.columns, columns=data.columns)
        
        # Compute correlation matrix
        corr_matrix = clean_data.corr(method=method)
        
        return corr_matrix
    
    def _format_correlation_matrix(self, corr_matrix: pd.DataFrame, method: str) -> List[str]:
        """Format correlation matrix for display."""
        output = []
        
        # Header
        output.append("Correlation Matrix")
        output.append("-" * 40)
        
        # Get variable names
        vars_list = corr_matrix.columns.tolist()
        n_vars = len(vars_list)
        
        # Create formatted table
        # Header row
        header = f"{'Variable':<12}"
        for var in vars_list:
            header += f"{var[:8]:>10}"
        output.append(header)
        output.append("-" * len(header))
        
        # Data rows
        for i, var1 in enumerate(vars_list):
            row = f"{var1[:12]:<12}"
            for j, var2 in enumerate(vars_list):
                if i == j:
                    row += f"{'1.0000':>10}"
                else:
                    corr_val = corr_matrix.loc[var1, var2]
                    if pd.isna(corr_val):
                        row += f"{'   .':>10}"
                    else:
                        row += f"{corr_val:>10.4f}"
            output.append(row)
        
        output.append("")
        
        # Add sample size information
        output.append("Sample Size Information")
        output.append("-" * 30)
        output.append(f"Number of observations used: {len(corr_matrix)}")
        output.append("")
        
        # Add method information
        output.append(f"Correlation method: {method.title()}")
        output.append("")
        
        return output
    
    def _format_with_correlation(self, corr_matrix: pd.DataFrame, var_vars: List[str], with_vars: List[str], method: str) -> List[str]:
        """Format VAR-WITH correlation table for display."""
        output = []
        
        # Header
        output.append("Correlation Analysis: VAR variables with WITH variables")
        output.append("-" * 60)
        
        # Create formatted table
        # Header row
        header = f"{'Variable':<12}"
        for var in with_vars:
            header += f"{var[:8]:>10}"
        output.append(header)
        output.append("-" * len(header))
        
        # Data rows
        for var in var_vars:
            row = f"{var[:12]:<12}"
            for with_var in with_vars:
                corr_val = corr_matrix.loc[var, with_var]
                if pd.isna(corr_val):
                    row += f"{'   .':>10}"
                else:
                    row += f"{corr_val:>10.4f}"
            output.append(row)
        
        output.append("")
        
        # Add sample size information
        output.append("Sample Size Information")
        output.append("-" * 30)
        output.append(f"Number of observations used: {len(corr_matrix)}")
        output.append("")
        
        # Add method information
        output.append(f"Correlation method: {method.title()}")
        output.append("")
        
        return output
