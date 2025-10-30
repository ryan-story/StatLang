"""
PROC FREQ Implementation for Open-SAS

This module implements SAS PROC FREQ functionality for frequency
tables and cross-tabulations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcFreq:
    """Implementation of SAS PROC FREQ procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC FREQ on the given data.
        
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
        
        # Get TABLES specification
        tables_spec = proc_info.options.get('tables', '')
        if not tables_spec:
            results['output_text'].append("ERROR: TABLES statement required for PROC FREQ.")
            return results
        
        # Parse tables specification
        # Handle options (everything after /)
        if '/' in tables_spec:
            table_part, options_part = tables_spec.split('/', 1)
            table_part = table_part.strip()
            options_part = options_part.strip()
        else:
            table_part = tables_spec.strip()
            options_part = ""
        
        # Parse table specification (variables)
        if '*' in table_part:
            # Two-way table
            vars_list = [var.strip() for var in table_part.split('*')]
            if len(vars_list) == 2:
                var1, var2 = vars_list
                if var1 in data.columns and var2 in data.columns:
                    results.update(self._create_crosstab(data, var1, var2, options_part))
                else:
                    results['output_text'].append(f"ERROR: Variables {var1} or {var2} not found in data.")
            else:
                results['output_text'].append("ERROR: Only two-way tables supported currently.")
        else:
            # One-way frequency
            var = table_part.strip()
            if var in data.columns:
                results.update(self._create_frequency_table(data, var, options_part))
            else:
                results['output_text'].append(f"ERROR: Variable {var} not found in data.")
        
        return results
    
    def _create_frequency_table(self, data: pd.DataFrame, var: str, options: str = "") -> Dict[str, Any]:
        """Create a one-way frequency table."""
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Calculate frequencies
        freq_table = data[var].value_counts().sort_index()
        total = len(data[var].dropna())
        
        results['output_text'].append(f"PROC FREQ - Frequency Table for {var}")
        results['output_text'].append("=" * 50)
        results['output_text'].append("")
        
        # Create formatted table
        lines = []
        lines.append(f"{'Value':<20} {'Frequency':<12} {'Percent':<10} {'Cumulative Percent':<18}")
        lines.append("-" * 60)
        
        cum_freq = 0
        for value, freq in freq_table.items():
            percent = (freq / total) * 100
            cum_freq += freq
            cum_percent = (cum_freq / total) * 100
            
            lines.append(f"{str(value):<20} {freq:<12} {percent:<10.1f} {cum_percent:<18.1f}")
        
        # Add total row
        lines.append("-" * 60)
        lines.append(f"{'Total':<20} {total:<12} {100.0:<10.1f} {100.0:<18.1f}")
        
        results['output_text'].extend(lines)
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'Value': freq_table.index,
            'Frequency': freq_table.values,
            'Percent': (freq_table.values / total) * 100
        })
        results['output_data'] = output_df
        
        return results
    
    def _create_crosstab(self, data: pd.DataFrame, var1: str, var2: str, options: str = "") -> Dict[str, Any]:
        """Create a two-way cross-tabulation table."""
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Parse options
        options_list = [opt.strip().lower() for opt in options.split()] if options else []
        nocol = 'nocol' in options_list
        nopercent = 'nopercent' in options_list
        chisq = 'chisq' in options_list
        exact = 'exact' in options_list
        
        # Create crosstab
        crosstab = pd.crosstab(data[var1], data[var2], margins=True, margins_name="Total")
        
        results['output_text'].append(f"PROC FREQ - Cross-tabulation: {var1} * {var2}")
        if options:
            results['output_text'].append(f"Options: {options}")
        results['output_text'].append("=" * 50)
        results['output_text'].append("")
        
        # Format crosstab for display
        lines = self._format_crosstab(crosstab, var1, var2, nocol, nopercent)
        results['output_text'].extend(lines)
        
        # Add Chi-square test if requested
        if chisq:
            chi_square_results = self._perform_chi_square_test(crosstab, var1, var2, exact)
            results['output_text'].extend(chi_square_results['output'])
            results['output_data'] = {
                'crosstab': crosstab,
                'chi_square': chi_square_results['stats']
            }
        else:
            results['output_data'] = crosstab
        
        return results
    
    def _format_crosstab(self, crosstab: pd.DataFrame, var1: str, var2: str, nocol: bool = False, nopercent: bool = False) -> List[str]:
        """Format crosstab DataFrame for display."""
        lines = []
        
        # Get column widths
        col_widths = {}
        for col in crosstab.columns:
            col_widths[col] = max(len(str(col)), crosstab[col].astype(str).str.len().max())
        
        # Create header
        header = f"{var1:<15} | " + " | ".join(f"{str(col):<{col_widths[col]}}" for col in crosstab.columns)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Add rows
        for idx, row in crosstab.iterrows():
            row_str = f"{str(idx):<15} | " + " | ".join(f"{str(val):<{col_widths[col]}}" for col, val in zip(crosstab.columns, row))
            lines.append(row_str)
        
        # Add statistics if not suppressed
        if not nopercent:
            lines.append("")
            lines.append("Statistics:")
            total = crosstab.loc["Total", "Total"]
            for idx, row in crosstab.iterrows():
                if idx != "Total":
                    row_total = row["Total"]
                    row_percent = (row_total / total) * 100
                    lines.append(f"  {idx}: {row_total} ({row_percent:.1f}%)")
        
        return lines
    
    def _perform_chi_square_test(self, crosstab: pd.DataFrame, var1: str, var2: str, exact: bool = False) -> Dict[str, Any]:
        """Perform Chi-square test of independence."""
        
        # Remove margins for chi-square test
        test_table = crosstab.drop('Total', axis=0).drop('Total', axis=1)
        
        # Check if table is 2x2 for Fisher's exact test
        is_2x2 = test_table.shape == (2, 2)
        
        output = []
        stats_dict = {}
        
        output.append("")
        output.append("Chi-Square Test of Independence")
        output.append("-" * 40)
        
        # Perform Chi-square test
        chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(test_table)
        
        output.append(f"Chi-square statistic: {chi2_stat:.4f}")
        output.append(f"Degrees of freedom: {dof}")
        output.append(f"p-value: {chi2_p:.6f}")
        output.append("")
        
        # Check expected frequencies
        min_expected = expected.min()
        output.append("Expected Frequencies")
        output.append("-" * 25)
        output.append(f"Minimum expected frequency: {min_expected:.2f}")
        
        if min_expected < 5:
            output.append("Warning: Some expected frequencies are less than 5.")
            if is_2x2:
                output.append("Consider Fisher's exact test for 2x2 tables.")
        else:
            output.append("All expected frequencies are >= 5.")
        
        output.append("")
        
        # Fisher's exact test for 2x2 tables
        if is_2x2 and exact:
            try:
                fisher_oddsratio, fisher_p = stats.fisher_exact(test_table)
                output.append("Fisher's Exact Test (2x2 table)")
                output.append("-" * 35)
                output.append(f"Odds ratio: {fisher_oddsratio:.4f}")
                output.append(f"p-value: {fisher_p:.6f}")
                output.append("")
                stats_dict['fisher_oddsratio'] = fisher_oddsratio
                stats_dict['fisher_p'] = fisher_p
            except Exception as e:
                output.append(f"Fisher's exact test failed: {str(e)}")
                output.append("")
        
        # Effect size (Cramér's V)
        n = test_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(test_table.shape) - 1)))
        output.append(f"Effect size (Cramér's V): {cramers_v:.4f}")
        output.append("")
        
        # Interpretation
        if chi2_p < 0.001:
            interpretation = "p < 0.001 (highly significant)"
        elif chi2_p < 0.01:
            interpretation = "p < 0.01 (very significant)"
        elif chi2_p < 0.05:
            interpretation = "p < 0.05 (significant)"
        else:
            interpretation = "p >= 0.05 (not significant)"
        
        output.append(f"Conclusion: {interpretation}")
        if chi2_p < 0.05:
            output.append("The variables are significantly associated.")
        else:
            output.append("No significant association between variables.")
        
        output.append("")
        
        # Store statistics
        stats_dict.update({
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_p,
            'degrees_of_freedom': dof,
            'min_expected_freq': min_expected,
            'cramers_v': cramers_v,
            'significant': chi2_p < 0.05,
            'is_2x2': is_2x2
        })
        
        return {
            'output': output,
            'stats': stats_dict
        }
