"""
PROC NPAR1WAY Implementation for Open-SAS

This module implements SAS PROC NPAR1WAY functionality for nonparametric
tests including Mann-Whitney U test and Kruskal-Wallis test.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcNpar1way:
    """Implementation of SAS PROC NPAR1WAY procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC NPAR1WAY on the given data.
        
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
        
        # Get VAR variables (outcome variables)
        var_vars = proc_info.options.get('var', [])
        if not var_vars:
            # If no VAR specified, use all numeric columns
            var_vars = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to only include variables that exist in the data
        var_vars = [var for var in var_vars if var in data.columns]
        
        if not var_vars:
            results['output_text'].append("ERROR: No valid numeric variables found for nonparametric analysis.")
            return results
        
        # Get CLASS variable (grouping variable)
        class_vars = proc_info.options.get('class', [])
        if not class_vars:
            results['output_text'].append("ERROR: CLASS variable is required for PROC NPAR1WAY.")
            return results
        
        class_var = class_vars[0]  # Use first CLASS variable
        if class_var not in data.columns:
            results['output_text'].append(f"ERROR: CLASS variable '{class_var}' not found in data.")
            return results
        
        # Get test type (default: wilcoxon)
        test_type = proc_info.options.get('test', 'wilcoxon').lower()
        if test_type not in ['wilcoxon', 'kruskal']:
            test_type = 'wilcoxon'
        
        results['output_text'].append("PROC NPAR1WAY - Nonparametric One-Way Analysis")
        results['output_text'].append("=" * 60)
        results['output_text'].append(f"Class variable: {class_var}")
        results['output_text'].append(f"Test type: {test_type.upper()}")
        results['output_text'].append("")
        
        # Get unique groups
        groups = data[class_var].dropna().unique()
        if len(groups) < 2:
            results['output_text'].append("ERROR: At least 2 groups required for nonparametric analysis.")
            return results
        
        # Analyze each VAR variable
        all_results = []
        for var in var_vars:
            var_results = self._analyze_variable(data, var, class_var, groups, test_type)
            results['output_text'].extend(var_results['output'])
            all_results.append(var_results['stats'])
            results['output_text'].append("")
        
        # Create combined output DataFrame
        if all_results:
            combined_results = pd.DataFrame(all_results)
            results['output_data'] = combined_results
        
        return results
    
    def _analyze_variable(self, data: pd.DataFrame, var_name: str, class_var: str, 
                         groups: np.ndarray, test_type: str) -> Dict[str, Any]:
        """Analyze a single variable with nonparametric tests."""
        
        # Remove missing values
        clean_data = data[[var_name, class_var]].dropna()
        
        if len(clean_data) == 0:
            return {
                'output': [f"Variable {var_name}: No valid observations"],
                'stats': {}
            }
        
        # Group data by class variable
        grouped_data = []
        group_names = []
        group_sizes = []
        
        for group in groups:
            group_data = clean_data[clean_data[class_var] == group][var_name]
            if len(group_data) > 0:
                grouped_data.append(group_data.values)
                group_names.append(str(group))
                group_sizes.append(len(group_data))
        
        if len(grouped_data) < 2:
            return {
                'output': [f"Variable {var_name}: Insufficient groups for analysis"],
                'stats': {}
            }
        
        # Format output
        output = []
        output.append(f"Variable: {var_name}")
        output.append("-" * 50)
        
        # Summary statistics by group
        output.append("Summary Statistics by Group")
        output.append("-" * 40)
        output.append(f"{'Group':<15} {'N':<8} {'Mean':<12} {'Median':<12} {'Std Dev':<12}")
        output.append("-" * 65)
        
        for i, (group_name, group_data) in enumerate(zip(group_names, grouped_data)):
            mean_val = np.mean(group_data)
            median_val = np.median(group_data)
            std_val = np.std(group_data, ddof=1)
            output.append(f"{group_name:<15} {len(group_data):<8} {mean_val:<12.4f} {median_val:<12.4f} {std_val:<12.4f}")
        
        output.append("")
        
        # Perform appropriate test
        if len(grouped_data) == 2 and test_type == 'wilcoxon':
            # Two-sample Mann-Whitney U test
            test_result = self._mann_whitney_test(grouped_data[0], grouped_data[1], group_names[0], group_names[1])
            output.extend(test_result['output'])
            stats_dict = test_result['stats']
        elif len(grouped_data) > 2 or test_type == 'kruskal':
            # Kruskal-Wallis test for multiple groups
            test_result = self._kruskal_wallis_test(grouped_data, group_names)
            output.extend(test_result['output'])
            stats_dict = test_result['stats']
        else:
            # Default to Mann-Whitney for two groups
            test_result = self._mann_whitney_test(grouped_data[0], grouped_data[1], group_names[0], group_names[1])
            output.extend(test_result['output'])
            stats_dict = test_result['stats']
        
        # Add variable name to stats
        stats_dict['Variable'] = var_name
        stats_dict['Class_Variable'] = class_var
        stats_dict['N_Groups'] = len(grouped_data)
        
        return {
            'output': output,
            'stats': stats_dict
        }
    
    def _mann_whitney_test(self, group1: np.ndarray, group2: np.ndarray, 
                          group1_name: str, group2_name: str) -> Dict[str, Any]:
        """Perform Mann-Whitney U test (Wilcoxon rank-sum test)."""
        
        # Perform the test
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Calculate effect size (r = Z / sqrt(N))
        n1, n2 = len(group1), len(group2)
        z_score = stats.norm.ppf(1 - p_value/2)  # Approximate Z-score
        effect_size = z_score / np.sqrt(n1 + n2)
        
        # Format output
        output = []
        output.append("Mann-Whitney U Test (Wilcoxon Rank-Sum Test)")
        output.append("-" * 50)
        output.append(f"Groups: {group1_name} vs {group2_name}")
        output.append(f"Sample sizes: {n1} vs {n2}")
        output.append("")
        output.append("Test Statistics")
        output.append("-" * 20)
        output.append(f"U statistic: {statistic:.4f}")
        output.append(f"Z-score: {z_score:.4f}")
        output.append(f"p-value: {p_value:.6f}")
        output.append(f"Effect size (r): {effect_size:.4f}")
        output.append("")
        
        # Interpretation
        if p_value < 0.001:
            interpretation = "p < 0.001 (highly significant)"
        elif p_value < 0.01:
            interpretation = "p < 0.01 (very significant)"
        elif p_value < 0.05:
            interpretation = "p < 0.05 (significant)"
        else:
            interpretation = "p >= 0.05 (not significant)"
        
        output.append(f"Conclusion: {interpretation}")
        output.append("")
        
        stats_dict = {
            'Test_Type': 'Mann-Whitney_U',
            'Group1': group1_name,
            'Group2': group2_name,
            'N1': n1,
            'N2': n2,
            'U_Statistic': statistic,
            'Z_Score': z_score,
            'P_Value': p_value,
            'Effect_Size': effect_size,
            'Significant': p_value < 0.05
        }
        
        return {
            'output': output,
            'stats': stats_dict
        }
    
    def _kruskal_wallis_test(self, grouped_data: List[np.ndarray], group_names: List[str]) -> Dict[str, Any]:
        """Perform Kruskal-Wallis test for multiple groups."""
        
        # Perform the test
        statistic, p_value = stats.kruskal(*grouped_data)
        
        # Calculate degrees of freedom
        df = len(grouped_data) - 1
        
        # Calculate effect size (eta-squared approximation)
        n_total = sum(len(group) for group in grouped_data)
        effect_size = (statistic - df) / (n_total - 1 - df) if n_total > 1 + df else 0
        
        # Format output
        output = []
        output.append("Kruskal-Wallis Test")
        output.append("-" * 30)
        output.append(f"Groups: {', '.join(group_names)}")
        output.append(f"Sample sizes: {[len(group) for group in grouped_data]}")
        output.append("")
        output.append("Test Statistics")
        output.append("-" * 20)
        output.append(f"H statistic: {statistic:.4f}")
        output.append(f"Degrees of freedom: {df}")
        output.append(f"p-value: {p_value:.6f}")
        output.append(f"Effect size (eta²): {effect_size:.4f}")
        output.append("")
        
        # Interpretation
        if p_value < 0.001:
            interpretation = "p < 0.001 (highly significant)"
        elif p_value < 0.01:
            interpretation = "p < 0.01 (very significant)"
        elif p_value < 0.05:
            interpretation = "p < 0.05 (significant)"
        else:
            interpretation = "p >= 0.05 (not significant)"
        
        output.append(f"Conclusion: {interpretation}")
        if p_value < 0.05:
            output.append("Note: Significant difference detected. Consider post-hoc pairwise comparisons.")
        output.append("")
        
        stats_dict = {
            'Test_Type': 'Kruskal_Wallis',
            'Groups': ', '.join(group_names),
            'N_Groups': len(grouped_data),
            'N_Total': sum(len(group) for group in grouped_data),
            'H_Statistic': statistic,
            'DF': df,
            'P_Value': p_value,
            'Effect_Size': effect_size,
            'Significant': p_value < 0.05
        }
        
        return {
            'output': output,
            'stats': stats_dict
        }
