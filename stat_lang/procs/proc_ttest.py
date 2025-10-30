"""
PROC TTEST Implementation for Open-SAS

This module implements SAS PROC TTEST functionality for t-tests including
independent samples, paired samples, and one-sample t-tests.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcTtest:
    """Implementation of SAS PROC TTEST procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC TTEST on the given data.
        
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
            results['output_text'].append("ERROR: No valid numeric variables found for t-test analysis.")
            return results
        
        # Get CLASS variable (for independent samples t-test)
        class_vars = proc_info.options.get('class', [])
        class_var = class_vars[0] if class_vars else None
        
        # Get PAIRED variables (for paired samples t-test)
        paired_vars = proc_info.options.get('paired', [])
        
        # Get H0 value for one-sample t-test
        h0_value = proc_info.options.get('h0', 0.0)
        
        # Determine test type
        if paired_vars and len(paired_vars) == 2:
            test_type = 'paired'
        elif class_var and class_var in data.columns:
            test_type = 'independent'
        else:
            test_type = 'onesample'
        
        results['output_text'].append("PROC TTEST - T-Test Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Test type: {test_type.upper()}")
        results['output_text'].append("")
        
        # Analyze each VAR variable
        all_results = []
        for var in var_vars:
            if test_type == 'paired':
                var_results = self._paired_ttest(data, var, paired_vars[0], paired_vars[1])
            elif test_type == 'independent':
                var_results = self._independent_ttest(data, var, class_var)
            else:
                var_results = self._onesample_ttest(data, var, h0_value)
            
            results['output_text'].extend(var_results['output'])
            all_results.append(var_results['stats'])
            results['output_text'].append("")
        
        # Create combined output DataFrame
        if all_results:
            combined_results = pd.DataFrame(all_results)
            results['output_data'] = combined_results
        
        return results
    
    def _independent_ttest(self, data: pd.DataFrame, var_name: str, class_var: str) -> Dict[str, Any]:
        """Perform independent samples t-test."""
        
        # Remove missing values
        clean_data = data[[var_name, class_var]].dropna()
        
        if len(clean_data) == 0:
            return {
                'output': [f"Variable {var_name}: No valid observations"],
                'stats': {}
            }
        
        # Get groups
        groups = clean_data[class_var].unique()
        if len(groups) != 2:
            return {
                'output': [f"Variable {var_name}: CLASS variable must have exactly 2 groups"],
                'stats': {}
            }
        
        group1_data = clean_data[clean_data[class_var] == groups[0]][var_name]
        group2_data = clean_data[clean_data[class_var] == groups[1]][var_name]
        
        if len(group1_data) < 2 or len(group2_data) < 2:
            return {
                'output': [f"Variable {var_name}: Insufficient data for t-test"],
                'stats': {}
            }
        
        # Perform both equal and unequal variance t-tests
        t_stat_equal, p_equal = stats.ttest_ind(group1_data, group2_data, equal_var=True)
        t_stat_unequal, p_unequal = stats.ttest_ind(group1_data, group2_data, equal_var=False)
        
        # Levene's test for equality of variances
        levene_stat, levene_p = stats.levene(group1_data, group2_data)
        
        # Calculate means and standard deviations
        mean1, mean2 = np.mean(group1_data), np.mean(group2_data)
        std1, std2 = np.std(group1_data, ddof=1), np.std(group2_data, ddof=1)
        n1, n2 = len(group1_data), len(group2_data)
        
        # Calculate degrees of freedom
        df_equal = n1 + n2 - 2
        df_unequal = (std1**2/n1 + std2**2/n2)**2 / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std
        
        # Format output
        output = []
        output.append(f"Variable: {var_name}")
        output.append("-" * 50)
        output.append("Independent Samples T-Test")
        output.append("")
        
        # Group statistics
        output.append("Group Statistics")
        output.append("-" * 30)
        output.append(f"{'Group':<15} {'N':<8} {'Mean':<12} {'Std Dev':<12} {'Std Error':<12}")
        output.append("-" * 65)
        output.append(f"{str(groups[0]):<15} {n1:<8} {mean1:<12.4f} {std1:<12.4f} {std1/np.sqrt(n1):<12.4f}")
        output.append(f"{str(groups[1]):<15} {n2:<8} {mean2:<12.4f} {std2:<12.4f} {std2/np.sqrt(n2):<12.4f}")
        output.append("")
        
        # Levene's test
        output.append("Levene's Test for Equality of Variances")
        output.append("-" * 45)
        output.append(f"F statistic: {levene_stat:.4f}")
        output.append(f"p-value: {levene_p:.6f}")
        output.append("")
        
        # T-test results
        output.append("T-Test Results")
        output.append("-" * 20)
        output.append(f"{'Test':<25} {'t':<10} {'df':<8} {'p-value':<12} {'Mean Diff':<12}")
        output.append("-" * 75)
        output.append(f"{'Equal variances':<25} {t_stat_equal:<10.4f} {df_equal:<8.0f} {p_equal:<12.6f} {mean1-mean2:<12.4f}")
        output.append(f"{'Unequal variances':<25} {t_stat_unequal:<10.4f} {df_unequal:<8.1f} {p_unequal:<12.6f} {mean1-mean2:<12.4f}")
        output.append("")
        
        # Effect size
        output.append(f"Effect size (Cohen's d): {cohens_d:.4f}")
        output.append("")
        
        # Interpretation
        if levene_p < 0.05:
            recommended_test = "Unequal variances (Welch's t-test)"
            final_p = p_unequal
        else:
            recommended_test = "Equal variances (Student's t-test)"
            final_p = p_equal
        
        output.append(f"Recommended test: {recommended_test}")
        
        if final_p < 0.001:
            interpretation = "p < 0.001 (highly significant)"
        elif final_p < 0.01:
            interpretation = "p < 0.01 (very significant)"
        elif final_p < 0.05:
            interpretation = "p < 0.05 (significant)"
        else:
            interpretation = "p >= 0.05 (not significant)"
        
        output.append(f"Conclusion: {interpretation}")
        output.append("")
        
        stats_dict = {
            'Variable': var_name,
            'Test_Type': 'Independent_Samples',
            'Group1': str(groups[0]),
            'Group2': str(groups[1]),
            'N1': n1,
            'N2': n2,
            'Mean1': mean1,
            'Mean2': mean2,
            'Std1': std1,
            'Std2': std2,
            'Mean_Difference': mean1 - mean2,
            'T_Stat_Equal': t_stat_equal,
            'T_Stat_Unequal': t_stat_unequal,
            'DF_Equal': df_equal,
            'DF_Unequal': df_unequal,
            'P_Equal': p_equal,
            'P_Unequal': p_unequal,
            'Levene_Stat': levene_stat,
            'Levene_P': levene_p,
            'Cohens_D': cohens_d,
            'Recommended_Test': recommended_test,
            'Final_P': final_p,
            'Significant': final_p < 0.05
        }
        
        return {
            'output': output,
            'stats': stats_dict
        }
    
    def _paired_ttest(self, data: pd.DataFrame, var_name: str, var1: str, var2: str) -> Dict[str, Any]:
        """Perform paired samples t-test."""
        
        # Check if both variables exist
        if var1 not in data.columns or var2 not in data.columns:
            return {
                'output': [f"Variable {var_name}: PAIRED variables '{var1}' and '{var2}' not found"],
                'stats': {}
            }
        
        # Remove missing values
        clean_data = data[[var1, var2]].dropna()
        
        if len(clean_data) < 2:
            return {
                'output': [f"Variable {var_name}: Insufficient paired observations"],
                'stats': {}
            }
        
        # Calculate differences
        differences = clean_data[var1] - clean_data[var2]
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(clean_data[var1], clean_data[var2])
        
        # Calculate statistics
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        std_error = std_diff / np.sqrt(n)
        df = n - 1
        
        # Calculate confidence interval
        t_critical = stats.t.ppf(0.975, df)
        ci_lower = mean_diff - t_critical * std_error
        ci_upper = mean_diff + t_critical * std_error
        
        # Format output
        output = []
        output.append(f"Variable: {var_name}")
        output.append("-" * 50)
        output.append("Paired Samples T-Test")
        output.append("")
        
        # Paired statistics
        output.append("Paired Statistics")
        output.append("-" * 25)
        output.append(f"{'Variable':<15} {'N':<8} {'Mean':<12} {'Std Dev':<12} {'Std Error':<12}")
        output.append("-" * 65)
        output.append(f"{var1:<15} {n:<8} {np.mean(clean_data[var1]):<12.4f} {np.std(clean_data[var1], ddof=1):<12.4f} {np.std(clean_data[var1], ddof=1)/np.sqrt(n):<12.4f}")
        output.append(f"{var2:<15} {n:<8} {np.mean(clean_data[var2]):<12.4f} {np.std(clean_data[var2], ddof=1):<12.4f} {np.std(clean_data[var2], ddof=1)/np.sqrt(n):<12.4f}")
        output.append("")
        
        # Difference statistics
        output.append("Difference Statistics")
        output.append("-" * 25)
        output.append(f"Mean difference: {mean_diff:.4f}")
        output.append(f"Std deviation: {std_diff:.4f}")
        output.append(f"Std error: {std_error:.4f}")
        output.append(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        output.append("")
        
        # T-test results
        output.append("T-Test Results")
        output.append("-" * 20)
        output.append(f"t statistic: {t_stat:.4f}")
        output.append(f"Degrees of freedom: {df}")
        output.append(f"p-value: {p_value:.6f}")
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
            'Variable': var_name,
            'Test_Type': 'Paired_Samples',
            'Variable1': var1,
            'Variable2': var2,
            'N': n,
            'Mean1': np.mean(clean_data[var1]),
            'Mean2': np.mean(clean_data[var2]),
            'Mean_Difference': mean_diff,
            'Std_Difference': std_diff,
            'Std_Error': std_error,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'T_Statistic': t_stat,
            'DF': df,
            'P_Value': p_value,
            'Significant': p_value < 0.05
        }
        
        return {
            'output': output,
            'stats': stats_dict
        }
    
    def _onesample_ttest(self, data: pd.DataFrame, var_name: str, h0_value: float) -> Dict[str, Any]:
        """Perform one-sample t-test."""
        
        # Remove missing values
        clean_data = data[var_name].dropna()
        
        if len(clean_data) < 2:
            return {
                'output': [f"Variable {var_name}: Insufficient data for t-test"],
                'stats': {}
            }
        
        # Perform one-sample t-test
        t_stat, p_value = stats.ttest_1samp(clean_data, h0_value)
        
        # Calculate statistics
        n = len(clean_data)
        mean_val = np.mean(clean_data)
        std_val = np.std(clean_data, ddof=1)
        std_error = std_val / np.sqrt(n)
        df = n - 1
        
        # Calculate confidence interval
        t_critical = stats.t.ppf(0.975, df)
        ci_lower = mean_val - t_critical * std_error
        ci_upper = mean_val + t_critical * std_error
        
        # Format output
        output = []
        output.append(f"Variable: {var_name}")
        output.append("-" * 50)
        output.append("One-Sample T-Test")
        output.append("")
        
        # Sample statistics
        output.append("Sample Statistics")
        output.append("-" * 20)
        output.append(f"N: {n}")
        output.append(f"Mean: {mean_val:.4f}")
        output.append(f"Std deviation: {std_val:.4f}")
        output.append(f"Std error: {std_error:.4f}")
        output.append(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        output.append("")
        
        # T-test results
        output.append("T-Test Results")
        output.append("-" * 20)
        output.append(f"Test value (H0): {h0_value:.4f}")
        output.append(f"t statistic: {t_stat:.4f}")
        output.append(f"Degrees of freedom: {df}")
        output.append(f"p-value: {p_value:.6f}")
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
            'Variable': var_name,
            'Test_Type': 'One_Sample',
            'H0_Value': h0_value,
            'N': n,
            'Mean': mean_val,
            'Std_Dev': std_val,
            'Std_Error': std_error,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'T_Statistic': t_stat,
            'DF': df,
            'P_Value': p_value,
            'Significant': p_value < 0.05
        }
        
        return {
            'output': output,
            'stats': stats_dict
        }
