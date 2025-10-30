"""
PROC UNIVARIATE Implementation for Open-SAS

This module implements SAS PROC UNIVARIATE functionality for detailed
univariate analysis including descriptive statistics and distribution tests.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcUnivariate:
    """Implementation of SAS PROC UNIVARIATE procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC UNIVARIATE on the given data.
        
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
            results['output_text'].append("ERROR: No valid numeric variables found for analysis.")
            return results
        
        results['output_text'].append("PROC UNIVARIATE - Univariate Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append("")
        
        # Analyze each variable
        all_stats = []
        for var in var_vars:
            var_stats = self._analyze_variable(data[var], var)
            results['output_text'].extend(var_stats['output'])
            all_stats.append(var_stats['stats'])
            results['output_text'].append("")
        
        # Create combined output DataFrame
        if all_stats:
            combined_stats = pd.DataFrame(all_stats)
            results['output_data'] = combined_stats
        
        return results
    
    def _analyze_variable(self, series: pd.Series, var_name: str) -> Dict[str, Any]:
        """Analyze a single variable."""
        # Remove missing values
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                'output': [f"Variable {var_name}: No valid observations"],
                'stats': {}
            }
        
        # Calculate basic statistics
        n = len(clean_series)
        mean = clean_series.mean()
        std = clean_series.std()
        var = clean_series.var()
        min_val = clean_series.min()
        max_val = clean_series.max()
        range_val = max_val - min_val
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(clean_series, percentiles)
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(clean_series)
        kurtosis = stats.kurtosis(clean_series)
        
        # Test for normality (Shapiro-Wilk test for small samples, otherwise Kolmogorov-Smirnov)
        if n <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(clean_series)
                normality_test = "Shapiro-Wilk"
                normality_stat = shapiro_stat
                normality_p = shapiro_p
            except:
                normality_test = "Not available"
                normality_stat = None
                normality_p = None
        else:
            # For large samples, use Kolmogorov-Smirnov test
            try:
                ks_stat, ks_p = stats.kstest(clean_series, 'norm', args=(mean, std))
                normality_test = "Kolmogorov-Smirnov"
                normality_stat = ks_stat
                normality_p = ks_p
            except:
                normality_test = "Not available"
                normality_stat = None
                normality_p = None
        
        # Format output
        output = []
        output.append(f"Variable: {var_name}")
        output.append("-" * 40)
        
        # Moments
        output.append("Moments")
        output.append(f"  N                          {n:>10}")
        output.append(f"  Mean                       {mean:>10.6f}")
        output.append(f"  Std Deviation              {std:>10.6f}")
        output.append(f"  Variance                   {var:>10.6f}")
        output.append(f"  Skewness                   {skewness:>10.6f}")
        output.append(f"  Uncorrected SS             {np.sum(clean_series**2):>10.6f}")
        output.append(f"  Corrected SS               {np.sum((clean_series - mean)**2):>10.6f}")
        output.append(f"  Coeff Variation            {(std/mean*100) if mean != 0 else 0:>10.6f}")
        output.append("")
        
        # Basic Statistical Measures
        output.append("Basic Statistical Measures")
        output.append(f"  Location                    Variability")
        output.append(f"  Mean      {mean:>10.6f}     Std Deviation     {std:>10.6f}")
        output.append(f"  Median    {np.median(clean_series):>10.6f}     Variance          {var:>10.6f}")
        output.append(f"  Mode      {stats.mode(clean_series, keepdims=True)[0][0]:>10.6f}     Range             {range_val:>10.6f}")
        output.append(f"                           Interquartile Range {np.percentile(clean_series, 75) - np.percentile(clean_series, 25):>10.6f}")
        output.append("")
        
        # Tests for Location
        output.append("Tests for Location: Mu0=0")
        t_stat, t_p = stats.ttest_1samp(clean_series, 0)
        output.append(f"  Test           -Statistic-    -----p Value------")
        output.append(f"  Student's t    {t_stat:>10.6f}    Pr > |t|    {t_p:>10.6f}")
        # Use binomial test from scipy.stats
        from scipy.stats import binom
        sign_count = np.sum(clean_series > 0)
        sign_p = 2 * min(binom.cdf(sign_count, n, 0.5), 1 - binom.cdf(sign_count, n, 0.5))
        output.append(f"  Sign           {sign_count:>10.0f}    Pr >= |M|   {sign_p:>10.6f}")
        output.append(f"  Signed Rank    {np.sum(np.sign(clean_series) * np.arange(1, n+1)):>10.0f}    Pr >= |S|   {2 * min(0.5, 0.5):>10.6f}")
        output.append("")
        
        # Quantiles
        output.append("Quantiles (Definition 5)")
        output.append("  Level         Quantile")
        for p, val in zip(percentiles, percentile_values):
            output.append(f"  {p:>3}%        {val:>10.6f}")
        output.append("")
        
        # Extreme Observations
        output.append("Extreme Observations")
        output.append("  ----Lowest----        ----Highest---")
        output.append("  Value      Obs        Value      Obs")
        
        # Get 5 lowest and 5 highest values with their positions
        sorted_series = clean_series.sort_values()
        lowest_5 = sorted_series.head(5)
        highest_5 = sorted_series.tail(5)
        
        for i in range(5):
            if i < len(lowest_5):
                low_val = lowest_5.iloc[i]
                low_obs = lowest_5.index[i] + 1  # Convert to 1-based indexing
            else:
                low_val = ""
                low_obs = ""
                
            if i < len(highest_5):
                high_val = highest_5.iloc[-(i+1)]
                high_obs = highest_5.index[-(i+1)] + 1  # Convert to 1-based indexing
            else:
                high_val = ""
                high_obs = ""
            
            output.append(f"  {low_val:>8.6f}  {low_obs:>6}    {high_val:>8.6f}  {high_obs:>6}")
        
        # Create stats dictionary for output DataFrame
        stats_dict = {
            'Variable': var_name,
            'N': n,
            'Mean': mean,
            'Std_Dev': std,
            'Variance': var,
            'Min': min_val,
            'Max': max_val,
            'Range': range_val,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'P1': percentile_values[0],
            'P5': percentile_values[1],
            'P10': percentile_values[2],
            'P25': percentile_values[3],
            'P50': percentile_values[4],
            'P75': percentile_values[5],
            'P90': percentile_values[6],
            'P95': percentile_values[7],
            'P99': percentile_values[8]
        }
        
        return {
            'output': output,
            'stats': stats_dict
        }
