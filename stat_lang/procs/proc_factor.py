"""
PROC FACTOR Implementation for Open-SAS

This module implements SAS PROC FACTOR functionality for principal component
analysis (PCA) and factor analysis using scikit-learn.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcFactor:
    """Implementation of SAS PROC FACTOR procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC FACTOR on the given data.
        
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
            results['output_text'].append("ERROR: No valid numeric variables found for factor analysis.")
            return results
        
        if len(var_vars) < 2:
            results['output_text'].append("ERROR: At least 2 variables required for factor analysis.")
            return results
        
        # Get analysis type (default: pca)
        method = proc_info.options.get('method', 'pca').lower()
        if method not in ['pca', 'factor']:
            method = 'pca'
        
        # Get number of factors/components
        n_factors = proc_info.options.get('nfactors', None)
        if n_factors is None:
            # Default to number of variables or use Kaiser criterion
            n_factors = min(len(var_vars), 10)  # Reasonable default
        
        # Get standardization option
        standardize = proc_info.options.get('standardize', True)
        
        results['output_text'].append("PROC FACTOR - Factor Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Method: {method.upper()}")
        results['output_text'].append(f"Number of factors: {n_factors}")
        results['output_text'].append(f"Standardize: {standardize}")
        results['output_text'].append("")
        
        # Prepare data
        analysis_data = data[var_vars].select_dtypes(include=[np.number])
        clean_data = analysis_data.dropna()
        
        if len(clean_data) < 2:
            results['output_text'].append("ERROR: Insufficient data for factor analysis.")
            return results
        
        # Standardize data if requested
        if standardize:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clean_data)
            analysis_data_scaled = pd.DataFrame(scaled_data, columns=var_vars, index=clean_data.index)
        else:
            analysis_data_scaled = clean_data
        
        # Perform analysis
        if method == 'pca':
            results.update(self._perform_pca(analysis_data_scaled, var_vars, n_factors))
        else:
            results.update(self._perform_factor_analysis(analysis_data_scaled, var_vars, n_factors))
        
        return results
    
    def _perform_pca(self, data: pd.DataFrame, var_names: List[str], n_components: int) -> Dict[str, Any]:
        """Perform Principal Component Analysis."""
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(data)
        
        # Get results
        components = pca.components_
        explained_variance = pca.explained_variance_
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Format output
        results['output_text'].append("Principal Component Analysis")
        results['output_text'].append("-" * 40)
        results['output_text'].append("")
        
        # Eigenvalues and variance explained
        results['output_text'].append("Eigenvalues and Variance Explained")
        results['output_text'].append("-" * 40)
        results['output_text'].append(f"{'Component':<12} {'Eigenvalue':<12} {'Proportion':<12} {'Cumulative':<12}")
        results['output_text'].append("-" * 50)
        
        for i in range(len(explained_variance)):
            results['output_text'].append(
                f"{i+1:<12} {explained_variance[i]:<12.4f} "
                f"{explained_variance_ratio[i]:<12.4f} {cumulative_variance[i]:<12.4f}"
            )
        
        results['output_text'].append("")
        
        # Component loadings
        results['output_text'].append("Component Loadings (Eigenvectors)")
        results['output_text'].append("-" * 40)
        
        # Header
        header = f"{'Variable':<12}"
        for i in range(n_components):
            header += f"{'Comp' + str(i+1):>10}"
        results['output_text'].append(header)
        results['output_text'].append("-" * len(header))
        
        # Loadings
        for i, var in enumerate(var_names):
            row = f"{var[:12]:<12}"
            for j in range(n_components):
                row += f"{components[j, i]:>10.4f}"
            results['output_text'].append(row)
        
        results['output_text'].append("")
        
        # Kaiser criterion
        kaiser_count = np.sum(explained_variance > 1.0)
        results['output_text'].append(f"Kaiser criterion (eigenvalue > 1): {kaiser_count} components")
        results['output_text'].append("")
        
        # Create output DataFrame
        loadings_df = pd.DataFrame(
            components.T,
            columns=[f'Component_{i+1}' for i in range(n_components)],
            index=var_names
        )
        
        variance_df = pd.DataFrame({
            'Eigenvalue': explained_variance,
            'Proportion': explained_variance_ratio,
            'Cumulative': cumulative_variance
        }, index=[f'Component_{i+1}' for i in range(len(explained_variance))])
        
        results['output_data'] = {
            'loadings': loadings_df,
            'variance': variance_df,
            'pca_model': pca
        }
        
        return results
    
    def _perform_factor_analysis(self, data: pd.DataFrame, var_names: List[str], n_factors: int) -> Dict[str, Any]:
        """Perform Factor Analysis."""
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Fit Factor Analysis
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(data)
        
        # Get results
        components = fa.components_
        explained_variance = fa.noise_variance_
        
        # Format output
        results['output_text'].append("Factor Analysis")
        results['output_text'].append("-" * 40)
        results['output_text'].append("")
        
        # Factor loadings
        results['output_text'].append("Factor Loadings")
        results['output_text'].append("-" * 40)
        
        # Header
        header = f"{'Variable':<12}"
        for i in range(n_factors):
            header += f"{'Factor' + str(i+1):>10}"
        results['output_text'].append(header)
        results['output_text'].append("-" * len(header))
        
        # Loadings
        for i, var in enumerate(var_names):
            row = f"{var[:12]:<12}"
            for j in range(n_factors):
                row += f"{components[j, i]:>10.4f}"
            results['output_text'].append(row)
        
        results['output_text'].append("")
        
        # Create output DataFrame
        loadings_df = pd.DataFrame(
            components.T,
            columns=[f'Factor_{i+1}' for i in range(n_factors)],
            index=var_names
        )
        
        results['output_data'] = {
            'loadings': loadings_df,
            'factor_model': fa
        }
        
        return results
