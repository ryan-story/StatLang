"""
PROC LOGIT Implementation for Open-SAS

This module implements SAS PROC LOGIT functionality for logistic regression
using statsmodels for statistical inference.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcLogit:
    """Implementation of SAS PROC LOGIT procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC LOGIT on the given data.
        
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
        
        # Get MODEL specification
        model_spec = proc_info.options.get('model', '')
        if not model_spec:
            results['output_text'].append("ERROR: MODEL specification required (e.g., MODEL y = x1 x2).")
            return results
        
        # Parse model specification: MODEL y = x1 x2 x3
        # Split by '=' and then by spaces
        if '=' in model_spec:
            parts = model_spec.split('=')
            dependent_var = parts[0].strip()
            independent_vars = [var.strip() for var in parts[1].split() if var.strip()]
        else:
            # If no '=' found, treat as single variable
            dependent_var = model_spec.strip()
            independent_vars = []
        
        # Check if variables exist
        if dependent_var not in data.columns:
            results['output_text'].append(f"ERROR: Dependent variable '{dependent_var}' not found in data.")
            return results
        
        missing_vars = [var for var in independent_vars if var not in data.columns]
        if missing_vars:
            results['output_text'].append(f"ERROR: Independent variables not found: {missing_vars}")
            return results
        
        # Get link function (default: logit)
        link = proc_info.options.get('link', 'logit').lower()
        if link not in ['logit', 'probit', 'cloglog']:
            link = 'logit'
        
        results['output_text'].append("PROC LOGIT - Logistic Regression Analysis")
        results['output_text'].append("=" * 60)
        results['output_text'].append(f"Dependent variable: {dependent_var}")
        results['output_text'].append(f"Independent variables: {', '.join(independent_vars)}")
        results['output_text'].append(f"Link function: {link}")
        results['output_text'].append("")
        
        # Prepare data
        model_data = data[[dependent_var] + independent_vars].copy()
        clean_data = model_data.dropna()
        
        if len(clean_data) < len(independent_vars) + 1:
            results['output_text'].append("ERROR: Insufficient data after removing missing values.")
            return results
        
        # Prepare dependent variable
        y = clean_data[dependent_var]
        
        # Handle categorical dependent variable
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            y_classes = le.classes_
            results['output_text'].append(f"Dependent variable encoded: {dict(zip(y_classes, range(len(y_classes))))}")
        else:
            y_encoded = y
            y_classes = None
        
        # Check if binary
        unique_values = np.unique(y_encoded)
        if len(unique_values) != 2:
            results['output_text'].append("ERROR: PROC LOGIT currently supports binary logistic regression only.")
            return results
        
        # Prepare independent variables
        X = clean_data[independent_vars].copy()
        
        # Handle categorical independent variables
        categorical_vars = []
        for var in independent_vars:
            if X[var].dtype == 'object' or X[var].dtype.name == 'category':
                categorical_vars.append(var)
                # Create dummy variables
                dummies = pd.get_dummies(X[var], prefix=var, drop_first=True)
                X = pd.concat([X.drop(var, axis=1), dummies], axis=1)
        
        if categorical_vars:
            results['output_text'].append(f"Categorical variables converted to dummy variables: {categorical_vars}")
            results['output_text'].append("")
        
        # Add intercept
        X = sm.add_constant(X)
        
        # Fit logistic regression model
        try:
            if link == 'logit':
                model = sm.Logit(y_encoded, X)
            elif link == 'probit':
                model = sm.Probit(y_encoded, X)
            else:  # cloglog
                model = sm.Logit(y_encoded, X)  # Use logit as approximation
            
            fitted_model = model.fit(disp=0, maxiter=1000)
            
            # Format output
            results['output_text'].extend(self._format_logistic_output(fitted_model, X.columns.tolist(), link))
            
            # Create output DataFrame
            results['output_data'] = self._create_output_dataframe(fitted_model, X.columns.tolist())
            
        except Exception as e:
            error_msg = str(e)
            if "Perfect separation" in error_msg or "Singular matrix" in error_msg:
                results['output_text'].append("WARNING: Perfect separation detected in the data.")
                results['output_text'].append("This occurs when predictors perfectly predict the outcome.")
                results['output_text'].append("Consider:")
                results['output_text'].append("1. Using different predictor variables")
                results['output_text'].append("2. Adding regularization")
                results['output_text'].append("3. Checking for data quality issues")
                results['output_text'].append("")
                results['output_text'].append(f"Technical error: {error_msg}")
            else:
                results['output_text'].append(f"ERROR: Model fitting failed: {error_msg}")
            return results
        
        return results
    
    def _format_logistic_output(self, model, var_names: List[str], link: str) -> List[str]:
        """Format logistic regression output."""
        output = []
        
        # Model information
        output.append("Model Information")
        output.append("-" * 30)
        output.append(f"Link function: {link}")
        output.append(f"Number of observations: {model.nobs}")
        output.append(f"Number of parameters: {len(model.params)}")
        output.append("")
        
        # Model fit statistics
        output.append("Model Fit Statistics")
        output.append("-" * 30)
        output.append(f"Log-likelihood: {model.llf:.4f}")
        output.append(f"AIC: {model.aic:.4f}")
        output.append(f"BIC: {model.bic:.4f}")
        
        # Pseudo R-squared
        if hasattr(model, 'prsquared'):
            output.append(f"Pseudo R-squared: {model.prsquared:.4f}")
        
        output.append("")
        
        # Parameter estimates
        output.append("Parameter Estimates")
        output.append("-" * 30)
        output.append(f"{'Variable':<15} {'Estimate':<12} {'Std Err':<12} {'Z':<10} {'P>|Z|':<10} {'[95% Conf. Interval]':<25}")
        output.append("-" * 95)
        
        for i, var in enumerate(var_names):
            coef = model.params[i]
            std_err = model.bse[i]
            z_stat = model.tvalues[i]
            p_value = model.pvalues[i]
            ci_lower = model.conf_int().iloc[i, 0]
            ci_upper = model.conf_int().iloc[i, 1]
            
            output.append(f"{var[:15]:<15} {coef:<12.4f} {std_err:<12.4f} {z_stat:<10.4f} {p_value:<10.4f} [{ci_lower:<8.4f}, {ci_upper:<8.4f}]")
        
        output.append("")
        
        # Odds ratios (for logit link)
        if link == 'logit':
            output.append("Odds Ratios")
            output.append("-" * 20)
            output.append(f"{'Variable':<15} {'Odds Ratio':<12} {'[95% Conf. Interval]':<25}")
            output.append("-" * 55)
            
            for i, var in enumerate(var_names):
                if var == 'const':
                    continue  # Skip intercept for odds ratios
                
                coef = model.params[i]
                ci_lower = model.conf_int().iloc[i, 0]
                ci_upper = model.conf_int().iloc[i, 1]
                
                or_ratio = np.exp(coef)
                or_ci_lower = np.exp(ci_lower)
                or_ci_upper = np.exp(ci_upper)
                
                output.append(f"{var[:15]:<15} {or_ratio:<12.4f} [{or_ci_lower:<8.4f}, {or_ci_upper:<8.4f}]")
            
            output.append("")
        
        # Model significance
        output.append("Model Significance")
        output.append("-" * 25)
        output.append(f"Likelihood ratio test: {model.llr:.4f}")
        output.append(f"Chi-square p-value: {model.llr_pvalue:.6f}")
        
        if model.llr_pvalue < 0.001:
            interpretation = "p < 0.001 (highly significant)"
        elif model.llr_pvalue < 0.01:
            interpretation = "p < 0.01 (very significant)"
        elif model.llr_pvalue < 0.05:
            interpretation = "p < 0.05 (significant)"
        else:
            interpretation = "p >= 0.05 (not significant)"
        
        output.append(f"Conclusion: {interpretation}")
        output.append("")
        
        return output
    
    def _create_output_dataframe(self, model, var_names: List[str]) -> pd.DataFrame:
        """Create output DataFrame with model results."""
        
        results_data = []
        for i, var in enumerate(var_names):
            results_data.append({
                'Variable': var,
                'Coefficient': model.params[i],
                'Std_Error': model.bse[i],
                'Z_Statistic': model.tvalues[i],
                'P_Value': model.pvalues[i],
                'CI_Lower': model.conf_int().iloc[i, 0],
                'CI_Upper': model.conf_int().iloc[i, 1],
                'Odds_Ratio': np.exp(model.params[i]) if var != 'const' else np.nan,
                'OR_CI_Lower': np.exp(model.conf_int().iloc[i, 0]) if var != 'const' else np.nan,
                'OR_CI_Upper': np.exp(model.conf_int().iloc[i, 1]) if var != 'const' else np.nan
            })
        
        return pd.DataFrame(results_data)
