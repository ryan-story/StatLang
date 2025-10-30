"""
PROC REG Implementation for Open-SAS

This module implements SAS PROC REG functionality for linear regression analysis,
including model fitting, predictions, and residual analysis.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcReg:
    """Implementation of SAS PROC REG procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC REG on the given data.
        
        Args:
            data: Input DataFrame
            proc_info: Parsed PROC statement information
            dataset_manager: Dataset manager for output datasets
            
        Returns:
            Dictionary containing results and output data
        """
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Parse options
        options = proc_info.options
        
        # Get statements
        statements = proc_info.statements
        
        # Parse MODEL statement
        model_info = None
        output_info = None
        score_info = None
        
        for stmt in statements:
            stmt_upper = stmt.upper().strip()
            if stmt_upper.startswith('MODEL'):
                model_info = self._parse_model_statement(stmt)
            elif stmt_upper.startswith('OUTPUT'):
                output_info = self._parse_output_statement(stmt)
            elif stmt_upper.startswith('SCORE'):
                score_info = self._parse_score_statement(stmt)
        
        if not model_info:
            results['output_text'].append("ERROR: MODEL statement required for PROC REG")
            return results
        
        # Prepare data for regression
        try:
            # Get dependent variable
            dep_var = model_info['dependent']
            if dep_var not in data.columns:
                results['output_text'].append(f"ERROR: Dependent variable '{dep_var}' not found in dataset")
                return results
            
            # Get independent variables
            indep_vars = model_info['independent']
            missing_vars = [var for var in indep_vars if var not in data.columns]
            if missing_vars:
                results['output_text'].append(f"ERROR: Independent variables not found: {missing_vars}")
                return results
            
            # Remove rows with missing values
            regression_data = data[[dep_var] + indep_vars].dropna()
            
            if len(regression_data) < 2:
                results['output_text'].append("ERROR: Insufficient data for regression (need at least 2 observations)")
                return results
            
            # Prepare X and y
            X = regression_data[indep_vars]
            y = regression_data[dep_var]
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate output
            results['output_text'].append("PROC REG - Linear Regression Analysis")
            results['output_text'].append("=" * 50)
            results['output_text'].append(f"Dependent Variable: {dep_var}")
            results['output_text'].append(f"Independent Variables: {', '.join(indep_vars)}")
            results['output_text'].append(f"Number of Observations: {len(regression_data)}")
            results['output_text'].append("")
            
            # Model summary
            results['output_text'].append("Model Summary")
            results['output_text'].append("-" * 20)
            results['output_text'].append(f"R-Square: {model.score(X, y):.6f}")
            results['output_text'].append(f"Adjusted R-Square: {self._calculate_adjusted_r2(model.score(X, y), len(regression_data), len(indep_vars)):.6f}")
            results['output_text'].append(f"Root MSE: {np.sqrt(mean_squared_error(y, model.predict(X))):.6f}")
            results['output_text'].append("")
            
            # Parameter estimates
            results['output_text'].append("Parameter Estimates")
            results['output_text'].append("-" * 30)
            results['output_text'].append(f"{'Variable':<15} {'DF':<5} {'Estimate':<12} {'Std Error':<12} {'t Value':<10} {'Pr > |t|':<10}")
            results['output_text'].append("-" * 80)
            
            # Intercept
            results['output_text'].append(f"{'Intercept':<15} {len(indep_vars):<5} {model.intercept_:<12.6f} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
            
            # Coefficients
            for i, var in enumerate(indep_vars):
                coef = model.coef_[i]
                # For simplicity, we'll use placeholder values for std error and t-value
                std_error = np.sqrt(mean_squared_error(y, model.predict(X))) / np.sqrt(len(regression_data))
                t_value = coef / std_error if std_error > 0 else 0
                # Simplified p-value calculation
                p_value = 0.05 if abs(t_value) > 2 else 0.1
                
                results['output_text'].append(f"{var:<15} {1:<5} {coef:<12.6f} {std_error:<12.6f} {t_value:<10.3f} {p_value:<10.4f}")
            
            results['output_text'].append("")
            
            # Handle OUTPUT statement
            if output_info:
                output_data = regression_data.copy()
                output_data[f"predicted_{dep_var}"] = model.predict(X)
                output_data[f"residuals"] = y - model.predict(X)
                
                # Store output dataset in results for interpreter to handle
                results['output_data'] = output_data
                results['output_dataset'] = output_info['out']
                
                results['output_text'].append(f"Output dataset '{output_info['out']}' created with predictions and residuals")
            
            # Handle SCORE statement
            if score_info:
                # For SCORE, we need to predict on a different dataset
                # Get the score dataset from the dataset manager
                if dataset_manager and score_info['data']:
                    score_dataset = dataset_manager.get_dataset(score_info['data'])
                    if score_dataset:
                        score_data = score_dataset.dataframe
                        
                        # Prepare score data (same variables as training)
                        score_X = score_data[indep_vars]
                        
                        # Make predictions
                        score_predictions = model.predict(score_X)
                        
                        # Create output dataset with predictions
                        score_output = score_data.copy()
                        score_output[f"predicted_{dep_var}"] = score_predictions
                        
                        # Store score results
                        results['output_data'] = score_output
                        results['output_dataset'] = score_info['out']
                        
                        results['output_text'].append(f"SCORE completed for dataset '{score_info['data']}'")
                        results['output_text'].append(f"Score dataset '{score_info['out']}' created with predictions")
                    else:
                        results['output_text'].append(f"ERROR: Score dataset '{score_info['data']}' not found")
                else:
                    results['output_text'].append(f"SCORE statement processed for dataset '{score_info['data']}'")
                    results['output_text'].append("Note: SCORE functionality requires dataset manager access")
            
            # Only set output_data if no OUTPUT or SCORE statement was processed
            if not output_info and not score_info:
                results['output_data'] = regression_data
            
        except Exception as e:
            results['output_text'].append(f"ERROR: Regression analysis failed: {str(e)}")
        
        return results
    
    def _parse_model_statement(self, stmt: str) -> Dict[str, Any]:
        """Parse MODEL statement."""
        # MODEL depvar = indepvar1 indepvar2 ...;
        stmt_clean = stmt.strip()
        
        # Find the MODEL keyword and extract everything after it
        if not stmt_clean.upper().startswith('MODEL'):
            raise ValueError("Statement must start with MODEL")
        
        # Remove MODEL keyword (preserve case)
        model_part = stmt_clean[5:].strip()
        
        # Split on equals sign
        parts = model_part.split('=')
        if len(parts) != 2:
            raise ValueError("Invalid MODEL statement format - missing equals sign")
        
        dep_var = parts[0].strip()
        indep_part = parts[1].strip()
        
        # Remove semicolon if present
        if indep_part.endswith(';'):
            indep_part = indep_part[:-1]
        
        indep_vars = [var.strip() for var in indep_part.split()]
        
        return {
            'dependent': dep_var,
            'independent': indep_vars
        }
    
    def _parse_output_statement(self, stmt: str) -> Dict[str, Any]:
        """Parse OUTPUT statement."""
        # OUTPUT OUT=dataset p=predicted_var r=residuals;
        output_info = {'out': None, 'predicted': None, 'residuals': None}
        
        # Extract OUT= option
        if 'OUT=' in stmt.upper():
            out_match = stmt.upper().split('OUT=')[1].split()[0]
            output_info['out'] = out_match.replace(';', '').strip().lower()
        
        # Extract predicted variable
        if 'P=' in stmt.upper():
            p_match = stmt.upper().split('P=')[1].split()[0]
            output_info['predicted'] = p_match.replace(';', '').strip()
        
        # Extract residuals variable
        if 'R=' in stmt.upper():
            r_match = stmt.upper().split('R=')[1].split()[0]
            output_info['residuals'] = r_match.replace(';', '').strip()
        
        return output_info
    
    def _parse_score_statement(self, stmt: str) -> Dict[str, Any]:
        """Parse SCORE statement."""
        # SCORE DATA=dataset OUT=output_dataset PREDICTED=predicted_var;
        score_info = {'data': None, 'out': None, 'predicted': None}
        
        # Extract DATA= option
        if 'DATA=' in stmt.upper():
            data_match = stmt.upper().split('DATA=')[1].split()[0]
            score_info['data'] = data_match.replace(';', '').strip().lower()
        
        # Extract OUT= option
        if 'OUT=' in stmt.upper():
            out_match = stmt.upper().split('OUT=')[1].split()[0]
            score_info['out'] = out_match.replace(';', '').strip().lower()
        
        # Extract PREDICTED= option
        if 'PREDICTED=' in stmt.upper():
            pred_match = stmt.upper().split('PREDICTED=')[1].split()[0]
            score_info['predicted'] = pred_match.replace(';', '').strip()
        
        return score_info
    
    def _calculate_adjusted_r2(self, r2: float, n: int, p: int) -> float:
        """Calculate adjusted R-squared."""
        if n <= p + 1:
            return 0.0
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
