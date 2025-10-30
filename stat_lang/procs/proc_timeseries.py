"""
PROC TIMESERIES Implementation for Open-SAS

This module implements SAS PROC TIMESERIES and PROC ARIMA functionality
for time series analysis including seasonal decomposition and ARIMA modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcTimeseries:
    """Implementation of SAS PROC TIMESERIES procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC TIMESERIES on the given data.
        
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
            results['output_text'].append("ERROR: No valid numeric variables found for time series analysis.")
            return results
        
        # Get analysis type
        analysis_type = proc_info.options.get('type', 'decompose').lower()
        if analysis_type not in ['decompose', 'arima', 'acf', 'pacf']:
            analysis_type = 'decompose'
        
        # Get time variable (optional)
        time_var = proc_info.options.get('time', None)
        
        results['output_text'].append("PROC TIMESERIES - Time Series Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Analysis type: {analysis_type.upper()}")
        results['output_text'].append("")
        
        # Analyze each variable
        all_results = []
        for var in var_vars:
            if analysis_type == 'decompose':
                var_results = self._seasonal_decompose(data, var, time_var)
            elif analysis_type == 'arima':
                var_results = self._arima_analysis(data, var, time_var, proc_info.options)
            elif analysis_type == 'acf':
                var_results = self._acf_analysis(data, var, time_var)
            else:  # pacf
                var_results = self._pacf_analysis(data, var, time_var)
            
            results['output_text'].extend(var_results['output'])
            all_results.append(var_results['stats'])
            results['output_text'].append("")
        
        # Create combined output DataFrame
        if all_results:
            combined_results = pd.DataFrame(all_results)
            results['output_data'] = combined_results
        
        return results
    
    def _seasonal_decompose(self, data: pd.DataFrame, var_name: str, time_var: Optional[str] = None) -> Dict[str, Any]:
        """Perform seasonal decomposition."""
        
        # Prepare time series data
        if time_var and time_var in data.columns:
            # Use specified time variable
            ts_data = data.set_index(time_var)[var_name].dropna()
        else:
            # Use index as time
            ts_data = data[var_name].dropna()
        
        if len(ts_data) < 12:
            return {
                'output': [f"Variable {var_name}: Insufficient data for seasonal decomposition (need at least 12 observations)"],
                'stats': {}
            }
        
        # Determine period
        period = 12  # Default monthly
        if len(ts_data) >= 24:
            # Try to detect period
            if len(ts_data) % 12 == 0:
                period = 12
            elif len(ts_data) % 4 == 0:
                period = 4
            else:
                period = min(12, len(ts_data) // 2)
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=period)
            
            # Format output
            output = []
            output.append(f"Variable: {var_name}")
            output.append("-" * 50)
            output.append("Seasonal Decomposition")
            output.append("")
            
            # Summary statistics
            output.append("Component Summary Statistics")
            output.append("-" * 40)
            output.append(f"{'Component':<15} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
            output.append("-" * 65)
            
            components = {
                'Original': ts_data,
                'Trend': decomposition.trend.dropna(),
                'Seasonal': decomposition.seasonal.dropna(),
                'Residual': decomposition.resid.dropna()
            }
            
            for comp_name, comp_data in components.items():
                if len(comp_data) > 0:
                    output.append(f"{comp_name:<15} {comp_data.mean():<12.4f} {comp_data.std():<12.4f} {comp_data.min():<12.4f} {comp_data.max():<12.4f}")
            
            output.append("")
            
            # Variance explained
            total_var = ts_data.var()
            trend_var = decomposition.trend.dropna().var()
            seasonal_var = decomposition.seasonal.dropna().var()
            residual_var = decomposition.resid.dropna().var()
            
            output.append("Variance Explained")
            output.append("-" * 25)
            output.append(f"Trend: {trend_var/total_var*100:.1f}%")
            output.append(f"Seasonal: {seasonal_var/total_var*100:.1f}%")
            output.append(f"Residual: {residual_var/total_var*100:.1f}%")
            output.append("")
            
            # Create output DataFrame
            decomp_df = pd.DataFrame({
                'Original': ts_data,
                'Trend': decomposition.trend,
                'Seasonal': decomposition.seasonal,
                'Residual': decomposition.resid
            })
            
            stats_dict = {
                'Variable': var_name,
                'Analysis_Type': 'Seasonal_Decomposition',
                'Period': period,
                'N_Observations': len(ts_data),
                'Trend_Variance_Pct': trend_var/total_var*100,
                'Seasonal_Variance_Pct': seasonal_var/total_var*100,
                'Residual_Variance_Pct': residual_var/total_var*100
            }
            
            return {
                'output': output,
                'stats': stats_dict,
                'decomposition': decomp_df
            }
            
        except Exception as e:
            return {
                'output': [f"Variable {var_name}: Seasonal decomposition failed - {str(e)}"],
                'stats': {}
            }
    
    def _arima_analysis(self, data: pd.DataFrame, var_name: str, time_var: Optional[str], options: Dict) -> Dict[str, Any]:
        """Perform ARIMA analysis."""
        
        # Prepare time series data
        if time_var and time_var in data.columns:
            ts_data = data.set_index(time_var)[var_name].dropna()
        else:
            ts_data = data[var_name].dropna()
        
        if len(ts_data) < 10:
            return {
                'output': [f"Variable {var_name}: Insufficient data for ARIMA analysis (need at least 10 observations)"],
                'stats': {}
            }
        
        # Get ARIMA orders
        p = options.get('p', 1)
        d = options.get('d', 1)
        q = options.get('q', 1)
        
        try:
            # Fit ARIMA model
            model = ARIMA(ts_data, order=(p, d, q))
            fitted_model = model.fit()
            
            # Format output
            output = []
            output.append(f"Variable: {var_name}")
            output.append("-" * 50)
            output.append("ARIMA Model")
            output.append("")
            
            # Model information
            output.append("Model Information")
            output.append("-" * 25)
            output.append(f"ARIMA({p},{d},{q})")
            output.append(f"Number of observations: {len(ts_data)}")
            output.append("")
            
            # Model fit statistics
            output.append("Model Fit Statistics")
            output.append("-" * 25)
            output.append(f"Log-likelihood: {fitted_model.llf:.4f}")
            output.append(f"AIC: {fitted_model.aic:.4f}")
            output.append(f"BIC: {fitted_model.bic:.4f}")
            output.append("")
            
            # Parameter estimates
            output.append("Parameter Estimates")
            output.append("-" * 25)
            output.append(f"{'Parameter':<15} {'Estimate':<12} {'Std Error':<12} {'Z':<10} {'P>|Z|':<10}")
            output.append("-" * 65)
            
            for param_name, param_value in fitted_model.params.items():
                std_err = fitted_model.bse[param_name]
                z_stat = fitted_model.tvalues[param_name]
                p_value = fitted_model.pvalues[param_name]
                
                output.append(f"{param_name:<15} {param_value:<12.4f} {std_err:<12.4f} {z_stat:<10.4f} {p_value:<10.4f}")
            
            output.append("")
            
            # Forecast (if requested)
            forecast_steps = options.get('forecast', 0)
            if forecast_steps > 0:
                forecast = fitted_model.forecast(steps=forecast_steps)
                conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
                
                output.append(f"Forecast (next {forecast_steps} periods)")
                output.append("-" * 35)
                output.append(f"{'Period':<10} {'Forecast':<12} {'Lower CI':<12} {'Upper CI':<12}")
                output.append("-" * 50)
                
                for i in range(forecast_steps):
                    output.append(f"{i+1:<10} {forecast.iloc[i]:<12.4f} {conf_int.iloc[i,0]:<12.4f} {conf_int.iloc[i,1]:<12.4f}")
                
                output.append("")
            
            stats_dict = {
                'Variable': var_name,
                'Analysis_Type': 'ARIMA',
                'ARIMA_Order': f"({p},{d},{q})",
                'N_Observations': len(ts_data),
                'Log_Likelihood': fitted_model.llf,
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic,
                'Forecast_Steps': forecast_steps
            }
            
            return {
                'output': output,
                'stats': stats_dict,
                'model': fitted_model
            }
            
        except Exception as e:
            return {
                'output': [f"Variable {var_name}: ARIMA analysis failed - {str(e)}"],
                'stats': {}
            }
    
    def _acf_analysis(self, data: pd.DataFrame, var_name: str, time_var: Optional[str] = None) -> Dict[str, Any]:
        """Perform ACF analysis."""
        
        # Prepare time series data
        if time_var and time_var in data.columns:
            ts_data = data.set_index(time_var)[var_name].dropna()
        else:
            ts_data = data[var_name].dropna()
        
        if len(ts_data) < 10:
            return {
                'output': [f"Variable {var_name}: Insufficient data for ACF analysis"],
                'stats': {}
            }
        
        # Calculate ACF
        max_lags = min(40, len(ts_data) // 4)
        acf_values = sm.tsa.acf(ts_data, nlags=max_lags)
        
        # Format output
        output = []
        output.append(f"Variable: {var_name}")
        output.append("-" * 50)
        output.append("Autocorrelation Function (ACF)")
        output.append("")
        
        output.append(f"{'Lag':<8} {'ACF':<12} {'Significance':<15}")
        output.append("-" * 35)
        
        for lag in range(min(20, len(acf_values))):
            acf_val = acf_values[lag]
            # Simple significance test (approximate)
            significance = "Significant" if abs(acf_val) > 1.96 / np.sqrt(len(ts_data)) else "Not significant"
            output.append(f"{lag:<8} {acf_val:<12.4f} {significance:<15}")
        
        output.append("")
        
        stats_dict = {
            'Variable': var_name,
            'Analysis_Type': 'ACF',
            'N_Observations': len(ts_data),
            'Max_Lags': max_lags,
            'ACF_Values': acf_values[:20].tolist()
        }
        
        return {
            'output': output,
            'stats': stats_dict
        }
    
    def _pacf_analysis(self, data: pd.DataFrame, var_name: str, time_var: Optional[str] = None) -> Dict[str, Any]:
        """Perform PACF analysis."""
        
        # Prepare time series data
        if time_var and time_var in data.columns:
            ts_data = data.set_index(time_var)[var_name].dropna()
        else:
            ts_data = data[var_name].dropna()
        
        if len(ts_data) < 10:
            return {
                'output': [f"Variable {var_name}: Insufficient data for PACF analysis"],
                'stats': {}
            }
        
        # Calculate PACF
        max_lags = min(40, len(ts_data) // 4)
        pacf_values = sm.tsa.pacf(ts_data, nlags=max_lags)
        
        # Format output
        output = []
        output.append(f"Variable: {var_name}")
        output.append("-" * 50)
        output.append("Partial Autocorrelation Function (PACF)")
        output.append("")
        
        output.append(f"{'Lag':<8} {'PACF':<12} {'Significance':<15}")
        output.append("-" * 35)
        
        for lag in range(min(20, len(pacf_values))):
            pacf_val = pacf_values[lag]
            # Simple significance test (approximate)
            significance = "Significant" if abs(pacf_val) > 1.96 / np.sqrt(len(ts_data)) else "Not significant"
            output.append(f"{lag:<8} {pacf_val:<12.4f} {significance:<15}")
        
        output.append("")
        
        stats_dict = {
            'Variable': var_name,
            'Analysis_Type': 'PACF',
            'N_Observations': len(ts_data),
            'Max_Lags': max_lags,
            'PACF_Values': pacf_values[:20].tolist()
        }
        
        return {
            'output': output,
            'stats': stats_dict
        }
