"""
PROC SURVEYSELECT Implementation for Open-SAS

This module implements SAS PROC SURVEYSELECT functionality for random sampling
of observations from datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcSurveySelect:
    """Implementation of SAS PROC SURVEYSELECT procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC SURVEYSELECT on the given data.
        
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
        
        # Parse options
        options = proc_info.options
        
        # Get sampling parameters
        method = options.get('method', 'srs').lower()
        if method != 'srs':
            results['output_text'].append(f"ERROR: Only METHOD=SRS supported. Got METHOD={method}")
            return results
        
        # Get sample size parameters
        n_total = len(data)
        n_sample = None
        
        if 'n' in options:
            n_sample = int(options['n'])
        elif 'samprate' in options:
            samprate = float(options['samprate'])
            n_sample = int(samprate * n_total)
        else:
            results['output_text'].append("ERROR: Must specify either SAMPRATE= or N=")
            return results
        
        # Validate sample size
        if n_sample <= 0:
            results['output_text'].append(f"ERROR: Sample size must be positive. Got {n_sample}")
            return results
        
        if n_sample > n_total:
            results['output_text'].append(f"ERROR: Sample size ({n_sample}) cannot exceed total observations ({n_total})")
            return results
        
        # Set random seed for reproducibility
        seed = int(options.get('seed', 42))
        np.random.seed(seed)
        
        # Perform random sampling
        selected_indices = np.random.choice(n_total, size=n_sample, replace=False)
        selected_mask = pd.Series(False, index=data.index)
        selected_mask.iloc[selected_indices] = True
        
        # Create output dataset
        if options.get('outall', False):
            # Include all observations with selection indicator
            output_data = data.copy()
            output_data['selected'] = selected_mask.astype(int)
            sample_data = data[selected_mask]
        else:
            # Only include selected observations
            output_data = data[selected_mask]
            sample_data = output_data
        
        # Generate output
        results['output_text'].append("PROC SURVEYSELECT - Random Sampling Results")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Sampling Method: {method.upper()}")
        results['output_text'].append(f"Total Observations: {n_total}")
        results['output_text'].append(f"Sample Size: {n_sample}")
        results['output_text'].append(f"Sampling Rate: {n_sample/n_total:.3f}")
        results['output_text'].append(f"Random Seed: {seed}")
        results['output_text'].append("")
        
        if options.get('outall', False):
            results['output_text'].append(f"Output Dataset: {len(output_data)} observations (all with selection indicator)")
            results['output_text'].append(f"Selected Observations: {len(sample_data)}")
        else:
            results['output_text'].append(f"Output Dataset: {len(output_data)} observations (selected only)")
        
        results['output_text'].append("")
        
        # Display sample data
        if len(output_data) > 0:
            results['output_text'].append("Sample Data Preview:")
            results['output_text'].append("-" * 30)
            
            # Show first few rows
            preview_data = output_data.head(10)
            lines = self._format_dataframe(preview_data)
            results['output_text'].extend(lines)
            
            if len(output_data) > 10:
                results['output_text'].append(f"... and {len(output_data) - 10} more observations")
        else:
            results['output_text'].append("No observations selected.")
        
        results['output_data'] = output_data
        return results
    
    def _format_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Format DataFrame for display."""
        lines = []
        
        # Get column widths
        col_widths = {}
        for col in df.columns:
            col_widths[col] = max(len(str(col)), df[col].astype(str).str.len().max(), 8)
        
        # Create header
        header = " | ".join(f"{col:<{col_widths[col]}}" for col in df.columns)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Add rows
        for idx, row in df.iterrows():
            row_str = " | ".join(f"{str(val):<{col_widths[col]}}" for col, val in zip(df.columns, row))
            lines.append(row_str)
        
        return lines
