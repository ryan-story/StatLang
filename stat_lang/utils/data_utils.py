"""
Data Utilities for Open-SAS

This module provides utility functions for data manipulation,
formatting, and other data-related operations.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union


class DataUtils:
    """Utility functions for data manipulation."""
    
    @staticmethod
    def format_dataframe_for_display(df: pd.DataFrame, max_rows: int = 100) -> List[str]:
        """
        Format a DataFrame for text display.
        
        Args:
            df: DataFrame to format
            max_rows: Maximum number of rows to display
            
        Returns:
            List of formatted lines
        """
        if df.empty:
            return ["No data to display."]
        
        # Limit rows if necessary
        display_df = df.head(max_rows)
        
        # Get column widths
        col_widths = {}
        for col in display_df.columns:
            col_width = max(len(str(col)), display_df[col].astype(str).str.len().max())
            col_widths[col] = min(col_width, 20)  # Limit column width
        
        lines = []
        
        # Create header
        header = " | ".join(f"{str(col):<{col_widths[col]}}" for col in display_df.columns)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Add rows
        for idx, row in display_df.iterrows():
            row_str = " | ".join(f"{str(val):<{col_widths[col]}}" for col, val in zip(display_df.columns, row))
            lines.append(row_str)
        
        # Add summary if truncated
        if len(df) > max_rows:
            lines.append(f"... ({len(df) - max_rows} more rows)")
        
        return lines
    
    @staticmethod
    def create_summary_stats(df: pd.DataFrame, numeric_cols: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Create summary statistics for numeric columns.
        
        Args:
            df: DataFrame to analyze
            numeric_cols: List of numeric columns to analyze
            
        Returns:
            Dictionary of statistics for each column
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats = {}
        for col in numeric_cols:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    stats[col] = {
                        'N': len(series),
                        'Mean': series.mean(),
                        'Std': series.std(),
                        'Min': series.min(),
                        'Max': series.max(),
                        'Median': series.median()
                    }
        
        return stats
    
    @staticmethod
    def apply_where_condition(df: pd.DataFrame, condition: str, expression_parser) -> pd.DataFrame:
        """
        Apply a WHERE condition to a DataFrame.
        
        Args:
            df: DataFrame to filter
            condition: WHERE condition string
            expression_parser: ExpressionParser instance
            
        Returns:
            Filtered DataFrame
        """
        if not condition.strip():
            return df
        
        try:
            mask = expression_parser.parse_where_condition(condition, df)
            return df[mask]
        except Exception as e:
            print(f"Warning: Could not apply WHERE condition '{condition}': {e}")
            return df
    
    @staticmethod
    def create_frequency_table(df: pd.DataFrame, var_name: str) -> pd.DataFrame:
        """
        Create a frequency table for a variable.
        
        Args:
            df: DataFrame containing the data
            var_name: Name of the variable to analyze
            
        Returns:
            DataFrame with frequency counts and percentages
        """
        if var_name not in df.columns:
            return pd.DataFrame()
        
        # Calculate frequencies
        freq_counts = df[var_name].value_counts().sort_index()
        total = len(df[var_name].dropna())
        
        # Create result DataFrame
        result = pd.DataFrame({
            'Value': freq_counts.index,
            'Frequency': freq_counts.values,
            'Percent': (freq_counts.values / total) * 100
        })
        
        # Add cumulative percent
        result['Cumulative_Percent'] = result['Percent'].cumsum()
        
        return result
    
    @staticmethod
    def create_crosstab(df: pd.DataFrame, var1: str, var2: str) -> pd.DataFrame:
        """
        Create a cross-tabulation table.
        
        Args:
            df: DataFrame containing the data
            var1: First variable (rows)
            var2: Second variable (columns)
            
        Returns:
            Cross-tabulation DataFrame
        """
        if var1 not in df.columns or var2 not in df.columns:
            return pd.DataFrame()
        
        return pd.crosstab(df[var1], df[var2], margins=True, margins_name="Total")
    
    @staticmethod
    def sort_dataframe(df: pd.DataFrame, by_vars: List[str], ascending: bool = True) -> pd.DataFrame:
        """
        Sort a DataFrame by specified variables.
        
        Args:
            df: DataFrame to sort
            by_vars: List of variables to sort by
            ascending: Sort order
            
        Returns:
            Sorted DataFrame
        """
        # Filter to only include variables that exist
        valid_vars = [var for var in by_vars if var in df.columns]
        
        if not valid_vars:
            return df
        
        return df.sort_values(by=valid_vars, ascending=ascending)
    
    @staticmethod
    def drop_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        """
        Remove duplicate rows from a DataFrame.
        
        Args:
            df: DataFrame to process
            subset: List of columns to consider for duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        if subset:
            # Filter to only include variables that exist
            valid_subset = [var for var in subset if var in df.columns]
            if valid_subset:
                return df.drop_duplicates(subset=valid_subset)
        
        return df.drop_duplicates()
    
    @staticmethod
    def rename_columns(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
        """
        Rename columns in a DataFrame.
        
        Args:
            df: DataFrame to rename
            rename_map: Dictionary mapping old names to new names
            
        Returns:
            DataFrame with renamed columns
        """
        # Only rename columns that exist
        valid_rename_map = {old: new for old, new in rename_map.items() if old in df.columns}
        
        if valid_rename_map:
            return df.rename(columns=valid_rename_map)
        
        return df
    
    @staticmethod
    def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Select specific columns from a DataFrame.
        
        Args:
            df: DataFrame to select from
            columns: List of column names to select
            
        Returns:
            DataFrame with selected columns
        """
        # Only select columns that exist
        valid_columns = [col for col in columns if col in df.columns]
        
        if valid_columns:
            return df[valid_columns]
        
        return pd.DataFrame()
    
    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Drop specific columns from a DataFrame.
        
        Args:
            df: DataFrame to drop from
            columns: List of column names to drop
            
        Returns:
            DataFrame with dropped columns
        """
        # Only drop columns that exist
        valid_columns = [col for col in columns if col in df.columns]
        
        if valid_columns:
            return df.drop(columns=valid_columns)
        
        return df
