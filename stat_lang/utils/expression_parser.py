"""
Expression Parser for Open-SAS

This module provides functionality to parse and evaluate SAS expressions
for WHERE clauses, IF statements, and variable assignments.
"""

import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union
import operator


class ExpressionParser:
    """Parser for SAS expressions and conditions."""
    
    def __init__(self):
        # Define operators
        self.operators = {
            '=': operator.eq,
            '==': operator.eq,
            '^=': operator.ne,
            '~=': operator.ne,
            'ne': operator.ne,
            '>': operator.gt,
            '>=': operator.ge,
            'ge': operator.ge,
            '<': operator.lt,
            '<=': operator.le,
            'le': operator.le,
            'in': self._in_operator,
            'not in': self._not_in_operator,
            'contains': self._contains_operator,
            'like': self._like_operator
        }
        
        # Define logical operators
        self.logical_operators = {
            'and': operator.and_,
            'or': operator.or_,
            'not': operator.not_
        }
    
    def parse_where_condition(self, condition: str, data: pd.DataFrame) -> pd.Series:
        """
        Parse a WHERE condition and return a boolean Series.
        
        Args:
            condition: The WHERE condition string
            data: DataFrame to apply the condition to
            
        Returns:
            Boolean Series indicating which rows match the condition
        """
        condition = condition.strip()
        
        # Handle simple conditions first
        if self._is_simple_condition(condition):
            return self._parse_simple_condition(condition, data)
        
        # Handle complex conditions with AND/OR
        return self._parse_complex_condition(condition, data)
    
    def _is_simple_condition(self, condition: str) -> bool:
        """Check if condition is simple (no AND/OR)."""
        condition_lower = condition.lower()
        return ' and ' not in condition_lower and ' or ' not in condition_lower
    
    def _parse_simple_condition(self, condition: str, data: pd.DataFrame) -> pd.Series:
        """Parse a simple condition (no AND/OR)."""
        # Try to match various patterns
        patterns = [
            r'(\w+)\s*(>=|<=|!=|~=|==|=|>|<|ge|le|ne)\s*(.+)',  # var op value (longer operators first)
            r'(\w+)\s+(in|not in)\s*\(([^)]+)\)',  # var in (list)
            r'(\w+)\s+(contains|like)\s+(.+)',  # var contains/like value
        ]
        
        for pattern in patterns:
            match = re.match(pattern, condition, re.IGNORECASE)
            if match:
                if 'in' in pattern:
                    return self._parse_in_condition(match, data)
                elif 'contains' in pattern or 'like' in pattern:
                    return self._parse_string_condition(match, data)
                else:
                    return self._parse_comparison_condition(match, data)
        
        # If no pattern matches, return all True (no filtering)
        return pd.Series([True] * len(data), index=data.index)
    
    def _parse_comparison_condition(self, match: re.Match, data: pd.DataFrame) -> pd.Series:
        """Parse a comparison condition like 'age > 30'."""
        var_name = match.group(1)
        op = match.group(2).lower()
        value_str = match.group(3).strip()
        
        if var_name not in data.columns:
            return pd.Series([False] * len(data), index=data.index)
        
        # Parse the value
        value = self._parse_value(value_str)
        
        # Get the operator function
        op_func = self.operators.get(op)
        if not op_func:
            return pd.Series([False] * len(data), index=data.index)
        
        # Apply the condition
        try:
            return op_func(data[var_name], value)
        except:
            return pd.Series([False] * len(data), index=data.index)
    
    def _parse_in_condition(self, match: re.Match, data: pd.DataFrame) -> pd.Series:
        """Parse an IN condition like 'name in (John, Mary)'."""
        var_name = match.group(1)
        op = match.group(2).lower()
        values_str = match.group(3)
        
        if var_name not in data.columns:
            return pd.Series([False] * len(data), index=data.index)
        
        # Parse the values list
        values = [self._parse_value(v.strip()) for v in values_str.split(',')]
        
        # Apply the condition
        if op == 'in':
            return data[var_name].isin(values)
        else:  # not in
            return ~data[var_name].isin(values)
    
    def _parse_string_condition(self, match: re.Match, data: pd.DataFrame) -> pd.Series:
        """Parse string conditions like 'name contains John'."""
        var_name = match.group(1)
        op = match.group(2).lower()
        value_str = match.group(3).strip()
        
        if var_name not in data.columns:
            return pd.Series([False] * len(data), index=data.index)
        
        # Remove quotes if present
        value = value_str.strip('"\'').lower()
        
        if op == 'contains':
            return data[var_name].astype(str).str.lower().str.contains(value, na=False)
        elif op == 'like':
            # Convert SQL LIKE pattern to regex
            pattern = value.replace('%', '.*').replace('_', '.')
            return data[var_name].astype(str).str.lower().str.match(pattern, na=False)
        
        return pd.Series([False] * len(data), index=data.index)
    
    def _parse_complex_condition(self, condition: str, data: pd.DataFrame) -> pd.Series:
        """Parse complex conditions with AND/OR."""
        # Simple implementation - split on AND/OR and combine
        condition_lower = condition.lower()
        
        if ' and ' in condition_lower:
            parts = re.split(r'\s+and\s+', condition, flags=re.IGNORECASE)
            result = pd.Series([True] * len(data), index=data.index)
            for part in parts:
                part_result = self._parse_simple_condition(part.strip(), data)
                result = result & part_result
            return result
        
        elif ' or ' in condition_lower:
            parts = re.split(r'\s+or\s+', condition, flags=re.IGNORECASE)
            result = pd.Series([False] * len(data), index=data.index)
            for part in parts:
                part_result = self._parse_simple_condition(part.strip(), data)
                result = result | part_result
            return result
        
        # Fallback to simple condition
        return self._parse_simple_condition(condition, data)
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string into appropriate Python type."""
        value_str = value_str.strip()
        
        # Remove quotes if present
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        # Try to parse as number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # Return as string
        return value_str
    
    def _in_operator(self, series: pd.Series, values: List) -> pd.Series:
        """IN operator implementation."""
        return series.isin(values)
    
    def _not_in_operator(self, series: pd.Series, values: List) -> pd.Series:
        """NOT IN operator implementation."""
        return ~series.isin(values)
    
    def _contains_operator(self, series: pd.Series, value: str) -> pd.Series:
        """CONTAINS operator implementation."""
        return series.astype(str).str.contains(value, case=False, na=False)
    
    def _like_operator(self, series: pd.Series, pattern: str) -> pd.Series:
        """LIKE operator implementation."""
        # Convert SQL LIKE pattern to regex
        regex_pattern = pattern.replace('%', '.*').replace('_', '.')
        return series.astype(str).str.match(regex_pattern, case=False, na=False)
