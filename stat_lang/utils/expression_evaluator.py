"""
Expression Evaluator for Open-SAS

This module provides functionality to evaluate SAS expressions
for variable assignments, IF/THEN/ELSE statements, and other
data step operations.
"""

import pandas as pd
import numpy as np
import re
from typing import Any, Dict, List, Union, Optional
from .expression_parser import ExpressionParser


class ExpressionEvaluator:
    """Evaluator for SAS expressions in DATA steps."""
    
    def __init__(self):
        self.expression_parser = ExpressionParser()
        
        # Define SAS functions
        self.functions = {
            'sum': lambda *args: sum(args),
            'mean': lambda *args: np.mean(args),
            'min': lambda *args: min(args),
            'max': lambda *args: max(args),
            'abs': abs,
            'sqrt': np.sqrt,
            'round': round,
            'int': int,
            'length': len,
            'substr': self._substr,
            'index': self._index,
            'compress': self._compress,
            'trim': str.strip,
            'upcase': str.upper,
            'lowcase': str.lower,
            'ifc': self._ifc,
            'ifn': self._ifn,
        }
    
    def evaluate_assignment(self, assignment: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate a variable assignment statement.
        
        Args:
            assignment: Assignment statement (e.g., "new_var = old_var * 2")
            data: DataFrame to apply the assignment to
            
        Returns:
            DataFrame with the new variable added
        """
        # Parse assignment: variable = expression
        match = re.match(r'(\w+)\s*=\s*(.+)', assignment.strip())
        if not match:
            return data
        
        var_name = match.group(1)
        expression = match.group(2).strip()
        
        try:
            # Debug: Evaluating assignment
            # Make sure we're working with a copy to avoid modifying the original
            if not hasattr(data, '_is_copy'):
                data = data.copy()
                data._is_copy = True
            # Evaluate the expression for each row
            result = self._evaluate_expression(expression, data)
            # Debug: Result type and length
            data[var_name] = result
            return data
        except Exception as e:
            print(f"Warning: Could not evaluate assignment '{assignment}': {e}")
            import traceback
            traceback.print_exc()
            return data
    
    def evaluate_if_then_else(self, if_statement: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate an IF/THEN/ELSE statement.
        
        Args:
            if_statement: IF/THEN/ELSE statement
            data: DataFrame to apply the condition to
            
        Returns:
            DataFrame with conditional logic applied
        """
        # Parse IF/THEN/ELSE statement
        # Simple implementation for now
        if 'if' in if_statement.lower() and 'then' in if_statement.lower():
            # Extract condition and assignment
            match = re.match(r'if\s+(.+?)\s+then\s+(.+)', if_statement, re.IGNORECASE)
            if match:
                condition = match.group(1).strip()
                assignment = match.group(2).strip()
                
                # Parse assignment
                assign_match = re.match(r'(\w+)\s*=\s*(.+)', assignment)
                if assign_match:
                    var_name = assign_match.group(1)
                    value = assign_match.group(2).strip()
                    
                    # Create boolean mask for condition
                    mask = self.expression_parser.parse_where_condition(condition, data)
                    
                    # Apply conditional assignment
                    if var_name not in data.columns:
                        data[var_name] = None
                    
                    # Handle different value types
                    if value.startswith('"') and value.endswith('"'):
                        # String value
                        data.loc[mask, var_name] = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        # String value
                        data.loc[mask, var_name] = value[1:-1]
                    else:
                        # Try to evaluate as expression or numeric value
                        try:
                            if value.replace('.', '').replace('-', '').isdigit():
                                data.loc[mask, var_name] = float(value)
                            else:
                                # Try to evaluate as expression
                                result = self._evaluate_expression(value, data)
                                data.loc[mask, var_name] = result[mask]
                        except:
                            data.loc[mask, var_name] = value
        
        return data
    
    def _evaluate_expression(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate an expression for each row in the DataFrame.
        
        Args:
            expression: Expression to evaluate
            data: DataFrame to evaluate against
            
        Returns:
            Series with evaluated results
        """
        # Debug: Evaluating expression
        
        # Handle IFN functions first (they can contain complex expressions)
        if 'ifn(' in expression.lower():
            return self._evaluate_ifn_expression(expression, data)
        
        # Handle simple cases first
        if expression.strip() in data.columns:
            return data[expression.strip()]
        
        # Handle numeric literals
        if expression.replace('.', '').replace('-', '').isdigit():
            return pd.Series([float(expression)] * len(data), index=data.index)
        
        # Handle string literals
        if (expression.startswith('"') and expression.endswith('"')) or \
           (expression.startswith("'") and expression.endswith("'")):
            return pd.Series([expression[1:-1]] * len(data), index=data.index)
        
        # Handle arithmetic expressions
        if any(op in expression for op in ['+', '-', '*', '/', '**']):
            return self._evaluate_arithmetic(expression, data)
        
        # Handle function calls
        if '(' in expression and ')' in expression:
            return self._evaluate_function(expression, data)
        
        # Default: return as string
        return pd.Series([expression] * len(data), index=data.index)
    
    def _evaluate_arithmetic(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """Evaluate arithmetic expressions."""
        try:
            # Remove semicolon if present
            expression = expression.rstrip(';')
            # Debug: Evaluating arithmetic
            
            # Replace column names with their values
            result = expression
            # Sort columns by length (longest first) to avoid partial replacements
            sorted_cols = sorted(data.columns, key=len, reverse=True)
            for col in sorted_cols:
                if col in expression:
                    # Use word boundaries to avoid partial replacements
                    import re
                    pattern = r'\b' + re.escape(col) + r'\b'
                    result = re.sub(pattern, f"data['{col}']", result)
            
            # Debug: Arithmetic expression after column replacement
            
            # Evaluate the expression
            evaluated_result = eval(result)
            # Debug: Arithmetic result
            return evaluated_result
        except Exception as e:
            print(f"Error evaluating arithmetic expression '{expression}': {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return zeros
            return pd.Series([0] * len(data), index=data.index)
    
    def _evaluate_function(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """Evaluate function calls."""
        # Parse function call: function_name(arg1, arg2, ...)
        match = re.match(r'(\w+)\s*\(([^)]+)\)', expression)
        if not match:
            return pd.Series([0] * len(data), index=data.index)
        
        func_name = match.group(1).lower()
        args_str = match.group(2)
        
        if func_name not in self.functions:
            return pd.Series([0] * len(data), index=data.index)
        
        # Parse arguments
        args = [arg.strip() for arg in args_str.split(',')]
        
        try:
            func = self.functions[func_name]
            
            # Evaluate arguments
            evaluated_args = []
            for arg in args:
                if arg in data.columns:
                    evaluated_args.append(data[arg])
                elif arg.replace('.', '').replace('-', '').isdigit():
                    evaluated_args.append(float(arg))
                elif (arg.startswith('"') and arg.endswith('"')) or \
                     (arg.startswith("'") and arg.endswith("'")):
                    evaluated_args.append(arg[1:-1])
                else:
                    evaluated_args.append(arg)
            
            # Apply function
            if len(evaluated_args) == 1 and isinstance(evaluated_args[0], pd.Series):
                return evaluated_args[0].apply(lambda x: func(x))
            else:
                return pd.Series([func(*evaluated_args)] * len(data), index=data.index)
                
        except:
            return pd.Series([0] * len(data), index=data.index)
    
    def _substr(self, string: str, start: int, length: int = None) -> str:
        """SAS SUBSTR function."""
        if length is None:
            return string[start-1:]
        return string[start-1:start-1+length]
    
    def _index(self, string: str, substring: str) -> int:
        """SAS INDEX function."""
        return string.find(substring) + 1 if substring in string else 0
    
    def _compress(self, string: str, chars: str = None) -> str:
        """SAS COMPRESS function."""
        if chars is None:
            return string.replace(' ', '')
        return ''.join(c for c in string if c not in chars)
    
    def _ifc(self, condition: bool, true_value: str, false_value: str) -> str:
        """SAS IFC function."""
        return true_value if condition else false_value
    
    def _ifn(self, condition, true_value, false_value):
        """SAS IFN function - handles nested IFN calls."""
        # Handle nested IFN calls in false_value
        if isinstance(false_value, str) and 'ifn(' in false_value.lower():
            # Parse nested IFN: ifn(condition2, value2, value3)
            import re
            match = re.match(r'ifn\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', false_value, re.IGNORECASE)
            if match:
                cond2, val2, val3 = match.groups()
                # Evaluate the nested condition
                if isinstance(condition, pd.Series):
                    # For vectorized operations
                    result = pd.Series(index=condition.index, dtype=object)
                    result[condition] = true_value
                    # Evaluate nested condition for false cases
                    nested_condition = self._evaluate_condition(cond2.strip(), condition.index)
                    result[~condition & nested_condition] = val2.strip().strip('"\'')
                    result[~condition & ~nested_condition] = val3.strip().strip('"\'')
                    return result
                else:
                    # For scalar operations
                    if condition:
                        return true_value
                    else:
                        # Evaluate nested condition
                        nested_cond = self._evaluate_condition(cond2.strip(), None)
                        return val2.strip().strip('"\'') if nested_cond else val3.strip().strip('"\'')
        
        # Simple IFN case
        if isinstance(condition, pd.Series):
            result = pd.Series(index=condition.index, dtype=object)
            result[condition] = true_value
            result[~condition] = false_value
            return result
        else:
            return true_value if condition else false_value
    
    def _evaluate_condition(self, condition: str, index=None):
        """Evaluate a condition string."""
        # Handle simple comparisons
        if '>' in condition:
            parts = condition.split('>')
            if len(parts) == 2:
                var, val = parts[0].strip(), parts[1].strip()
                if index is not None:
                    # Vectorized comparison
                    return pd.Series([True] * len(index), index=index)  # Placeholder
                else:
                    return True  # Placeholder
        return True
    
    def _evaluate_ifn_expression(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """Evaluate IFN expressions with proper vectorization."""
        try:
            print(f"Evaluating IFN expression: {expression}")
            
            # Parse IFN expression: ifn(condition, true_value, false_value)
            import re
            # Try to match complete IFN first
            match = re.match(r'ifn\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', expression, re.IGNORECASE)
            if not match:
                # Try to match incomplete IFN (missing closing parenthesis)
                match = re.match(r'ifn\s*\(\s*([^,]+),\s*([^,]+),\s*(.+)$', expression, re.IGNORECASE)
                if match:
                    print(f"IFN pattern matched (incomplete): {expression}")
                else:
                    print(f"IFN pattern not matched for: {expression}")
                    return pd.Series([expression] * len(data), index=data.index)
            
            condition_str, true_value, false_value = match.groups()
            condition_str = condition_str.strip()
            true_value = true_value.strip().strip('"\'')
            false_value = false_value.strip()
            
            print(f"IFN parsed - condition: {condition_str}, true: {true_value}, false: {false_value}")
            
            # Evaluate the condition for each row
            condition_result = self._evaluate_condition_vectorized(condition_str, data)
            print(f"Condition result: {condition_result}")
            
            # Handle nested IFN in false_value
            if 'ifn(' in false_value.lower():
                print("Handling nested IFN")
                # Parse nested IFN - try different patterns
                nested_match = re.match(r'ifn\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', false_value, re.IGNORECASE)
                if not nested_match:
                    # Try without closing parenthesis
                    nested_match = re.match(r'ifn\s*\(\s*([^,]+),\s*([^,]+),\s*(.+)$', false_value, re.IGNORECASE)
                
                if nested_match:
                    cond2, val2, val3 = nested_match.groups()
                    cond2 = cond2.strip()
                    val2 = val2.strip().strip('"\'')
                    val3 = val3.strip().strip('"\'')
                    
                    print(f"Nested IFN - cond2: {cond2}, val2: {val2}, val3: {val3}")
                    
                    # Create result series
                    result = pd.Series(index=data.index, dtype=object)
                    
                    # True cases get true_value
                    result[condition_result] = true_value
                    
                    # False cases get evaluated with nested condition
                    false_mask = ~condition_result
                    if false_mask.any():
                        nested_condition = self._evaluate_condition_vectorized(cond2, data)
                        result[false_mask & nested_condition] = val2
                        result[false_mask & ~nested_condition] = val3
                    
                    print(f"Final IFN result: {result}")
                    return result
                else:
                    print(f"Could not parse nested IFN: {false_value}")
                    # Fall back to simple IFN
                    result = pd.Series(index=data.index, dtype=object)
                    result[condition_result] = true_value
                    result[~condition_result] = false_value
                    return result
            
            # Simple IFN case
            result = pd.Series(index=data.index, dtype=object)
            result[condition_result] = true_value
            result[~condition_result] = false_value.strip().strip('"\'')
            
            print(f"Simple IFN result: {result}")
            return result
            
        except Exception as e:
            print(f"Error evaluating IFN expression '{expression}': {e}")
            import traceback
            traceback.print_exc()
            return pd.Series([expression] * len(data), index=data.index)
    
    def _evaluate_condition_vectorized(self, condition: str, data: pd.DataFrame) -> pd.Series:
        """Evaluate a condition for each row in the DataFrame."""
        try:
            # Handle simple comparisons
            if '>' in condition:
                parts = condition.split('>')
                if len(parts) == 2:
                    var, val = parts[0].strip(), parts[1].strip()
                    if var in data.columns:
                        return data[var] > float(val)
            
            # Default: return all True
            return pd.Series([True] * len(data), index=data.index)
            
        except Exception as e:
            print(f"Error evaluating condition '{condition}': {e}")
            return pd.Series([True] * len(data), index=data.index)
