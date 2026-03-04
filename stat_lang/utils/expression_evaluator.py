"""
Expression Evaluator for StatLang

This module provides functionality to evaluate SAS expressions
for variable assignments, IF/THEN/ELSE statements, array references,
LAG/DIF functions, and other data step operations.
"""

import re
from collections import deque
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .expression_parser import ExpressionParser


class ExpressionEvaluator:
    """Evaluator for SAS expressions in DATA steps."""

    def __init__(self):
        self.expression_parser = ExpressionParser()

        # Array definitions: name -> list of variable names
        self._arrays: Dict[str, List[str]] = {}

        # LAG queues: (var, k) -> deque of last k values
        self._lag_queues: Dict[str, deque] = {}

        # Define SAS functions
        self.functions: Dict[str, Callable[..., Any]] = {
            'sum': lambda *args: sum(a for a in args if a is not None and not (isinstance(a, float) and np.isnan(a))),
            'mean': lambda *args: np.nanmean([a for a in args if a is not None]),
            'min': lambda *args: min(a for a in args if a is not None),
            'max': lambda *args: max(a for a in args if a is not None),
            'abs': abs,
            'sqrt': np.sqrt,
            'round': round,
            'int': int,
            'length': len,
            'substr': self._substr,
            'index': self._index,
            'compress': self._compress,
            'trim': lambda s: s.strip() if isinstance(s, str) else str(s).strip(),
            'upcase': lambda s: s.upper() if isinstance(s, str) else str(s).upper(),
            'lowcase': lambda s: s.lower() if isinstance(s, str) else str(s).lower(),
            'ifc': self._ifc,
            'ifn': self._ifn,
            'cat': lambda *args: ''.join(str(a) for a in args),
            'cats': lambda *args: ''.join(str(a).strip() for a in args),
            'catx': self._catx,
            'strip': lambda s: s.strip() if isinstance(s, str) else str(s).strip(),
            'left': lambda s: s.lstrip() if isinstance(s, str) else str(s).lstrip(),
            'right': lambda s: s.rstrip() if isinstance(s, str) else str(s).rstrip(),
            'scan': self._scan,
            'put': lambda val, fmt: str(val),
            'input': lambda val, fmt: float(val) if val else 0,
            'log': np.log,
            'log2': np.log2,
            'log10': np.log10,
            'exp': np.exp,
            'ceil': np.ceil,
            'floor': np.floor,
            'mod': lambda a, b: a % b,
            'missing': lambda x: x is None or (isinstance(x, float) and np.isnan(x)),
            'nmiss': lambda *args: sum(1 for a in args if a is None or (isinstance(a, float) and np.isnan(a))),
            'n': lambda *args: sum(1 for a in args if a is not None and not (isinstance(a, float) and np.isnan(a))),
            'coalesce': lambda *args: next((a for a in args if a is not None and not (isinstance(a, float) and np.isnan(a))), None),
            'tranwrd': lambda s, f, r: s.replace(f, r) if isinstance(s, str) else str(s).replace(f, r),
            'reverse': lambda s: s[::-1] if isinstance(s, str) else str(s)[::-1],
            'propcase': lambda s: s.title() if isinstance(s, str) else str(s).title(),
            'count': lambda s, sub: s.count(sub) if isinstance(s, str) else 0,
            'countw': lambda s, *args: len(s.split(args[0] if args else None)) if isinstance(s, str) else 0,
        }

    def register_arrays(self, arrays: List) -> None:
        """Register array definitions for use in expression evaluation."""
        for arr in arrays:
            self._arrays[arr.name.lower()] = arr.variables

    def reset_lag_queues(self) -> None:
        """Reset LAG/DIF state for a new DATA step."""
        self._lag_queues.clear()

    def evaluate_assignment(self, assignment: str, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate a variable assignment statement."""
        match = re.match(r'(\w+)\s*=\s*(.+)', assignment.strip())
        if not match:
            return data

        var_name = match.group(1)
        expression = match.group(2).strip()

        try:
            if not hasattr(data, '_is_copy'):
                data = data.copy()
                data._is_copy = True
            result = self._evaluate_expression(expression, data)
            data[var_name] = result
            return data
        except Exception as e:
            print(f"Warning: Could not evaluate assignment '{assignment}': {e}")
            return data

    def evaluate_if_then_else(self, if_statement: str, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate an IF/THEN/ELSE statement.

        Handles two SAS forms:
        - Subsetting IF:  ``if condition;``  — keeps only matching rows.
        - Conditional:    ``if condition then var = value;``
        """
        stmt_lower = if_statement.lower()

        # --- Conditional IF … THEN … ---
        if 'then' in stmt_lower:
            match = re.match(r'if\s+(.+?)\s+then\s+(.+)', if_statement, re.IGNORECASE)
            if match:
                condition = match.group(1).strip()
                assignment = match.group(2).strip()

                assign_match = re.match(r'(\w+)\s*=\s*(.+)', assignment)
                if assign_match:
                    var_name = assign_match.group(1)
                    value = assign_match.group(2).strip()

                    mask = self.expression_parser.parse_where_condition(condition, data)

                    if var_name not in data.columns:
                        data[var_name] = None

                    if value.startswith('"') and value.endswith('"'):
                        data.loc[mask, var_name] = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        data.loc[mask, var_name] = value[1:-1]
                    else:
                        try:
                            if value.replace('.', '').replace('-', '').isdigit():
                                data.loc[mask, var_name] = float(value)
                            else:
                                result = self._evaluate_expression(value, data)
                                data.loc[mask, var_name] = result[mask]
                        except Exception:
                            data.loc[mask, var_name] = value

            return data

        # --- Subsetting IF (no THEN) — filter rows ---
        m = re.match(r'if\s+(.+)', if_statement, re.IGNORECASE)
        if m:
            condition = m.group(1).strip().rstrip(';')
            try:
                mask = self.expression_parser.parse_where_condition(condition, data)
                data = data.loc[mask].reset_index(drop=True)
            except Exception:
                pass

        return data

    def evaluate_row_assignment(
        self, assignment: str, row: Dict[str, Any],
        arrays: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Evaluate assignment for a single row (used in row-by-row processing)."""
        # Handle array ref on left-hand side: xs(i) = expr
        arr_match = re.match(r'(\w+)\s*\(\s*(.+?)\s*\)\s*=\s*(.+)', assignment.strip())
        if arr_match and arrays:
            arr_name = arr_match.group(1).lower()
            idx_expr = arr_match.group(2).strip()
            expression = arr_match.group(3).strip()
            if arr_name in arrays:
                try:
                    idx = int(idx_expr) if idx_expr.isdigit() else int(row.get(idx_expr, 0))
                except (ValueError, TypeError):
                    idx = 0
                arr_vars = arrays[arr_name]
                if 1 <= idx <= len(arr_vars):
                    var_name = arr_vars[idx - 1]
                    value = self._evaluate_scalar_expression(expression, row, arrays)
                    row[var_name] = value
                return

        match = re.match(r'(\w+)\s*=\s*(.+)', assignment.strip())
        if not match:
            return

        var_name = match.group(1)
        expression = match.group(2).strip()

        try:
            value = self._evaluate_scalar_expression(expression, row, arrays)
            row[var_name] = value
        except Exception:
            pass

    def evaluate_row_if(
        self, if_stmt: str, row: Dict[str, Any],
        arrays: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Evaluate IF/THEN for a single row."""
        match = re.match(r'if\s+(.+?)\s+then\s+(.+)', if_stmt, re.IGNORECASE)
        if not match:
            return
        condition = match.group(1).strip()
        action = match.group(2).strip()

        if self._evaluate_scalar_condition(condition, row):
            assign_match = re.match(r'(\w+)\s*=\s*(.+)', action)
            if assign_match:
                var_name = assign_match.group(1)
                expr = assign_match.group(2).strip()
                row[var_name] = self._evaluate_scalar_expression(expr, row, arrays)

    def lag(self, var: str, k: int = 1, current_value: Any = None) -> Any:
        """Return the k-th lagged value and push current_value into the queue."""
        key = f'{var}_{k}'
        if key not in self._lag_queues:
            self._lag_queues[key] = deque(maxlen=k)

        q = self._lag_queues[key]
        result = q[0] if len(q) == k else None
        if current_value is not None:
            q.append(current_value)
        return result

    def dif(self, var: str, k: int = 1, current_value: Any = None) -> Any:
        """Return current - LAG(k)."""
        lagged = self.lag(var, k, current_value)
        if lagged is not None and current_value is not None:
            try:
                return current_value - lagged
            except (TypeError, ValueError):
                return None
        return None

    # ------------------------------------------------------------------
    # Scalar expression evaluation (for row-by-row processing)
    # ------------------------------------------------------------------
    def _evaluate_scalar_expression(
        self, expression: str, row: Dict[str, Any],
        arrays: Optional[Dict[str, List[str]]] = None,
    ) -> Any:
        """Evaluate expression against a single row dict."""
        expression = expression.rstrip(';').strip()

        # Resolve array references: arrayname(index)
        if arrays:
            expression = self._resolve_array_refs(expression, row, arrays)

        # LAG / DIF
        lag_m = re.match(r'lag(\d*)\s*\(\s*(\w+)\s*\)', expression, re.IGNORECASE)
        if lag_m:
            k = int(lag_m.group(1)) if lag_m.group(1) else 1
            var = lag_m.group(2)
            return self.lag(var, k, row.get(var))

        dif_m = re.match(r'dif(\d*)\s*\(\s*(\w+)\s*\)', expression, re.IGNORECASE)
        if dif_m:
            k = int(dif_m.group(1)) if dif_m.group(1) else 1
            var = dif_m.group(2)
            return self.dif(var, k, row.get(var))

        # String literal
        if (expression.startswith('"') and expression.endswith('"')) or \
           (expression.startswith("'") and expression.endswith("'")):
            return expression[1:-1]

        # Numeric literal
        try:
            if '.' in expression:
                return float(expression)
            return int(expression)
        except ValueError:
            pass

        # Variable reference
        if expression in row:
            return row[expression]

        # Function call
        func_m = re.match(r'(\w+)\s*\((.+)\)', expression, re.IGNORECASE)
        if func_m:
            fname = func_m.group(1).lower()
            if fname in self.functions:
                args_str = func_m.group(2)
                args = [self._evaluate_scalar_expression(a.strip(), row, arrays)
                        for a in self._split_func_args(args_str)]
                return self.functions[fname](*args)

        # Arithmetic
        if any(op in expression for op in ['+', '-', '*', '/', '**']):
            return self._evaluate_scalar_arithmetic(expression, row)

        return expression

    def _evaluate_scalar_arithmetic(self, expression: str, row: Dict[str, Any]) -> Any:
        """Evaluate arithmetic expression against a single row."""
        result = expression
        sorted_keys = sorted(row.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in result:
                val = row[key]
                if val is None:
                    val = 0
                pattern = r'\b' + re.escape(key) + r'\b'
                result = re.sub(pattern, str(val), result)
        try:
            return eval(result)  # noqa: S307
        except Exception:
            return 0

    def _evaluate_scalar_condition(self, condition: str, row: Dict[str, Any]) -> bool:
        """Evaluate a condition against a single row."""
        condition = condition.strip()

        # Handle AND / OR
        if ' and ' in condition.lower():
            parts = re.split(r'\s+and\s+', condition, flags=re.IGNORECASE)
            return all(self._evaluate_scalar_condition(p.strip(), row) for p in parts)
        if ' or ' in condition.lower():
            parts = re.split(r'\s+or\s+', condition, flags=re.IGNORECASE)
            return any(self._evaluate_scalar_condition(p.strip(), row) for p in parts)

        # Simple comparison
        m = re.match(r'(\w+)\s*(>=|<=|!=|~=|\^=|==|=|>|<|ge|le|ne|gt|lt|eq)\s*(.+)', condition, re.IGNORECASE)
        if m:
            var = m.group(1)
            op = m.group(2).lower()
            val_str = m.group(3).strip().strip("'\"")
            left = row.get(var, 0)

            # Coerce
            try:
                right: Any = float(val_str) if val_str.replace('.', '').replace('-', '').isdigit() else val_str
            except ValueError:
                right = val_str

            op_map = {'=': '==', 'eq': '==', 'ne': '!=', '^=': '!=', '~=': '!=',
                       'gt': '>', 'lt': '<', 'ge': '>=', 'le': '<=',
                       '>=': '>=', '<=': '<=', '>': '>', '<': '<', '==': '==', '!=': '!='}
            pyop = op_map.get(op, '==')
            try:
                return bool(eval(f'left {pyop} right', {'left': left, 'right': right}))  # noqa: S307
            except Exception:
                return False

        return True

    @staticmethod
    def _resolve_array_refs(
        expression: str, row: Dict[str, Any],
        arrays: Dict[str, List[str]],
    ) -> str:
        """Resolve array(index) references to actual variable names."""
        for arr_name, arr_vars in arrays.items():
            pattern = re.compile(rf'\b{re.escape(arr_name)}\s*\(\s*(.+?)\s*\)', re.IGNORECASE)
            for m in pattern.finditer(expression):
                idx_expr = m.group(1).strip()
                try:
                    idx = int(idx_expr) if idx_expr.isdigit() else int(row.get(idx_expr, 0))
                except (ValueError, TypeError):
                    idx = 0
                if 1 <= idx <= len(arr_vars):
                    var_name = arr_vars[idx - 1]
                    expression = expression[:m.start()] + var_name + expression[m.end():]
        return expression

    @staticmethod
    def _split_func_args(args_str: str) -> List[str]:
        """Split function arguments respecting nested parentheses."""
        args: List[str] = []
        depth = 0
        current = ''
        for ch in args_str:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                args.append(current)
                current = ''
            else:
                current += ch
        if current.strip():
            args.append(current)
        return args

    # ------------------------------------------------------------------
    # Vectorised expression evaluation (original interface, kept for compat)
    # ------------------------------------------------------------------
    def _evaluate_expression(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """Evaluate an expression for each row in the DataFrame."""
        if 'ifn(' in expression.lower():
            return self._evaluate_ifn_expression(expression, data)

        if expression.strip() in data.columns:
            return data[expression.strip()]

        if expression.replace('.', '').replace('-', '').isdigit():
            return pd.Series([float(expression)] * len(data), index=data.index)

        if (expression.startswith('"') and expression.endswith('"')) or \
           (expression.startswith("'") and expression.endswith("'")):
            return pd.Series([expression[1:-1]] * len(data), index=data.index)

        if any(op in expression for op in ['+', '-', '*', '/', '**']):
            return self._evaluate_arithmetic(expression, data)

        if '(' in expression and ')' in expression:
            return self._evaluate_function(expression, data)

        return pd.Series([expression] * len(data), index=data.index)

    def _evaluate_arithmetic(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """Evaluate arithmetic expressions."""
        try:
            expression = expression.rstrip(';')
            result = expression
            sorted_cols = sorted(data.columns, key=len, reverse=True)
            for col in sorted_cols:
                if col in expression:
                    pattern = r'\b' + re.escape(col) + r'\b'
                    result = re.sub(pattern, f"data['{col}']", result)
            return eval(result)  # noqa: S307
        except Exception:
            return pd.Series([0] * len(data), index=data.index)

    def _evaluate_function(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """Evaluate function calls."""
        match = re.match(r'(\w+)\s*\(([^)]+)\)', expression)
        if not match:
            return pd.Series([0] * len(data), index=data.index)

        func_name = match.group(1).lower()
        args_str = match.group(2)

        if func_name not in self.functions:
            return pd.Series([0] * len(data), index=data.index)

        args = [arg.strip() for arg in args_str.split(',')]

        try:
            func: Callable[..., Any] = self.functions[func_name]
            evaluated_args: List[Any] = []
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

            if len(evaluated_args) == 1 and isinstance(evaluated_args[0], pd.Series):
                return evaluated_args[0].apply(lambda x: func(x))
            else:
                return pd.Series([func(*evaluated_args)] * len(data), index=data.index)

        except Exception:
            return pd.Series([0] * len(data), index=data.index)

    # ------------------------------------------------------------------
    # SAS built-in functions
    # ------------------------------------------------------------------
    @staticmethod
    def _substr(string: str, start: int, length: Optional[int] = None) -> str:
        if length is None:
            return string[int(start) - 1:]
        return string[int(start) - 1:int(start) - 1 + int(length)]

    @staticmethod
    def _index(string: str, substring: str) -> int:
        return string.find(substring) + 1 if substring in string else 0

    @staticmethod
    def _compress(string: str, chars: Optional[str] = None) -> str:
        if chars is None:
            return string.replace(' ', '')
        return ''.join(c for c in string if c not in chars)

    @staticmethod
    def _ifc(condition: bool, true_value: str, false_value: str) -> str:
        return true_value if condition else false_value

    @staticmethod
    def _ifn(condition, true_value, false_value):
        if isinstance(condition, pd.Series):
            result = pd.Series(index=condition.index, dtype=object)
            result[condition] = true_value
            result[~condition] = false_value
            return result
        return true_value if condition else false_value

    @staticmethod
    def _catx(sep: str, *args: Any) -> str:
        return sep.join(str(a).strip() for a in args if a is not None and str(a).strip())

    @staticmethod
    def _scan(string: str, n: int, delim: Optional[str] = None) -> str:
        parts = string.split(delim) if delim else string.split()
        idx = int(n)
        if idx > 0 and idx <= len(parts):
            return parts[idx - 1]
        if idx < 0 and abs(idx) <= len(parts):
            return parts[idx]
        return ''

    # ------------------------------------------------------------------
    # IFN vectorized evaluation
    # ------------------------------------------------------------------
    def _evaluate_ifn_expression(self, expression: str, data: pd.DataFrame) -> pd.Series:
        try:
            match = re.match(r'ifn\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', expression, re.IGNORECASE)
            if not match:
                match = re.match(r'ifn\s*\(\s*([^,]+),\s*([^,]+),\s*(.+)$', expression, re.IGNORECASE)
                if not match:
                    return pd.Series([expression] * len(data), index=data.index)

            condition_str, true_value, false_value = match.groups()
            condition_str = condition_str.strip()
            true_value = true_value.strip().strip('"\'')
            false_value = false_value.strip()

            condition_result = self._evaluate_condition_vectorized(condition_str, data)

            if 'ifn(' in false_value.lower():
                nested_match = re.match(r'ifn\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', false_value, re.IGNORECASE)
                if not nested_match:
                    nested_match = re.match(r'ifn\s*\(\s*([^,]+),\s*([^,]+),\s*(.+)$', false_value, re.IGNORECASE)

                if nested_match:
                    cond2, val2, val3 = nested_match.groups()
                    val2 = val2.strip().strip('"\'')
                    val3 = val3.strip().strip('"\'')

                    result = pd.Series(index=data.index, dtype=object)
                    result[condition_result] = true_value
                    false_mask = ~condition_result
                    if false_mask.any():
                        nested_condition = self._evaluate_condition_vectorized(cond2.strip(), data)
                        result[false_mask & nested_condition] = val2
                        result[false_mask & ~nested_condition] = val3
                    return result

            result = pd.Series(index=data.index, dtype=object)
            result[condition_result] = true_value
            result[~condition_result] = false_value.strip().strip('"\'')
            return result

        except Exception:
            return pd.Series([expression] * len(data), index=data.index)

    def _evaluate_condition_vectorized(self, condition: str, data: pd.DataFrame) -> pd.Series:
        try:
            if '>' in condition:
                parts = condition.split('>')
                if len(parts) == 2:
                    var, val = parts[0].strip(), parts[1].strip()
                    if var in data.columns:
                        return data[var] > float(val)
            if '<' in condition:
                parts = condition.split('<')
                if len(parts) == 2:
                    var, val = parts[0].strip(), parts[1].strip()
                    if var in data.columns:
                        return data[var] < float(val)
            if '=' in condition:
                parts = condition.split('=')
                if len(parts) == 2:
                    var, val = parts[0].strip(), parts[1].strip()
                    if var in data.columns:
                        try:
                            return data[var] == float(val)
                        except ValueError:
                            return data[var] == val
            return pd.Series([True] * len(data), index=data.index)
        except Exception:
            return pd.Series([True] * len(data), index=data.index)
