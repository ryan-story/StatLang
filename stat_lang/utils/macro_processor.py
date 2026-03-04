"""
Macro facility implementation for StatLang

This module implements the macro system including:
- Macro definitions (%MACRO/%MEND)
- Macro variables (%LET, & substitution)
- Conditional logic (%IF/%THEN/%ELSE)
- Loops (%DO/%END)
- %INCLUDE execution (injection into code stream)
- %SYSEVALF (numeric expression evaluation)
- %SYSFUNC (built-in function calls)
- %GLOBAL / %LOCAL scope declarations
- Built-in system macro variables (SYSLAST, SYSDATE9, etc.)
"""

import ast
import datetime
import math
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from stat_lang import __version__


@dataclass
class MacroDefinition:
    """Represents a macro definition."""
    name: str
    parameters: List[str]
    body: List[str]
    is_global: bool = True


@dataclass
class MacroVariable:
    """Represents a macro variable."""
    name: str
    value: str
    scope: str = 'global'


class MacroProcessor:
    """Macro Processor for StatLang."""

    # Maximum %INCLUDE recursion depth
    MAX_INCLUDE_DEPTH = 10

    def __init__(self):
        self.macros: Dict[str, MacroDefinition] = {}
        self.global_variables: Dict[str, str] = {}
        self.local_scopes: deque = deque()
        self._include_depth = 0

        # Built-in function registry for %SYSFUNC
        self._sysfunc_registry: Dict[str, Callable[..., str]] = {
            'today': self._fn_today,
            'date': self._fn_today,
            'time': self._fn_time,
            'datetime': self._fn_datetime,
            'substr': self._fn_substr,
            'scan': self._fn_scan,
            'cat': self._fn_cat,
            'cats': self._fn_cats,
            'catt': self._fn_catt,
            'catx': self._fn_catx,
            'trim': self._fn_trim,
            'left': self._fn_left,
            'compress': self._fn_compress,
            'upcase': self._fn_upcase,
            'lowcase': self._fn_lowcase,
            'propcase': self._fn_propcase,
            'length': self._fn_length,
            'index': self._fn_index,
            'reverse': self._fn_reverse,
            'tranwrd': self._fn_tranwrd,
            'put': self._fn_put,
            'input': self._fn_input,
            'abs': lambda x: str(abs(float(x))),
            'ceil': lambda x: str(math.ceil(float(x))),
            'floor': lambda x: str(math.floor(float(x))),
            'round': lambda x, *a: str(round(float(x), int(a[0]) if a else 0)),
            'int': lambda x: str(int(float(x))),
            'sqrt': lambda x: str(math.sqrt(float(x))),
            'log': lambda x: str(math.log(float(x))),
            'exp': lambda x: str(math.exp(float(x))),
            'mod': lambda x, y: str(float(x) % float(y)),
            'min': lambda *a: str(min(float(x) for x in a)),
            'max': lambda *a: str(max(float(x) for x in a)),
            'sum': lambda *a: str(sum(float(x) for x in a)),
            'mean': lambda *a: str(sum(float(x) for x in a) / len(a)),
            'getoption': lambda opt: '',
            'sysmsg': lambda: '',
        }

        self._initialize_system_variables()

    # ------------------------------------------------------------------
    # System variables
    # ------------------------------------------------------------------
    def _initialize_system_variables(self):
        """Initialize built-in system macro variables."""
        now = datetime.datetime.now()
        self.global_variables.update({
            'SYSVER': f'StatLang v{__version__}',
            'SYSDATE': now.strftime('%d%b%y').upper(),
            'SYSDATE9': now.strftime('%d%b%Y').upper(),
            'SYSTIME': now.strftime('%H:%M:%S'),
            'SYSUSERID': os.environ.get('USER', os.environ.get('USERNAME', 'user')),
            'SYSPROCESSID': str(os.getpid()),
            'SYSPROCESSNAME': 'statlang',
            'SYSLAST': '_NULL_',
            'SYSCC': '0',
            'SYSJOBID': str(os.getpid()),
        })

    # ------------------------------------------------------------------
    # Variable management
    # ------------------------------------------------------------------
    def define_macro(self, name: str, parameters: List[str], body: List[str]) -> None:
        self.macros[name] = MacroDefinition(
            name=name, parameters=parameters, body=body, is_global=True,
        )

    def set_variable(self, name: str, value: str, scope: str = 'auto') -> None:
        if scope == 'auto':
            scope = 'local' if self.local_scopes else 'global'
        if scope == 'local' and self.local_scopes:
            self.local_scopes[-1][name] = value
        else:
            self.global_variables[name] = value

    def get_variable(self, name: str) -> Optional[str]:
        for scope in reversed(self.local_scopes):
            if name in scope:
                return str(scope[name])
        return self.global_variables.get(name)

    def push_local_scope(self, parameters: Optional[Dict[str, str]] = None) -> None:
        self.local_scopes.append(parameters or {})

    def pop_local_scope(self) -> None:
        if self.local_scopes:
            self.local_scopes.pop()

    # ------------------------------------------------------------------
    # Macro expansion
    # ------------------------------------------------------------------
    def expand_macro_call(self, macro_name: str, arguments: List[str]) -> List[str]:
        if macro_name not in self.macros:
            raise ValueError(f"Macro {macro_name} not defined")

        macro_def = self.macros[macro_name]
        param_map: Dict[str, str] = {}
        for i, param in enumerate(macro_def.parameters):
            param_map[param] = arguments[i] if i < len(arguments) else ''

        self.push_local_scope(param_map)
        try:
            expanded_code: List[str] = []
            i = 0
            while i < len(macro_def.body):
                line = macro_def.body[i]
                stripped = line.strip()

                if stripped.upper().startswith('%IF'):
                    i = self._process_if_statement(macro_def.body, i, expanded_code)
                elif stripped.upper().startswith('%DO'):
                    i = self._process_do_loop(macro_def.body, i, expanded_code)
                elif stripped.upper().startswith('%LET'):
                    self._parse_let_statement(stripped)
                elif stripped.upper().startswith('%PUT'):
                    self._parse_put_statement(stripped)
                else:
                    expanded_code.append(self._substitute_variables(line))
                i += 1
            return expanded_code
        finally:
            self.pop_local_scope()

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------
    def _process_if_statement(self, body: List[str], start_idx: int, output: List[str]) -> int:
        if_line = body[start_idx].strip()
        condition = self._parse_if_condition(if_line)

        then_idx = else_idx = end_idx = None
        for i in range(start_idx + 1, len(body)):
            stripped = body[i].strip()
            if stripped.upper().startswith('%THEN'):
                then_idx = i
            elif stripped.upper().startswith('%ELSE'):
                else_idx = i
            elif stripped.upper().startswith('%END'):
                end_idx = i
                break

        if then_idx is None:
            raise ValueError("Missing %THEN in %IF statement")

        if self._evaluate_condition(condition):
            end_block = else_idx if else_idx is not None else (end_idx or len(body))
            for i in range(then_idx + 1, end_block):
                line = body[i]
                if not line.strip().startswith('%'):
                    output.append(self._substitute_variables(line))
        else:
            if else_idx is not None:
                end_block = end_idx or len(body)
                for i in range(else_idx + 1, end_block):
                    line = body[i]
                    if not line.strip().startswith('%'):
                        output.append(self._substitute_variables(line))

        return end_idx if end_idx else len(body) - 1

    def _process_do_loop(self, body: List[str], start_idx: int, output: List[str]) -> int:
        do_line = body[start_idx].strip()
        loop_params = self._parse_do_loop(do_line)

        end_idx = None
        depth = 1
        for i in range(start_idx + 1, len(body)):
            stripped = body[i].strip().upper()
            if stripped.startswith('%DO'):
                depth += 1
            if stripped.startswith('%END'):
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        if end_idx is None:
            raise ValueError("Missing %END for %DO loop")

        start_val = int(self._substitute_variables(loop_params['start']))
        end_val = int(self._substitute_variables(loop_params['end']))
        step = int(self._substitute_variables(loop_params.get('step', '1')))

        for loop_val in range(start_val, end_val + 1, step):
            self.set_variable(loop_params['var'], str(loop_val), 'local')
            for i in range(start_idx + 1, end_idx):
                line = body[i]
                if not line.strip().startswith('%'):
                    output.append(self._substitute_variables(line))

        return end_idx

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_if_condition(self, if_line: str) -> str:
        match = re.search(r'%IF\s+(.+?)\s+%THEN', if_line, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        raise ValueError("Invalid %IF statement format")

    def _parse_do_loop(self, do_line: str) -> Dict[str, str]:
        pattern = r'%DO\s+(\w+)\s*=\s*([^%]+?)\s+%TO\s+([^%]+?)(?:\s+%BY\s+([^%]+?))?'
        match = re.search(pattern, do_line, re.IGNORECASE)
        if match:
            return {
                'var': match.group(1),
                'start': match.group(2).strip(),
                'end': match.group(3).strip(),
                'step': match.group(4).strip() if match.group(4) else '1',
            }
        raise ValueError("Invalid %DO loop format")

    def _evaluate_condition(self, condition: str) -> bool:
        condition = self._substitute_variables(condition)
        # Normalise operators
        condition = re.sub(r'(?<!=)=(?!=)', '==', condition)
        condition = condition.replace('^=', '!=').replace('~=', '!=')
        condition = re.sub(r'\bEQ\b', '==', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bNE\b', '!=', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bGT\b', '>', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bLT\b', '<', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bGE\b', '>=', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bLE\b', '<=', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bAND\b', 'and', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bOR\b', 'or', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bNOT\b', 'not', condition, flags=re.IGNORECASE)
        try:
            return bool(eval(condition))  # noqa: S307
        except Exception:
            try:
                return bool(ast.literal_eval(condition))
            except Exception:
                return condition.strip().lower() in ('true', '1', 'yes')

    def _substitute_variables(self, text: str) -> str:
        """Substitute macro variables, %SYSEVALF, and %SYSFUNC in text."""
        # Process %SYSEVALF first
        text = self._process_sysevalf(text)
        # Process %SYSFUNC
        text = self._process_sysfunc(text)

        # Handle multiple ampersands
        while '&&' in text:
            text = text.replace('&&', '&')

        def replace_var(match):
            var_name = match.group(1)
            value = self.get_variable(var_name)
            return value if value is not None else match.group(0)

        pattern = r'&(\w+)\.?'
        text = re.sub(pattern, replace_var, text)
        return text

    # ------------------------------------------------------------------
    # %SYSEVALF
    # ------------------------------------------------------------------
    def _process_sysevalf(self, text: str) -> str:
        """Process %SYSEVALF(expression) in text."""
        while '%sysevalf(' in text.lower():
            m = re.search(r'%sysevalf\((.+?)\)', text, re.IGNORECASE)
            if not m:
                break
            expr = m.group(1)
            # Substitute variables in the expression first
            expr = re.sub(r'&(\w+)\.?',
                          lambda mv: self.get_variable(mv.group(1)) or mv.group(0),
                          expr)
            try:
                result = str(eval(expr))  # noqa: S307
            except Exception:
                result = '0'
            text = text[:m.start()] + result + text[m.end():]
        return text

    # ------------------------------------------------------------------
    # %SYSFUNC
    # ------------------------------------------------------------------
    def _process_sysfunc(self, text: str) -> str:
        """Process %SYSFUNC(function(args)) in text."""
        while '%sysfunc(' in text.lower():
            # Find matching outer paren (handle nested parens)
            start_idx = text.lower().find('%sysfunc(')
            if start_idx == -1:
                break
            paren_start = start_idx + len('%sysfunc(')
            depth = 1
            pos = paren_start
            while pos < len(text) and depth > 0:
                if text[pos] == '(':
                    depth += 1
                elif text[pos] == ')':
                    depth -= 1
                pos += 1
            if depth != 0:
                break
            inner = text[paren_start:pos - 1].strip()
            m_start = start_idx
            m_end = pos
            # Parse function call
            fm = re.match(r'(\w+)\s*\(([^)]*)\)', inner)
            if fm:
                fname = fm.group(1).lower()
                args_str = fm.group(2)
                args = [a.strip().strip("'\"") for a in args_str.split(',') if a.strip()] if args_str else []
                # Substitute variables in args
                args = [re.sub(r'&(\w+)\.?',
                               lambda mv: str(self.get_variable(str(mv.group(1))) or mv.group(0)),
                               a) for a in args]
                if fname in self._sysfunc_registry:
                    try:
                        result = self._sysfunc_registry[fname](*args)
                    except Exception:
                        result = ''
                else:
                    result = ''
            else:
                # No-arg function
                fname = inner.lower()
                if fname in self._sysfunc_registry:
                    try:
                        result = self._sysfunc_registry[fname]()
                    except Exception:
                        result = ''
                else:
                    result = ''
            text = text[:m_start] + result + text[m_end:]
        return text

    # %SYSFUNC built-in function implementations
    @staticmethod
    def _fn_today() -> str:
        return datetime.date.today().isoformat()

    @staticmethod
    def _fn_time() -> str:
        return datetime.datetime.now().strftime('%H:%M:%S')

    @staticmethod
    def _fn_datetime() -> str:
        return datetime.datetime.now().isoformat()

    @staticmethod
    def _fn_substr(s: str, start: str, *args: str) -> str:
        start_i = int(start) - 1
        if args:
            return s[start_i:start_i + int(args[0])]
        return s[start_i:]

    @staticmethod
    def _fn_scan(s: str, n: str, *args: str) -> str:
        delim = args[0] if args else None
        parts = s.split(delim) if delim else s.split()
        idx = int(n)
        if 1 <= idx <= len(parts):
            return parts[idx - 1]
        return ''

    @staticmethod
    def _fn_cat(*args: str) -> str:
        return ''.join(args)

    @staticmethod
    def _fn_cats(*args: str) -> str:
        return ''.join(a.strip() for a in args)

    @staticmethod
    def _fn_catt(*args: str) -> str:
        return ''.join(a.rstrip() for a in args)

    @staticmethod
    def _fn_catx(sep: str, *args: str) -> str:
        return sep.join(a.strip() for a in args if a.strip())

    @staticmethod
    def _fn_trim(s: str) -> str:
        return s.rstrip()

    @staticmethod
    def _fn_left(s: str) -> str:
        return s.lstrip()

    @staticmethod
    def _fn_compress(s: str, *args: str) -> str:
        if not args:
            return s.replace(' ', '')
        return ''.join(c for c in s if c not in args[0])

    @staticmethod
    def _fn_upcase(s: str) -> str:
        return s.upper()

    @staticmethod
    def _fn_lowcase(s: str) -> str:
        return s.lower()

    @staticmethod
    def _fn_propcase(s: str) -> str:
        return s.title()

    @staticmethod
    def _fn_length(s: str) -> str:
        return str(len(s))

    @staticmethod
    def _fn_index(s: str, sub: str) -> str:
        idx = s.find(sub)
        return str(idx + 1 if idx >= 0 else 0)

    @staticmethod
    def _fn_reverse(s: str) -> str:
        return s[::-1]

    @staticmethod
    def _fn_tranwrd(s: str, old: str, new: str) -> str:
        return s.replace(old, new)

    @staticmethod
    def _fn_put(val: str, fmt: str) -> str:
        return str(val)

    @staticmethod
    def _fn_input(val: str, fmt: str) -> str:
        return str(val)

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------
    def process_macro_statements(self, code_lines: List[str]) -> List[str]:
        """Process all macro statements and return expanded code."""
        # First pass: define macros, process %LET, %GLOBAL/%LOCAL, %INCLUDE
        i = 0
        expanded_lines = list(code_lines)  # Work on a copy

        # Handle %INCLUDE injection first (recursive)
        expanded_lines = self._inject_includes(expanded_lines)

        # Collect macro definitions
        i = 0
        while i < len(expanded_lines):
            line = expanded_lines[i].strip()

            if line.upper().startswith('%MACRO'):
                i = self._parse_macro_definition(expanded_lines, i)
            elif line.upper().startswith('%LET'):
                self._parse_let_statement(line)
            elif line.upper().startswith('%PUT'):
                self._parse_put_statement(line)
            elif line.upper().startswith('%GLOBAL'):
                self._parse_global_statement(line)
            elif line.upper().startswith('%LOCAL'):
                self._parse_local_statement(line)
            i += 1

        # Second pass: expand macro calls and substitute variables
        result: List[str] = []
        i = 0
        while i < len(expanded_lines):
            line = expanded_lines[i].strip()

            if line.upper().startswith('%MACRO'):
                i = self._skip_macro_definition(expanded_lines, i)
            elif line.upper().startswith(('%LET', '%PUT', '%GLOBAL', '%LOCAL')):
                i += 1
            elif line.upper().startswith('%INCLUDE'):
                i += 1  # Already processed
            elif '%' in line and ('(' in line or re.match(r'%\w+', line)):
                expanded = self._try_expand_macro_call(line)
                if expanded:
                    result.extend(expanded)
                else:
                    result.append(self._substitute_variables(line))
                i += 1
            else:
                result.append(self._substitute_variables(line))
                i += 1

        return result

    # ------------------------------------------------------------------
    # %INCLUDE injection
    # ------------------------------------------------------------------
    def _inject_includes(self, lines: List[str]) -> List[str]:
        """Inject %INCLUDE file contents into the code stream."""
        if self._include_depth >= self.MAX_INCLUDE_DEPTH:
            print("WARNING: Maximum %INCLUDE depth reached, skipping further includes")
            return lines

        result: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith('%INCLUDE'):
                included = self._parse_include_statement(stripped)
                if included:
                    self._include_depth += 1
                    # Recursively process includes in the included file
                    included_clean = [line.rstrip('\n') for line in included]
                    included_expanded = self._inject_includes(included_clean)
                    result.extend(included_expanded)
                    self._include_depth -= 1
                # If include failed, skip the line
            else:
                result.append(line)
        return result

    # ------------------------------------------------------------------
    # %GLOBAL / %LOCAL
    # ------------------------------------------------------------------
    def _parse_global_statement(self, line: str) -> None:
        """Parse %GLOBAL var1 var2 ..."""
        m = re.match(r'%GLOBAL\s+(.+?);?\s*$', line, re.IGNORECASE)
        if m:
            var_names = m.group(1).split()
            for vname in var_names:
                if vname not in self.global_variables:
                    self.global_variables[vname] = ''

    def _parse_local_statement(self, line: str) -> None:
        """Parse %LOCAL var1 var2 ..."""
        m = re.match(r'%LOCAL\s+(.+?);?\s*$', line, re.IGNORECASE)
        if m:
            if self.local_scopes:
                var_names = m.group(1).split()
                for vname in var_names:
                    if vname not in self.local_scopes[-1]:
                        self.local_scopes[-1][vname] = ''

    # ------------------------------------------------------------------
    # Existing helpers (kept from original)
    # ------------------------------------------------------------------
    def _skip_macro_definition(self, code_lines: List[str], start_idx: int) -> int:
        i = start_idx + 1
        while i < len(code_lines):
            if code_lines[i].strip().upper().startswith('%MEND'):
                return i + 1
            i += 1
        return len(code_lines)

    def _parse_macro_definition(self, code_lines: List[str], start_idx: int) -> int:
        macro_line = code_lines[start_idx].strip()
        match = re.match(r'%MACRO\s+(\w+)(?:\s*\(([^)]*)\))?', macro_line, re.IGNORECASE)
        if not match:
            raise ValueError("Invalid %MACRO statement")

        macro_name = match.group(1)
        param_str = match.group(2) or ''
        parameters = [p.strip() for p in param_str.split(',') if p.strip()]

        body: List[str] = []
        i = start_idx + 1
        while i < len(code_lines):
            if code_lines[i].strip().upper().startswith('%MEND'):
                break
            body.append(code_lines[i])
            i += 1

        self.define_macro(macro_name, parameters, body)
        return i

    def _parse_let_statement(self, line: str) -> None:
        match = re.match(r'%LET\s+(\w+)\s*=\s*(.*?);?\s*$', line, re.IGNORECASE)
        if match:
            self.set_variable(match.group(1), match.group(2).strip())

    def _parse_put_statement(self, line: str) -> None:
        match = re.match(r'%PUT\s+(.*?);?\s*$', line, re.IGNORECASE)
        if match:
            print(f"MACRO: {self._substitute_variables(match.group(1).strip())}")

    def _parse_include_statement(self, line: str) -> List[str]:
        match = re.match(r'%INCLUDE\s+[\'"]([^\'"]+)[\'"];?\s*$', line, re.IGNORECASE)
        if match:
            filepath = match.group(1)
            try:
                with open(filepath, 'r') as f:
                    return f.readlines()
            except FileNotFoundError:
                print(f"Warning: Include file {filepath} not found")
                return []
        return []

    def _try_expand_macro_call(self, line: str) -> Optional[List[str]]:
        line = line.rstrip(';')

        # %macroname(arg1, arg2, ...)
        match = re.match(r'%(\w+)\s*\(([^)]*)\)', line)
        if match:
            macro_name = match.group(1)
            args_str = match.group(2) or ''
            if macro_name in self.macros:
                arguments = [a.strip() for a in args_str.split(',') if a.strip()] if args_str.strip() else []
                return self.expand_macro_call(macro_name, arguments)

        # %macroname (no args)
        match = re.match(r'%(\w+)', line)
        if match:
            macro_name = match.group(1)
            if macro_name in self.macros:
                return self.expand_macro_call(macro_name, [])

        return None

    def list_macros(self) -> Dict[str, List[str]]:
        return {name: md.parameters for name, md in self.macros.items()}

    def list_variables(self) -> Dict[str, str]:
        all_vars = self.global_variables.copy()
        for scope in self.local_scopes:
            all_vars.update(scope)
        return all_vars
