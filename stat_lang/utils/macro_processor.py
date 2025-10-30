"""
Macro facility implementation for StatLang

This module implements the macro system including:
- Macro definitions (%MACRO/%MEND)
- Macro variables (%LET, & substitution)
- Conditional logic (%IF/%THEN/%ELSE)
- Loops (%DO/%END)
- Macro execution and code generation
"""

import re
import ast
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from collections import deque


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
    scope: str = 'global'  # 'global' or 'local'


class MacroProcessor:
    """Macro Processor for StatLang."""
    
    def __init__(self):
        # Global macro registry
        self.macros: Dict[str, MacroDefinition] = {}
        
        # Macro variable storage
        self.global_variables: Dict[str, str] = {}
        self.local_scopes: deque = deque()  # Stack of local variable dictionaries
        
        # Built-in system variables
        self._initialize_system_variables()
    
    def _initialize_system_variables(self):
        """Initialize system variables."""
        self.global_variables.update({
            'SYSVER': 'StatLang v0.1.2',
            'SYSDATE': '2024-01-01',
            'SYSTIME': '12:00:00',
            'SYSUSERID': 'user',
            'SYSPROCESSID': '1',
            'SYSPROCESSNAME': 'statlang'
        })
    
    def define_macro(self, name: str, parameters: List[str], body: List[str]) -> None:
        """Define a new macro."""
        macro_def = MacroDefinition(
            name=name,
            parameters=parameters,
            body=body,
            is_global=True
        )
        self.macros[name] = macro_def
    
    def set_variable(self, name: str, value: str, scope: str = 'auto') -> None:
        """Set a macro variable."""
        if scope == 'auto':
            # If we're in a local scope, use local; otherwise global
            scope = 'local' if self.local_scopes else 'global'
        
        if scope == 'local' and self.local_scopes:
            self.local_scopes[-1][name] = value
        else:
            self.global_variables[name] = value
    
    def get_variable(self, name: str) -> Optional[str]:
        """Get a macro variable value."""
        # Check local scopes first (most recent first)
        for scope in reversed(self.local_scopes):
            if name in scope:
                return scope[name]
        
        # Check global variables
        return self.global_variables.get(name)
    
    def push_local_scope(self, parameters: Dict[str, str] = None) -> None:
        """Push a new local scope onto the stack."""
        local_vars = parameters or {}
        self.local_scopes.append(local_vars)
    
    def pop_local_scope(self) -> None:
        """Pop the current local scope from the stack."""
        if self.local_scopes:
            self.local_scopes.pop()
    
    def expand_macro_call(self, macro_name: str, arguments: List[str]) -> List[str]:
        """Expand a macro call into executable code."""
        if macro_name not in self.macros:
            raise ValueError(f"Macro {macro_name} not defined")
        
        macro_def = self.macros[macro_name]
        
        # Create parameter mapping
        param_map = {}
        for i, param in enumerate(macro_def.parameters):
            if i < len(arguments):
                param_map[param] = arguments[i]
            else:
                param_map[param] = ''  # Empty string for missing parameters
        
        # Push local scope with parameters
        self.push_local_scope(param_map)
        
        try:
            # Process macro body
            expanded_code = []
            i = 0
            while i < len(macro_def.body):
                line = macro_def.body[i]
                
                # Handle macro control structures
                if line.strip().startswith('%IF'):
                    i = self._process_if_statement(macro_def.body, i, expanded_code)
                elif line.strip().startswith('%DO'):
                    i = self._process_do_loop(macro_def.body, i, expanded_code)
                else:
                    # Regular macro line - substitute variables
                    expanded_line = self._substitute_variables(line)
                    expanded_code.append(expanded_line)
                
                i += 1
            
            return expanded_code
        
        finally:
            # Always pop the local scope
            self.pop_local_scope()
    
    def _process_if_statement(self, body: List[str], start_idx: int, output: List[str]) -> int:
        """Process %IF/%THEN/%ELSE statement."""
        if_line = body[start_idx].strip()
        
        # Parse condition
        condition = self._parse_if_condition(if_line)
        
        # Find %THEN and %ELSE
        then_idx = None
        else_idx = None
        end_idx = None
        
        for i in range(start_idx + 1, len(body)):
            line = body[i].strip()
            if line.startswith('%THEN'):
                then_idx = i
            elif line.startswith('%ELSE'):
                else_idx = i
            elif line.startswith('%END'):
                end_idx = i
                break
        
        if then_idx is None:
            raise ValueError("Missing %THEN in %IF statement")
        
        # Evaluate condition
        if self._evaluate_condition(condition):
            # Process %THEN block
            if else_idx is not None:
                end_block = else_idx
            else:
                end_block = end_idx if end_idx else len(body)
            
            for i in range(then_idx + 1, end_block):
                line = body[i]
                if not line.strip().startswith('%'):
                    expanded_line = self._substitute_variables(line)
                    output.append(expanded_line)
        else:
            # Process %ELSE block if present
            if else_idx is not None:
                end_block = end_idx if end_idx else len(body)
                for i in range(else_idx + 1, end_block):
                    line = body[i]
                    if not line.strip().startswith('%'):
                        expanded_line = self._substitute_variables(line)
                        output.append(expanded_line)
        
        return end_idx if end_idx else len(body) - 1
    
    def _process_do_loop(self, body: List[str], start_idx: int, output: List[str]) -> int:
        """Process %DO/%END loop."""
        do_line = body[start_idx].strip()
        
        # Parse loop parameters
        loop_params = self._parse_do_loop(do_line)
        
        # Find matching %END
        end_idx = None
        for i in range(start_idx + 1, len(body)):
            if body[i].strip().startswith('%END'):
                end_idx = i
                break
        
        if end_idx is None:
            raise ValueError("Missing %END for %DO loop")
        
        # Execute loop
        start_val = int(self._substitute_variables(loop_params['start']))
        end_val = int(self._substitute_variables(loop_params['end']))
        step = int(self._substitute_variables(loop_params.get('step', '1')))
        
        for loop_val in range(start_val, end_val + 1, step):
            # Set loop variable
            self.set_variable(loop_params['var'], str(loop_val), 'local')
            
            # Process loop body
            for i in range(start_idx + 1, end_idx):
                line = body[i]
                if not line.strip().startswith('%'):
                    expanded_line = self._substitute_variables(line)
                    output.append(expanded_line)
        
        return end_idx
    
    def _parse_if_condition(self, if_line: str) -> str:
        """Parse condition from %IF statement."""
        # Extract condition between %IF and %THEN
        match = re.search(r'%IF\s+(.+?)\s+%THEN', if_line, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        raise ValueError("Invalid %IF statement format")
    
    def _parse_do_loop(self, do_line: str) -> Dict[str, str]:
        """Parse %DO loop parameters."""
        # Pattern: %DO var = start %TO end %BY step
        pattern = r'%DO\s+(\w+)\s*=\s*([^%]+?)\s+%TO\s+([^%]+?)(?:\s+%BY\s+([^%]+?))?'
        match = re.search(pattern, do_line, re.IGNORECASE)
        
        if match:
            return {
                'var': match.group(1),
                'start': match.group(2).strip(),
                'end': match.group(3).strip(),
                'step': match.group(4).strip() if match.group(4) else '1'
            }
        raise ValueError("Invalid %DO loop format")
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a macro condition."""
        # Substitute variables in condition
        condition = self._substitute_variables(condition)
        
        # Handle common operators
        condition = condition.replace('=', '==')
        condition = condition.replace('&', 'and')
        condition = condition.replace('|', 'or')
        
        # Simple evaluation using Python's eval (with safety considerations)
        try:
            # Only allow safe operations
            allowed_names = {
                '__builtins__': {},
                'True': True,
                'False': False,
                'None': None
            }
            return bool(ast.literal_eval(condition))
        except:
            # Fallback to string comparison
            return condition.lower() in ('true', '1', 'yes')
    
    def _substitute_variables(self, text: str) -> str:
        """Substitute macro variables in text."""
        # Handle multiple ampersands (&&var -> &var after first pass)
        while '&&' in text:
            text = text.replace('&&', '&')
        
        # Substitute single ampersands
        def replace_var(match):
            var_name = match.group(1)
            value = self.get_variable(var_name)
            return value if value is not None else match.group(0)
        
        # Pattern for &variable (including with period delimiter)
        pattern = r'&(\w+)\.?'
        text = re.sub(pattern, replace_var, text)
        
        return text
    
    def process_macro_statements(self, code_lines: List[str]) -> List[str]:
        """Process all macro statements in code and return expanded code."""
        # First pass: Define all macros and process other macro statements
        i = 0
        while i < len(code_lines):
            line = code_lines[i].strip()
            
            if line.upper().startswith('%MACRO'):
                # Define macro
                i = self._parse_macro_definition(code_lines, i)
            elif line.upper().startswith('%LET'):
                # Set macro variable
                self._parse_let_statement(line)
            elif line.upper().startswith('%PUT'):
                # Print macro variable
                self._parse_put_statement(line)
            elif line.upper().startswith('%INCLUDE'):
                # Include external file
                self._parse_include_statement(line)
            
            i += 1
        
        # Second pass: Expand macro calls and process regular lines
        result = []
        i = 0
        
        while i < len(code_lines):
            line = code_lines[i].strip()
            
            if line.upper().startswith('%MACRO'):
                # Skip macro definition lines (already processed)
                i = self._skip_macro_definition(code_lines, i)
                # Don't increment i here since _skip_macro_definition already positioned us correctly
            elif line.upper().startswith('%LET'):
                # Skip %LET lines (already processed)
                i += 1
            elif line.upper().startswith('%PUT'):
                # Skip %PUT lines (already processed)
                i += 1
            elif line.upper().startswith('%INCLUDE'):
                # Skip %INCLUDE lines (already processed)
                i += 1
            elif '%' in line and ('(' in line or re.match(r'%\w+', line)):
                # Potential macro call
                expanded = self._try_expand_macro_call(line)
                if expanded:
                    result.extend(expanded)
                else:
                    # Regular line with variable substitution
                    expanded_line = self._substitute_variables(line)
                    result.append(expanded_line)
                i += 1
            else:
                # Regular line with variable substitution
                expanded_line = self._substitute_variables(line)
                result.append(expanded_line)
                i += 1
        
        return result
    
    def _skip_macro_definition(self, code_lines: List[str], start_idx: int) -> int:
        """Skip macro definition from %MACRO to %MEND."""
        i = start_idx + 1
        while i < len(code_lines):
            line = code_lines[i].strip()
            if line.upper().startswith('%MEND'):
                return i + 1
            i += 1
        return len(code_lines)
    
    def _parse_macro_definition(self, code_lines: List[str], start_idx: int) -> int:
        """Parse macro definition from %MACRO to %MEND."""
        macro_line = code_lines[start_idx].strip()
        
        # Extract macro name and parameters
        match = re.match(r'%MACRO\s+(\w+)(?:\s*\(([^)]*)\))?', macro_line, re.IGNORECASE)
        if not match:
            raise ValueError("Invalid %MACRO statement")
        
        macro_name = match.group(1)
        param_str = match.group(2) or ''
        
        # Parse parameters
        parameters = []
        if param_str.strip():
            params = [p.strip() for p in param_str.split(',')]
            parameters = [p for p in params if p]
        
        # Collect macro body until %MEND
        body = []
        i = start_idx + 1
        while i < len(code_lines):
            line = code_lines[i].strip()
            if line.startswith('%MEND'):
                break
            body.append(code_lines[i])  # Keep original formatting
            i += 1
        
        # Define the macro
        self.define_macro(macro_name, parameters, body)
        
        return i
    
    def _parse_let_statement(self, line: str) -> None:
        """Parse %LET statement."""
        # Pattern: %LET variable = value;
        match = re.match(r'%LET\s+(\w+)\s*=\s*(.*?);?\s*$', line, re.IGNORECASE)
        if match:
            var_name = match.group(1)
            var_value = match.group(2).strip()
            self.set_variable(var_name, var_value)
    
    def _parse_put_statement(self, line: str) -> None:
        """Parse %PUT statement."""
        # Pattern: %PUT text;
        match = re.match(r'%PUT\s+(.*?);?\s*$', line, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            expanded_text = self._substitute_variables(text)
            print(f"MACRO: {expanded_text}")
    
    def _parse_include_statement(self, line: str) -> List[str]:
        """Parse %INCLUDE statement."""
        # Pattern: %INCLUDE 'filepath';
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
        """Try to expand a macro call."""
        # Remove trailing semicolon if present
        line = line.rstrip(';')
        
        # Pattern 1: %macroname(arg1, arg2, ...) - with parameters
        match = re.match(r'%(\w+)\s*\(([^)]*)\)', line)
        if match:
            macro_name = match.group(1)
            args_str = match.group(2) or ''
            
            if macro_name in self.macros:
                # Parse arguments
                arguments = []
                if args_str.strip():
                    # Simple argument parsing (doesn't handle quoted strings with commas)
                    args = [arg.strip() for arg in args_str.split(',')]
                    arguments = [arg for arg in args if arg]
                
                # Expand the macro
                return self.expand_macro_call(macro_name, arguments)
        
        # Pattern 2: %macroname - without parameters
        match = re.match(r'%(\w+)', line)
        if match:
            macro_name = match.group(1)
            
            if macro_name in self.macros:
                # Expand the macro without arguments
                return self.expand_macro_call(macro_name, [])
        
        return None
    
    def list_macros(self) -> Dict[str, List[str]]:
        """List all defined macros and their parameters."""
        result = {}
        for name, macro_def in self.macros.items():
            result[name] = macro_def.parameters
        return result
    
    def list_variables(self) -> Dict[str, str]:
        """List all macro variables."""
        all_vars = self.global_variables.copy()
        
        # Add local variables
        for scope in self.local_scopes:
            all_vars.update(scope)
        
        return all_vars
