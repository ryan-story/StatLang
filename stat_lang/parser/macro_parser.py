"""
Macro Parser for Open-SAS

This module handles SAS macro language processing including
macro variable resolution, macro definitions, and macro calls.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MacroDefinition:
    """Represents a SAS macro definition."""
    name: str
    parameters: List[str]
    body: str
    line_start: int
    line_end: int


@dataclass
class MacroCall:
    """Represents a SAS macro call."""
    name: str
    arguments: Dict[str, str]
    line_number: int


class MacroParser:
    """Parser and processor for SAS macro language."""
    
    def __init__(self):
        self.macro_variables: Dict[str, str] = {}
        self.macro_definitions: Dict[str, MacroDefinition] = {}
        
    def resolve_macro_variables(self, code: str) -> str:
        """
        Resolve macro variable references in code.
        
        Args:
            code: Code containing macro variable references
            
        Returns:
            Code with macro variables resolved
        """
        # Pattern to match macro variable references like &var or &var.
        pattern = r'&([a-zA-Z_][a-zA-Z0-9_]*)\.?'
        
        def replace_macro_var(match):
            var_name = match.group(1).upper()  # Convert to uppercase for lookup
            return self.macro_variables.get(var_name, f"&{match.group(1)}")
        
        return re.sub(pattern, replace_macro_var, code)
    
    def parse_let_statement(self, statement: str) -> Tuple[str, str]:
        """
        Parse a %LET statement.
        
        Args:
            statement: The %LET statement to parse
            
        Returns:
            Tuple of (variable_name, value)
        """
        # Pattern: %LET variable = value;
        match = re.match(r'%let\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*?);?$', statement, re.IGNORECASE)
        if match:
            var_name = match.group(1).upper()
            value = match.group(2).strip()
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            return var_name, value
        else:
            raise ValueError(f"Invalid %LET statement: {statement}")
    
    def parse_put_statement(self, statement: str) -> str:
        """
        Parse a %PUT statement.
        
        Args:
            statement: The %PUT statement to parse
            
        Returns:
            Message to output
        """
        # Pattern: %PUT message;
        match = re.match(r'%put\s+(.*?);?$', statement, re.IGNORECASE)
        if match:
            message = match.group(1).strip()
            # Resolve any macro variables in the message
            return self.resolve_macro_variables(message)
        else:
            raise ValueError(f"Invalid %PUT statement: {statement}")
    
    def parse_macro_definition(self, code: str) -> MacroDefinition:
        """
        Parse a macro definition (%MACRO ... %MEND).
        
        Args:
            code: The macro definition code
            
        Returns:
            MacroDefinition object
        """
        lines = code.split('\n')
        
        # Find %MACRO statement
        macro_start = None
        macro_name = None
        parameters = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.upper().startswith('%MACRO '):
                macro_start = i
                # Parse macro name and parameters
                match = re.match(r'%macro\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\(([^)]*)\))?', line, re.IGNORECASE)
                if match:
                    macro_name = match.group(1).upper()
                    params_str = match.group(2)
                    if params_str:
                        # Parse parameters (simple comma-separated list for now)
                        parameters = [p.strip() for p in params_str.split(',')]
                break
        
        if macro_start is None:
            raise ValueError("No %MACRO statement found")
            
        # Find %MEND statement
        mend_line = None
        for i in range(macro_start + 1, len(lines)):
            line = lines[i].strip()
            if line.upper().startswith('%MEND'):
                mend_line = i
                break
        
        if mend_line is None:
            raise ValueError("No %MEND statement found")
            
        # Extract macro body
        body_lines = lines[macro_start + 1:mend_line]
        body = '\n'.join(body_lines)
        
        return MacroDefinition(
            name=macro_name,
            parameters=parameters,
            body=body,
            line_start=macro_start,
            line_end=mend_line
        )
    
    def parse_macro_call(self, statement: str) -> MacroCall:
        """
        Parse a macro call.
        
        Args:
            statement: The macro call statement
            
        Returns:
            MacroCall object
        """
        # Pattern: %macroname or %macroname(arg1, arg2) or %macroname(param1=value1, param2=value2)
        match = re.match(r'%([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\(([^)]*)\))?', statement)
        if match:
            macro_name = match.group(1).upper()
            args_str = match.group(2)
            arguments = {}
            
            if args_str:
                # Parse arguments
                args = [arg.strip() for arg in args_str.split(',')]
                for arg in args:
                    if '=' in arg:
                        # Named parameter
                        key, value = arg.split('=', 1)
                        arguments[key.strip()] = value.strip()
                    else:
                        # Positional parameter
                        arguments[f"param{len(arguments) + 1}"] = arg
            
            return MacroCall(
                name=macro_name,
                arguments=arguments,
                line_number=0  # Would need to track line numbers in full implementation
            )
        else:
            raise ValueError(f"Invalid macro call: {statement}")
    
    def expand_macro(self, macro_call: MacroCall) -> str:
        """
        Expand a macro call into its body with parameter substitution.
        
        Args:
            macro_call: The macro call to expand
            
        Returns:
            Expanded macro body
        """
        if macro_call.name not in self.macro_definitions:
            raise ValueError(f"Macro {macro_call.name} not defined")
            
        macro_def = self.macro_definitions[macro_call.name]
        body = macro_def.body
        
        # Substitute parameters
        for param_name, param_value in macro_call.arguments.items():
            if param_name in macro_def.parameters:
                # Replace parameter references in body
                pattern = f"&{param_name}\\b"
                body = re.sub(pattern, param_value, body)
        
        # Resolve any remaining macro variables
        body = self.resolve_macro_variables(body)
        
        return body
    
    def process_macro_code(self, code: str) -> str:
        """
        Process code containing macro language constructs.
        
        Args:
            code: Code containing macro constructs
            
        Returns:
            Code with macros processed and expanded
        """
        lines = code.split('\n')
        processed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Handle %LET statements
            if line.upper().startswith('%LET '):
                try:
                    var_name, value = self.parse_let_statement(line)
                    self.macro_variables[var_name] = value
                    # Don't include %LET in output
                except ValueError as e:
                    processed_lines.append(f"/* ERROR: {e} */")
                i += 1
                
            # Handle %PUT statements
            elif line.upper().startswith('%PUT '):
                try:
                    message = self.parse_put_statement(line)
                    processed_lines.append(f"/* %PUT: {message} */")
                except ValueError as e:
                    processed_lines.append(f"/* ERROR: {e} */")
                i += 1
                
            # Handle macro definitions
            elif line.upper().startswith('%MACRO '):
                # Find the complete macro definition
                macro_lines = []
                j = i
                while j < len(lines):
                    macro_lines.append(lines[j])
                    if lines[j].strip().upper().startswith('%MEND'):
                        break
                    j += 1
                
                if j < len(lines):
                    macro_code = '\n'.join(macro_lines)
                    try:
                        macro_def = self.parse_macro_definition(macro_code)
                        self.macro_definitions[macro_def.name] = macro_def
                        # Don't include macro definition in output
                    except ValueError as e:
                        processed_lines.append(f"/* ERROR: {e} */")
                    i = j + 1
                else:
                    processed_lines.append(line)
                    i += 1
                    
            # Handle macro calls
            elif '%' in line and re.search(r'%[a-zA-Z_][a-zA-Z0-9_]*', line):
                # Check if this is a macro call
                macro_match = re.search(r'%([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\([^)]*\))?', line)
                if macro_match and macro_match.group(1).upper() in self.macro_definitions:
                    try:
                        macro_call = self.parse_macro_call(macro_match.group(0))
                        expanded_code = self.expand_macro(macro_call)
                        # Replace the macro call with expanded code
                        line = line.replace(macro_match.group(0), expanded_code)
                        processed_lines.append(line)
                    except ValueError as e:
                        processed_lines.append(f"/* ERROR: {e} */")
                else:
                    # Just resolve macro variables
                    processed_lines.append(self.resolve_macro_variables(line))
                i += 1
                
            else:
                # Regular line, just resolve macro variables
                processed_lines.append(self.resolve_macro_variables(line))
                i += 1
        
        return '\n'.join(processed_lines)
