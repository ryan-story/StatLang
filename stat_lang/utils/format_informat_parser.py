"""
SAS FORMAT and INFORMAT Statement Parser

This module parses FORMAT and INFORMAT statements in DATA steps and PROC steps.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FormatStatement:
    """Represents a parsed FORMAT statement."""
    variables: List[str]
    formats: List[str]
    statement_type: str = 'FORMAT'


@dataclass
class InformatStatement:
    """Represents a parsed INFORMAT statement."""
    variables: List[str]
    informats: List[str]
    statement_type: str = 'INFORMAT'


class FormatInformatParser:
    """Parser for FORMAT and INFORMAT statements."""
    
    def __init__(self):
        pass
    
    def parse_format_statement(self, statement: str) -> FormatStatement:
        """Parse a FORMAT statement."""
        # Remove FORMAT keyword and clean up
        statement = statement.strip()
        if statement.upper().startswith('FORMAT'):
            statement = statement[6:].strip()
        
        # Remove trailing semicolon
        if statement.endswith(';'):
            statement = statement[:-1].strip()
        
        # Parse variable-format pairs
        variables, formats = self._parse_variable_format_pairs(statement)
        
        return FormatStatement(
            variables=variables,
            formats=formats,
            statement_type='FORMAT'
        )
    
    def parse_informat_statement(self, statement: str) -> InformatStatement:
        """Parse an INFORMAT statement."""
        # Remove INFORMAT keyword and clean up
        statement = statement.strip()
        if statement.upper().startswith('INFORMAT'):
            statement = statement[8:].strip()
        
        # Remove trailing semicolon
        if statement.endswith(';'):
            statement = statement[:-1].strip()
        
        # Parse variable-informat pairs
        variables, informats = self._parse_variable_format_pairs(statement)
        
        return InformatStatement(
            variables=variables,
            informats=informats,
            statement_type='INFORMAT'
        )
    
    def _parse_variable_format_pairs(self, statement: str) -> Tuple[List[str], List[str]]:
        """Parse variable-format pairs from statement."""
        variables = []
        formats = []
        
        # Split by spaces
        tokens = statement.split()
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check if this looks like a format (contains a dot or is a known format)
            if self._is_format_token(token):
                # This is a format - it should be paired with the previous variable
                if variables:
                    # Apply this format to the last variable
                    formats.append(token)
                else:
                    # Format without preceding variable - this shouldn't happen
                    pass
            else:
                # This is a variable name
                variables.append(token)
            
            i += 1
        
        # Ensure we have the same number of variables and formats
        while len(formats) < len(variables):
            formats.append('BEST12.')  # Default format
        
        return variables, formats
    
    def _is_format_token(self, token: str) -> bool:
        """Check if a token looks like a format."""
        # Common format patterns - be more specific to avoid false positives
        format_patterns = [
            r'^[A-Z$]+(\d+)(?:\.(\d+))?\.?$',  # DOLLAR10.2, DATE9., etc.
            r'^\$\d+\.?$',  # $10., $CHAR10.
            r'^[A-Z]+\d+\.?$'  # BEST12., COMMA10., etc. (must have digits)
        ]
        
        # Known format names without digits
        known_formats = ['BEST', 'COMMA', 'DOLLAR', 'PERCENT', 'DATE', 'TIME', 'DATETIME']
        
        for pattern in format_patterns:
            if re.match(pattern, token.upper()):
                return True
        
        # Check if it's a known format name
        if token.upper() in known_formats:
            return True
        
        return False
    
    def extract_format_statements(self, code_lines: List[str]) -> List[FormatStatement]:
        """Extract all FORMAT statements from code."""
        format_statements = []
        
        for line in code_lines:
            line = line.strip()
            if line.upper().startswith('FORMAT'):
                try:
                    format_stmt = self.parse_format_statement(line)
                    format_statements.append(format_stmt)
                except Exception as e:
                    print(f"Warning: Could not parse FORMAT statement: {line}")
        
        return format_statements
    
    def extract_informat_statements(self, code_lines: List[str]) -> List[InformatStatement]:
        """Extract all INFORMAT statements from code."""
        informat_statements = []
        
        for line in code_lines:
            line = line.strip()
            if line.upper().startswith('INFORMAT'):
                try:
                    informat_stmt = self.parse_informat_statement(line)
                    informat_statements.append(informat_stmt)
                except Exception as e:
                    print(f"Warning: Could not parse INFORMAT statement: {line}")
        
        return informat_statements
    
    def apply_format_statements_to_dataset(self, dataset, format_statements: List[FormatStatement]) -> None:
        """Apply format statements to a dataset."""
        for stmt in format_statements:
            for i, variable in enumerate(stmt.variables):
                if i < len(stmt.formats):
                    format_str = stmt.formats[i]
                    dataset.set_format(variable, format_str)
    
    def apply_informat_statements_to_dataset(self, dataset, informat_statements: List[InformatStatement]) -> None:
        """Apply informat statements to a dataset."""
        for stmt in informat_statements:
            for i, variable in enumerate(stmt.variables):
                if i < len(stmt.informats):
                    informat_str = stmt.informats[i]
                    dataset.set_informat(variable, informat_str)
