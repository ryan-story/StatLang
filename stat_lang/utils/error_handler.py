"""
Error Handler for Open-SAS

This module provides comprehensive error handling and validation
for SAS code execution.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ErrorType(Enum):
    """Types of errors in SAS code."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    WARNING = "warning"
    NOTE = "note"


@dataclass
class SASError:
    """Represents a SAS error or warning."""
    error_type: ErrorType
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None


class ErrorHandler:
    """Comprehensive error handler for SAS code."""
    
    def __init__(self):
        self.errors: List[SASError] = []
        self.warnings: List[SASError] = []
        self.notes: List[SASError] = []
    
    def add_error(self, error_type: ErrorType, message: str, 
                  line_number: Optional[int] = None, 
                  column: Optional[int] = None,
                  code_snippet: Optional[str] = None):
        """Add an error, warning, or note."""
        error = SASError(
            error_type=error_type,
            message=message,
            line_number=line_number,
            column=column,
            code_snippet=code_snippet
        )
        
        if error_type == ErrorType.SYNTAX_ERROR or error_type == ErrorType.RUNTIME_ERROR:
            self.errors.append(error)
        elif error_type == ErrorType.WARNING:
            self.warnings.append(error)
        elif error_type == ErrorType.NOTE:
            self.notes.append(error)
    
    def validate_syntax(self, code: str) -> List[SASError]:
        """
        Validate SAS code syntax.
        
        Args:
            code: SAS code to validate
            
        Returns:
            List of syntax errors found
        """
        syntax_errors = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('/*'):
                continue
            
            # Check for common syntax errors
            errors = self._validate_line_syntax(line, i)
            syntax_errors.extend(errors)
        
        return syntax_errors
    
    def _validate_line_syntax(self, line: str, line_number: int) -> List[SASError]:
        """Validate syntax of a single line."""
        errors = []
        
        # Check for missing semicolons
        if self._requires_semicolon(line) and not line.endswith(';'):
            errors.append(SASError(
                error_type=ErrorType.SYNTAX_ERROR,
                message="Statement not terminated with semicolon",
                line_number=line_number,
                code_snippet=line
            ))
        
        # Check for unmatched quotes
        if self._has_unmatched_quotes(line):
            errors.append(SASError(
                error_type=ErrorType.SYNTAX_ERROR,
                message="Unmatched quotes in statement",
                line_number=line_number,
                code_snippet=line
            ))
        
        # Check for invalid DATA step syntax
        if line.upper().startswith('DATA '):
            if not re.match(r'data\s+\w+', line, re.IGNORECASE):
                errors.append(SASError(
                    error_type=ErrorType.SYNTAX_ERROR,
                    message="Invalid DATA statement syntax",
                    line_number=line_number,
                    code_snippet=line
                ))
        
        # Check for invalid PROC syntax
        if line.upper().startswith('PROC '):
            if not re.match(r'proc\s+\w+', line, re.IGNORECASE):
                errors.append(SASError(
                    error_type=ErrorType.SYNTAX_ERROR,
                    message="Invalid PROC statement syntax",
                    line_number=line_number,
                    code_snippet=line
                ))
        
        # Check for invalid %LET syntax
        if line.upper().startswith('%LET '):
            if not re.match(r'%let\s+\w+\s*=\s*.+', line, re.IGNORECASE):
                errors.append(SASError(
                    error_type=ErrorType.SYNTAX_ERROR,
                    message="Invalid %LET statement syntax",
                    line_number=line_number,
                    code_snippet=line
                ))
        
        return errors
    
    def _requires_semicolon(self, line: str) -> bool:
        """Check if a line requires a semicolon."""
        line_upper = line.upper()
        
        # Statements that require semicolons
        semicolon_required = [
            'DATA ', 'PROC ', 'SET ', 'MERGE ', 'WHERE ', 'IF ',
            'DROP ', 'KEEP ', 'RENAME ', 'BY ', 'VAR ', 'TABLES ',
            'CLASS ', 'MODEL ', 'OUTPUT ', '%LET ', 'LIBNAME '
        ]
        
        for stmt in semicolon_required:
            if line_upper.startswith(stmt):
                return True
        
        return False
    
    def _has_unmatched_quotes(self, line: str) -> bool:
        """Check for unmatched quotes in a line."""
        single_quotes = line.count("'")
        double_quotes = line.count('"')
        
        return single_quotes % 2 != 0 or double_quotes % 2 != 0
    
    def validate_data_step(self, data_step_code: str) -> List[SASError]:
        """
        Validate DATA step syntax.
        
        Args:
            data_step_code: DATA step code to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        lines = data_step_code.split('\n')
        
        has_data_statement = False
        has_run_statement = False
        in_datalines = False
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            line_upper = line.upper()
            
            # Check for DATA statement
            if line_upper.startswith('DATA '):
                has_data_statement = True
                if not re.match(r'data\s+\w+', line, re.IGNORECASE):
                    errors.append(SASError(
                        error_type=ErrorType.SYNTAX_ERROR,
                        message="Invalid DATA statement",
                        line_number=i,
                        code_snippet=line
                    ))
            
            # Check for RUN statement
            elif line_upper == 'RUN;':
                has_run_statement = True
            
            # Check for DATALINES
            elif line_upper in ['DATALINES;', 'CARDS;']:
                in_datalines = True
            
            # Check for INPUT statement
            elif line_upper.startswith('INPUT '):
                if not re.match(r'input\s+.+', line, re.IGNORECASE):
                    errors.append(SASError(
                        error_type=ErrorType.SYNTAX_ERROR,
                        message="Invalid INPUT statement",
                        line_number=i,
                        code_snippet=line
                    ))
        
        # Check for required statements
        if not has_data_statement:
            errors.append(SASError(
                error_type=ErrorType.SYNTAX_ERROR,
                message="DATA step must start with DATA statement"
            ))
        
        if not has_run_statement:
            errors.append(SASError(
                error_type=ErrorType.SYNTAX_ERROR,
                message="DATA step must end with RUN statement"
            ))
        
        return errors
    
    def validate_proc(self, proc_code: str) -> List[SASError]:
        """
        Validate PROC syntax.
        
        Args:
            proc_code: PROC code to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        lines = proc_code.split('\n')
        
        has_proc_statement = False
        has_run_statement = False
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            line_upper = line.upper()
            
            # Check for PROC statement
            if line_upper.startswith('PROC '):
                has_proc_statement = True
                if not re.match(r'proc\s+\w+', line, re.IGNORECASE):
                    errors.append(SASError(
                        error_type=ErrorType.SYNTAX_ERROR,
                        message="Invalid PROC statement",
                        line_number=i,
                        code_snippet=line
                    ))
            
            # Check for RUN statement
            elif line_upper in ['RUN;', 'QUIT;']:
                has_run_statement = True
        
        # Check for required statements
        if not has_proc_statement:
            errors.append(SASError(
                error_type=ErrorType.SYNTAX_ERROR,
                message="PROC must start with PROC statement"
            ))
        
        if not has_run_statement:
            errors.append(SASError(
                error_type=ErrorType.SYNTAX_ERROR,
                message="PROC must end with RUN or QUIT statement"
            ))
        
        return errors
    
    def format_errors(self) -> str:
        """Format all errors, warnings, and notes for display."""
        output = []
        
        if self.errors:
            output.append("ERRORS:")
            for error in self.errors:
                output.append(f"  ERROR: {error.message}")
                if error.line_number:
                    output.append(f"    Line {error.line_number}: {error.code_snippet}")
        
        if self.warnings:
            output.append("\nWARNINGS:")
            for warning in self.warnings:
                output.append(f"  WARNING: {warning.message}")
                if warning.line_number:
                    output.append(f"    Line {warning.line_number}: {warning.code_snippet}")
        
        if self.notes:
            output.append("\nNOTES:")
            for note in self.notes:
                output.append(f"  NOTE: {note.message}")
                if note.line_number:
                    output.append(f"    Line {note.line_number}: {note.code_snippet}")
        
        return "\n".join(output)
    
    def clear(self):
        """Clear all errors, warnings, and notes."""
        self.errors.clear()
        self.warnings.clear()
        self.notes.clear()
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
