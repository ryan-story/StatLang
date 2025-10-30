"""
PROC Parser for Open-SAS

This module parses SAS PROC procedure syntax and extracts parameters
for execution by the appropriate PROC implementation.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ProcStatement:
    """Represents a parsed PROC statement."""
    proc_name: str
    options: Dict[str, Any]
    statements: List[str]
    data_option: Optional[str] = None
    output_option: Optional[str] = None


class ProcParser:
    """Parser for SAS PROC procedure syntax."""
    
    def __init__(self):
        pass
        
    def parse_proc(self, code: str) -> ProcStatement:
        """
        Parse a PROC procedure.
        
        Args:
            code: The PROC code to parse
            
        Returns:
            ProcStatement object containing parsed information
        """
        lines = code.split('\n')
        
        # Find the PROC statement
        proc_line = None
        proc_name = None
        data_option = None
        output_option = None
        options = {}
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('PROC '):
                proc_line = line
                # Extract PROC name and options
                match = re.match(r'proc\s+(\w+)(?:\s+(.+?))?(?:\s*;)?$', line, re.IGNORECASE)
                if match:
                    proc_name = match.group(1).upper()
                    options_str = match.group(2) if match.group(2) else ""
                    
                    # Parse DATA= option
                    data_match = re.search(r'data\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if data_match:
                        data_option = data_match.group(1)
                        
                    # Parse OUT= option
                    out_match = re.search(r'out\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if out_match:
                        output_option = out_match.group(1)
                    
                    # Parse NOPRINT option
                    if re.search(r'\bnoprint\b', options_str, re.IGNORECASE):
                        options['noprint'] = True
                    
                    # Parse PROMPT option
                    prompt_match = re.search(r'prompt\s*=\s*[\'"]([^\'"]+)[\'"]', options_str, re.IGNORECASE)
                    if prompt_match:
                        options['prompt'] = prompt_match.group(1)
                    
                    # Parse MODEL option
                    model_match = re.search(r'model\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if model_match:
                        options['model'] = model_match.group(1)
                    
                    # Parse MODE option
                    mode_match = re.search(r'mode\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if mode_match:
                        options['mode'] = mode_match.group(1)
                    
                    # Parse SURVEYSELECT-specific options
                    # METHOD option
                    method_match = re.search(r'method\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if method_match:
                        options['method'] = method_match.group(1)
                    
                    # SAMPRATE option
                    samprate_match = re.search(r'samprate\s*=\s*([\d.]+)', options_str, re.IGNORECASE)
                    if samprate_match:
                        options['samprate'] = samprate_match.group(1)
                    
                    # N option
                    n_match = re.search(r'n\s*=\s*(\d+)', options_str, re.IGNORECASE)
                    if n_match:
                        options['n'] = n_match.group(1)
                    
                    # SEED option
                    seed_match = re.search(r'seed\s*=\s*(\d+)', options_str, re.IGNORECASE)
                    if seed_match:
                        options['seed'] = seed_match.group(1)
                    
                    # OUTALL option
                    if re.search(r'\boutall\b', options_str, re.IGNORECASE):
                        options['outall'] = True
                    
                    # Parse PROMPT option
                    prompt_match = re.search(r'prompt\s*=\s*[\'"]([^\'"]+)[\'"]', options_str, re.IGNORECASE)
                    if prompt_match:
                        options['prompt'] = prompt_match.group(1)
                    
                    # Parse MODEL option
                    model_match = re.search(r'model\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if model_match:
                        options['model'] = model_match.group(1)
                    
                    # Parse MODE option
                    mode_match = re.search(r'mode\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if mode_match:
                        options['mode'] = mode_match.group(1)
                    
                    # Parse SURVEYSELECT-specific options
                    # METHOD option
                    method_match = re.search(r'method\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if method_match:
                        options['method'] = method_match.group(1)
                    
                    # SAMPRATE option
                    samprate_match = re.search(r'samprate\s*=\s*([\d.]+)', options_str, re.IGNORECASE)
                    if samprate_match:
                        options['samprate'] = samprate_match.group(1)
                    
                    # N option
                    n_match = re.search(r'n\s*=\s*(\d+)', options_str, re.IGNORECASE)
                    if n_match:
                        options['n'] = n_match.group(1)
                    
                    # SEED option
                    seed_match = re.search(r'seed\s*=\s*(\d+)', options_str, re.IGNORECASE)
                    if seed_match:
                        options['seed'] = seed_match.group(1)
                    
                    # OUTALL option
                    if re.search(r'\boutall\b', options_str, re.IGNORECASE):
                        options['outall'] = True
                    
                    # Parse NOPRINT option
                    if re.search(r'\bnoprint\b', options_str, re.IGNORECASE):
                        options['noprint'] = True
                    
                    # Parse PROMPT option
                    prompt_match = re.search(r'prompt\s*=\s*[\'"]([^\'"]+)[\'"]', options_str, re.IGNORECASE)
                    if prompt_match:
                        options['prompt'] = prompt_match.group(1)
                    
                    # Parse MODEL option
                    model_match = re.search(r'model\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if model_match:
                        options['model'] = model_match.group(1)
                    
                    # Parse MODE option
                    mode_match = re.search(r'mode\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if mode_match:
                        options['mode'] = mode_match.group(1)
                    
                    # Parse SURVEYSELECT-specific options
                    # METHOD option
                    method_match = re.search(r'method\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if method_match:
                        options['method'] = method_match.group(1)
                    
                    # SAMPRATE option
                    samprate_match = re.search(r'samprate\s*=\s*([\d.]+)', options_str, re.IGNORECASE)
                    if samprate_match:
                        options['samprate'] = samprate_match.group(1)
                    
                    # N option
                    n_match = re.search(r'n\s*=\s*(\d+)', options_str, re.IGNORECASE)
                    if n_match:
                        options['n'] = n_match.group(1)
                    
                    # SEED option
                    seed_match = re.search(r'seed\s*=\s*(\d+)', options_str, re.IGNORECASE)
                    if seed_match:
                        options['seed'] = seed_match.group(1)
                    
                    # OUTALL option
                    if re.search(r'\boutall\b', options_str, re.IGNORECASE):
                        options['outall'] = True
                    
                    # Parse SURVEYSELECT-specific options
                    # METHOD option
                    method_match = re.search(r'method\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if method_match:
                        options['method'] = method_match.group(1)
                    
                    # SAMPRATE option
                    samprate_match = re.search(r'samprate\s*=\s*([\d.]+)', options_str, re.IGNORECASE)
                    if samprate_match:
                        options['samprate'] = samprate_match.group(1)
                    
                    # N option
                    n_match = re.search(r'n\s*=\s*(\d+)', options_str, re.IGNORECASE)
                    if n_match:
                        options['n'] = n_match.group(1)
                    
                    # SEED option
                    seed_match = re.search(r'seed\s*=\s*(\d+)', options_str, re.IGNORECASE)
                    if seed_match:
                        options['seed'] = seed_match.group(1)
                    
                    # OUTALL option
                    if re.search(r'\boutall\b', options_str, re.IGNORECASE):
                        options['outall'] = True
                break
        
        # If no PROC line found, try to parse the first line as a combined PROC statement
        if not proc_line and lines:
            first_line = lines[0].strip()
            if first_line.upper().startswith('PROC '):
                proc_line = first_line
                # Extract PROC name and options
                match = re.match(r'proc\s+(\w+)(?:\s+(.+?))?(?:\s*;)?$', first_line, re.IGNORECASE)
                if match:
                    proc_name = match.group(1).upper()
                    options_str = match.group(2) if match.group(2) else ""
                    
                    # Parse DATA= option
                    data_match = re.search(r'data\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if data_match:
                        data_option = data_match.group(1)
                        
                    # Parse OUT= option
                    out_match = re.search(r'out\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if out_match:
                        output_option = out_match.group(1)
                    
                    # Parse NOPRINT option
                    if re.search(r'\bnoprint\b', options_str, re.IGNORECASE):
                        options['noprint'] = True
                    
                    # Parse PROMPT option
                    prompt_match = re.search(r'prompt\s*=\s*[\'"]([^\'"]+)[\'"]', options_str, re.IGNORECASE)
                    if prompt_match:
                        options['prompt'] = prompt_match.group(1)
                    
                    # Parse MODEL option
                    model_match = re.search(r'model\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if model_match:
                        options['model'] = model_match.group(1)
                    
                    # Parse MODE option
                    mode_match = re.search(r'mode\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if mode_match:
                        options['mode'] = mode_match.group(1)
                    
                    # Parse SURVEYSELECT-specific options
                    # METHOD option
                    method_match = re.search(r'method\s*=\s*([\w.]+)', options_str, re.IGNORECASE)
                    if method_match:
                        options['method'] = method_match.group(1)
                    
                    # SAMPRATE option
                    samprate_match = re.search(r'samprate\s*=\s*([\d.]+)', options_str, re.IGNORECASE)
                    if samprate_match:
                        options['samprate'] = samprate_match.group(1)
                    
                    # N option
                    n_match = re.search(r'n\s*=\s*(\d+)', options_str, re.IGNORECASE)
                    if n_match:
                        options['n'] = n_match.group(1)
                    
                    # SEED option
                    seed_match = re.search(r'seed\s*=\s*(\d+)', options_str, re.IGNORECASE)
                    if seed_match:
                        options['seed'] = seed_match.group(1)
                    
                    # OUTALL option
                    if re.search(r'\boutall\b', options_str, re.IGNORECASE):
                        options['outall'] = True
        
        if not proc_line:
            raise ValueError("No PROC statement found")
            
        # Parse remaining statements
        statements = []
        
        # Special handling for PROC SQL - treat everything as SQL statements
        if proc_name == 'SQL':
            # For PROC SQL, collect all lines between PROC SQL and RUN as SQL statements
            sql_lines = []
            in_sql_block = False
            
            for line in lines:
                line = line.strip()
                if line.upper().startswith('PROC SQL'):
                    in_sql_block = True
                    continue
                elif line.upper() in ['RUN;', 'QUIT;']:
                    break
                elif in_sql_block and line and not line.startswith('*') and not line.startswith('/*'):
                    sql_lines.append(line)
            
            # Join SQL lines and split by semicolons
            sql_text = ' '.join(sql_lines)
            if sql_text.strip():
                # For PROC SQL, treat the entire text as one statement unless there are semicolons
                if ';' in sql_text:
                    # Split by semicolons
                    parts = sql_text.split(';')
                    statements = [part.strip() for part in parts if part.strip()]
                else:
                    # Single statement
                    statements = [sql_text.strip()]
        
        # Check if the PROC line contains additional statements
        if proc_line:
            # Extract TABLES statement from the PROC line
            if 'tables' in proc_line.lower():
                tables_match = re.search(r'tables\s+([^;]+)', proc_line, re.IGNORECASE)
                if tables_match:
                    options['tables'] = tables_match.group(1).strip()
            
            # Extract WHERE statement from the PROC line
            if 'where' in proc_line.lower():
                where_match = re.search(r'where\s+([^;]+)', proc_line, re.IGNORECASE)
                if where_match:
                    options['where'] = where_match.group(1).strip()
        
        # Skip regular statement parsing for PROC SQL (already handled above)
        if proc_name != 'SQL':
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.upper().startswith('PROC '):
                    continue
                    
                if line.upper() in ['RUN;', 'QUIT;']:
                    break
                    
                # Skip empty lines and comments
                if not line or line.startswith('*') or line.startswith('/*'):
                    continue
                    
                statements.append(line)
                
                # Parse common options
                if line.upper().startswith('VAR '):
                    var_match = re.match(r'var\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                    if var_match:
                        options['var'] = [v.strip() for v in var_match.group(1).split()]
                        
                elif line.upper().startswith('BY '):
                    by_match = re.match(r'by\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                    if by_match:
                        by_content = by_match.group(1).strip()
                        # Parse BY variables with ascending/descending modifiers
                        by_vars = []
                        by_ascending = []
                        
                        # Split by spaces and process each token
                        tokens = by_content.split()
                        i = 0
                        while i < len(tokens):
                            token = tokens[i].strip()
                            if token.upper() == 'DESCENDING':
                                # Next token should be the variable name
                                if i + 1 < len(tokens):
                                    var_name = tokens[i + 1].strip()
                                    by_vars.append(var_name)
                                    by_ascending.append(False)  # False = descending
                                    i += 2  # Skip both 'descending' and variable name
                                else:
                                    # Malformed BY statement
                                    break
                            elif token.upper() == 'ASCENDING':
                                # Next token should be the variable name
                                if i + 1 < len(tokens):
                                    var_name = tokens[i + 1].strip()
                                    by_vars.append(var_name)
                                    by_ascending.append(True)  # True = ascending
                                    i += 2  # Skip both 'ascending' and variable name
                                else:
                                    # Malformed BY statement
                                    break
                            else:
                                # Regular variable name (default ascending)
                                by_vars.append(token)
                                by_ascending.append(True)  # Default ascending
                                i += 1
                        
                        options['by'] = by_vars
                        options['by_ascending'] = by_ascending
                        
                elif line.upper().startswith('CLASS '):
                    class_match = re.match(r'class\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                    if class_match:
                        options['class'] = [v.strip() for v in class_match.group(1).split()]
                        
                elif line.upper().startswith('TABLES '):
                    tables_match = re.match(r'tables\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                    if tables_match:
                        options['tables'] = tables_match.group(1).strip()
                        
                elif line.upper().startswith('MODEL '):
                    model_match = re.match(r'model\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                    if model_match:
                        options['model'] = model_match.group(1).strip()
                        
                elif line.upper().startswith('OUTPUT '):
                    # Handle multi-line OUTPUT statement
                    output_content = []
                    output_match = re.match(r'output\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                    if output_match:
                        output_content.append(output_match.group(1).strip())
                    
                    # Look ahead for continuation lines (until we hit a semicolon or next statement)
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if not next_line or next_line.startswith('*') or next_line.startswith('/*'):
                            j += 1
                            continue
                        if next_line.upper().startswith(('PROC ', 'DATA ', 'RUN', 'QUIT')):
                            break
                        if ';' in next_line:
                            # This line ends the OUTPUT statement
                            output_content.append(next_line.rstrip(';').strip())
                            break
                        else:
                            # Continuation line
                            output_content.append(next_line.strip())
                            j += 1
                    
                    # Join all OUTPUT content
                    full_output = ' '.join(output_content)
                    options['output'] = full_output
                        
                elif line.upper().startswith('WHERE '):
                    where_match = re.match(r'where\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                    if where_match:
                        options['where'] = where_match.group(1).strip()
        
        # Debug: Parsed options and statements are available
        
        return ProcStatement(
            proc_name=proc_name,
            options=options,
            statements=statements,
            data_option=data_option,
            output_option=output_option
        )
