"""
DATA Step Parser for StatLang

This module parses SAS DATA step syntax and converts it to executable
Python operations on DataFrames.

Supports: SET, MERGE, ARRAY, RETAIN, DO loops, LAG/DIF, INFILE/FILE,
IF/THEN/ELSE, DROP/KEEP/RENAME, BY, WHERE, DATALINES/CARDS,
FIRST./LAST. variables, INPUT, PUT.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class DataStepStatement:
    """Represents a parsed DATA step statement."""
    type: str
    content: str
    line_number: int


@dataclass
class DoBlock:
    """Represents a DO loop block."""
    var: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    step: str = '1'
    condition: Optional[str] = None   # for DO WHILE / DO UNTIL
    loop_type: str = 'iterative'      # 'iterative', 'while', 'until'
    body: List[str] = field(default_factory=list)
    nested: List['DoBlock'] = field(default_factory=list)


@dataclass
class ArrayDef:
    """Represents an ARRAY definition."""
    name: str
    variables: List[str]
    dim: Optional[int] = None


@dataclass
class InfileSpec:
    """Represents an INFILE specification."""
    path: str
    delimiter: str = ' '
    dsd: bool = False
    firstobs: int = 1
    obs: Optional[int] = None
    missover: bool = False


@dataclass
class FileSpec:
    """Represents a FILE specification."""
    path: str
    delimiter: str = ' '
    dsd: bool = False


@dataclass
class DataStepInfo:
    """Information about a DATA step."""
    output_dataset: str
    statements: List[str]
    set_datasets: List[str]
    where_conditions: List[str]
    variable_assignments: List[str]
    drop_vars: List[str]
    keep_vars: List[str]
    rename_vars: Dict[str, str]
    by_vars: List[str]
    # New fields for enhanced DATA step
    merge_datasets: List[str] = field(default_factory=list)
    merge_by: List[str] = field(default_factory=list)
    array_defs: List[ArrayDef] = field(default_factory=list)
    retain_vars: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    do_blocks: List[DoBlock] = field(default_factory=list)
    infile_spec: Optional[InfileSpec] = None
    input_spec: List[str] = field(default_factory=list)
    file_spec: Optional[FileSpec] = None
    put_spec: List[str] = field(default_factory=list)
    has_lag: bool = False
    has_dif: bool = False


class DataStepParser:
    """Parser for SAS DATA step syntax."""

    def __init__(self):
        self.current_line = 0

    def parse_data_step(self, code: str) -> DataStepInfo:
        """Parse a complete DATA step."""
        lines = code.split('\n')

        # Find output dataset name
        output_dataset: Optional[str] = None
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith('DATA '):
                m = re.match(r'data\s+([^;]+)', stripped, re.IGNORECASE)
                if m:
                    output_dataset = m.group(1).strip()
                break

        if output_dataset is None:
            raise ValueError("No DATA statement found or missing output dataset name")

        # Collect all statements between DATA and RUN
        full_text = ' '.join(lines)
        data_start = full_text.upper().find('DATA ')
        if data_start == -1:
            raise ValueError("No DATA statement found")

        content_start = full_text.find(';', data_start) + 1
        run_pos = full_text.upper().find('RUN;', content_start)
        if run_pos == -1:
            raise ValueError("No RUN statement found")

        data_content = full_text[content_start:run_pos].strip()
        raw_stmts = [s.strip() for s in data_content.split(';') if s.strip()]

        # Parse each statement
        info = DataStepInfo(
            output_dataset=output_dataset,
            statements=raw_stmts,
            set_datasets=[],
            where_conditions=[],
            variable_assignments=[],
            drop_vars=[],
            keep_vars=[],
            rename_vars={},
            by_vars=[],
        )

        i = 0
        while i < len(raw_stmts):
            stmt = raw_stmts[i].strip()
            upper = stmt.upper()
            i += 1

            # SET
            if upper.startswith('SET '):
                m = re.match(r'set\s+(.+)', stmt, re.IGNORECASE)
                if m:
                    datasets = m.group(1).split()
                    info.set_datasets.extend(datasets)

            # MERGE
            elif upper.startswith('MERGE '):
                m = re.match(r'merge\s+(.+)', stmt, re.IGNORECASE)
                if m:
                    datasets = m.group(1).split()
                    info.merge_datasets.extend(datasets)

            # BY
            elif upper.startswith('BY '):
                m = re.match(r'by\s+(.+)', stmt, re.IGNORECASE)
                if m:
                    by_vars = m.group(1).split()
                    info.by_vars.extend(by_vars)
                    if info.merge_datasets:
                        info.merge_by.extend(by_vars)

            # WHERE
            elif upper.startswith('WHERE '):
                m = re.match(r'where\s+(.+)', stmt, re.IGNORECASE)
                if m:
                    info.where_conditions.append(m.group(1).strip())

            # ARRAY
            elif upper.startswith('ARRAY '):
                arr = self._parse_array(stmt)
                if arr:
                    info.array_defs.append(arr)

            # RETAIN
            elif upper.startswith('RETAIN '):
                retains = self._parse_retain(stmt)
                info.retain_vars.extend(retains)

            # DO block
            elif upper.startswith('DO '):
                block, i = self._parse_do_block(raw_stmts, i - 1)
                if block:
                    info.do_blocks.append(block)

            # INFILE
            elif upper.startswith('INFILE '):
                info.infile_spec = self._parse_infile(stmt)

            # INPUT (for INFILE)
            elif upper.startswith('INPUT '):
                m = re.match(r'input\s+(.+)', stmt, re.IGNORECASE)
                if m:
                    info.input_spec = m.group(1).split()

            # FILE
            elif upper.startswith('FILE '):
                info.file_spec = self._parse_file_stmt(stmt)

            # PUT
            elif upper.startswith('PUT '):
                m = re.match(r'put\s+(.+)', stmt, re.IGNORECASE)
                if m:
                    info.put_spec.append(m.group(1).strip())

            # DROP
            elif upper.startswith('DROP '):
                m = re.match(r'drop\s+(.+)', stmt, re.IGNORECASE)
                if m:
                    info.drop_vars.extend(m.group(1).split())

            # KEEP
            elif upper.startswith('KEEP '):
                m = re.match(r'keep\s+(.+)', stmt, re.IGNORECASE)
                if m:
                    info.keep_vars.extend(m.group(1).split())

            # RENAME
            elif upper.startswith('RENAME '):
                m = re.match(r'rename\s+(.+)', stmt, re.IGNORECASE)
                if m:
                    for pair in m.group(1).split():
                        if '=' in pair:
                            old, new = pair.split('=', 1)
                            info.rename_vars[old.strip()] = new.strip()

            # IF/THEN/ELSE
            elif upper.startswith('IF '):
                info.variable_assignments.append(stmt)

            # Variable assignments
            elif '=' in stmt and not upper.startswith(('DROP', 'KEEP', 'RENAME', 'BY')):
                info.variable_assignments.append(stmt)

        # Check for LAG/DIF usage in assignments
        all_code = ' '.join(info.variable_assignments)
        if re.search(r'\blag\d*\s*\(', all_code, re.IGNORECASE):
            info.has_lag = True
        if re.search(r'\bdif\d*\s*\(', all_code, re.IGNORECASE):
            info.has_dif = True

        return info

    # ------------------------------------------------------------------
    # ARRAY parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_array(stmt: str) -> Optional[ArrayDef]:
        """Parse ARRAY name{dim} var1 var2 ... or ARRAY name(*) var1 var2 ..."""
        # ARRAY name{n} var1 var2 ...
        m = re.match(
            r'array\s+(\w+)\s*[\{\[\(]\s*(\d+|\*)\s*[\}\]\)]\s*(.*)',
            stmt, re.IGNORECASE,
        )
        if m:
            name = m.group(1)
            dim_str = m.group(2)
            vars_str = m.group(3).strip()
            variables = vars_str.split() if vars_str else []
            dim = None if dim_str == '*' else int(dim_str)
            if dim is None:
                dim = len(variables) if variables else None
            return ArrayDef(name=name, variables=variables, dim=dim)

        # ARRAY name var1 var2 ... (no dimension)
        m = re.match(r'array\s+(\w+)\s+(.+)', stmt, re.IGNORECASE)
        if m:
            name = m.group(1)
            variables = m.group(2).split()
            return ArrayDef(name=name, variables=variables, dim=len(variables))
        return None

    # ------------------------------------------------------------------
    # RETAIN parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_retain(stmt: str) -> List[Tuple[str, Optional[str]]]:
        """Parse RETAIN var1 0 var2 '' ..."""
        m = re.match(r'retain\s+(.*)', stmt, re.IGNORECASE)
        if not m:
            return []
        tokens = m.group(1).split()
        result: List[Tuple[str, Optional[str]]] = []
        idx = 0
        while idx < len(tokens):
            tok = tokens[idx]
            # Check if next token is an initial value
            if idx + 1 < len(tokens):
                next_tok = tokens[idx + 1]
                # Numeric or quoted string initial value
                if (next_tok.replace('.', '').replace('-', '').isdigit()
                        or next_tok.startswith("'") or next_tok.startswith('"')):
                    result.append((tok, next_tok.strip("'\"")))
                    idx += 2
                    continue
            result.append((tok, None))
            idx += 1
        return result

    # ------------------------------------------------------------------
    # DO block parsing
    # ------------------------------------------------------------------
    def _parse_do_block(
        self, stmts: List[str], start: int
    ) -> Tuple[Optional[DoBlock], int]:
        """Parse a DO/END block from the statement list."""
        stmt = stmts[start].strip()

        block = DoBlock()

        # DO WHILE(condition)
        m = re.match(r'do\s+while\s*\((.+)\)', stmt, re.IGNORECASE)
        if m:
            block.loop_type = 'while'
            block.condition = m.group(1).strip()
        else:
            # DO UNTIL(condition)
            m = re.match(r'do\s+until\s*\((.+)\)', stmt, re.IGNORECASE)
            if m:
                block.loop_type = 'until'
                block.condition = m.group(1).strip()
            else:
                # DO var = start TO end BY step
                m = re.match(
                    r'do\s+(\w+)\s*=\s*(.+?)\s+to\s+(.+?)(?:\s+by\s+(.+?))?$',
                    stmt, re.IGNORECASE,
                )
                if m:
                    block.loop_type = 'iterative'
                    block.var = m.group(1)
                    block.start = m.group(2).strip()
                    block.end = m.group(3).strip()
                    block.step = m.group(4).strip() if m.group(4) else '1'
                else:
                    return None, start + 1

        # Collect body until matching END
        i = start + 1
        depth = 1
        while i < len(stmts):
            s = stmts[i].strip().upper()
            if s.startswith('DO '):
                depth += 1
            if s == 'END' or s.startswith('END'):
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            block.body.append(stmts[i].strip())
            i += 1

        return block, i

    # ------------------------------------------------------------------
    # INFILE parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_infile(stmt: str) -> InfileSpec:
        """Parse INFILE 'path' options."""
        m = re.match(r"infile\s+['\"]([^'\"]+)['\"](.*)$", stmt, re.IGNORECASE)
        if not m:
            # Try without quotes
            m = re.match(r'infile\s+(\S+)(.*)', stmt, re.IGNORECASE)
        if not m:
            return InfileSpec(path='')

        path = m.group(1)
        opts_str = m.group(2).strip() if m.group(2) else ''

        spec = InfileSpec(path=path)

        # DLM=
        dm = re.search(r"dlm\s*=\s*['\"](.+?)['\"]", opts_str, re.IGNORECASE)
        if dm:
            spec.delimiter = dm.group(1)

        # DSD
        if re.search(r'\bdsd\b', opts_str, re.IGNORECASE):
            spec.dsd = True
            if spec.delimiter == ' ':
                spec.delimiter = ','

        # FIRSTOBS=
        fo = re.search(r'firstobs\s*=\s*(\d+)', opts_str, re.IGNORECASE)
        if fo:
            spec.firstobs = int(fo.group(1))

        # OBS=
        ob = re.search(r'obs\s*=\s*(\d+)', opts_str, re.IGNORECASE)
        if ob:
            spec.obs = int(ob.group(1))

        # MISSOVER
        if re.search(r'\bmissover\b', opts_str, re.IGNORECASE):
            spec.missover = True

        return spec

    # ------------------------------------------------------------------
    # FILE statement parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_file_stmt(stmt: str) -> FileSpec:
        """Parse FILE 'path' options."""
        m = re.match(r"file\s+['\"]([^'\"]+)['\"](.*)$", stmt, re.IGNORECASE)
        if not m:
            m = re.match(r'file\s+(\S+)(.*)', stmt, re.IGNORECASE)
        if not m:
            return FileSpec(path='')

        path = m.group(1)
        opts_str = m.group(2).strip() if m.group(2) else ''
        spec = FileSpec(path=path)

        dm = re.search(r"dlm\s*=\s*['\"](.+?)['\"]", opts_str, re.IGNORECASE)
        if dm:
            spec.delimiter = dm.group(1)

        if re.search(r'\bdsd\b', opts_str, re.IGNORECASE):
            spec.dsd = True
            if spec.delimiter == ' ':
                spec.delimiter = ','

        return spec

    # ------------------------------------------------------------------
    # DATALINES parsing (unchanged from original)
    # ------------------------------------------------------------------
    def parse_datalines(self, code: str) -> pd.DataFrame:
        """Parse DATALINES/CARDS section to create a DataFrame."""
        lines = code.split('\n')
        data_lines: List[str] = []
        in_datalines = False
        input_statement: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith('INPUT '):
                input_statement = stripped.rstrip(';')
                continue
            if stripped.upper() in ('DATALINES;', 'CARDS;'):
                in_datalines = True
                continue
            if stripped == ';' and in_datalines:
                break
            if in_datalines and stripped:
                data_lines.append(stripped)

        if not data_lines or not input_statement:
            return pd.DataFrame()

        # Parse INPUT statement
        input_parts = input_statement[6:].strip().split()
        var_names: List[str] = []
        var_types: Dict[str, str] = {}

        idx = 0
        while idx < len(input_parts):
            part = input_parts[idx]
            if idx + 1 < len(input_parts) and input_parts[idx + 1] == '$':
                var_names.append(part)
                var_types[part] = 'str'
                idx += 2
            else:
                var_names.append(part)
                var_types[part] = 'float'
                idx += 1

        # Parse data lines
        rows: List[Dict[str, Any]] = []
        for line in data_lines:
            values = line.split()
            if len(values) == len(var_names):
                row: Dict[str, Any] = {}
                for j, vname in enumerate(var_names):
                    if var_types[vname] == 'str':
                        row[vname] = values[j]
                    else:
                        try:
                            row[vname] = float(values[j])
                        except ValueError:
                            row[vname] = None
                rows.append(row)

        return pd.DataFrame(rows)
