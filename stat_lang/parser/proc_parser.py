"""
PROC Parser for StatLang

This module parses SAS PROC procedure syntax and extracts parameters
for execution by the appropriate PROC implementation.

Supports generic OPTION=value parsing on the PROC line so new procedures
can receive arbitrary options without adding proc-specific regex.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ProcStatement:
    """Represents a parsed PROC statement."""
    proc_name: str
    options: Dict[str, Any]
    statements: List[str]
    data_option: Optional[str] = None
    output_option: Optional[str] = None


# Known boolean flags that do not take a value
_BOOLEAN_FLAGS = frozenset({
    'noprint', 'outall', 'nodup', 'nodupkey', 'noduprec',
    'descending', 'ascending', 'force', 'noint',
})

# Known sub-statement keywords (parsed into options with special handling)
_KNOWN_STATEMENTS = frozenset({
    'var', 'by', 'class', 'tables', 'model', 'output', 'where',
    'input', 'target', 'architecture', 'score', 'strata',
    'lsmeans', 'means', 'random', 'repeated', 'baseline',
    'id', 'copy', 'weight', 'exact', 'paired',
    'state', 'actions', 'reward', 'prompt', 'text', 'image',
})


def _coerce_value(value: str) -> Any:
    """Coerce a string value to int, float, or leave as str."""
    # Try int
    try:
        return int(value)
    except ValueError:
        pass
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_generic_options(text: str) -> Dict[str, Any]:
    """
    Scan *text* for all ``IDENTIFIER=value`` and ``IDENTIFIER='quoted'``
    pairs and bare boolean flags.  Returns a dict with lowercase keys.

    This is the single generic scanner used for PROC-line options and
    for sub-statement option extraction so new procs get options without
    adding proc-specific regex.
    """
    opts: Dict[str, Any] = {}
    if not text:
        return opts

    # Strip trailing semicolon
    text = text.rstrip(';').strip()

    # 1) key = 'quoted' or key = "quoted"
    for m in re.finditer(
        r'\b(\w+)\s*=\s*([\'"])(.+?)\2', text, re.IGNORECASE
    ):
        opts[m.group(1).lower()] = m.group(3)

    # 2) key = value (unquoted, stops at whitespace or end)
    for m in re.finditer(
        r'\b(\w+)\s*=\s*(?![\'"])(\S+)', text, re.IGNORECASE
    ):
        key = m.group(1).lower()
        if key not in opts:  # quoted match takes priority
            opts[key] = _coerce_value(m.group(2).rstrip(';'))

    # 3) bare boolean flags
    for token in re.findall(r'\b(\w+)\b', text):
        if token.lower() in _BOOLEAN_FLAGS and token.lower() not in opts:
            opts[token.lower()] = True

    return opts


class ProcParser:
    """Parser for SAS PROC procedure syntax."""

    def parse_proc(self, code: str) -> ProcStatement:
        """
        Parse a PROC procedure block.

        Uses a generic option scanner on the PROC line so any
        ``OPTION=value`` is captured automatically, then parses known
        sub-statements (VAR, BY, CLASS, MODEL, OUTPUT, WHERE, TABLES,
        INPUT, TARGET, ARCHITECTURE, etc.) into ``options`` while also
        keeping raw statement strings in ``statements``.
        """
        lines = code.split('\n')

        proc_line: Optional[str] = None
        proc_name: Optional[str] = None
        data_option: Optional[str] = None
        output_option: Optional[str] = None
        options: Dict[str, Any] = {}

        # --- locate PROC line and extract name + generic options ----------
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith('PROC '):
                proc_line = stripped
                match = re.match(
                    r'proc\s+(\w+)(?:\s+(.+?))?(?:\s*;)?\s*$',
                    stripped, re.IGNORECASE,
                )
                if match:
                    proc_name = match.group(1).upper()
                    rest = match.group(2) or ''
                    # Generic option scan on the PROC line
                    options = _parse_generic_options(rest)
                    # Pull DATA= and OUT= into dedicated fields
                    if 'data' in options:
                        data_option = str(options.pop('data'))
                    if 'out' in options:
                        output_option = str(options.pop('out'))
                break

        if not proc_line:
            raise ValueError("No PROC statement found")
        if proc_name is None:
            raise ValueError("PROC statement missing procedure name")

        proc_name = str(proc_name)

        # --- collect sub-statements ---------------------------------------
        statements: List[str] = []

        # Special handling for PROC SQL
        if proc_name == 'SQL':
            statements = self._parse_sql_statements(lines)
        else:
            statements = self._parse_sub_statements(lines, options)

        # Promote DATA= and OUT= that appeared on continuation lines
        if data_option is None and 'data' in options:
            data_option = str(options.pop('data'))
        if output_option is None and 'out' in options:
            output_option = str(options.pop('out'))

        return ProcStatement(
            proc_name=proc_name,
            options=options,
            statements=statements,
            data_option=data_option,
            output_option=output_option,
        )

    # ------------------------------------------------------------------
    # SQL-specific parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_sql_statements(lines: List[str]) -> List[str]:
        """Collect SQL statements between PROC SQL and QUIT/RUN."""
        sql_lines: List[str] = []
        in_sql = False
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith('PROC SQL'):
                in_sql = True
                continue
            if stripped.upper() in ('RUN;', 'QUIT;'):
                break
            if in_sql and stripped and not stripped.startswith('*') and not stripped.startswith('/*'):
                sql_lines.append(stripped)
        sql_text = ' '.join(sql_lines)
        if not sql_text.strip():
            return []
        if ';' in sql_text:
            return [p.strip() for p in sql_text.split(';') if p.strip()]
        return [sql_text.strip()]

    # ------------------------------------------------------------------
    # General sub-statement parsing
    # ------------------------------------------------------------------
    def _parse_sub_statements(
        self, lines: List[str], options: Dict[str, Any]
    ) -> List[str]:
        """Parse sub-statements inside a PROC block."""
        statements: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.upper().startswith('PROC '):
                continue
            if line.upper() in ('RUN;', 'QUIT;'):
                break
            if line.startswith('*') or line.startswith('/*'):
                continue

            statements.append(line)

            upper = line.upper()

            # ----- VAR -----
            if upper.startswith('VAR '):
                m = re.match(r'var\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['var'] = m.group(1).split()

            # ----- BY -----
            elif upper.startswith('BY '):
                m = re.match(r'by\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    self._parse_by_vars(m.group(1).strip(), options)

            # ----- CLASS -----
            elif upper.startswith('CLASS '):
                m = re.match(r'class\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['class'] = m.group(1).split()

            # ----- TABLES -----
            elif upper.startswith('TABLES '):
                m = re.match(r'tables\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['tables'] = m.group(1).strip()

            # ----- MODEL -----
            elif upper.startswith('MODEL '):
                m = re.match(r'model\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['model'] = m.group(1).strip()

            # ----- OUTPUT -----
            elif upper.startswith('OUTPUT '):
                full_output = self._collect_output_statement(line, lines, i)
                options['output'] = full_output

            # ----- WHERE -----
            elif upper.startswith('WHERE '):
                m = re.match(r'where\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['where'] = m.group(1).strip()

            # ----- INPUT -----
            elif upper.startswith('INPUT '):
                m = re.match(r'input\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['input'] = m.group(1).split()

            # ----- TARGET -----
            elif upper.startswith('TARGET '):
                m = re.match(r'target\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['target'] = m.group(1).strip()

            # ----- ARCHITECTURE -----
            elif upper.startswith('ARCHITECTURE '):
                m = re.match(r'architecture\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    vals = m.group(1).split()
                    options['architecture'] = [_coerce_value(v) for v in vals]

            # ----- SCORE -----
            elif upper.startswith('SCORE '):
                m = re.match(r'score\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['score'] = _parse_generic_options(m.group(1))

            # ----- STRATA -----
            elif upper.startswith('STRATA '):
                m = re.match(r'strata\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['strata'] = m.group(1).split()

            # ----- LSMEANS -----
            elif upper.startswith('LSMEANS '):
                m = re.match(r'lsmeans\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['lsmeans'] = m.group(1).strip()

            # ----- RANDOM -----
            elif upper.startswith('RANDOM '):
                m = re.match(r'random\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['random'] = m.group(1).strip()

            # ----- REPEATED -----
            elif upper.startswith('REPEATED '):
                m = re.match(r'repeated\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['repeated'] = m.group(1).strip()

            # ----- BASELINE -----
            elif upper.startswith('BASELINE '):
                m = re.match(r'baseline\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['baseline'] = m.group(1).strip()

            # ----- ID -----
            elif upper.startswith('ID '):
                m = re.match(r'id\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['id'] = m.group(1).split()

            # ----- COPY -----
            elif upper.startswith('COPY '):
                m = re.match(r'copy\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['copy'] = m.group(1).split()

            # ----- WEIGHT -----
            elif upper.startswith('WEIGHT '):
                m = re.match(r'weight\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['weight'] = m.group(1).strip()

            # ----- PAIRED -----
            elif upper.startswith('PAIRED '):
                m = re.match(r'paired\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['paired'] = m.group(1).strip()

            # ----- STATE -----
            elif upper.startswith('STATE '):
                m = re.match(r'state\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['state'] = m.group(1).split()

            # ----- ACTIONS -----
            elif upper.startswith('ACTIONS '):
                m = re.match(r'actions\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['actions'] = m.group(1).split()

            # ----- REWARD -----
            elif upper.startswith('REWARD '):
                m = re.match(r'reward\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['reward'] = m.group(1).strip()

            # ----- TEXT -----
            elif upper.startswith('TEXT '):
                m = re.match(r'text\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['text'] = m.group(1).strip()

            # ----- IMAGE -----
            elif upper.startswith('IMAGE '):
                m = re.match(r'image\s+(.+?)(?:\s*;)?$', line, re.IGNORECASE)
                if m:
                    options['image'] = m.group(1).strip()

            # ----- Generic: any other statement with OPTION=value -----
            else:
                # Try to extract key=value pairs from unknown statements
                line_opts = _parse_generic_options(line)
                for k, v in line_opts.items():
                    if k not in options:
                        options[k] = v

        return statements

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_by_vars(by_content: str, options: Dict[str, Any]) -> None:
        """Parse BY variables with ascending/descending modifiers."""
        by_vars: List[str] = []
        by_ascending: List[bool] = []
        tokens = by_content.split()
        idx = 0
        while idx < len(tokens):
            tok = tokens[idx].upper()
            if tok == 'DESCENDING':
                if idx + 1 < len(tokens):
                    by_vars.append(tokens[idx + 1])
                    by_ascending.append(False)
                    idx += 2
                else:
                    break
            elif tok == 'ASCENDING':
                if idx + 1 < len(tokens):
                    by_vars.append(tokens[idx + 1])
                    by_ascending.append(True)
                    idx += 2
                else:
                    break
            else:
                by_vars.append(tokens[idx])
                by_ascending.append(True)
                idx += 1
        options['by'] = by_vars
        options['by_ascending'] = by_ascending

    @staticmethod
    def _collect_output_statement(
        first_line: str, lines: List[str], next_idx: int
    ) -> str:
        """Collect a possibly multi-line OUTPUT statement."""
        parts: List[str] = []
        m = re.match(r'output\s+(.+?)(?:\s*;)?$', first_line, re.IGNORECASE)
        if m:
            parts.append(m.group(1).strip())
        j = next_idx
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt or nxt.startswith('*') or nxt.startswith('/*'):
                j += 1
                continue
            if nxt.upper().startswith(('PROC ', 'DATA ', 'RUN', 'QUIT')):
                break
            if ';' in nxt:
                parts.append(nxt.rstrip(';').strip())
                break
            parts.append(nxt)
            j += 1
        return ' '.join(parts)
