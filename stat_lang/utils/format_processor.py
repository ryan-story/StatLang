"""
SAS Format and Informat System Implementation for Open-SAS

This module implements the SAS format system including:
- Built-in date/time formats (DATE9., MMDDYY10., WEEKDATE., DATETIME20., TIME5.)
- Built-in numeric formats (DOLLARw.d, COMMAw.d, BESTw., PERCENTw.d)
- Format/informat parsing and application
- Dataset metadata for format persistence
- Integration with pandas DataFrames
"""

import re
import locale
from datetime import datetime, date, time
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class FormatDefinition:
    """Represents a SAS format definition."""
    name: str
    width: int
    decimal: int = 0
    format_type: str = 'numeric'  # 'numeric', 'character', 'date', 'datetime', 'time'
    is_informat: bool = False


class FormatProcessor:
    """SAS Format Processor for Open-SAS."""
    
    def __init__(self):
        # Format registry
        self.formats: Dict[str, Callable] = {}
        self.informats: Dict[str, Callable] = {}
        
        # Initialize built-in formats
        self._initialize_builtin_formats()
        self._initialize_builtin_informats()
    
    def _initialize_builtin_formats(self):
        """Initialize built-in SAS formats."""
        # Date formats
        self.formats['DATE9'] = self._format_date9
        self.formats['MMDDYY10'] = self._format_mmddyy10
        self.formats['WEEKDATE'] = self._format_weekdate
        self.formats['YYMMDD'] = self._format_yymmdd
        
        # Datetime formats
        self.formats['DATETIME20'] = self._format_datetime20
        self.formats['DATETIME'] = self._format_datetime
        
        # Time formats
        self.formats['TIME5'] = self._format_time5
        self.formats['TIME'] = self._format_time
        
        # Numeric formats
        self.formats['DOLLAR'] = self._format_dollar
        self.formats['COMMA'] = self._format_comma
        self.formats['BEST'] = self._format_best
        self.formats['PERCENT'] = self._format_percent
        
        # Character formats
        self.formats['$'] = self._format_character
    
    def _initialize_builtin_informats(self):
        """Initialize built-in SAS informats."""
        # Date informats
        self.informats['MMDDYY10'] = self._informat_mmddyy10
        self.informats['DATE9'] = self._informat_date9
        self.informats['YYMMDD'] = self._informat_yymmdd
        
        # Numeric informats
        self.informats['COMMA'] = self._informat_comma
        self.informats['DOLLAR'] = self._informat_dollar
        self.informats['PERCENT'] = self._informat_percent
        
        # Character informats
        self.informats['$'] = self._informat_character
    
    def parse_format(self, format_str: str) -> FormatDefinition:
        """Parse a format string like 'DOLLAR10.2' or 'DATE9.'."""
        # Pattern: NAMEw.d or NAMEw
        pattern = r'^([A-Z$]+)(\d+)(?:\.(\d+))?\.?$'
        match = re.match(pattern, format_str.upper())
        
        if not match:
            raise ValueError(f"Invalid format: {format_str}")
        
        name = match.group(1)
        width = int(match.group(2))
        decimal = int(match.group(3)) if match.group(3) else 0
        
        return FormatDefinition(
            name=name,
            width=width,
            decimal=decimal,
            format_type=self._get_format_type(name)
        )
    
    def _get_format_type(self, name: str) -> str:
        """Determine format type from name."""
        date_formats = ['DATE', 'MMDDYY', 'WEEKDATE', 'YYMMDD']
        datetime_formats = ['DATETIME']
        time_formats = ['TIME']
        
        if any(df in name for df in date_formats):
            return 'date'
        elif any(dtf in name for dtf in datetime_formats):
            return 'datetime'
        elif any(tf in name for tf in time_formats):
            return 'time'
        elif name.startswith('$'):
            return 'character'
        else:
            return 'numeric'
    
    def apply_format(self, value: Any, format_def: FormatDefinition) -> str:
        """Apply a format to a value."""
        if pd.isna(value) or value is None:
            return '.'
        
        format_name = format_def.name
        
        # Handle parameterized formats
        if format_name in ['DOLLAR', 'COMMA', 'BEST', 'PERCENT']:
            return self.formats[format_name](value, format_def.width, format_def.decimal)
        elif format_name.startswith('$'):
            return self.formats['$'](value, format_def.width)
        else:
            # Direct format lookup
            if format_name in self.formats:
                return self.formats[format_name](value, format_def.width)
            else:
                return str(value)
    
    def apply_informat(self, text: str, informat_def: FormatDefinition) -> Any:
        """Apply an informat to parse text into a value."""
        if not text or text.strip() == '':
            return None
        
        informat_name = informat_def.name
        
        if informat_name in self.informats:
            return self.informats[informat_name](text, informat_def.width)
        else:
            # Default parsing
            return text
    
    # Date Format Implementations
    def _format_date9(self, value: Any, width: int = 9) -> str:
        """Format as DATE9. (ddMONyyyy)."""
        if isinstance(value, (datetime, date)):
            dt = value if isinstance(value, datetime) else datetime.combine(value, time.min)
            return dt.strftime('%d%b%Y').upper()
        elif isinstance(value, str):
            try:
                dt = datetime.strptime(value, '%Y-%m-%d')
                return dt.strftime('%d%b%Y').upper()
            except:
                return str(value)
        else:
            return str(value)
    
    def _format_mmddyy10(self, value: Any, width: int = 10) -> str:
        """Format as MMDDYY10. (mm/dd/yyyy)."""
        if isinstance(value, (datetime, date)):
            dt = value if isinstance(value, datetime) else datetime.combine(value, time.min)
            return dt.strftime('%m/%d/%Y')
        elif isinstance(value, str):
            try:
                dt = datetime.strptime(value, '%Y-%m-%d')
                return dt.strftime('%m/%d/%Y')
            except:
                return str(value)
        else:
            return str(value)
    
    def _format_weekdate(self, value: Any, width: int = 29) -> str:
        """Format as WEEKDATE. (Dayname, Monthname dd, yyyy)."""
        if isinstance(value, (datetime, date)):
            dt = value if isinstance(value, datetime) else datetime.combine(value, time.min)
            return dt.strftime('%A, %B %d, %Y')
        elif isinstance(value, str):
            try:
                dt = datetime.strptime(value, '%Y-%m-%d')
                return dt.strftime('%A, %B %d, %Y')
            except:
                return str(value)
        else:
            return str(value)
    
    def _format_yymmdd(self, value: Any, width: int = 8) -> str:
        """Format as YYMMDD (yymmdd)."""
        if isinstance(value, (datetime, date)):
            dt = value if isinstance(value, datetime) else datetime.combine(value, time.min)
            return dt.strftime('%y%m%d')
        elif isinstance(value, str):
            try:
                dt = datetime.strptime(value, '%Y-%m-%d')
                return dt.strftime('%y%m%d')
            except:
                return str(value)
        else:
            return str(value)
    
    # Datetime Format Implementations
    def _format_datetime20(self, value: Any, width: int = 20) -> str:
        """Format as DATETIME20. (ddMONyyyy:hh:mm:ss)."""
        if isinstance(value, datetime):
            return value.strftime('%d%b%Y:%H:%M:%S').upper()
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return dt.strftime('%d%b%Y:%H:%M:%S').upper()
            except:
                return str(value)
        else:
            return str(value)
    
    def _format_datetime(self, value: Any, width: int = 16) -> str:
        """Format as DATETIME (ddMONyy:hh:mm:ss)."""
        if isinstance(value, datetime):
            return value.strftime('%d%b%y:%H:%M:%S').upper()
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return dt.strftime('%d%b%y:%H:%M:%S').upper()
            except:
                return str(value)
        else:
            return str(value)
    
    # Time Format Implementations
    def _format_time5(self, value: Any, width: int = 5) -> str:
        """Format as TIME5. (h:mm or hh:mm)."""
        if isinstance(value, time):
            return value.strftime('%-H:%M') if value.hour < 10 else value.strftime('%H:%M')
        elif isinstance(value, datetime):
            return value.strftime('%-H:%M') if value.hour < 10 else value.strftime('%H:%M')
        elif isinstance(value, str):
            try:
                if ':' in value:
                    t = datetime.strptime(value, '%H:%M:%S').time()
                else:
                    t = datetime.strptime(value, '%H:%M').time()
                return t.strftime('%-H:%M') if t.hour < 10 else t.strftime('%H:%M')
            except:
                return str(value)
        else:
            return str(value)
    
    def _format_time(self, value: Any, width: int = 8) -> str:
        """Format as TIME (hh:mm:ss)."""
        if isinstance(value, time):
            return value.strftime('%H:%M:%S')
        elif isinstance(value, datetime):
            return value.strftime('%H:%M:%S')
        elif isinstance(value, str):
            try:
                if ':' in value:
                    t = datetime.strptime(value, '%H:%M:%S').time()
                else:
                    t = datetime.strptime(value, '%H:%M').time()
                return t.strftime('%H:%M:%S')
            except:
                return str(value)
        else:
            return str(value)
    
    # Numeric Format Implementations
    def _format_dollar(self, value: Any, width: int, decimal: int) -> str:
        """Format as DOLLARw.d ($1,234.56)."""
        try:
            num_val = float(value)
            if num_val < 0:
                formatted = f"${abs(num_val):,.{decimal}f}"
                return f"-{formatted}"
            else:
                return f"${num_val:,.{decimal}f}"
        except (ValueError, TypeError):
            return str(value)
    
    def _format_comma(self, value: Any, width: int, decimal: int) -> str:
        """Format as COMMAw.d (1,234.56)."""
        try:
            num_val = float(value)
            return f"{num_val:,.{decimal}f}"
        except (ValueError, TypeError):
            return str(value)
    
    def _format_best(self, value: Any, width: int, decimal: int = 0) -> str:
        """Format as BESTw. (best notation)."""
        try:
            num_val = float(value)
            # Use Python's general format
            return f"{num_val:.{width}g}"
        except (ValueError, TypeError):
            return str(value)
    
    def _format_percent(self, value: Any, width: int, decimal: int) -> str:
        """Format as PERCENTw.d (83.50%)."""
        try:
            num_val = float(value)
            percent_val = num_val * 100
            if percent_val < 0:
                formatted = f"{abs(percent_val):.{decimal}f}%"
                return f"({formatted})"
            else:
                return f"{percent_val:.{decimal}f}%"
        except (ValueError, TypeError):
            return str(value)
    
    # Character Format Implementation
    def _format_character(self, value: Any, width: int) -> str:
        """Format as $w. (character)."""
        str_val = str(value) if value is not None else ''
        return str_val[:width].ljust(width)
    
    # Informat Implementations
    def _informat_mmddyy10(self, text: str, width: int) -> Optional[date]:
        """Parse MMDDYY10. informat (mm/dd/yyyy)."""
        try:
            # Handle various separators
            for sep in ['/', '-', '']:
                if sep in text or sep == '':
                    if sep == '':
                        # No separator, assume MMDDYYYY
                        if len(text) == 8:
                            return datetime.strptime(text, '%m%d%Y').date()
                    else:
                        return datetime.strptime(text, f'%m{sep}%d{sep}%Y').date()
        except ValueError:
            pass
        return None
    
    def _informat_date9(self, text: str, width: int) -> Optional[date]:
        """Parse DATE9. informat (ddMONyyyy)."""
        try:
            return datetime.strptime(text.upper(), '%d%b%Y').date()
        except ValueError:
            return None
    
    def _informat_yymmdd(self, text: str, width: int) -> Optional[date]:
        """Parse YYMMDD informat (yymmdd)."""
        try:
            return datetime.strptime(text, '%y%m%d').date()
        except ValueError:
            return None
    
    def _informat_comma(self, text: str, width: int) -> Optional[float]:
        """Parse COMMA informat (1,234.56)."""
        try:
            # Remove commas and convert
            cleaned = text.replace(',', '')
            return float(cleaned)
        except ValueError:
            return None
    
    def _informat_dollar(self, text: str, width: int) -> Optional[float]:
        """Parse DOLLAR informat ($1,234.56)."""
        try:
            # Remove $ and commas
            cleaned = text.replace('$', '').replace(',', '')
            return float(cleaned)
        except ValueError:
            return None
    
    def _informat_percent(self, text: str, width: int) -> Optional[float]:
        """Parse PERCENT informat (83.50%)."""
        try:
            # Remove % and divide by 100
            cleaned = text.replace('%', '')
            return float(cleaned) / 100
        except ValueError:
            return None
    
    def _informat_character(self, text: str, width: int) -> str:
        """Parse $w. informat (character)."""
        return text[:width] if width > 0 else text
    
    def apply_formats_to_dataframe(self, df: pd.DataFrame, formats: Dict[str, str]) -> pd.DataFrame:
        """Apply formats to a DataFrame."""
        formatted_df = df.copy()
        
        for column, format_str in formats.items():
            if column in formatted_df.columns:
                try:
                    format_def = self.parse_format(format_str)
                    formatted_df[column] = formatted_df[column].apply(
                        lambda x: self.apply_format(x, format_def)
                    )
                except Exception as e:
                    print(f"Warning: Could not apply format {format_str} to column {column}: {e}")
        
        return formatted_df
    
    def register_custom_format(self, name: str, formatter: Callable) -> None:
        """Register a custom format."""
        self.formats[name] = formatter
    
    def register_custom_informat(self, name: str, parser: Callable) -> None:
        """Register a custom informat."""
        self.informats[name] = parser
