"""PROC EXPORT — Export dataset to external file (CSV, Excel, JSON)."""
from typing import Any, Dict

import pandas as pd

from ..parser.proc_parser import ProcStatement


class ProcExport:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}

        outfile = str(proc_info.options.get('outfile', proc_info.options.get('file', '')))
        dbms = str(proc_info.options.get('dbms', 'csv')).lower()
        delimiter = str(proc_info.options.get('delimiter', ','))

        if not outfile:
            results['output_text'].append("ERROR: OUTFILE= required")
            return results

        outfile = outfile.strip("'\"")

        try:
            if dbms in ('csv', 'dlm', 'tab'):
                sep = '\t' if dbms == 'tab' else delimiter
                data.to_csv(outfile, index=False, sep=sep)
            elif dbms in ('xlsx', 'excel'):
                data.to_excel(outfile, index=False)
            elif dbms == 'json':
                data.to_json(outfile, orient='records', indent=2)
            elif dbms == 'parquet':
                data.to_parquet(outfile, index=False)
            else:
                data.to_csv(outfile, index=False)

            results['output_text'].append(f"PROC EXPORT: {len(data)} obs written to {outfile} ({dbms})")
        except Exception as e:
            results['output_text'].append(f"ERROR: {e}")
        return results
