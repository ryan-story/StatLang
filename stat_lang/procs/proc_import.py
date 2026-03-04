"""PROC IMPORT — Import external file to dataset (CSV, Excel, JSON)."""
from typing import Any, Dict

import pandas as pd

from ..parser.proc_parser import ProcStatement


class ProcImport:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}

        datafile = str(proc_info.options.get('datafile', proc_info.options.get('file', '')))
        dbms = str(proc_info.options.get('dbms', 'csv')).lower()
        out_name = proc_info.output_option or str(proc_info.options.get('out', ''))
        delimiter = str(proc_info.options.get('delimiter', ','))
        getnames = str(proc_info.options.get('getnames', 'yes')).lower()

        if not datafile:
            results['output_text'].append("ERROR: DATAFILE= required")
            return results

        datafile = datafile.strip("'\"")
        header = 0 if getnames == 'yes' else None

        try:
            if dbms in ('csv', 'dlm', 'tab'):
                sep = '\t' if dbms == 'tab' else delimiter
                df = pd.read_csv(datafile, sep=sep, header=header)
            elif dbms in ('xlsx', 'excel'):
                df = pd.read_excel(datafile, header=header)
            elif dbms == 'json':
                df = pd.read_json(datafile)
            elif dbms == 'parquet':
                df = pd.read_parquet(datafile)
            else:
                df = pd.read_csv(datafile, header=header)

            results['output_text'].append(f"PROC IMPORT: {len(df)} obs read from {datafile} ({dbms})")
            results['output_data'] = df
            if out_name:
                results['output_dataset'] = out_name
        except Exception as e:
            results['output_text'].append(f"ERROR: {e}")
        return results
