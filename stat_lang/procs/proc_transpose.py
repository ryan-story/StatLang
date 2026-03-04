"""PROC TRANSPOSE — Reshape data (wide ↔ long)."""
from typing import Any, Dict, List

import pandas as pd

from ..parser.proc_parser import ProcStatement


class ProcTranspose:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}

        var_vars = proc_info.options.get('var', [])
        by_vars = proc_info.options.get('by', [])
        id_vars = proc_info.options.get('id', [])
        prefix = str(proc_info.options.get('prefix', 'COL'))
        name_var = str(proc_info.options.get('name', '_NAME_'))

        if not var_vars:
            var_vars = data.select_dtypes(include='number').columns.tolist()
        missing = [v for v in var_vars if v not in data.columns]
        if missing:
            results['output_text'].append(f"ERROR: VAR variables not found: {missing}")
            return results

        try:
            if by_vars:
                valid_by = [v for v in by_vars if v in data.columns]
                if not valid_by:
                    results['output_text'].append("ERROR: BY variables not found")
                    return results
                groups = []
                for group_keys, grp in data.groupby(valid_by):
                    t = self._transpose_group(grp, var_vars, id_vars, prefix, name_var)
                    if not isinstance(group_keys, tuple):
                        group_keys = (group_keys,)
                    for i, bv in enumerate(valid_by):
                        t[bv] = group_keys[i]
                    groups.append(t)
                result_df = pd.concat(groups, ignore_index=True)
            else:
                result_df = self._transpose_group(data, var_vars, id_vars, prefix, name_var)

            results['output_text'].append("PROC TRANSPOSE completed")
            results['output_text'].append(f"Output shape: {result_df.shape}")
            results['output_data'] = result_df
        except Exception as e:
            results['output_text'].append(f"ERROR: {e}")
        return results

    @staticmethod
    def _transpose_group(
        data: pd.DataFrame, var_vars: List[str],
        id_vars: List[str], prefix: str, name_var: str,
    ) -> pd.DataFrame:
        if id_vars:
            valid_id = [v for v in id_vars if v in data.columns]
            if valid_id:
                subset = data[valid_id + var_vars].set_index(valid_id)
                transposed = subset[var_vars].T
                transposed.index.name = name_var
                return transposed.reset_index()

        transposed = data[var_vars].T
        transposed.columns = [f"{prefix}{i + 1}" for i in range(len(transposed.columns))]
        transposed.index.name = name_var
        return transposed.reset_index()
