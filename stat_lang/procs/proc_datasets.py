"""PROC DATASETS — Manage datasets (delete, rename, list)."""
from typing import Any, Dict

import pandas as pd

from ..parser.proc_parser import ProcStatement


class ProcDatasets:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        data_sets = kw.get('data_sets', {})
        dataset_manager = kw.get('dataset_manager')

        results['output_text'].append("PROC DATASETS")
        results['output_text'].append("=" * 40)

        # Parse sub-statements for DELETE, CHANGE, CONTENTS
        for stmt in proc_info.statements:
            upper = stmt.strip().upper()
            if upper.startswith('DELETE '):
                names = stmt.strip()[7:].rstrip(';').split()
                for name in names:
                    name = name.strip()
                    if name in data_sets:
                        del data_sets[name]
                        if dataset_manager:
                            dataset_manager.delete_dataset(name)
                        results['output_text'].append(f"  Deleted: {name}")
                    else:
                        results['output_text'].append(f"  Not found: {name}")

            elif upper.startswith('CHANGE '):
                # CHANGE old=new
                import re
                pairs = re.findall(r'(\w+)\s*=\s*(\w+)', stmt)
                for old, new in pairs:
                    if old in data_sets:
                        data_sets[new] = data_sets.pop(old)
                        if dataset_manager:
                            ds = dataset_manager.datasets.pop(old, None)
                            if ds:
                                ds.name = new
                                dataset_manager.datasets[new] = ds
                        results['output_text'].append(f"  Renamed: {old} -> {new}")
                    else:
                        results['output_text'].append(f"  Not found: {old}")

            elif upper.startswith('CONTENTS'):
                # List contents
                for name in sorted(data_sets.keys()):
                    df = data_sets[name]
                    results['output_text'].append(f"  {name}: {len(df)} obs, {len(df.columns)} vars")

        # Default: list all datasets
        if not proc_info.statements:
            for name in sorted(data_sets.keys()):
                df = data_sets[name]
                results['output_text'].append(f"  {name}: {len(df)} obs, {len(df.columns)} vars")

        return results
