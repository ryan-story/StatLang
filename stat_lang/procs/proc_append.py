"""PROC APPEND — Append one dataset to another."""
from typing import Any, Dict

import pandas as pd

from ..parser.proc_parser import ProcStatement


class ProcAppend:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        data_sets = kw.get('data_sets', {})
        dataset_manager = kw.get('dataset_manager')

        base_name = proc_info.options.get('base', '')
        # For APPEND, DATA= means the source dataset to append.
        # The generic parser pops data= into data_option, so check both.
        append_name = proc_info.options.get('data', '') or proc_info.data_option or ''
        force = proc_info.options.get('force', False)

        if not base_name:
            results['output_text'].append("ERROR: BASE= option required")
            return results
        if not append_name:
            results['output_text'].append("ERROR: DATA= option required")
            return results

        base_df = data_sets.get(str(base_name))
        append_df = data_sets.get(str(append_name))

        if append_df is None:
            results['output_text'].append(f"ERROR: Dataset '{append_name}' not found")
            return results

        if base_df is None:
            # Create new base from append
            result_df = append_df.copy()
        else:
            if not force:
                # Check column compatibility
                base_cols = set(base_df.columns)
                append_cols = set(append_df.columns)
                if base_cols != append_cols:
                    extra = append_cols - base_cols
                    if extra:
                        results['output_text'].append(
                            f"WARNING: Extra columns in DATA will be dropped: {extra}. Use FORCE to keep."
                        )
                        append_df = append_df[list(base_cols & append_cols)]
            result_df = pd.concat([base_df, append_df], ignore_index=True)

        # Store result
        data_sets[str(base_name)] = result_df
        if dataset_manager:
            from ..utils.statlang_dataset import SasDataset
            dataset_manager.datasets[str(base_name)] = SasDataset(
                name=str(base_name), dataframe=result_df
            )

        results['output_text'].append(f"PROC APPEND: {len(append_df)} obs appended to {base_name}")
        results['output_text'].append(f"Result: {len(result_df)} obs total")
        results['output_data'] = result_df
        results['overwrite_input'] = True
        return results
