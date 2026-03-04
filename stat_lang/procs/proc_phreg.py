"""PROC PHREG — Cox Proportional Hazards Regression."""
import re
from typing import Any, Dict

import pandas as pd

from ..parser.proc_parser import ProcStatement

try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


class ProcPhreg:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        model_spec = proc_info.options.get('model', '')
        if not model_spec or '=' not in model_spec:
            results['output_text'].append("ERROR: MODEL statement required")
            return results

        if not HAS_LIFELINES:
            results['output_text'].append("ERROR: lifelines package not installed")
            return results

        parts = model_spec.split('=')
        lhs = parts[0].strip()
        indep = [v.strip() for v in parts[1].split() if v.strip()]

        m = re.match(r'(\w+)\s*\*\s*(\w+)\s*\((\d+)\)', lhs)
        if m:
            time_var = m.group(1)
            censor_var = m.group(2)
            censor_val = int(m.group(3))
        else:
            time_var = lhs
            censor_var = None
            censor_val = 0

        if time_var not in data.columns:
            results['output_text'].append(f"ERROR: Time variable '{time_var}' not found")
            return results

        cols = [time_var] + indep
        if censor_var and censor_var in data.columns:
            cols.append(censor_var)
        clean = data[cols].dropna()

        if censor_var and censor_var in clean.columns:
            clean['_event'] = (clean[censor_var] != censor_val).astype(int)
        else:
            clean['_event'] = 1

        try:
            cph = CoxPHFitter()
            cph.fit(clean[indep + [time_var, '_event']], duration_col=time_var, event_col='_event')

            results['output_text'].append("PROC PHREG - Cox Proportional Hazards")
            results['output_text'].append("=" * 50)
            results['output_text'].append(f"Time variable: {time_var}")
            results['output_text'].append(f"N observations: {len(clean)}")
            results['output_text'].append(f"N events: {clean['_event'].sum()}")
            results['output_text'].append(f"Concordance: {cph.concordance_index_:.4f}")
            results['output_text'].append("")
            results['output_text'].append(str(cph.summary))

            results['model_object'] = cph
            results['model_name'] = f'phreg_{time_var}'
            results['model_metadata'] = {'proc': 'PHREG'}

        except Exception as e:
            results['output_text'].append(f"ERROR: {e}")
        return results
