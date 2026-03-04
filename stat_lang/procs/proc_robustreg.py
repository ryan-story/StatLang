"""PROC ROBUSTREG — Robust Regression (M-estimation)."""
from typing import Any, Dict

import pandas as pd

from ..parser.proc_parser import ProcStatement

try:
    import statsmodels.api as sm
    from statsmodels.robust.robust_linear_model import RLM
    HAS_SM = True
except ImportError:
    HAS_SM = False


class ProcRobustreg:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        model_spec = proc_info.options.get('model', '')
        if not model_spec or '=' not in model_spec:
            results['output_text'].append("ERROR: MODEL statement required")
            return results

        parts = model_spec.split('=')
        dep = parts[0].strip()
        indep = [v.strip() for v in parts[1].split() if v.strip()]
        missing = [v for v in [dep] + indep if v not in data.columns]
        if missing:
            results['output_text'].append(f"ERROR: Variables not found: {missing}")
            return results

        clean = data[[dep] + indep].dropna()
        if not HAS_SM:
            results['output_text'].append("ERROR: statsmodels not installed")
            return results

        y = clean[dep]
        X = sm.add_constant(clean[indep])
        try:
            model = RLM(y, X, M=sm.robust.norms.HuberT()).fit()
            results['output_text'].append("PROC ROBUSTREG - Robust Regression (M-estimation)")
            results['output_text'].append("=" * 50)
            results['output_text'].append(f"Dependent Variable: {dep}")
            results['output_text'].append(f"Scale estimate: {model.scale:.6f}")
            results['output_text'].append("")
            results['output_text'].append("Parameter Estimates")
            results['output_text'].append("-" * 60)
            results['output_text'].append(f"{'Variable':<15} {'Estimate':<12} {'Std Err':<12} {'z Value':<10} {'Pr>|z|':<10}")
            for name in model.params.index:
                coef = model.params[name]
                se = model.bse[name]
                z = model.tvalues[name]
                p = model.pvalues[name]
                results['output_text'].append(f"{str(name):<15} {coef:<12.6f} {se:<12.6f} {z:<10.3f} {p:<10.4f}")

            out_spec = proc_info.options.get('output', '')
            if out_spec or proc_info.output_option:
                out = clean.copy()
                out[f'predicted_{dep}'] = model.fittedvalues
                out['residual'] = model.resid
                results['output_data'] = out

            results['model_object'] = model
            results['model_name'] = f'robustreg_{dep}'
            results['model_metadata'] = {'proc': 'ROBUSTREG', 'dep': dep, 'indep': indep}
        except Exception as e:
            results['output_text'].append(f"ERROR: {e}")
        return results
