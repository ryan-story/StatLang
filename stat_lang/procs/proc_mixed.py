"""PROC MIXED — Mixed Linear Models (random effects)."""
from typing import Any, Dict

import pandas as pd

from ..parser.proc_parser import ProcStatement

try:
    from statsmodels.formula.api import mixedlm
    HAS_SM = True
except ImportError:
    HAS_SM = False


class ProcMixed:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        model_spec = proc_info.options.get('model', '')
        if not model_spec or '=' not in model_spec:
            results['output_text'].append("ERROR: MODEL statement required")
            return results
        if not HAS_SM:
            results['output_text'].append("ERROR: statsmodels not installed")
            return results

        parts = model_spec.split('=')
        dep = parts[0].strip()
        indep = [v.strip() for v in parts[1].split() if v.strip()]
        class_vars = proc_info.options.get('class', [])
        random_spec = proc_info.options.get('random', '')

        # Determine grouping variable from RANDOM or CLASS
        group_var = None
        if random_spec:
            tokens = random_spec.split()
            for t in tokens:
                if t in data.columns:
                    group_var = t
                    break
        if not group_var and class_vars:
            for cv in class_vars:
                if cv in data.columns and cv not in indep:
                    group_var = cv
                    break
        if not group_var:
            results['output_text'].append("ERROR: RANDOM or CLASS grouping variable required")
            return results

        missing = [v for v in [dep, group_var] + indep if v not in data.columns]
        if missing:
            results['output_text'].append(f"ERROR: Variables not found: {missing}")
            return results

        clean = data[[dep, group_var] + indep].dropna()
        terms = []
        for v in indep:
            if v in class_vars:
                terms.append(f"C({v})")
            else:
                terms.append(v)
        formula = f"{dep} ~ {' + '.join(terms)}" if terms else f"{dep} ~ 1"

        try:
            model = mixedlm(formula, data=clean, groups=clean[group_var]).fit()
            results['output_text'].append("PROC MIXED - Mixed Linear Model")
            results['output_text'].append("=" * 50)
            results['output_text'].append(f"Dependent Variable: {dep}")
            results['output_text'].append(f"Group Variable: {group_var}")
            results['output_text'].append(f"N groups: {clean[group_var].nunique()}")
            results['output_text'].append(f"N observations: {len(clean)}")
            results['output_text'].append("")
            results['output_text'].append("Fixed Effects")
            results['output_text'].append("-" * 60)
            results['output_text'].append(f"{'Variable':<20} {'Estimate':<12} {'Std Err':<12} {'z':<10} {'Pr>|z|':<10}")
            for name in model.fe_params.index:
                results['output_text'].append(
                    f"{str(name):<20} {model.fe_params[name]:<12.6f} "
                    f"{model.bse_fe[name]:<12.6f} {model.tvalues[name]:<10.3f} "
                    f"{model.pvalues[name]:<10.4f}"
                )
            results['output_text'].append("")
            results['output_text'].append("Random Effects Variance")
            results['output_text'].append(f"Group variance: {model.cov_re.iloc[0, 0]:.6f}")

            results['model_object'] = model
            results['model_name'] = f'mixed_{dep}'
            results['model_metadata'] = {'proc': 'MIXED', 'group': group_var}
        except Exception as e:
            results['output_text'].append(f"ERROR: {e}")
        return results
