"""PROC ANOVA — Balanced Analysis of Variance."""
from typing import Any, Dict

import pandas as pd

from ..parser.proc_parser import ProcStatement

try:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    HAS_SM = True
except ImportError:
    HAS_SM = False


class ProcANOVA:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        model_spec = proc_info.options.get('model', '')
        if not model_spec or '=' not in model_spec:
            results['output_text'].append("ERROR: MODEL statement required")
            return results

        parts = model_spec.split('=')
        dep = parts[0].strip()
        indep = [v.strip() for v in parts[1].split() if v.strip()]
        class_vars = proc_info.options.get('class', [])

        if not class_vars:
            class_vars = list(indep)

        missing = [v for v in [dep] + indep if v not in data.columns]
        if missing:
            results['output_text'].append(f"ERROR: Variables not found: {missing}")
            return results

        clean = data[[dep] + indep].dropna()
        if not HAS_SM:
            results['output_text'].append("ERROR: statsmodels not installed")
            return results

        terms = [f"C({v})" for v in indep]
        formula = f"{dep} ~ {' + '.join(terms)}"
        try:
            model = ols(formula, data=clean).fit()
            table = anova_lm(model)
            results['output_text'].append("PROC ANOVA - Analysis of Variance")
            results['output_text'].append("=" * 50)
            results['output_text'].append(f"Dependent Variable: {dep}")
            results['output_text'].append("")
            results['output_text'].append(str(table))

            means_spec = proc_info.options.get('means', '') or proc_info.options.get('lsmeans', '')
            if means_spec:
                results['output_text'].append("")
                results['output_text'].append("Group Means")
                results['output_text'].append("-" * 30)
                for v in indep:
                    if v in data.columns:
                        grp = clean.groupby(v)[dep].agg(['mean', 'std', 'count'])
                        results['output_text'].append(f"\n{v}:")
                        results['output_text'].append(str(grp))

            results['output_data'] = clean
        except Exception as e:
            results['output_text'].append(f"ERROR: {e}")
        return results
