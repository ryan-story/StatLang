"""PROC GLM — General Linear Models (regression, ANOVA, ANCOVA)."""
from typing import Any, Dict

import pandas as pd

from ..parser.proc_parser import ProcStatement

try:
    from statsmodels.formula.api import ols
    HAS_SM = True
except ImportError:
    HAS_SM = False


class ProcGLM:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        model_spec = proc_info.options.get('model', '')
        if not model_spec or '=' not in model_spec:
            results['output_text'].append("ERROR: MODEL statement required (e.g., MODEL y = x1 x2)")
            return results

        parts = model_spec.split('=')
        dep = parts[0].strip()
        indep = [v.strip() for v in parts[1].split() if v.strip()]
        class_vars = proc_info.options.get('class', [])

        missing = [v for v in [dep] + indep if v not in data.columns]
        if missing:
            results['output_text'].append(f"ERROR: Variables not found: {missing}")
            return results

        clean = data[[dep] + indep].dropna()
        if len(clean) < 3:
            results['output_text'].append("ERROR: Insufficient data")
            return results

        if HAS_SM:
            # Build formula
            terms = []
            for v in indep:
                if v in class_vars:
                    terms.append(f"C({v})")
                else:
                    terms.append(v)
            formula = f"{dep} ~ {' + '.join(terms)}"
            try:
                model = ols(formula, data=clean).fit()
                results['output_text'].append("PROC GLM - General Linear Model")
                results['output_text'].append("=" * 50)
                results['output_text'].append(f"Dependent Variable: {dep}")
                results['output_text'].append(f"R-Square: {model.rsquared:.6f}")
                results['output_text'].append(f"Adj R-Square: {model.rsquared_adj:.6f}")
                results['output_text'].append(f"F Value: {model.fvalue:.4f}")
                results['output_text'].append(f"Pr > F: {model.f_pvalue:.4f}")
                results['output_text'].append("")
                results['output_text'].append("Parameter Estimates")
                results['output_text'].append("-" * 60)
                results['output_text'].append(f"{'Variable':<20} {'Estimate':<12} {'Std Err':<12} {'t Value':<10} {'Pr>|t|':<10}")
                for name, coef in model.params.items():
                    se = model.bse.get(name, 0)
                    tv = model.tvalues.get(name, 0)
                    pv = model.pvalues.get(name, 0)
                    results['output_text'].append(f"{str(name):<20} {coef:<12.6f} {se:<12.6f} {tv:<10.3f} {pv:<10.4f}")

                # ANOVA table
                results['output_text'].append("")
                results['output_text'].append("Type III SS ANOVA")
                results['output_text'].append("-" * 40)
                try:
                    from statsmodels.stats.anova import anova_lm
                    anova_table = anova_lm(model, typ=3)
                    results['output_text'].append(str(anova_table))
                except Exception:
                    pass

                # Output predictions
                output_spec = proc_info.options.get('output', '')
                if output_spec or proc_info.output_option:
                    out = clean.copy()
                    out[f'predicted_{dep}'] = model.fittedvalues
                    out['residual'] = model.resid
                    results['output_data'] = out

                # Store model
                results['model_object'] = model
                results['model_name'] = proc_info.options.get('model_name', f'glm_{dep}')
                results['model_metadata'] = {'proc': 'GLM', 'dep': dep, 'indep': indep}

            except Exception as e:
                results['output_text'].append(f"ERROR: {e}")
        else:
            results['output_text'].append("ERROR: statsmodels not installed")
        return results
