"""PROC GENMOD — Generalized Linear Models."""
from typing import Any, Dict

import pandas as pd

from ..parser.proc_parser import ProcStatement

try:
    import statsmodels.api as sm
    import statsmodels.genmod.families as families
    from statsmodels.genmod.generalized_linear_model import GLM as SM_GLM
    HAS_SM = True
except ImportError:
    HAS_SM = False


_FAMILY_MAP = {}
if HAS_SM:
    _FAMILY_MAP = {
        'normal': families.Gaussian,
        'gaussian': families.Gaussian,
        'binomial': families.Binomial,
        'poisson': families.Poisson,
        'gamma': families.Gamma,
        'inversegaussian': families.InverseGaussian,
        'negbin': families.NegativeBinomial,
        'tweedie': families.Tweedie,
    }

_LINK_MAP = {}
if HAS_SM:
    _LINK_MAP = {
        'identity': families.links.Identity(),
        'log': families.links.Log(),
        'logit': families.links.Logit(),
        'probit': families.links.Probit(),
        'cloglog': families.links.CLogLog(),
        'inverse': families.links.InversePower(),
        'sqrt': families.links.Sqrt(),
    }


class ProcGenmod:
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

        missing = [v for v in [dep] + indep if v not in data.columns]
        if missing:
            results['output_text'].append(f"ERROR: Variables not found: {missing}")
            return results

        clean = data[[dep] + indep].dropna()
        y = clean[dep]

        # Handle class variables with dummies
        X_parts = []
        for v in indep:
            if v in class_vars:
                dummies = pd.get_dummies(clean[v], prefix=v, drop_first=True, dtype=float)
                X_parts.append(dummies)
            else:
                X_parts.append(clean[[v]])
        X = pd.concat(X_parts, axis=1)
        X = sm.add_constant(X)

        dist = str(proc_info.options.get('dist', 'normal')).lower()
        link = str(proc_info.options.get('link', '')).lower()

        family_cls = _FAMILY_MAP.get(dist, families.Gaussian)
        if link and link in _LINK_MAP:
            family = family_cls(link=_LINK_MAP[link])
        else:
            family = family_cls()

        try:
            model = SM_GLM(y, X, family=family).fit()
            results['output_text'].append("PROC GENMOD - Generalized Linear Model")
            results['output_text'].append("=" * 50)
            results['output_text'].append(f"Distribution: {dist}")
            results['output_text'].append(f"Link: {link or 'default'}")
            results['output_text'].append(f"Dependent Variable: {dep}")
            results['output_text'].append(f"N: {len(clean)}")
            results['output_text'].append(f"Deviance: {model.deviance:.4f}")
            results['output_text'].append(f"AIC: {model.aic:.4f}")
            results['output_text'].append("")
            results['output_text'].append("Parameter Estimates")
            results['output_text'].append("-" * 60)
            results['output_text'].append(f"{'Variable':<20} {'Estimate':<12} {'Std Err':<12} {'z':<10} {'Pr>|z|':<10}")
            for name in model.params.index:
                results['output_text'].append(
                    f"{str(name):<20} {model.params[name]:<12.6f} "
                    f"{model.bse[name]:<12.6f} {model.tvalues[name]:<10.3f} "
                    f"{model.pvalues[name]:<10.4f}"
                )

            if proc_info.output_option:
                out = clean.copy()
                out[f'predicted_{dep}'] = model.fittedvalues
                out['residual'] = model.resid_response
                results['output_data'] = out

            results['model_object'] = model
            results['model_name'] = f'genmod_{dep}'
            results['model_metadata'] = {'proc': 'GENMOD', 'dist': dist, 'link': link}
        except Exception as e:
            results['output_text'].append(f"ERROR: {e}")
        return results
