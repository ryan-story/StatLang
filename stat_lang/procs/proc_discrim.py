"""PROC DISCRIM — Discriminant Analysis."""
from typing import Any, Dict

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from ..parser.proc_parser import ProcStatement


class ProcDiscrim:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        class_vars = proc_info.options.get('class', [])
        var_vars = proc_info.options.get('var', [])
        method = str(proc_info.options.get('method', 'linear')).lower()

        if not class_vars:
            results['output_text'].append("ERROR: CLASS statement required")
            return results
        target = class_vars[0]
        if target not in data.columns:
            results['output_text'].append(f"ERROR: CLASS variable '{target}' not found")
            return results
        if not var_vars:
            var_vars = [c for c in data.select_dtypes(include='number').columns if c != target]
        missing = [v for v in var_vars if v not in data.columns]
        if missing:
            results['output_text'].append(f"ERROR: Variables not found: {missing}")
            return results

        clean = data[[target] + var_vars].dropna()
        le = LabelEncoder()
        y = le.fit_transform(clean[target].astype(str))
        X = clean[var_vars].values

        if method == 'quadratic':
            model = QuadraticDiscriminantAnalysis()
        else:
            model = LinearDiscriminantAnalysis()

        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)

        results['output_text'].append("PROC DISCRIM - Discriminant Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Method: {method.title()}")
        results['output_text'].append(f"Target: {target}")
        results['output_text'].append(f"Predictors: {', '.join(var_vars)}")
        results['output_text'].append(f"Resubstitution Accuracy: {acc:.4f}")
        results['output_text'].append("")
        results['output_text'].append(classification_report(y, preds, target_names=le.classes_, zero_division=0))

        out = clean.copy()
        out[f'predicted_{target}'] = le.inverse_transform(preds)
        results['output_data'] = out

        model_store = kw.get('model_store')
        if model_store:
            results['model_object'] = model
            results['model_name'] = proc_info.options.get('model_name', f'discrim_{target}')
            results['model_metadata'] = {'proc': 'DISCRIM', 'le': le, 'vars': var_vars}

        return results
