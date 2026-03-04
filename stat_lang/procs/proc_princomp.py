"""PROC PRINCOMP — Principal Component Analysis."""
from typing import Any, Dict

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..parser.proc_parser import ProcStatement


class ProcPrincomp:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        var_vars = proc_info.options.get('var', [])
        n_comp = proc_info.options.get('n', None)

        if not var_vars:
            var_vars = data.select_dtypes(include='number').columns.tolist()
        missing = [v for v in var_vars if v not in data.columns]
        if missing:
            results['output_text'].append(f"ERROR: Variables not found: {missing}")
            return results

        clean = data[var_vars].dropna()
        if len(clean) < 2:
            results['output_text'].append("ERROR: Insufficient data")
            return results

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clean)

        if n_comp is None:
            n_comp = min(len(var_vars), len(clean))
        n_comp = int(n_comp)

        pca = PCA(n_components=n_comp)
        scores = pca.fit_transform(X_scaled)

        results['output_text'].append("PROC PRINCOMP - Principal Component Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Variables: {', '.join(var_vars)}")
        results['output_text'].append(f"Components: {n_comp}")
        results['output_text'].append("")
        results['output_text'].append("Eigenvalues and Variance Explained")
        results['output_text'].append("-" * 40)
        results['output_text'].append(f"{'Component':<12} {'Eigenvalue':<12} {'Proportion':<12} {'Cumulative':<12}")
        cum = 0.0
        for i in range(n_comp):
            ev = pca.explained_variance_[i]
            prop = pca.explained_variance_ratio_[i]
            cum += prop
            results['output_text'].append(f"{'Prin' + str(i+1):<12} {ev:<12.4f} {prop:<12.4f} {cum:<12.4f}")

        results['output_text'].append("")
        results['output_text'].append("Eigenvectors (Loadings)")
        results['output_text'].append("-" * 40)
        header = f"{'Variable':<15}" + ''.join(f"{'Prin' + str(i+1):<12}" for i in range(n_comp))
        results['output_text'].append(header)
        for j, v in enumerate(var_vars):
            row_str = f"{v:<15}" + ''.join(f"{pca.components_[i][j]:<12.4f}" for i in range(n_comp))
            results['output_text'].append(row_str)

        # Output dataset with scores
        score_cols = [f'Prin{i+1}' for i in range(n_comp)]
        out = clean.copy()
        for i, col in enumerate(score_cols):
            out[col] = scores[:, i]
        results['output_data'] = out

        return results
