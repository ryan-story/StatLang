"""PROC DNN — Deep Neural Network training and prediction.

Supports SAS-style options:
    PROC DNN DATA=ds;
        INPUT x1 x2 x3;
        TARGET y;
        ARCHITECTURE 64 32;  /* hidden layer sizes */
        MODEL epochs=50 lr=0.01 optimizer=adam;
    RUN;
"""
from typing import Any, Dict, List

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..parser.proc_parser import ProcStatement

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _SimpleNet(nn.Module):
    """Simple feedforward network built from a list of layer sizes."""
    def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int, task: str = 'regression'):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.task = task

    def forward(self, x):
        return self.net(x)


class ProcDNN:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}
        if not HAS_TORCH:
            results['output_text'].append("ERROR: PyTorch not installed")
            return results

        input_vars = proc_info.options.get('input', [])
        target_var = proc_info.options.get('target', '')
        arch = proc_info.options.get('architecture', [64, 32])
        epochs = int(proc_info.options.get('epochs', 50))
        lr = float(proc_info.options.get('lr', proc_info.options.get('learningrate', 0.01)))
        optimizer_name = str(proc_info.options.get('optimizer', 'adam')).lower()

        # Fall back to MODEL for target/input
        model_spec = proc_info.options.get('model', '')
        if not target_var and model_spec and '=' in model_spec:
            parts = model_spec.split('=')
            target_var = parts[0].strip()
            if not input_vars:
                input_vars = parts[1].split()

        if not target_var:
            results['output_text'].append("ERROR: TARGET or MODEL statement required")
            return results
        if not input_vars:
            input_vars = [c for c in data.select_dtypes(include='number').columns if c != target_var]

        missing = [v for v in input_vars + [target_var] if v not in data.columns]
        if missing:
            results['output_text'].append(f"ERROR: Variables not found: {missing}")
            return results

        clean = data[input_vars + [target_var]].dropna()
        if len(clean) < 10:
            results['output_text'].append("ERROR: Insufficient data")
            return results

        # Determine task type
        y_raw = clean[target_var]
        is_classification = y_raw.dtype == 'object' or y_raw.nunique() < 10

        # Prepare features
        scaler = StandardScaler()
        X = scaler.fit_transform(clean[input_vars].values)
        X_tensor = torch.FloatTensor(X)

        le = None
        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
            y_tensor: torch.Tensor = torch.LongTensor(y)
            output_dim = len(le.classes_)
            criterion: nn.Module = nn.CrossEntropyLoss()
        else:
            y = y_raw.values.astype(float)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            output_dim = 1
            criterion = nn.MSELoss()

        hidden = [int(h) for h in arch] if isinstance(arch, list) else [64, 32]
        model = _SimpleNet(len(input_vars), hidden, output_dim,
                           'classification' if is_classification else 'regression')

        opt: optim.Optimizer
        if optimizer_name == 'sgd':
            opt = optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == 'rmsprop':
            opt = optim.RMSprop(model.parameters(), lr=lr)
        else:
            opt = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        model.train()
        losses: List[float] = []
        for epoch in range(epochs):
            opt.zero_grad()
            out = model(X_tensor)
            loss = criterion(out, y_tensor)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        results['output_text'].append("PROC DNN - Deep Neural Network")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Task: {'Classification' if is_classification else 'Regression'}")
        results['output_text'].append(f"Architecture: {hidden}")
        results['output_text'].append(f"Epochs: {epochs}")
        results['output_text'].append(f"Learning Rate: {lr}")
        results['output_text'].append(f"Final Loss: {losses[-1]:.6f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            preds = model(X_tensor)
        if is_classification:
            pred_labels = preds.argmax(dim=1).numpy()
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y, pred_labels)
            results['output_text'].append(f"Accuracy: {acc:.4f}")
            out_df = clean.copy()
            out_df[f'predicted_{target_var}'] = le.inverse_transform(pred_labels)  # type: ignore[union-attr]
        else:
            pred_vals = preds.squeeze().numpy()
            from sklearn.metrics import r2_score
            r2 = r2_score(y, pred_vals)
            results['output_text'].append(f"R-squared: {r2:.4f}")
            out_df = clean.copy()
            out_df[f'predicted_{target_var}'] = pred_vals

        results['output_data'] = out_df
        results['model_object'] = {'model': model, 'scaler': scaler, 'le': le}
        results['model_name'] = proc_info.options.get('model_name', f'dnn_{target_var}')
        results['model_metadata'] = {'proc': 'DNN', 'input': input_vars, 'target': target_var}
        return results
