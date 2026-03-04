"""
Model Store for StatLang

Provides in-memory model storage with optional pickle persistence.
Used by trainable procs (REG, TREE, FOREST, BOOST, DNN, NLP, etc.)
to save models and by SCORE to load them for prediction.
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StoredModel:
    """A stored model with metadata."""
    name: str
    model: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    proc_type: str = ''
    feature_vars: List[str] = field(default_factory=list)
    target_var: str = ''


class ModelStore:
    """In-memory model store with optional pickle persistence."""

    def __init__(self, persist_dir: Optional[str] = None):
        self._models: Dict[str, StoredModel] = {}
        self._persist_dir = persist_dir

    def save(
        self, name: str, model: Any,
        metadata: Optional[Dict[str, Any]] = None,
        proc_type: str = '',
        feature_vars: Optional[List[str]] = None,
        target_var: str = '',
        persist: bool = False,
    ) -> None:
        """Save a model to the store."""
        stored = StoredModel(
            name=name,
            model=model,
            metadata=metadata or {},
            proc_type=proc_type,
            feature_vars=feature_vars or [],
            target_var=target_var,
        )
        self._models[name] = stored

        if persist and self._persist_dir:
            self._persist_model(name, stored)

    def load(self, name: str) -> Optional[StoredModel]:
        """Load a model from the store."""
        if name in self._models:
            return self._models[name]
        # Try loading from disk
        if self._persist_dir:
            return self._load_persisted(name)
        return None

    def get_model(self, name: str) -> Optional[Any]:
        """Get just the model object."""
        stored = self.load(name)
        return stored.model if stored else None

    def list_models(self) -> List[str]:
        """List all stored model names."""
        return list(self._models.keys())

    def delete(self, name: str) -> bool:
        """Delete a model from the store."""
        if name in self._models:
            del self._models[name]
            if self._persist_dir:
                path = os.path.join(self._persist_dir, f"{name}.pkl")
                if os.path.exists(path):
                    os.remove(path)
            return True
        return False

    def clear(self) -> None:
        """Clear all models."""
        self._models.clear()

    def _persist_model(self, name: str, stored: StoredModel) -> None:
        """Persist a model to disk using pickle."""
        if not self._persist_dir:
            return
        os.makedirs(self._persist_dir, exist_ok=True)
        path = os.path.join(self._persist_dir, f"{name}.pkl")
        try:
            with open(path, 'wb') as f:
                pickle.dump(stored, f)
        except Exception as e:
            print(f"Warning: Could not persist model '{name}': {e}")

    def _load_persisted(self, name: str) -> Optional[StoredModel]:
        """Load a persisted model from disk."""
        if not self._persist_dir:
            return None
        path = os.path.join(self._persist_dir, f"{name}.pkl")
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                stored: StoredModel = pickle.load(f)  # noqa: S301
            self._models[name] = stored
            return stored
        except Exception as e:
            print(f"Warning: Could not load model '{name}': {e}")
            return None
