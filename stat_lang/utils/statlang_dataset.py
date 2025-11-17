"""
Enhanced dataset class with format support (StatLang)

This module extends the basic dataset functionality to include:
- Format and informat metadata
- Format persistence across operations
- Integration with macro and format processors
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .format_processor import FormatProcessor


@dataclass
class SasDataset:
    """Enhanced dataset with format metadata."""
    name: str
    dataframe: pd.DataFrame
    formats: Dict[str, str] = field(default_factory=dict)
    informats: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    lengths: Dict[str, int] = field(default_factory=dict)
    types: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize dataset metadata."""
        if not self.formats:
            self.formats = {}
        if not self.informats:
            self.informats = {}
        if not self.labels:
            self.labels = {}
        if not self.types:
            self.types = {}
    
    def set_format(self, column: str, format_str: str) -> None:
        """Set format for a column."""
        self.formats[column] = format_str
    
    def set_informat(self, column: str, informat_str: str) -> None:
        """Set informat for a column."""
        self.informats[column] = informat_str
    
    def set_label(self, column: str, label: str) -> None:
        """Set label for a column."""
        self.labels[column] = label
    
    def get_format(self, column: str) -> Optional[str]:
        """Get format for a column."""
        return self.formats.get(column)
    
    def get_informat(self, column: str) -> Optional[str]:
        """Get informat for a column."""
        return self.informats.get(column)
    
    def get_label(self, column: str) -> Optional[str]:
        """Get label for a column."""
        return self.labels.get(column)
    
    def apply_formats(self, format_processor: FormatProcessor) -> pd.DataFrame:
        """Apply formats to the dataset."""
        return format_processor.apply_formats_to_dataframe(self.dataframe, self.formats)
    
    def copy_with_formats(self, new_name: Optional[str] = None) -> 'SasDataset':
        """Create a copy of the dataset with format metadata."""
        return SasDataset(
            name=new_name or f"{self.name}_copy",
            dataframe=self.dataframe.copy(),
            formats=self.formats.copy(),
            informats=self.informats.copy(),
            labels=self.labels.copy(),
            lengths=self.lengths.copy(),
            types=self.types.copy()
        )
    
    def inherit_formats_from(self, source_dataset: 'SasDataset', columns: Optional[List[str]] = None) -> None:
        """Inherit formats from another dataset."""
        if columns is None:
            columns = list(source_dataset.dataframe.columns)
        
        for column in columns:
            if column in source_dataset.formats:
                self.formats[column] = source_dataset.formats[column]
            if column in source_dataset.informats:
                self.informats[column] = source_dataset.informats[column]
            if column in source_dataset.labels:
                self.labels[column] = source_dataset.labels[column]
    
    def get_formatted_dataframe(self, format_processor: FormatProcessor) -> pd.DataFrame:
        """Get a formatted version of the dataframe."""
        return self.apply_formats(format_processor)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary for serialization."""
        return {
            'name': self.name,
            'data': self.dataframe.to_dict('records'),
            'formats': self.formats,
            'informats': self.informats,
            'labels': self.labels,
            'lengths': self.lengths,
            'types': self.types
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SasDataset':
        """Create dataset from dictionary."""
        df = pd.DataFrame(data['data'])
        return cls(
            name=data['name'],
            dataframe=df,
            formats=data.get('formats', {}),
            informats=data.get('informats', {}),
            labels=data.get('labels', {}),
            lengths=data.get('lengths', {}),
            types=data.get('types', {})
        )


class SasDatasetManager:
    """Manager for datasets with format support."""
    
    def __init__(self):
        self.datasets: Dict[str, SasDataset] = {}
        self.format_processor = FormatProcessor()
    
    def create_dataset(self, name: str, dataframe: pd.DataFrame) -> SasDataset:
        """Create a new dataset."""
        dataset = SasDataset(name=name, dataframe=dataframe)
        self.datasets[name] = dataset
        return dataset
    
    def get_dataset(self, name: str) -> Optional[SasDataset]:
        """Get a dataset by name."""
        return self.datasets.get(name)
    
    def set_format(self, dataset_name: str, column: str, format_str: str) -> bool:
        """Set format for a column in a dataset."""
        dataset = self.get_dataset(dataset_name)
        if dataset:
            dataset.set_format(column, format_str)
            return True
        return False
    
    def set_informat(self, dataset_name: str, column: str, informat_str: str) -> bool:
        """Set informat for a column in a dataset."""
        dataset = self.get_dataset(dataset_name)
        if dataset:
            dataset.set_informat(column, informat_str)
            return True
        return False
    
    def apply_formats_to_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Apply formats to a dataset."""
        dataset = self.get_dataset(dataset_name)
        if dataset:
            return dataset.apply_formats(self.format_processor)
        return None
    
    def list_datasets(self) -> List[str]:
        """List all dataset names."""
        return list(self.datasets.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a dataset."""
        dataset = self.get_dataset(dataset_name)
        if dataset:
            return {
                'name': dataset.name,
                'rows': len(dataset.dataframe),
                'columns': list(dataset.dataframe.columns),
                'formats': dataset.formats,
                'informats': dataset.informats,
                'labels': dataset.labels
            }
        return None
    
    def copy_dataset(self, source_name: str, target_name: str) -> bool:
        """Copy a dataset with all metadata."""
        source = self.get_dataset(source_name)
        if source:
            target = source.copy_with_formats(target_name)
            self.datasets[target_name] = target
            return True
        return False
    
    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset."""
        if name in self.datasets:
            del self.datasets[name]
            return True
        return False
