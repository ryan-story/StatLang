"""
LIBNAME Manager for Open-SAS

This module provides functionality to manage SAS libraries (LIBNAME)
with persistent storage using Parquet format.
"""

import os
import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path


class LibnameManager:
    """Manager for SAS libraries and persistent data storage."""
    
    def __init__(self, default_work_dir: str = None):
        """
        Initialize the LIBNAME manager.
        
        Args:
            default_work_dir: Default directory for WORK library
        """
        self.libraries: Dict[str, str] = {}
        self.default_work_dir = default_work_dir or os.path.join(os.getcwd(), 'work')
        
        # Initialize WORK library
        self.create_library('work', self.default_work_dir)
    
    def create_library(self, libname: str, path: str) -> bool:
        """
        Create a new library mapping.
        
        Args:
            libname: Library name
            path: Directory path for the library
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(path).mkdir(parents=True, exist_ok=True)
            self.libraries[libname.upper()] = path
            return True
        except Exception as e:
            print(f"ERROR: Could not create library {libname}: {e}")
            return False
    
    def get_library_path(self, libname: str) -> Optional[str]:
        """
        Get the path for a library.
        
        Args:
            libname: Library name
            
        Returns:
            Library path or None if not found
        """
        return self.libraries.get(libname.upper())
    
    def list_libraries(self) -> Dict[str, str]:
        """
        List all available libraries.
        
        Returns:
            Dictionary of library names and paths
        """
        return self.libraries.copy()
    
    def save_dataset(self, libname: str, dataset_name: str, data: pd.DataFrame) -> bool:
        """
        Save a dataset to a library.
        
        Args:
            libname: Library name
            dataset_name: Dataset name
            data: DataFrame to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            lib_path = self.get_library_path(libname)
            if not lib_path:
                print(f"ERROR: Library {libname} not found")
                return False
            
            # Create file path
            file_path = os.path.join(lib_path, f"{dataset_name}.parquet")
            
            # Save as Parquet
            data.to_parquet(file_path, index=False)
            return True
            
        except Exception as e:
            print(f"ERROR: Could not save dataset {libname}.{dataset_name}: {e}")
            return False
    
    def load_dataset(self, libname: str, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load a dataset from a library.
        
        Args:
            libname: Library name
            dataset_name: Dataset name
            
        Returns:
            DataFrame or None if not found
        """
        try:
            lib_path = self.get_library_path(libname)
            if not lib_path:
                print(f"ERROR: Library {libname} not found")
                return None
            
            # Try Parquet first
            parquet_path = os.path.join(lib_path, f"{dataset_name}.parquet")
            if os.path.exists(parquet_path):
                return pd.read_parquet(parquet_path)
            
            # Try CSV as fallback
            csv_path = os.path.join(lib_path, f"{dataset_name}.csv")
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
            
            return None
            
        except Exception as e:
            print(f"ERROR: Could not load dataset {libname}.{dataset_name}: {e}")
            return None
    
    def list_datasets(self, libname: str) -> List[str]:
        """
        List all datasets in a library.
        
        Args:
            libname: Library name
            
        Returns:
            List of dataset names
        """
        try:
            lib_path = self.get_library_path(libname)
            if not lib_path:
                return []
            
            datasets = []
            for file in os.listdir(lib_path):
                if file.endswith('.parquet') or file.endswith('.csv'):
                    dataset_name = os.path.splitext(file)[0]
                    datasets.append(dataset_name)
            
            return sorted(datasets)
            
        except Exception as e:
            print(f"ERROR: Could not list datasets in library {libname}: {e}")
            return []
    
    def get_library_datasets(self, libname: str) -> Dict[str, pd.DataFrame]:
        """
        Get all datasets from a library as a dictionary.
        
        Args:
            libname: Library name
            
        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        datasets = {}
        try:
            dataset_names = self.list_datasets(libname)
            for dataset_name in dataset_names:
                df = self.load_dataset(libname, dataset_name)
                if df is not None:
                    datasets[dataset_name] = df
        except Exception as e:
            print(f"ERROR: Could not get datasets from library {libname}: {e}")
        
        return datasets
    
    def delete_dataset(self, libname: str, dataset_name: str) -> bool:
        """
        Delete a dataset from a library.
        
        Args:
            libname: Library name
            dataset_name: Dataset name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            lib_path = self.get_library_path(libname)
            if not lib_path:
                return False
            
            # Try to delete both Parquet and CSV files
            parquet_path = os.path.join(lib_path, f"{dataset_name}.parquet")
            csv_path = os.path.join(lib_path, f"{dataset_name}.csv")
            
            deleted = False
            if os.path.exists(parquet_path):
                os.remove(parquet_path)
                deleted = True
            
            if os.path.exists(csv_path):
                os.remove(csv_path)
                deleted = True
            
            return deleted
            
        except Exception as e:
            print(f"ERROR: Could not delete dataset {libname}.{dataset_name}: {e}")
            return False
    
    def clear_work_library(self) -> bool:
        """
        Clear all datasets from the WORK library.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            work_path = self.get_library_path('WORK')
            if not work_path:
                return False
            
            # Delete all files in work directory
            for file in os.listdir(work_path):
                file_path = os.path.join(work_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Could not clear WORK library: {e}")
            return False
    
    def parse_libname_statement(self, statement: str) -> Optional[tuple]:
        """
        Parse a LIBNAME statement.
        
        Args:
            statement: LIBNAME statement to parse
            
        Returns:
            Tuple of (libname, path) or None if parsing fails
        """
        import re
        
        # Pattern: LIBNAME libname 'path';
        pattern = r'libname\s+(\w+)\s+["\']([^"\']+)["\']\s*;?'
        match = re.match(pattern, statement, re.IGNORECASE)
        
        if match:
            libname = match.group(1)
            path = match.group(2)
            return (libname, path)
        
        return None
