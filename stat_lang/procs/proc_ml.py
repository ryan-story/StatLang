"""
PROC TREE, PROC FOREST, PROC BOOST Implementation for Open-SAS

This module implements SAS PROC TREE, PROC FOREST, and PROC BOOST functionality
for machine learning using scikit-learn.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcTree:
    """Implementation of SAS PROC TREE procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC TREE on the given data.
        
        Args:
            data: Input DataFrame
            proc_info: Parsed PROC statement information
            
        Returns:
            Dictionary containing results and output data
        """
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Get MODEL specification
        model_spec = proc_info.options.get('model', '')
        if not model_spec:
            results['output_text'].append("ERROR: MODEL specification required (e.g., MODEL y = x1 x2).")
            return results
        
        # Parse model specification: MODEL y = x1 x2 x3
        # Split by '=' and then by spaces
        if '=' in model_spec:
            parts = model_spec.split('=')
            target_var = parts[0].strip()
            feature_vars = [var.strip() for var in parts[1].split() if var.strip()]
        else:
            # If no '=' found, treat as single variable
            target_var = model_spec.strip()
            feature_vars = []
        
        # Check if variables exist
        if target_var not in data.columns:
            results['output_text'].append(f"ERROR: Target variable '{target_var}' not found in data.")
            return results
        
        missing_vars = [var for var in feature_vars if var not in data.columns]
        if missing_vars:
            results['output_text'].append(f"ERROR: Feature variables not found: {missing_vars}")
            return results
        
        # Get tree parameters
        max_depth = proc_info.options.get('maxdepth', 5)
        min_samples_split = proc_info.options.get('minsplit', 2)
        min_samples_leaf = proc_info.options.get('minleaf', 1)
        
        results['output_text'].append("PROC TREE - Decision Tree Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Target variable: {target_var}")
        results['output_text'].append(f"Feature variables: {', '.join(feature_vars)}")
        results['output_text'].append(f"Max depth: {max_depth}")
        results['output_text'].append("")
        
        # Prepare data
        model_data = data[[target_var] + feature_vars].copy()
        clean_data = model_data.dropna()
        
        if len(clean_data) < 10:
            results['output_text'].append("ERROR: Insufficient data for tree analysis.")
            return results
        
        # Prepare target and features
        y = clean_data[target_var]
        X = clean_data[feature_vars]
        
        # Handle categorical variables
        X_encoded = X.copy()
        label_encoders = {}
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                label_encoders[col] = le
        
        # Determine if classification or regression
        if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 10:
            # Classification
            is_classification = True
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y.astype(str))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
            
            # Fit decision tree
            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            tree.fit(X_train, y_train)
            
            # Predictions
            y_pred = tree.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Format output
            results['output_text'].extend(self._format_classification_output(tree, X_encoded.columns, accuracy, y_test, y_pred, le_target))
            
        else:
            # Regression
            is_classification = False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
            
            # Fit decision tree
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            tree.fit(X_train, y_train)
            
            # Predictions
            y_pred = tree.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Format output
            results['output_text'].extend(self._format_regression_output(tree, X_encoded.columns, mse, r2, y_test, y_pred))
        
        # Create output DataFrame
        feature_importance = pd.DataFrame({
            'Feature': X_encoded.columns,
            'Importance': tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        results['output_data'] = {
            'feature_importance': feature_importance,
            'model': tree,
            'is_classification': is_classification
        }
        
        return results
    
    def _format_classification_output(self, tree, feature_names, accuracy, y_test, y_pred, le_target):
        """Format classification tree output."""
        output = []
        
        output.append("Decision Tree Classification Results")
        output.append("-" * 40)
        output.append(f"Training accuracy: {accuracy:.4f}")
        output.append(f"Tree depth: {tree.get_depth()}")
        output.append(f"Number of leaves: {tree.get_n_leaves()}")
        output.append("")
        
        # Feature importance
        output.append("Feature Importance")
        output.append("-" * 20)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            output.append(f"{row['Feature']:<15} {row['Importance']:<10.4f}")
        
        output.append("")
        
        # Classification report
        output.append("Classification Report")
        output.append("-" * 25)
        class_names = le_target.classes_
        
        # Get unique classes in test data
        unique_test_classes = np.unique(y_test)
        unique_pred_classes = np.unique(y_pred)
        
        # Only include classes that appear in both test and prediction
        valid_classes = np.intersect1d(unique_test_classes, unique_pred_classes)
        valid_class_names = [class_names[i] for i in valid_classes]
        
        if len(valid_classes) > 0:
            report = classification_report(y_test, y_pred, labels=valid_classes, target_names=valid_class_names, output_dict=True, zero_division=0)
            
            output.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            output.append("-" * 55)
            
            for class_name in valid_class_names:
                if class_name in report:
                    output.append(f"{class_name:<15} {report[class_name]['precision']:<10.4f} {report[class_name]['recall']:<10.4f} {report[class_name]['f1-score']:<10.4f} {report[class_name]['support']:<10.0f}")
        else:
            output.append("No valid classes found in test data for classification report.")
        
        output.append("")
        
        return output
    
    def _format_regression_output(self, tree, feature_names, mse, r2, y_test, y_pred):
        """Format regression tree output."""
        output = []
        
        output.append("Decision Tree Regression Results")
        output.append("-" * 35)
        output.append(f"Mean Squared Error: {mse:.4f}")
        output.append(f"R-squared: {r2:.4f}")
        output.append(f"Tree depth: {tree.get_depth()}")
        output.append(f"Number of leaves: {tree.get_n_leaves()}")
        output.append("")
        
        # Feature importance
        output.append("Feature Importance")
        output.append("-" * 20)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            output.append(f"{row['Feature']:<15} {row['Importance']:<10.4f}")
        
        output.append("")
        
        return output


class ProcForest:
    """Implementation of SAS PROC FOREST procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC FOREST on the given data.
        
        Args:
            data: Input DataFrame
            proc_info: Parsed PROC statement information
            
        Returns:
            Dictionary containing results and output data
        """
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Get MODEL specification
        model_spec = proc_info.options.get('model', '')
        if not model_spec:
            results['output_text'].append("ERROR: MODEL specification required (e.g., MODEL y = x1 x2).")
            return results
        
        # Parse model specification: MODEL y = x1 x2 x3
        # Split by '=' and then by spaces
        if '=' in model_spec:
            parts = model_spec.split('=')
            target_var = parts[0].strip()
            feature_vars = [var.strip() for var in parts[1].split() if var.strip()]
        else:
            # If no '=' found, treat as single variable
            target_var = model_spec.strip()
            feature_vars = []
        
        # Check if variables exist
        if target_var not in data.columns:
            results['output_text'].append(f"ERROR: Target variable '{target_var}' not found in data.")
            return results
        
        missing_vars = [var for var in feature_vars if var not in data.columns]
        if missing_vars:
            results['output_text'].append(f"ERROR: Feature variables not found: {missing_vars}")
            return results
        
        # Get forest parameters
        n_estimators = proc_info.options.get('ntrees', 100)
        max_depth = proc_info.options.get('maxdepth', None)
        min_samples_split = proc_info.options.get('minsplit', 2)
        min_samples_leaf = proc_info.options.get('minleaf', 1)
        
        results['output_text'].append("PROC FOREST - Random Forest Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Target variable: {target_var}")
        results['output_text'].append(f"Feature variables: {', '.join(feature_vars)}")
        results['output_text'].append(f"Number of trees: {n_estimators}")
        results['output_text'].append("")
        
        # Prepare data
        model_data = data[[target_var] + feature_vars].copy()
        clean_data = model_data.dropna()
        
        if len(clean_data) < 10:
            results['output_text'].append("ERROR: Insufficient data for forest analysis.")
            return results
        
        # Prepare target and features
        y = clean_data[target_var]
        X = clean_data[feature_vars]
        
        # Handle categorical variables
        X_encoded = X.copy()
        label_encoders = {}
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                label_encoders[col] = le
        
        # Determine if classification or regression
        if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 10:
            # Classification
            is_classification = True
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y.astype(str))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
            
            # Fit random forest
            forest = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                oob_score=True
            )
            forest.fit(X_train, y_train)
            
            # Predictions
            y_pred = forest.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Format output
            results['output_text'].extend(self._format_classification_output(forest, X_encoded.columns, accuracy, y_test, y_pred, le_target))
            
        else:
            # Regression
            is_classification = False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
            
            # Fit random forest
            forest = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                oob_score=True
            )
            forest.fit(X_train, y_train)
            
            # Predictions
            y_pred = forest.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Format output
            results['output_text'].extend(self._format_regression_output(forest, X_encoded.columns, mse, r2, y_test, y_pred))
        
        # Create output DataFrame
        feature_importance = pd.DataFrame({
            'Feature': X_encoded.columns,
            'Importance': forest.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        results['output_data'] = {
            'feature_importance': feature_importance,
            'model': forest,
            'is_classification': is_classification
        }
        
        return results
    
    def _format_classification_output(self, forest, feature_names, accuracy, y_test, y_pred, le_target):
        """Format classification forest output."""
        output = []
        
        output.append("Random Forest Classification Results")
        output.append("-" * 40)
        output.append(f"Training accuracy: {accuracy:.4f}")
        output.append(f"Out-of-bag score: {forest.oob_score_:.4f}")
        output.append(f"Number of trees: {forest.n_estimators}")
        output.append("")
        
        # Feature importance
        output.append("Feature Importance")
        output.append("-" * 20)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': forest.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            output.append(f"{row['Feature']:<15} {row['Importance']:<10.4f}")
        
        output.append("")
        
        return output
    
    def _format_regression_output(self, forest, feature_names, mse, r2, y_test, y_pred):
        """Format regression forest output."""
        output = []
        
        output.append("Random Forest Regression Results")
        output.append("-" * 35)
        output.append(f"Mean Squared Error: {mse:.4f}")
        output.append(f"R-squared: {r2:.4f}")
        output.append(f"Out-of-bag score: {forest.oob_score_:.4f}")
        output.append(f"Number of trees: {forest.n_estimators}")
        output.append("")
        
        # Feature importance
        output.append("Feature Importance")
        output.append("-" * 20)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': forest.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            output.append(f"{row['Feature']:<15} {row['Importance']:<10.4f}")
        
        output.append("")
        
        return output


class ProcBoost:
    """Implementation of SAS PROC BOOST procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC BOOST on the given data.
        
        Args:
            data: Input DataFrame
            proc_info: Parsed PROC statement information
            
        Returns:
            Dictionary containing results and output data
        """
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Get MODEL specification
        model_spec = proc_info.options.get('model', '')
        if not model_spec:
            results['output_text'].append("ERROR: MODEL specification required (e.g., MODEL y = x1 x2).")
            return results
        
        # Parse model specification: MODEL y = x1 x2 x3
        # Split by '=' and then by spaces
        if '=' in model_spec:
            parts = model_spec.split('=')
            target_var = parts[0].strip()
            feature_vars = [var.strip() for var in parts[1].split() if var.strip()]
        else:
            # If no '=' found, treat as single variable
            target_var = model_spec.strip()
            feature_vars = []
        
        # Check if variables exist
        if target_var not in data.columns:
            results['output_text'].append(f"ERROR: Target variable '{target_var}' not found in data.")
            return results
        
        missing_vars = [var for var in feature_vars if var not in data.columns]
        if missing_vars:
            results['output_text'].append(f"ERROR: Feature variables not found: {missing_vars}")
            return results
        
        # Get boosting parameters
        n_estimators = proc_info.options.get('ntrees', 100)
        learning_rate = proc_info.options.get('learningrate', 0.1)
        max_depth = proc_info.options.get('maxdepth', 3)
        min_samples_split = proc_info.options.get('minsplit', 2)
        min_samples_leaf = proc_info.options.get('minleaf', 1)
        
        results['output_text'].append("PROC BOOST - Gradient Boosting Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Target variable: {target_var}")
        results['output_text'].append(f"Feature variables: {', '.join(feature_vars)}")
        results['output_text'].append(f"Number of estimators: {n_estimators}")
        results['output_text'].append(f"Learning rate: {learning_rate}")
        results['output_text'].append("")
        
        # Prepare data
        model_data = data[[target_var] + feature_vars].copy()
        clean_data = model_data.dropna()
        
        if len(clean_data) < 10:
            results['output_text'].append("ERROR: Insufficient data for boosting analysis.")
            return results
        
        # Prepare target and features
        y = clean_data[target_var]
        X = clean_data[feature_vars]
        
        # Handle categorical variables
        X_encoded = X.copy()
        label_encoders = {}
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                label_encoders[col] = le
        
        # Determine if classification or regression
        if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 10:
            # Classification
            is_classification = True
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y.astype(str))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
            
            # Fit gradient boosting
            boost = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            boost.fit(X_train, y_train)
            
            # Predictions
            y_pred = boost.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Format output
            results['output_text'].extend(self._format_classification_output(boost, X_encoded.columns, accuracy, y_test, y_pred, le_target))
            
        else:
            # Regression
            is_classification = False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
            
            # Fit gradient boosting
            boost = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            boost.fit(X_train, y_train)
            
            # Predictions
            y_pred = boost.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Format output
            results['output_text'].extend(self._format_regression_output(boost, X_encoded.columns, mse, r2, y_test, y_pred))
        
        # Create output DataFrame
        feature_importance = pd.DataFrame({
            'Feature': X_encoded.columns,
            'Importance': boost.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        results['output_data'] = {
            'feature_importance': feature_importance,
            'model': boost,
            'is_classification': is_classification
        }
        
        return results
    
    def _format_classification_output(self, boost, feature_names, accuracy, y_test, y_pred, le_target):
        """Format classification boosting output."""
        output = []
        
        output.append("Gradient Boosting Classification Results")
        output.append("-" * 45)
        output.append(f"Training accuracy: {accuracy:.4f}")
        output.append(f"Number of estimators: {boost.n_estimators}")
        output.append(f"Learning rate: {boost.learning_rate}")
        output.append("")
        
        # Feature importance
        output.append("Feature Importance")
        output.append("-" * 20)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': boost.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            output.append(f"{row['Feature']:<15} {row['Importance']:<10.4f}")
        
        output.append("")
        
        return output
    
    def _format_regression_output(self, boost, feature_names, mse, r2, y_test, y_pred):
        """Format regression boosting output."""
        output = []
        
        output.append("Gradient Boosting Regression Results")
        output.append("-" * 40)
        output.append(f"Mean Squared Error: {mse:.4f}")
        output.append(f"R-squared: {r2:.4f}")
        output.append(f"Number of estimators: {boost.n_estimators}")
        output.append(f"Learning rate: {boost.learning_rate}")
        output.append("")
        
        # Feature importance
        output.append("Feature Importance")
        output.append("-" * 20)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': boost.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            output.append(f"{row['Feature']:<15} {row['Importance']:<10.4f}")
        
        output.append("")
        
        return output
