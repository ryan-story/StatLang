"""
PROC CLUSTER Implementation for Open-SAS

This module implements SAS PROC CLUSTER functionality for clustering analysis
including k-means, hierarchical clustering, and HDBSCAN.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional
from ..parser.proc_parser import ProcStatement


class ProcCluster:
    """Implementation of SAS PROC CLUSTER procedure."""
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, dataset_manager=None) -> Dict[str, Any]:
        """
        Execute PROC CLUSTER on the given data.
        
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
        
        # Get VAR variables
        var_vars = proc_info.options.get('var', [])
        if not var_vars:
            # If no VAR specified, use all numeric columns
            var_vars = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to only include variables that exist in the data
        var_vars = [var for var in var_vars if var in data.columns]
        
        if not var_vars:
            results['output_text'].append("ERROR: No valid numeric variables found for clustering.")
            return results
        
        if len(var_vars) < 2:
            results['output_text'].append("ERROR: At least 2 variables required for clustering.")
            return results
        
        # Get clustering method (default: kmeans)
        method = proc_info.options.get('method', 'kmeans').lower()
        if method not in ['kmeans', 'hierarchical', 'hdbscan']:
            method = 'kmeans'
        
        # Get number of clusters
        n_clusters = proc_info.options.get('nclusters', 3)
        if n_clusters < 2:
            n_clusters = 3
        
        # Get standardization option
        standardize = proc_info.options.get('standardize', True)
        
        results['output_text'].append("PROC CLUSTER - Clustering Analysis")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"Method: {method.upper()}")
        results['output_text'].append(f"Number of clusters: {n_clusters}")
        results['output_text'].append(f"Standardize: {standardize}")
        results['output_text'].append("")
        
        # Prepare data
        analysis_data = data[var_vars].select_dtypes(include=[np.number])
        clean_data = analysis_data.dropna()
        
        if len(clean_data) < 2:
            results['output_text'].append("ERROR: Insufficient data for clustering.")
            return results
        
        # Standardize data if requested
        if standardize:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clean_data)
            analysis_data_scaled = pd.DataFrame(scaled_data, columns=var_vars, index=clean_data.index)
        else:
            analysis_data_scaled = clean_data
        
        # Perform clustering
        if method == 'kmeans':
            results.update(self._perform_kmeans(analysis_data_scaled, var_vars, n_clusters))
        elif method == 'hierarchical':
            results.update(self._perform_hierarchical(analysis_data_scaled, var_vars, n_clusters))
        else:  # hdbscan
            results.update(self._perform_hdbscan(analysis_data_scaled, var_vars))
        
        return results
    
    def _perform_kmeans(self, data: pd.DataFrame, var_names: List[str], n_clusters: int) -> Dict[str, Any]:
        """Perform K-means clustering."""
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        centroids = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        
        # Format output
        results['output_text'].append("K-Means Clustering Results")
        results['output_text'].append("-" * 40)
        results['output_text'].append("")
        
        # Cluster sizes
        results['output_text'].append("Cluster Sizes")
        results['output_text'].append("-" * 20)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            results['output_text'].append(f"Cluster {label}: {count} observations")
        results['output_text'].append("")
        
        # Cluster centroids
        results['output_text'].append("Cluster Centroids")
        results['output_text'].append("-" * 30)
        
        # Header
        header = f"{'Variable':<12}"
        for i in range(n_clusters):
            header += f"{'Cluster' + str(i):>12}"
        results['output_text'].append(header)
        results['output_text'].append("-" * len(header))
        
        # Centroids
        for i, var in enumerate(var_names):
            row = f"{var[:12]:<12}"
            for j in range(n_clusters):
                row += f"{centroids[j, i]:>12.4f}"
            results['output_text'].append(row)
        
        results['output_text'].append("")
        
        # Model statistics
        results['output_text'].append("Model Statistics")
        results['output_text'].append("-" * 20)
        results['output_text'].append(f"Within-cluster sum of squares: {inertia:.4f}")
        results['output_text'].append(f"Number of iterations: {kmeans.n_iter_}")
        results['output_text'].append("")
        
        # Create output DataFrame
        cluster_df = data.copy()
        cluster_df['Cluster'] = cluster_labels
        
        centroids_df = pd.DataFrame(
            centroids,
            columns=var_names,
            index=[f'Cluster_{i}' for i in range(n_clusters)]
        )
        
        results['output_data'] = {
            'clustered_data': cluster_df,
            'centroids': centroids_df,
            'model': kmeans,
            'inertia': inertia
        }
        
        return results
    
    def _perform_hierarchical(self, data: pd.DataFrame, var_names: List[str], n_clusters: int) -> Dict[str, Any]:
        """Perform Hierarchical clustering."""
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Fit Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = hierarchical.fit_predict(data)
        
        # Format output
        results['output_text'].append("Hierarchical Clustering Results")
        results['output_text'].append("-" * 40)
        results['output_text'].append("")
        
        # Cluster sizes
        results['output_text'].append("Cluster Sizes")
        results['output_text'].append("-" * 20)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            results['output_text'].append(f"Cluster {label}: {count} observations")
        results['output_text'].append("")
        
        # Calculate cluster centroids manually
        centroids = []
        for i in range(n_clusters):
            cluster_data = data[cluster_labels == i]
            if len(cluster_data) > 0:
                centroid = cluster_data.mean().values
                centroids.append(centroid)
            else:
                centroids.append(np.zeros(len(var_names)))
        
        centroids = np.array(centroids)
        
        # Cluster centroids
        results['output_text'].append("Cluster Centroids")
        results['output_text'].append("-" * 30)
        
        # Header
        header = f"{'Variable':<12}"
        for i in range(n_clusters):
            header += f"{'Cluster' + str(i):>12}"
        results['output_text'].append(header)
        results['output_text'].append("-" * len(header))
        
        # Centroids
        for i, var in enumerate(var_names):
            row = f"{var[:12]:<12}"
            for j in range(n_clusters):
                row += f"{centroids[j, i]:>12.4f}"
            results['output_text'].append(row)
        
        results['output_text'].append("")
        
        # Create output DataFrame
        cluster_df = data.copy()
        cluster_df['Cluster'] = cluster_labels
        
        centroids_df = pd.DataFrame(
            centroids,
            columns=var_names,
            index=[f'Cluster_{i}' for i in range(n_clusters)]
        )
        
        results['output_data'] = {
            'clustered_data': cluster_df,
            'centroids': centroids_df,
            'model': hierarchical
        }
        
        return results
    
    def _perform_hdbscan(self, data: pd.DataFrame, var_names: List[str]) -> Dict[str, Any]:
        """Perform HDBSCAN clustering."""
        results = {
            'output_text': [],
            'output_data': None
        }
        
        # Fit HDBSCAN
        hdbscan = HDBSCAN(min_cluster_size=5, min_samples=3)
        cluster_labels = hdbscan.fit_predict(data)
        
        # Format output
        results['output_text'].append("HDBSCAN Clustering Results")
        results['output_text'].append("-" * 40)
        results['output_text'].append("")
        
        # Cluster sizes (including noise points)
        results['output_text'].append("Cluster Sizes")
        results['output_text'].append("-" * 20)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label == -1:
                results['output_text'].append(f"Noise points: {count} observations")
            else:
                results['output_text'].append(f"Cluster {label}: {count} observations")
        results['output_text'].append("")
        
        # Number of clusters found
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        results['output_text'].append(f"Number of clusters found: {n_clusters_found}")
        results['output_text'].append("")
        
        # Calculate cluster centroids for non-noise clusters
        if n_clusters_found > 0:
            centroids = []
            cluster_ids = []
            for label in unique_labels:
                if label != -1:  # Skip noise points
                    cluster_data = data[cluster_labels == label]
                    if len(cluster_data) > 0:
                        centroid = cluster_data.mean().values
                        centroids.append(centroid)
                        cluster_ids.append(label)
            
            if centroids:
                centroids = np.array(centroids)
                
                # Cluster centroids
                results['output_text'].append("Cluster Centroids")
                results['output_text'].append("-" * 30)
                
                # Header
                header = f"{'Variable':<12}"
                for cluster_id in cluster_ids:
                    header += f"{'Cluster' + str(cluster_id):>12}"
                results['output_text'].append(header)
                results['output_text'].append("-" * len(header))
                
                # Centroids
                for i, var in enumerate(var_names):
                    row = f"{var[:12]:<12}"
                    for j in range(len(cluster_ids)):
                        row += f"{centroids[j, i]:>12.4f}"
                    results['output_text'].append(row)
                
                results['output_text'].append("")
                
                # Create centroids DataFrame
                centroids_df = pd.DataFrame(
                    centroids,
                    columns=var_names,
                    index=[f'Cluster_{cluster_id}' for cluster_id in cluster_ids]
                )
            else:
                centroids_df = pd.DataFrame()
        else:
            centroids_df = pd.DataFrame()
        
        # Create output DataFrame
        cluster_df = data.copy()
        cluster_df['Cluster'] = cluster_labels
        
        results['output_data'] = {
            'clustered_data': cluster_df,
            'centroids': centroids_df,
            'model': hdbscan,
            'n_clusters': n_clusters_found
        }
        
        return results
