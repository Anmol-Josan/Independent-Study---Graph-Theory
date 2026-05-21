#!/usr/bin/env python3
"""
Gene Co-expression Networks: Spectral Clustering and Dimensionality Reduction

This pipeline processes single-cell RNA sequencing data. It downloads the dataset,
extracts it, and identifies cell types, functional clusters, and patient response.
It also compares multiple dimensionality reduction techniques for performance.
"""

import os
import time
import logging
import argparse
import tarfile
import urllib.request
from typing import Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class DataManager:
    """Handles downloading, extracting, and loading expression data and metadata."""
    
    def __init__(self, cache_dir: str, data_filename: str, meta_filename: str, url: str = None):
        self.cache_dir = cache_dir
        self.data_path = os.path.join(cache_dir, data_filename)
        self.meta_path = os.path.join(cache_dir, meta_filename)
        self.url = url

    def fetch_and_extract(self):
        """Downloads the tarball and extracts it into the cache directory."""
        if not self.url:
            logger.warning("No URL provided. Attempting to load from local cache only.")
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        tar_path = os.path.join(self.cache_dir, "dataset.tar.gz")
        
        if os.path.exists(self.data_path) and os.path.exists(self.meta_path):
            logger.info("Data already exists in cache. Skipping download.")
            return

        logger.info(f"Downloading dataset from {self.url}...")
        try:
            urllib.request.urlretrieve(self.url, tar_path)
            logger.info("Download complete. Extracting archive...")
            
            with tarfile.open(tar_path, "r:gz") as archive:
                archive.extractall(path=self.cache_dir)
            logger.info("Extraction finished.")
        except Exception as e:
            logger.error(f"Failed to fetch or extract data: {str(e)}")
            raise

    def load_data(self) -> Tuple[Union[np.ndarray, sp.spmatrix], pd.DataFrame]:
        """Loads the feature matrix and clinical metadata."""
        self.fetch_and_extract()
        
        if not os.path.exists(self.data_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Data files are missing. Ensure the download and extraction completed successfully.")

        logger.info("Loading data from cache.")
        
        if self.data_path.endswith('.npy'):
            X = np.load(self.data_path)
        elif self.data_path.endswith('.npz'):
            X = sp.load_npz(self.data_path)
        else:
            X = pd.read_csv(self.data_path, index_col=0).values

        meta = pd.read_csv(self.meta_path)
        return X, meta


class DimReductionComparator:
    """Manages dimensionality reduction comparisons for performance and quality."""
    
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: Union[np.ndarray, sp.spmatrix], method: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reduces dimensionality using the specified method. 
        Applies a preliminary TruncatedSVD for non-linear methods if features are high
        to preserve memory and runtime efficiency.
        """
        if method not in ['pca', 'tsne', 'umap', 'isomap']:
            raise ValueError(f"Dimensionality reduction method '{method}' is not supported.")

        start_time = time.time()
        metrics = {}
        X_processed = X

        if method in ['tsne', 'umap', 'isomap'] and X.shape[1] > 50:
            logger.info(f"Applying TruncatedSVD before {method.upper()} to limit computational overhead.")
            svd = TruncatedSVD(n_components=50, random_state=self.random_state)
            X_processed = svd.fit_transform(X)

        if method == 'pca':
            if sp.issparse(X_processed):
                model = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
            else:
                model = PCA(n_components=self.n_components, random_state=self.random_state)
            embedding = model.fit_transform(X_processed)
            metrics['explained_variance_ratio'] = float(np.sum(model.explained_variance_ratio_))
            
        elif method == 'tsne':
            model = TSNE(n_components=self.n_components, random_state=self.random_state, init='pca', learning_rate='auto')
            embedding = model.fit_transform(X_processed)
            metrics['kl_divergence'] = float(model.kl_divergence_)
            
        elif method == 'umap':
            if not HAS_UMAP:
                raise ImportError("UMAP library is missing. Please run `pip install umap-learn`.")
            model = umap.UMAP(n_components=self.n_components, random_state=self.random_state)
            embedding = model.fit_transform(X_processed)
            
        elif method == 'isomap':
            model = Isomap(n_components=self.n_components, n_jobs=-1)
            embedding = model.fit_transform(X_processed)

        elapsed_time = time.time() - start_time
        metrics['time_seconds'] = elapsed_time
        logger.info(f"{method.upper()} completed in {elapsed_time:.2f} seconds.")
        
        return embedding, metrics


class SpectralClusterer:
    """Executes scalable spectral clustering on single-cell data."""
    
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def cluster(self, X: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
        """
        Runs spectral clustering.
        Using 'nearest_neighbors' affinity creates a sparse adjacency graph natively.
        This is significantly lighter on memory than the standard RBF kernel.
        """
        logger.info(f"Starting Spectral Clustering with k={self.n_clusters}")
        start_time = time.time()
        
        model = SpectralClustering(
            n_clusters=self.n_clusters, 
            affinity='nearest_neighbors',
            n_neighbors=15,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        labels = model.fit_predict(X)
        logger.info(f"Clustering finished in {time.time() - start_time:.2f} seconds.")
        return labels


class EvaluationEngine:
    """Calculates metrics and profiles clusters against clinical and biological labels."""
    
    @staticmethod
    def calculate_silhouette(X: Union[np.ndarray, sp.spmatrix], labels: np.ndarray) -> float:
        """Computes the silhouette score. Converts sparse to dense only when required."""
        logger.info("Calculating Silhouette Score.")
        X_eval = X.toarray() if sp.issparse(X) else X
        score = silhouette_score(X_eval, labels)
        return float(score)

    @staticmethod
    def characterize_clusters(meta: pd.DataFrame, cluster_labels: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Maps cluster assignments to biological and clinical states."""
        meta_temp = meta.copy()
        meta_temp['Cluster'] = cluster_labels
        
        response_profile = pd.crosstab(
            meta_temp['Cluster'], 
            meta_temp['response_status'], 
            normalize='index'
        ) * 100
        
        cell_type_profile = pd.crosstab(
            meta_temp['Cluster'], 
            meta_temp['cell_type'], 
            normalize='index'
        ) * 100
        
        return response_profile, cell_type_profile


class Visualizer:
    """Generates analytical plots for the pipeline."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_embedding(self, embedding: np.ndarray, labels: np.ndarray, title: str, filename: str):
        """Creates a 2D scatter plot of the dimensional reduction colored by clusters."""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=embedding[:, 0], y=embedding[:, 1], 
            hue=labels, palette='tab10', s=15, alpha=0.8, legend='full'
        )
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved visualization to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Spectral Clustering and Dim Reduction Pipeline")
    parser.add_argument("--download-dir", type=str, default="./cache", help="Directory for downloaded data")
    parser.add_argument("--data-file", type=str, default="data.npy", help="Name of the expression matrix file")
    parser.add_argument("--meta-file", type=str, default="meta.csv", help="Name of the clinical metadata file")
    parser.add_argument("--url", type=str, default="", help="URL to download the dataset tarball")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters for Spectral Clustering")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory for output files")
    args = parser.parse_args()

    # 1. Download and Load Data
    manager = DataManager(
        cache_dir=args.download_dir, 
        data_filename=args.data_file, 
        meta_filename=args.meta_file, 
        url=args.url if args.url else None
    )
    X, meta = manager.load_data()
    
    logger.info("Normalizing features for consistent distance metrics.")
    if sp.issparse(X):
        X_scaled = X.copy()
        X_scaled.data = X_scaled.data / np.std(X_scaled.data)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # 2. Dimensionality Reduction Comparison
    reducer = DimReductionComparator(n_components=2)
    methods = ['pca', 'tsne', 'umap', 'isomap'] if HAS_UMAP else ['pca', 'tsne', 'isomap']
    
    embeddings = {}
    dr_metrics = {}
    
    for method in methods:
        try:
            emb, metrics = reducer.fit_transform(X_scaled, method)
            embeddings[method] = emb
            dr_metrics[method] = metrics
        except Exception as e:
            logger.error(f"Failed to run {method}: {str(e)}")

    print("\nDimensionality Reduction Performance:")
    for m, metrics in dr_metrics.items():
        print(f" - {m.upper()}: {metrics['time_seconds']:.2f} seconds")

    # 3. Spectral Clustering
    clustering_input, _ = reducer.fit_transform(X_scaled, 'pca') 
    
    clusterer = SpectralClusterer(n_clusters=args.clusters)
    labels = clusterer.cluster(clustering_input)

    # 4. Evaluation & Profiling
    engine = EvaluationEngine()
    sil_score = engine.calculate_silhouette(clustering_input, labels)
    print(f"\nSpectral Clustering Silhouette Score: {sil_score:.4f}")

    response_profile, cell_type_profile = engine.characterize_clusters(meta, labels)
    print("\nResponder Distribution per Cluster (%):")
    print(response_profile.round(2))
    
    print("\nCell Type Distribution per Cluster (%):")
    print(cell_type_profile.round(2))

    # 5. Visualization
    viz = Visualizer(args.output_dir)
    for method, emb in embeddings.items():
        viz.plot_embedding(emb, labels, f"Spectral Clusters in {method.upper()} Space", f"cluster_{method}.png")
        
        if 'response_status' in meta.columns:
            viz.plot_embedding(emb, meta['response_status'].values, f"Clinical Response in {method.upper()} Space", f"response_{method}.png")

if __name__ == "__main__":
    main()