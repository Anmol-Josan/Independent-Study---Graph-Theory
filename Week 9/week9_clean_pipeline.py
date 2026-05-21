#!/usr/bin/env python3
"""
Optimized Spectral Clustering Pipeline for GSE300475
Following the architectural patterns and clinical metadata structure of hr-cancer.py
"""

import argparse
import gc
import gzip
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
try:
    matplotlib.use('Agg')
except Exception as e:
    print(f"[*] Backend notice: {e}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
from scipy.stats import chi2_contingency

# Memory & Threading Optimizations
os.environ['PYTHONHASHSEED'] = '0'
os.environ['OMP_NUM_THREADS'] = '4'

import psutil
def print_memory_usage() -> float:
    """Monitor runtime memory footprints to guarantee safety."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    vm = psutil.virtual_memory()
    print(f"[Memory] Current: {mem_mb:.2f} MB | System: {vm.percent}% used")
    return mem_mb

# Framework Dependencies
try:
    import scanpy as sc
    import anndata as ad
    # Add this setting right after your imports to handle modern pandas string arrays
    try:
        ad.settings.allow_write_nullable_strings = True
    except AttributeError:
        pass
except ImportError:
    print("[-] Scanpy or AnnData missing. Please run: pip install scanpy anndata")
    sys.exit(1)

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import pacmap
    HAS_PACMAP = True
except ImportError:
    HAS_PACMAP = False

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class DatasetManager:
    """Handles dataset verification, asset downloads, and archive extractions."""
    
    @staticmethod
    def download_file(url: str, dest_path: str, force: bool = False) -> bool:
        if os.path.exists(dest_path) and not force:
            print(f"[*] File already exists: {dest_path}. Skipping download.")
            return True
        print(f"[+] Downloading raw files from: {url}")
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
                chunk_size = 1024 * 1024
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            print("[+] Download completed.")
            return True
        except Exception as e:
            print(f"[-] Download error: {e}")
            return False

    @staticmethod
    def extract_tar(tar_path: str, extract_dir: str) -> bool:
        if os.path.exists(extract_dir) and os.listdir(extract_dir):
            print(f"[*] Directory {extract_dir} populated. Skipping extraction.")
            return True
        print(f"[+] Extracting source archive: {tar_path}")
        try:
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(tar_path, "r") as tar:
                try:
                    tar.extractall(path=extract_dir, filter='data')
                except TypeError:
                    tar.extractall(path=extract_dir)
            print("[+] Extraction finished.")
            return True
        except Exception as e:
            print(f"[-] Extraction error: {e}")
            return False


def get_clinical_metadata() -> pd.DataFrame:
    """Constructs the standard mapping for the 11 clinical trial trial samples."""
    return pd.DataFrame([
        {'S_Number': 'S1',  'GEX_Sample_ID': 'GSM9061665', 'Patient_ID': 'PT1', 'Timepoint': 'Baseline',   'Response': 'Responder'},
        {'S_Number': 'S2',  'GEX_Sample_ID': 'GSM9061666', 'Patient_ID': 'PT1', 'Timepoint': 'Post-Tx',    'Response': 'Responder'},
        {'S_Number': 'S3',  'GEX_Sample_ID': 'GSM9061667', 'Patient_ID': 'PT1', 'Timepoint': 'Recurrence', 'Response': 'Responder'},
        {'S_Number': 'S4',  'GEX_Sample_ID': 'GSM9061668', 'Patient_ID': 'PT2', 'Timepoint': 'Baseline',   'Response': 'Responder'},
        {'S_Number': 'S5',  'GEX_Sample_ID': 'GSM9061669', 'Patient_ID': 'PT2', 'Timepoint': 'Post-Tx',    'Response': 'Responder'},
        {'S_Number': 'S6',  'GEX_Sample_ID': 'GSM9061670', 'Patient_ID': 'PT3', 'Timepoint': 'Baseline',   'Response': 'Non-Responder'},
        {'S_Number': 'S7',  'GEX_Sample_ID': 'GSM9061671', 'Patient_ID': 'PT3', 'Timepoint': 'Post-Tx',    'Response': 'Non-Responder'},
        {'S_Number': 'S8',  'GEX_Sample_ID': 'GSM9061672', 'Patient_ID': 'PT3', 'Timepoint': 'Recurrence', 'Response': 'Non-Responder'},
        {'S_Number': 'S9',  'GEX_Sample_ID': 'GSM9061673', 'Patient_ID': 'PT4', 'Timepoint': 'Baseline',   'Response': 'Non-Responder'},
        {'S_Number': 'S10', 'GEX_Sample_ID': 'GSM9061674', 'Patient_ID': 'PT4', 'Timepoint': 'Post-Tx',    'Response': 'Non-Responder'},
        {'S_Number': 'S11', 'GEX_Sample_ID': 'GSM9061675', 'Patient_ID': 'PT4', 'Timepoint': 'Recurrence', 'Response': 'Non-Responder'},
    ])


def run_map_reduce_loading(metadata_df: pd.DataFrame, raw_data_dir: Path, max_cells: int) -> ad.AnnData:
    """
    Implements a disk-backed map-reduce workflow. Maps file loading and cell-level 
    QC constraints per-sample before merging into a unified tracking matrix.
    """
    temp_chunk_dir = Path("temp")
    if temp_chunk_dir.exists():
        shutil.rmtree(temp_chunk_dir)
    temp_chunk_dir.mkdir(exist_ok=True)
    
    chunk_files = []
    print("[+] Executing Map Phase: Processing sample chunks safely on disk...")
    
    for idx, row in metadata_df.iterrows():
        gex_id = row['GEX_Sample_ID']
        s_num = row['S_Number']
        prefix = f"{gex_id}_{s_num}"
        
        # Discover specific file path mappings matching the archive configuration
        matrix_file = None
        for ext in ['matrix.mtx.gz', 'matrix.mtx']:
            candidate = raw_data_dir / f"{prefix}_{ext}"
            if candidate.exists():
                matrix_file = candidate
                break
        if not matrix_file:
            possible = list(raw_data_dir.glob(f"*{gex_id}*{ext}"))
            if possible:
                matrix_file = possible[0]
                
        if not matrix_file:
            print(f"[*] Skipping sample path {prefix}: Matrix assets not discovered.")
            continue
            
        try:
            # Read utilizing integrated single-cell text-stream engines
            adata_sample = sc.read_10x_mtx(
                matrix_file.parent,
                var_names='gene_symbols',
                prefix=matrix_file.name.replace('matrix.mtx', '').replace('.gz', ''),
                cache=True
            )
            
            # Enforce compressed standard float32 precision immediately
            adata_sample.X = sp.csr_matrix(adata_sample.X, dtype=np.float32)
            
            # Affix categorical trial tracks
            adata_sample.obs['sample_id'] = str(gex_id)
            adata_sample.obs['patient_id'] = str(row['Patient_ID'])
            adata_sample.obs['timepoint'] = str(row['Timepoint'])
            adata_sample.obs['response'] = str(row['Response'])

            # Clean object data types to ensure strict backward compatibility
            for col in adata_sample.obs.columns:
                if adata_sample.obs[col].dtype.name in ['string', 'object']:
                    # Force conversion to base Python string format to clear pandas Nullable wrapper
                    adata_sample.obs[col] = adata_sample.obs[col].astype(str)
            
            # Early QC Filtering matching the notebook constraints
            sc.pp.filter_cells(adata_sample, min_genes=200)
            sc.pp.filter_genes(adata_sample, min_cells=3)
            adata_sample.var_names_make_unique()
            
            # Save chunk securely
            chunk_path = temp_chunk_dir / f"chunk_{idx}_{gex_id}.h5ad"
            adata_sample.write_h5ad(chunk_path, compression='gzip')
            chunk_files.append(chunk_path)
            
            del adata_sample
            gc.collect()
        except Exception as e:
            print(f"[-] Failed processing sample {prefix}: {e}")
            continue

    print("[+] Executing Reduce Phase: Merging matrix chunks safely...")
    if not chunk_files:
        raise RuntimeError("No valid data chunks were successfully mapped.")
        
    adatas = [sc.read_h5ad(f) for f in chunk_files]
    adata = ad.concat(adatas, join="outer", label="batch")
    adata.obs_names_make_unique()
    
    # Free chunk storage resources
    shutil.rmtree(temp_chunk_dir)
    
    # Cap size boundaries to enforce safe execution limits
    if adata.n_obs > max_cells:
        print(f"[*] Subsampling cell manifold down from {adata.n_obs} to {max_cells} targets.")
        np.random.seed(42)
        idx_choice = np.random.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx_choice, :].copy()
        
    print_memory_usage()
    return adata


def preprocess_adata_manifold(adata: ad.AnnData, n_top_features: int) -> Tuple[np.ndarray, ad.AnnData]:
    """
    Executes biology-preserving variance transformations directly inside 
    the unified expression framework.
    """
    print("[+] Executing single-cell preprocessing pipeline...")
    # Library size normalization matching the standard 10k target scale
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select variance-stabilized highly variable genes (HVGs) 
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, flavor='seurat')
    
    # Isolate feature tracking representations
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    
    # Scale feature coordinates to balance overall variance impacts across dimensions
    X_dense = adata_hvg.X.toarray() if hasattr(adata_hvg.X, 'toarray') else np.asarray(adata_hvg.X)
    X_mean = X_dense.mean(axis=0)
    X_std = X_dense.std(axis=0)
    X_std = np.where(X_std == 0, 1.0, X_std)
    X_scaled = (X_dense - X_mean) / X_std
    
    return X_scaled, adata_hvg


def spectral_cluster_optimized(
    X_scaled: np.ndarray, 
    n_clusters: Optional[int], 
    n_neighbors: int, 
    random_state: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Performs spectral embedding using a localized Zelnik-Manor scaling 
    approach alongside automated structural Eigengap evaluation.
    """
    print("[+] Building localized affinity matrix tracking topologies...")
    n_pc = min(30, X_scaled.shape[1], X_scaled.shape[0] - 1)
    X_pc = PCA(n_components=n_pc, random_state=random_state).fit_transform(X_scaled)
    
    n_samples = X_pc.shape[0]
    tree = cKDTree(X_pc)
    k_search = min(n_neighbors + 1, n_samples)
    dists, inds = tree.query(X_pc, k=k_search)

    # Calculate individual scaling radii across target boundaries
    local_k = min(7, dists.shape[1] - 1)
    sigmas = dists[:, local_k]
    sigmas = np.where(sigmas <= 0, 1e-6, sigmas)

    rows, cols, vals = [], [], []
    for i in range(n_samples):
        for idx, j in enumerate(inds[i, 1:]):
            dist = dists[i, idx + 1]
            w = np.exp(-(dist * dist) / (sigmas[i] * sigmas[j]))
            rows.append(i)
            cols.append(j)
            vals.append(w)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples), dtype=np.float32)
    A = 0.5 * (A + A.T)

    degree = np.asarray(A.sum(axis=1)).ravel()
    degree = np.where(degree <= 0, 1e-6, degree)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degree))
    L_sym = D_inv_sqrt @ A @ D_inv_sqrt

    max_k = min(30, n_samples - 2)
    eigvals, eigvecs = eigsh(L_sym, k=max_k, which="LA")
    
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Automated structure determination via clear mathematical eigengap optimization
    if n_clusters is None or n_clusters <= 0:
        gaps = np.diff(eigvals)
        search_limit = min(15, len(gaps) - 1)
        optimal_k = int(np.argmin(gaps[1:search_limit]) + 2)
        print(f"[+] Eigengap selection isolated structural target at: {optimal_k} clusters.")
    else:
        optimal_k = n_clusters

    U = eigvecs[:, :optimal_k]
    row_norm = np.linalg.norm(U, axis=1, keepdims=True)
    row_norm = np.where(row_norm == 0, 1e-6, row_norm)
    U_normalized = U / row_norm

    labels = KMeans(n_clusters=optimal_k, n_init=20, random_state=random_state).fit_predict(U_normalized)
    return labels, eigvals[:max_k], optimal_k


def run_clinical_analysis(adata: ad.AnnData, labels: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes rigorous cross-tabulations and chi-squared contingency statistics 
    to evaluate response patterns.
    """
    print("[+] Evaluating clinical cohort associations across discovered states...")
    adata.obs['Spectral_Cluster'] = labels.astype(str)
    
    # 1. Patient distribution profiling
    composition_df = pd.crosstab(
        adata.obs['Spectral_Cluster'], 
        adata.obs['patient_id'], 
        normalize='index'
    ) * 100
    
    # 2. Chi-Squared evaluation of immunotherapy response profiles
    enrichment_rows = []
    unique_clusters = sorted(adata.obs['Spectral_Cluster'].unique(), key=int)
    
    for cluster in unique_clusters:
        in_cluster = adata.obs['Spectral_Cluster'] == cluster
        
        r_in = int((adata.obs['response'][in_cluster] == 'Responder').sum())
        nr_in = int((adata.obs['response'][in_cluster] == 'Non-Responder').sum())
        r_out = int((adata.obs['response'][~in_cluster] == 'Responder').sum())
        nr_out = int((adata.obs['response'][~in_cluster] == 'Non-Responder').sum())
        
        eps = 1e-6
        fold_ratio = ((r_in + eps) / (nr_in + eps)) / ((r_out + eps) / (nr_out + eps))
        
        contingency_table = np.array([[r_in, nr_in], [r_out, nr_out]])
        p_val = 1.0
        if contingency_table.sum() > 0 and np.all(contingency_table.sum(axis=0) > 0) and np.all(contingency_table.sum(axis=1) > 0):
            _, p_val, _, _ = chi2_contingency(contingency_table)
            
        status = "Response Associated" if fold_ratio > 1.2 else ("Resistance Associated" if fold_ratio < 0.8 else "Neutral")
        
        # Extrapolate dominant timeline tracking
        dominant_timepoint = adata.obs['timepoint'][in_cluster].mode().iloc[0]
        
        enrichment_rows.append({
            'Cluster': cluster,
            'Dominant_Timepoint': dominant_timepoint,
            'Responders': r_in,
            'NonResponders': nr_in,
            'OddsRatio': fold_ratio,
            'PValue': p_val,
            'Clinical_Classification': status
        })
        
    return composition_df, pd.DataFrame(enrichment_rows)


def compute_embeddings(X_scaled: np.ndarray, random_state: int) -> Dict[str, np.ndarray]:
    """Generates down-projected representations to compare cell layouts."""
    emb = {}
    print("[+] Projecting multidimensional structural visualizations...")
    emb["pca"] = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)
    
    n_samples = X_scaled.shape[0]
    perplexity = float(max(10, min(50, n_samples // 30)))
    emb["tsne"] = TSNE(
        n_components=2, random_state=random_state, init="pca", 
        perplexity=perplexity, learning_rate="auto"
    ).fit_transform(X_scaled)

    if HAS_UMAP:
        emb["umap"] = umap.UMAP(n_components=2, random_state=random_state).fit_transform(X_scaled)
    if HAS_PACMAP:
        emb["pacmap"] = pacmap.PaCMAP(n_components=2, random_state=random_state).fit_transform(X_scaled)
        
    return emb


def generate_publication_plots(
    embeddings: Dict[str, np.ndarray], 
    adata: ad.AnnData, 
    clinical_df: pd.DataFrame, 
    out_dir: Path
) -> None:
    """Exports structured figures matching professional scientific presentation formats."""
    poster_dir = out_dir / "plots"
    poster_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # 1. Cluster layout vs Clinical Trial Layout comparisons
    for name, coords in embeddings.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
        
        # Cluster view
        scat1 = axes[0].scatter(
            coords[:, 0], coords[:, 1], 
            c=adata.obs['Spectral_Cluster'].astype(int), 
            s=8, cmap="tab10", alpha=0.75, linewidths=0
        )
        axes[0].set_title(f"{name.upper()} Layout - Discovered Spectral Clusters")
        fig.colorbar(scat1, ax=axes[0], label="Cluster Index")
        
        # Clinical response target view
        resp_color = np.where(adata.obs['response'] == 'Responder', 1, 0)
        scat2 = axes[1].scatter(
            coords[:, 0], coords[:, 1], 
            c=resp_color, s=8, cmap="coolwarm", alpha=0.75, linewidths=0
        )
        axes[1].set_title(f"{name.upper()} Layout - Clinical Response Space")
        cbar = fig.colorbar(scat2, ax=axes[1])
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Non-Responder', 'Responder'])
        
        plt.tight_layout()
        plt.savefig(poster_dir / f"clinical_projection_{name}.png", dpi=300)
        plt.close()
        
    # 2. Longitudinal Timeline Composition Profile
    timeline_comp = pd.crosstab(adata.obs['Spectral_Cluster'], adata.obs['timepoint'], normalize='index') * 100
    timeline_comp.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
    plt.title("Longitudinal Trial Timeline Composition by Spectral Cluster")
    plt.xlabel("Spectral Cluster Index")
    plt.ylabel("Composition Share (%)")
    plt.legend(title="Trial Stage")
    plt.tight_layout()
    plt.savefig(poster_dir / "trial_timeline_distribution.png", dpi=300)
    plt.close()


def run(args: argparse.Namespace) -> None:
    print("=== Initial Setup & Verification ===")
    print_memory_usage()
    
    download_path = Path(args.download_dir)
    output_path = Path(args.output_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tar_url = args.tar_url
    if "format=file" in tar_url and "acc=" in tar_url:
        parsed = urllib.parse.urlparse(tar_url)
        q = urllib.parse.parse_qs(parsed.query)
        acc = q.get("acc", [None])[0]
        if acc:
            tar_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:6]}nnn/{acc}/suppl/{acc}_RAW.tar"

    tar_file = download_path / "GSE300475_RAW.tar"
    extract_dir = download_path / "GSE300475_RAW_extracted"
    
    if not DatasetManager.download_file(tar_url, str(tar_file), force=args.force_download):
        raise RuntimeError("Download asset request terminated unexpectedly.")
    if not DatasetManager.extract_tar(str(tar_file), str(extract_dir)):
        raise RuntimeError("Extraction request terminated unexpectedly.")
        
    # Fetch clean clinical reference dataframes
    metadata_df = get_clinical_metadata()
    
    # Perform map-reduce data transformations
    adata = run_map_reduce_loading(metadata_df, extract_dir, max_cells=args.max_cells)
    
    # Process gene expression manifolds safely without losing information
    X_scaled, adata_hvg = preprocess_adata_manifold(adata, args.n_top_features)
    
    # Perform optimized partitioning calculations
    cluster_target = None if args.auto_tune else args.n_clusters
    labels, eigvals, optimal_k = spectral_cluster_optimized(
        X_scaled, cluster_target, args.n_neighbors, args.random_state
    )
    
    # Extract structural analysis readouts
    composition_df, clinical_df = run_clinical_analysis(adata, labels)
    
    print("\n=== Discovered Quality Evaluation ===")
    embeddings = compute_embeddings(X_scaled, args.random_state)
    for name, coords in embeddings.items():
        score = silhouette_score(coords, labels)
        print(f" Silhouette Separation Profile ({name.upper()} Manifold): {score:.4f}")
        
    print("\n=== Clinical Response Contingency Analysis ===")
    print(clinical_df.to_string(index=False))
    
    # Save processed tabular summaries and graphics
    clinical_df.to_csv(output_path / "clinical_enrichment_readouts.csv", index=False)
    composition_df.to_csv(output_path / "patient_share_compositions.csv")
    generate_publication_plots(embeddings, adata, clinical_df, output_path)
    
    print(f"\n[+] Tabular analytics exported to: {output_path.resolve()}")
    print(f"[+] Graphical layouts exported to: {(output_path / 'plots').resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean single-cell clinical trial analysis pipeline")
    p.add_argument("--download-dir", default="cache")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--tar-url", default="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE300475&format=file")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--max-cells", type=int, default=3000, help="Manifold subsample threshold ceiling")
    p.add_argument("--n-top-features", type=int, default=2000, help="Highly variable gene feature count limit")
    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--auto-tune", action="store_true", help="Automate structural state optimization via graph gaps")
    p.add_argument("--random-state", type=int, default=42)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())