# Implementation Plan - Multi-Reduction Comparative Analysis & Immunotherapy Responder Enrichment

This revised plan outlines a comprehensive, scientifically rigorous comparative analysis of single-cell dimensionality reductions alongside a detailed clinical responder/non-responder biomarker analysis. We prioritize deep analysis, extensive logging, and clean, modular high-resolution poster figure exports.

## Core Features of the Analysis

### 1. Multi-Reduction Embedding Suite
We will run four major dimensionality reduction methods to project the 30-PC preprocessed gene expression manifold into 2D:
- **PCA (Principal Component Analysis)**: Purely linear projection capturing global variance.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Non-linear projection emphasizing local neighborhood preservation.
- **UMAP (Uniform Manifold Approximation and Projection)**: Preserves both local and global topology. (Falls back to **Spectral Embedding** if `umap-learn` is not installed).
- **PaCMAP (Pairwise Controlled Manifold Approximation)**: Employs a three-stage pairwise force optimization scheduler.

### 2. Full Quantitative Analysis per Reduction Type
To do the work right, we will perform a full mathematical and visual analysis of **each** reduction type:
- **Logging & Diagnostics**: Log execution milestones, input shapes, execution durations, and final coordinate stats.
- **Cluster Preserved Separation Metric**: For each 2D embedding space, we will compute the **Mean Silhouette Score** using the spectral clustering labels. This quantifies how well each dimensionality reduction method preserves the network clustering boundaries in the low-dimensional visualization plane.
- **Dual Poster Figure Exports**: Save separate, high-resolution (300 DPI) figures for **each** method:
  1. An embedding scatter plot colored by **Spectral Cluster** (with custom-colored Convex Hulls and centroid labels).
  2. An embedding scatter plot colored by **Biological Cell Type** (matching consistent cell-type markers).

### 3. Patient Cohort & Clinical Responder Modeling
We partition cells across 6 patients with a simulated immunotherapy outcome:
- **Responders (pCR)**: `Pt_01`, `Pt_02`, `Pt_03`
- **Non-Responders (No-pCR)**: `Pt_04`, `Pt_05`, `Pt_06`

To simulate a realistic biological microenvironment for breast cancer immunotherapy:
- **T Cells** and **B Cells** are seeded with a 70% probability into the **Responder** patients.
- **Tumor/Epithelial Cells** are seeded with a 70% probability into the **Non-Responder** patients.
- All other cell types (Myeloid, Endothelial, Stromal) are distributed uniformly.

### 4. Clinical Cluster Importance & Statistical Enrichment
Using the consensus spectral clustering partitions, we evaluate therapy biomarkers:
- **Odds Ratio (OR) / Fold Ratio**:
  $$\text{Fold Ratio}_c = \frac{P_{resp, c} + \epsilon}{P_{non\_resp, c} + \epsilon}$$
  Ratio $> 1.2 \implies$ **Response Biomarker** (enriched in responders).
  Ratio $< 0.8 \implies$ **Resistance Biomarker** (enriched in non-responders).
- **Chi-Square Contingency Test**: Computes a $p$-value for each cluster to evaluate if its enrichment in responders/non-responders is statistically significant.
- **Tabular Report**: Prints a clean clinical biomarker summary table to the console.
- **Separate Poster Figures**:
  - `poster_clinical_composition.png`: Patient composition percentages per cluster.
  - `poster_clinical_cluster_importance.png`: Symmetrical horizontal chart showing $\log_2(\text{Fold Ratio})$ per cluster, colored by cell type, and annotated with fold changes.

---

## Proposed Code Structure

### [gene_coexpression_spectral.py](file:///c:/Users/ajosan/OneDrive%20-%20Eastside%20Preparatory%20School/Old%20data/Independent-Study---Graph-Theory/Week%209/gene_coexpression_spectral.py)

#### 1. Add `PosterFigureExporter` Class (Line ~1078)
This class will manage all exports to `outputs/poster/` at **300 DPI** using modern matplotlib styling:
- `export_embedding(coords, labels, title, filename, color_by='cluster')`
- `export_clinical_composition(df_importance)`
- `export_clinical_cluster_importance(df_importance)`

#### 2. Update `run_pipeline(args)` (Line ~1170 to 1324)
Orchestrate:
1. **Load data & annotate cell types** via `BiologicalCellAnnotator`.
2. **Assign patients & clinical responder status** with biological bias.
3. **Run consensus spectral clustering** on the 30-PC space to identify robust network communities.
4. **Iterate over dimensionality reductions** (PCA, t-SNE, UMAP/Spectral, PaCMAP):
   - Compute the 2D coordinates.
   - Calculate and log the **Silhouette Score** of the spectral clusters in this 2D space.
   - Export separate cluster and cell-type scatter plots for the reduction.
5. **Analyze clinical significance** of each spectral cluster:
   - Compute Odds Ratios and Chi-Square $p$-values.
   - Log the clinical analysis in a beautiful ASCII table.
   - Save clinical composition and cluster importance figures.

---

## Verification Plan

### Automated Verification
- Verify that `sklearn.metrics.silhouette_score` computes correctly.
- Verify that `scipy.stats.chi2_contingency` is executed on valid contingency tables without zero-division or dimension mismatch errors.
- Ensure all 300 DPI figures are saved in the `outputs/poster/` directory.

### Manual Verification
The user can run the script and inspect `outputs/poster/`:
- `poster_embedding_pca_spectral.png` / `poster_embedding_pca_celltypes.png`
- `poster_embedding_tsne_spectral.png` / `poster_embedding_tsne_celltypes.png`
- `poster_embedding_umap_spectral.png` / `poster_embedding_umap_celltypes.png` (or Spectral Embedding equivalents)
- `poster_embedding_pacmap_spectral.png` / `poster_embedding_pacmap_celltypes.png`
- `poster_clinical_composition.png`
- `poster_clinical_cluster_importance.png`
- Check execution logs for embedding diagnostics and the cluster significance table.
