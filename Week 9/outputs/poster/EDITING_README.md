# Editing the Spring 2026 IS Poster

## Final PDF
- `spring_2026_is_poster.pdf`

## Preview Image
- `spring_2026_is_poster_preview.png`

## Best Editable Source Right Now
- `build_spring_2026_poster_pdf.py`

This script generates both the PDF and PNG preview. To change text, colors, panel sizes, or image placements, edit the clearly named sections in the script and rerun:

```powershell
python build_spring_2026_poster_pdf.py
```

## LaTeX Source
- `spring_2026_is_poster.tex`

I also created a LaTeX poster source with the same content and image references. Your machine does not currently have `pdflatex` on PATH, and Docker Desktop was not running, so I could not compile the LaTeX version here. Once TeX or Docker is available, compile from this folder with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error spring_2026_is_poster.tex
```

## Image Slots to Replace
- Add Image 1: manual graph sketch or star graph Laplacian from Weeks 1-2.
- Add Image 2: NetworkX graph or centrality heat map from Weeks 3-4.
- Add Image 3: targeted attack survival curve from robustness experiments.
- Add Image 4: eigenvalue or spectral gap diagnostic.

## Images Already Used
- `../plots/dots_umap_clusters_and_response.png`
- `../plots/dots_tsne_clusters_and_response.png`
- `../plots/dots_response_enrichment_volcano.png`
- `../plots/dots_patient_composition_by_cluster.png`
- `poster_clinical_cluster_importance.png`
- `poster_clinical_composition.png`
