# SCENE

This repository contains the Python package for SCENE (Single-Cell Euclidean
Network Embedding), a graph representation learning method for single-cell
RNA sequencing data. SCENE places cells and genes in a shared Euclidean space
directly from raw UMI counts, enabling analysis of cell–gene relationships
through distances in the learned representation.

<div align="center">
  <video
    src="https://github.com/user-attachments/assets/ce0ae060-26c8-46d7-ac57-f26da177736b"
    width="360"
    controls>
  </video>
</div>

The [IFN-β tutorial](examples/scene_quickstart_ifnb.ipynb) walks through loading
raw PBMC counts, fitting SCENE, and interpreting the results.

For the paper's datasets, analyses, and figures, see the separate
[SCENE-reproducibility](https://github.com/oscarmoeberg/SCENE-reproducibility)
repository.

## Installation

From the repository root:

```bash
python -m pip install .
```

For visualization, install the plotting extra:

```bash
python -m pip install ".[plotting]"
```

To run the tutorial with JupyterLab:

```bash
python -m pip install -e ".[tutorial]"
python -m jupyter lab examples/scene_quickstart_ifnb.ipynb
```

Install SCENE from PyPI:

```bash
pip install scene-ldm
```

For plotting and UMAP visualization:

```bash
pip install "scene-ldm[plotting]"
```


## Quick start

Start with an `AnnData` object containing quality-controlled raw UMI counts
in `adata.layers["counts"]`. If your counts are in `adata.X`, use `layer=None`.

### Fit a model

```python
from scene import SCENE

model = SCENE(latent_dim=16, likelihood="ZIP")
model.fit(adata, layer="counts", epochs=600)
```

Cell coordinates are stored in `adata.obsm["SCENE"]` and gene coordinates
in `adata.varm["SCENE"]`.

To account for donor batches, add `batch_keys=["donor"]` to `.fit()`,
where `"donor"` is a column in `adata.obs`.

### Explore nearby genes

With cell-type annotations in `adata.obs["cell_type"]`, find genes near
the centroid of a cell population:

```python
from scene import nearest_genes

b_cells = adata.obs_names[adata.obs["cell_type"] == "B cells"]
population_genes = nearest_genes(adata, cells=b_cells, n_genes=10)
```

Results are ranked by Euclidean distance in the fitted embedding.
Proximity does not imply differential expression or statistical significance.

### Visualize cells and genes

With the plotting extra installed, create a shared UMAP and label selected
genes:

```python
from scene import joint_umap, plot_joint_embedding

joint_umap(adata, seed=42)
ax = plot_joint_embedding(
    adata,
    color="cell_type",
    genes=[adata.var_names[0]],
)
ax.figure.savefig("scene_joint.png", bbox_inches="tight")
```

UMAP is used for visualization. Gene queries and distances use the original
SCENE coordinates.

## Documentation and citation

See the [IFN-β tutorial](examples/scene_quickstart_ifnb.ipynb) for a complete
workflow and the [API reference](docs/api.md) for model options, neighbor
queries, count prediction, validation, and saving and loading results.

Software citation details are in [CITATION.cff](CITATION.cff).
SCENE is distributed under the [MIT license](LICENSE).
