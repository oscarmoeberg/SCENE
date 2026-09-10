# API and storage reference

Construct a model with `SCENE(latent_dim=16, likelihood="ZIP")`, then fit it
with `model.fit(adata, ...)`. The constructor defines the statistical model;
`.fit()` selects the data, initialization, and optimization settings. Queries use names from `obs_names` and `var_names`.
SCENE uses raw counts without normalization, log transformation, scaling,
filtering, gene selection, or PCA. All supplied cells and genes are retained.
The parameter reference below lists defaults and accepted values.

## AnnData storage

For a fit named `key` (default `"SCENE"`):

| Location | Contents |
|---|---|
| `obsm[key]` | Cell coordinates, `(n_cells, latent_dim)` |
| `varm[key]` | Gene coordinates, `(n_genes, latent_dim)` |
| `uns[key]["format_version"]` | Integer schema version, currently 2 |
| `uns[key]["package_version"]` | Version that produced the fitted model |
| `uns[key]["cell_names"]`, `["gene_names"]` | Original fitted identifiers in decoder order |
| `uns[key]["config"]` | JSON string with model, fit, and effective execution settings |
| `uns[key]["batch_size"]` | Category counts keyed by fitted batch column |
| `uns[key]["batch_categories"]` | JSON string mapping columns to ordered category values |
| `uns[key]["batch_codes"]` | One integer-code array per batch column, in fitted cell order |
| `uns[key]["training_history"]["train"]` | Column arrays: epoch, loss, optimizer_updates |
| `uns[key]["training_history"]["validation"]` | Column arrays: epoch, AUC, PR AUC, max F1, PoissonDeviance |
| `uns[key]["parameters"]` | alpha, cell_random_effects, gene_random_effects, batch_effects |
| `uns[key]["parameters"]["batch_effects"][column]` | type plus gamma (full), U/V (lowrank), or no arrays (none) |

JSON strings preserve optional nulls and scalar category types across supported
AnnData versions. Decode these two fields with `json.loads`. For example:

```python
import json
import pandas as pd
config = json.loads(adata.uns["SCENE"]["config"])
validation = pd.DataFrame(adata.uns["SCENE"]["training_history"]["validation"])
```

`config` has `model` (the five constructor settings), `fit` (cell and gene
initialization, resolved split seed, input layer, ordered batch keys, optimizer and reporting settings, key, device), and
`effective` (clamped cell batch size, effective loader settings, extraction
backend). Category values must be JSON scalar strings, numbers, or booleans;
missing values are rejected. Batch keys and output keys cannot contain `/`.

Write-back replaces the selected fit's metadata and coordinates. Unrelated fits
and input counts/annotations are retained. Output arrays do not share mutable
memory with model tensors. AnnData subsetting slices coordinates normally; the
metadata names/codes still describe the original fit. Geometry queries use the
current AnnData axes, not the original metadata identifiers.

`joint_umap` adds `obsm[key+"_umap"]`, `varm[key+"_umap"]`, and
`uns[key]["projection"]` (method, key, dimensions, metric, requested/effective
neighbors, min_dist, seed, initialization, worker count). Recompute the projection
after refitting. Plotting requires projection metadata and never runs UMAP itself.

## Model checkpoints

`save` writes `metadata.json` and `weights.pt`. Metadata uses the same format
version and field definitions as H5AD, but retains native JSON objects for config
and batch categories. Undefined metrics are JSON null and become NaN on load.
Weights contain only the exact decoder state dictionary. Metadata records a
SHA-256 checksum; mismatched or partial checkpoints are rejected. A load uses
`torch.load(..., weights_only=True)`, validates identifiers, batch codes, tensor
shapes/dtypes and history, and reconstructs without random or Laplacian initialization.

The original producing package version and run settings are preserved on load;
`device` selects inference placement without rewriting training provenance.
Counts and optimizer state are not included. Predictions only address fitted
identifiers. Calling fit on a loaded object performs a fresh fit.

H5AD stores portable geometry and diagnostics. To restore a model for
prediction, pass a model directory created by `SCENE.save` to `SCENE.load`.

## Counts and numerical interpretation

Counts must be finite and non-negative. Fractional entries produce a warning
and are used without rounding. The count likelihood is defined on integers. The predictor is negative Euclidean distance plus
enabled cell/gene random effects and batch effects; lambda is its exponential.

Poisson expectation is lambda. ZIP uses pi = sigmoid(alpha * predictor),
P(0) = 1 - pi, and a zero-truncated Poisson positive-count component, giving
expectation pi * lambda / (1 - exp(-lambda)). alpha is positive. These formulas
define the ZIP count model.

Validation PR AUC uses trapezoidal integration, max F1 is the maximum across
thresholds, and PoissonDeviance uses positive held-out counts with their
conditional positive-count mean. If no positives or sampled zeros are available,
metrics are undefined. Neither validation nor geometric ranking establishes
statistical or biological significance.

## Public reference

### SCENE

```python
SCENE(*, latent_dim=16, likelihood='ZIP', batch_effect_type='lowrank', batch_rank=2, use_random_effects=True)
```

```text
Joint cell–gene model fitted directly to raw UMI counts.

Parameters
----------
latent_dim : int, default 16
    Number of shared Euclidean dimensions.
likelihood : {"Poisson", "ZIP"}, default "ZIP"
    Count distribution; names are case-insensitive. ZIP models the
    probability of a nonzero count and uses a zero-truncated Poisson
    distribution for positive counts.
batch_effect_type : str or sequence of str, default "lowrank"
    "none" omits batch effects; "full" fits an offset per gene and batch
    category; "lowrank" factorizes these offsets using batch_rank dimensions.
    Supply one type for all batch_keys or one per key, in the same order.
batch_rank : int, default 2
    Factorization rank for lowrank batch effects.
use_random_effects : bool, default True
    Fit cell and gene random effects.

Attributes
----------
training_history : dict
    After fitting: train and validation column dictionaries, suitable for
    pandas.DataFrame. Validation columns include "PR AUC" and "max F1".
config : dict
    Fitted model, data, optimization, and effective loader settings.
batch_size : dict
    Inferred category counts by batch key; unrelated to cell_batch_size.
cell_names, gene_names : pandas.Index
    Fitted identifiers in decoder order.
batch_categories : dict
    Category labels in fitted encoding order.

Notes
-----
Each fit starts fresh and returns self. SCENE fits all supplied cells and
genes without normalization, log transformation, scaling, gene selection,
or PCA. Quality control is performed before supplying data to SCENE.
The model does not retain the input AnnData.
```

### SCENE.fit

```python
SCENE.fit(adata, *, layer='counts', batch_keys=None, cell_init='random', gene_init='random', epochs=600, lr=0.005, seed=42, split_seed=None, key='SCENE', validate=False, val_frac=0.1, val_interval=10, print_every=50, validation_mode='flipped', neg_ratio=1.0, cell_batch_size=1024, loader_workers=None, loader_prefetch_batches=2, loader_pin_memory=None, validation_edge_batch_size=100000, write_back=True, weight_decay=0.001, device='cpu')
```

```text
Fit from scratch, optionally write results into adata, and return self.

Parameters
----------
adata : anndata.AnnData
    Raw UMI counts with unique obs_names and var_names. All supplied
    cells and genes are retained, including zero-count rows and columns.
layer : str or None, default "counts"
    Count layer; None explicitly selects X.
batch_keys : str or sequence of str or None, default None
    Observation columns for batch effects, in model order.
cell_init, gene_init : {"random", "laplacian"}, default "random"
    Independent starting coordinates for this fit. Laplacian uses
    the symmetric normalized graph Laplacian of the training counts;
    it does not transform the counts used in the likelihood. Requires
    latent_dim < min(n_cells, n_genes) - 1 and at least one nonzero count.
epochs : int, default 600
    Number of complete training passes.
lr : float, default 0.005
    Positive AdamW learning rate.
seed : int, default 42
    Initialization and sampling seed in [0, 2**32).
split_seed : int or None, default None
    Validation split seed; None uses seed.
key : str, default "SCENE"
    Result key for obsm, varm, and uns; no '/' characters.
validate : bool, default False
    Evaluate held-out positive counts against sampled zero entries.
val_frac : float, default 0.10
    Fraction of positive entries held out, in [0, 1).
val_interval : int, default 10
    Validation interval; also evaluate first and last epochs.
print_every : int or None, default 50
    Reporting interval, including first/last epochs; None is quiet.
validation_mode : {"flipped", "masked"}, default "flipped"
    In "flipped" mode, held-out positive counts are treated as zeros
    during training. In "masked" mode, those entries are excluded from
    the loss. Sampled zero entries remain in training. Input counts
    in adata are unchanged in both modes.
neg_ratio : float, default 1.0
    Non-negative ratio of sampled zeros to held-out positives.
cell_batch_size : int or None, default 1024
    Cells per optimization step, including all genes for each selected cell.
    None uses one full-matrix step per epoch, as in the manuscript.
loader_workers : int or None, default None
    Worker count; None uses zero workers.
loader_prefetch_batches : int, default 2
    Positive prefetch count for multiprocessing loaders.
loader_pin_memory : bool or None, default None
    None enables pinning for CUDA cell blocks; otherwise request it
    explicitly. Effective pinning is disabled outside CUDA blocks.
validation_edge_batch_size : int, default 100000
    Maximum paired entries per validation chunk.
write_back : bool, default True
    Write embeddings to obsm[key]/varm[key] and metadata to uns[key].
    False leaves adata unchanged; the fitted model remains usable.
weight_decay : float, default 0.001
    Non-negative AdamW decay, excluding random effects and ZIP alpha.
device : str or torch.device, default "cpu"
    Training device.

Returns
-------
SCENE
    This fitted object. Existing fitted state is replaced on success.
```

### SCENE.expected_counts

```python
SCENE.expected_counts(*, cells, genes, use_random_effects=True, use_batch_effects=True)
```

```text
Predict expectations for named fitted cell–gene pairs.

Supply cell and gene names as strings or equal-length sequences.
Two sequences are paired by position; a single string is paired with
every name in the other sequence. Empty pairs are allowed.
use_random_effects and use_batch_effects (both True) include fitted
terms; batch categories are resolved automatically. Returns a DataFrame
with cell, gene, expected_count columns in request order. Chunked at
100,000 pairs, with no gradients or implicit full-matrix allocation.
Poisson expectation is lambda; ZIP is pi*lambda/(1-exp(-lambda)).
Unknown names are rejected; this does not embed new cells or genes.
```

### SCENE.linear_predictor_pairs

```python
SCENE.linear_predictor_pairs(*, cells, genes, use_random_effects=True, use_batch_effects=True)
```

```text
Return the fitted model score for each requested cell–gene pair.

The score is negative Euclidean distance across all latent dimensions,
plus any enabled cell, gene and batch effects. It is the linear predictor,
before conversion to expected counts or probabilities.

Arguments and defaults match expected_counts. Returns a table with
cell, gene and linear_predictor columns.
```

### SCENE.save

```python
SCENE.save(path, *, overwrite=False)
```

```text
Save metadata.json and weights.pt to path; return None.

path is a directory path (str or pathlib.Path). Existing checkpoints
require overwrite=True; unrelated directories are never overwritten.
Saves exact fitted tensors, identifiers, batch encoding, training history, and
configuration. Counts and optimizer state are not saved.
```

### SCENE.load

```python
SCENE.load(path, *, device='cpu')
```

```text
Load an inference-ready SCENE from a checkpoint directory.

path is a string or pathlib.Path pointing to a directory created by
save. device (default CPU) selects tensor placement; tensor dtype is
preserved. Restores fitted parameters for prediction. Counts and
optimizer state are not restored; calling fit starts a new fit.
Uses weights-only loading and rejects incompatible checkpoint state.
```

### nearest_genes

```python
nearest_genes(adata, *, cells=None, gene=None, n_genes=20, key='SCENE')
```

```text
Return genes nearest to cells' centroid or a gene in all fitted dimensions.

Parameters
----------
adata : anndata.AnnData
    Holds cell coordinates in obsm[key] and gene coordinates in varm[key].
cells : str or sequence of str, optional
    One obs_name or a non-empty sequence, averaged to a centroid.
gene : str, optional
    One var_name. Supply exactly one of cells or gene.
n_genes : int, default 20
    Maximum number of results. A gene query excludes the queried gene.
key : str, default "SCENE"
    Saved embedding key.

Returns
-------
pandas.DataFrame
    Gene-name index and a distance column, ascending with stable ties.
    These are Euclidean distances, not differential-expression statistics.
    Does not modify adata; works after H5AD reload.
```

### nearest_cells

```python
nearest_cells(adata, *, cells=None, gene=None, n_cells=20, key='SCENE')
```

```text
Return nearest cells, excluding queried cells from the candidates.

Parameters match nearest_genes, except n_cells (default 20) limits cell
results. Supply cells (one obs_name or a non-empty sequence whose centroid
is used) or gene (one var_name), exclusively. key selects obsm/varm arrays.
Returns a DataFrame indexed by cell with an ascending distance column;
ties preserve observation order. Uses all fitted dimensions without
modifying adata. No statistical significance is implied.
```

### cell_gene_distances

```python
cell_gene_distances(adata, *, cells, genes, key='SCENE')
```

```text
Return Euclidean distances for named pairs in all fitted dimensions.

adata holds obsm[key]/varm[key] (key defaults to "SCENE"). cells and genes
are obs_names and var_names: equal-length sequences or a scalar string
broadcast against the other sequence. Empty paired requests are allowed.
Returns a DataFrame with cell, gene, distance columns in request order.
Does not modify adata or allocate a full cell–gene distance matrix.
```

### laplacian_init

```python
laplacian_init(counts: scipy.sparse._csr.csr_matrix, latent_dim: int)
```

```text
Initialize cell and gene coordinates from a raw count matrix using
the symmetric normalized bipartite graph Laplacian.

Returns (cell_coordinates, gene_coordinates), each with latent_dim
columns and rows in input order. Counts must be finite and non-negative;
latent_dim must be positive and below min(counts.shape) - 1.

Uses eigenvectors corresponding to the smallest positive eigenvalues,
excluding zero modes from all connected components.
```

### joint_umap

```python
joint_umap(adata, *, key='SCENE', n_neighbors=15, min_dist=0.3, seed=42)
```

```text
Project cell and gene embeddings together into a shared 2D UMAP.

Reads embeddings from adata.obsm[key] and adata.varm[key]
(default key: "SCENE"). Stores the projected coordinates in
obsm[key + "_umap"] and varm[key + "_umap"], and returns a combined
array with cells first, followed by genes.

n_neighbors and min_dist control the projection; seed controls
reproducibility. Requires the plotting extra.

Use this projection for visualization only. Any distance based analysis should come
from the original embeddings.
```

### plot_joint_embedding

```python
plot_joint_embedding(adata, *, key='SCENE', color=None, genes=None, ax=None)
```

```text
Plot the shared cell–gene UMAP created by joint_umap.

Use color to color cells by an adata.obs column and genes to mark
and label selected gene names. Numeric colors use a continuous scale;
categorical colors use a legend.

key selects the embedding (default: "SCENE"). Optionally provide ax
to plot on existing Matplotlib axes. Returns the axes.

Requires the plotting extra. Display or save the figure separately.
```
