"""Optional joint-embedding projection and plotting."""

import numpy as np

from ._names import joint_embeddings, resolve_names
from ._validation import finite_real, positive_int, seed_value, storage_key


def joint_umap(adata, *, key="SCENE", n_neighbors=15, min_dist=0.3, seed=42):
    """Fit one joint UMAP to cells and genes, then store their coordinates separately.

    adata contains obsm[key]/varm[key], with key defaulting to "SCENE".
    n_neighbors (default 15, integer >=2) and min_dist (default 0.3, in [0,1])
    control UMAP. seed (default 42) fixes its random state. Euclidean metric,
    two dimensions, random initialization, and one worker are used. Requires
    the plotting extra. Returns an (n_cells+n_genes, 2) array, cells first.

    Writes obsm[key+"_umap"], varm[key+"_umap"], and uns[key]["projection"].
    Neighbor count is capped at total points minus one and recorded. At least
    three joint points are required. Original fitted coordinates are unchanged;
    UMAP is a visualization, not the geometry used for scientific distances.
    """
    storage_key(key)
    _, _, cells, genes = joint_embeddings(adata, key)
    n_neighbors = positive_int(n_neighbors, "n_neighbors")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    min_dist = finite_real(min_dist, "min_dist")
    if min_dist > 1:
        raise ValueError("min_dist must not exceed 1")
    seed = seed_value(seed, "seed")
    joint = np.vstack([cells, genes])
    if len(joint) < 3:
        raise ValueError("joint_umap requires at least three combined cells and genes")
    try:
        from umap import UMAP
    except ImportError as error:
        raise ImportError('joint_umap requires pip install "scene-ldm[plotting]"') from error
    effective_neighbors = min(n_neighbors, len(joint) - 1)
    coordinates = UMAP(n_components=2, metric="euclidean", n_neighbors=effective_neighbors,
                       min_dist=min_dist, random_state=seed, init="random", n_jobs=1).fit_transform(joint)
    projection_key = f"{key}_umap"
    adata.obsm[projection_key] = coordinates[:len(cells)].copy()
    adata.varm[projection_key] = coordinates[len(cells):].copy()
    adata.uns.setdefault(key, {})["projection"] = {
        "method": "umap", "key": projection_key, "n_components": 2, "metric": "euclidean",
        "n_neighbors": n_neighbors, "effective_n_neighbors": effective_neighbors,
        "min_dist": min_dist, "seed": seed, "init": "random", "n_jobs": 1,
    }
    return coordinates


def plot_joint_embedding(adata, *, key="SCENE", color=None, genes=None, ax=None):
    """Plot stored joint UMAP coordinates and return Matplotlib Axes.

    adata must have coordinates from joint_umap; key (default "SCENE") selects
    the source fit. color is an obs column name or None (one cell color).
    Numeric columns use a continuous color scale; other columns use categories
    with a legend outside the axes. Nearby gene labels are separated vertically.
    genes is one var_name or a sequence to mark and label; None labels no genes.
    ax optionally supplies existing Axes. Requires the plotting extra. Does
    not recompute coordinates, modify adata, call show(), or save a figure.
    """
    if key not in adata.uns or "projection" not in adata.uns[key]:
        raise ValueError("No joint projection found; call joint_umap first")
    projection_key = adata.uns[key]["projection"]["key"]
    _, gene_names, cells, gene_values = joint_embeddings(adata, projection_key)
    if cells.shape[1] != 2:
        raise ValueError("Joint plotting requires a two-dimensional projection")
    if color is not None and color not in adata.obs:
        raise KeyError(f"Unknown observation column: {color}")
    selected, positions = resolve_names([] if genes is None else genes, gene_names, "genes")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError('plot_joint_embedding requires pip install "scene-ldm[plotting]"') from error
    from pandas.api.types import is_numeric_dtype
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    if color is None:
        ax.scatter(*cells.T, s=8, alpha=0.5, rasterized=True)
    elif is_numeric_dtype(adata.obs[color]) and adata.obs[color].dtype.name != "category":
        points = ax.scatter(*cells.T, c=adata.obs[color].to_numpy(dtype=float, na_value=np.nan),
                            s=8, alpha=0.6, rasterized=True)
        ax.figure.colorbar(points, ax=ax, label=color)
    else:
        labels = adata.obs[color].astype("string").fillna("<missing>")
        for label in labels.unique():
            mask = (labels == label).to_numpy(dtype=bool)
            ax.scatter(*cells[mask].T, s=8, alpha=0.5, label=str(label), rasterized=True)
        ax.legend(title=color, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    if len(positions):
        xy = gene_values[positions]
        ax.scatter(*xy.T, marker="x", color="black", s=35)
        annotations = [ax.annotate(
            name, point, xytext=(5, 5), textcoords="offset points", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5},
            arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.5},
        ) for name, point in zip(selected, xy)]
    ax.set(xlabel="Joint UMAP 1", ylabel="Joint UMAP 2", title=color or "SCENE joint embedding")
    ax.set_aspect("equal", adjustable="box")
    if len(positions):
        # Stack colliding labels in display space, independently of gene names
        # and UMAP scale. Leaders retain the association with each coordinate.
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        placed = []
        for annotation in annotations:
            for step in range(len(annotations) + 1):
                annotation.set_position((5, 5 + 13 * step))
                annotation.update_bbox_position_size(renderer)
                bounds = annotation.get_bbox_patch().get_window_extent(renderer).expanded(1.05, 1.1)
                if not any(bounds.overlaps(other) for other in placed):
                    break
            placed.append(bounds)
    return ax
