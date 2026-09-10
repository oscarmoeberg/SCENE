"""Geometric interpretation of saved joint cell–gene embeddings."""

import numpy as np
import pandas as pd

from ._names import embedding, joint_embeddings, resolve_names, resolve_pairs
from ._validation import positive_int


def _nearest(adata, cells, gene, n, key, target):
    if (cells is None) == (gene is None):
        raise ValueError("Supply exactly one of cells or gene")
    names, values = embedding(adata, key, target)
    candidates = np.arange(len(names))
    if gene is not None:
        if not isinstance(gene, str):
            raise ValueError("gene must be one gene name")
        gene_names, gene_values = embedding(adata, key, "gene")
        _, positions = resolve_names(gene, gene_names, "gene")
        query = gene_values[positions[0]]
        if target == "gene":
            candidates = candidates[candidates != positions[0]]
    else:
        cell_names, cell_values = embedding(adata, key, "cell")
        _, positions = resolve_names(cells, cell_names, "cells")
        if not len(positions):
            raise ValueError("cells must not be empty")
        query = cell_values[positions].mean(axis=0)
        if target == "cell":
            candidates = candidates[~np.isin(candidates, positions)]
    if query.shape[0] != values.shape[1]:
        raise ValueError("Cell and gene embeddings must have matching dimensions")
    distances = np.linalg.norm(values[candidates] - query, axis=1)
    order = np.argsort(distances, kind="stable")[:n]
    return pd.DataFrame({"distance": distances[order]},
                        index=pd.Index(names[candidates[order]], name=target))


def nearest_genes(adata, *, cells=None, gene=None, n_genes=20, key="SCENE"):
    """Return genes nearest to cells' centroid or a gene in all fitted dimensions.

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
    """
    return _nearest(adata, cells, gene, positive_int(n_genes, "n_genes"), key, "gene")


def nearest_cells(adata, *, cells=None, gene=None, n_cells=20, key="SCENE"):
    """Return nearest cells, excluding queried cells from the candidates.

    Parameters match nearest_genes, except n_cells (default 20) limits cell
    results. Supply cells (one obs_name or a non-empty sequence whose centroid
    is used) or gene (one var_name), exclusively. key selects obsm/varm arrays.
    Returns a DataFrame indexed by cell with an ascending distance column;
    ties preserve observation order. Uses all fitted dimensions without
    modifying adata. No statistical significance is implied.
    """
    return _nearest(adata, cells, gene, positive_int(n_cells, "n_cells"), key, "cell")


def cell_gene_distances(adata, *, cells, genes, key="SCENE"):
    """Return Euclidean distances for named pairs in all fitted dimensions.

    adata holds obsm[key]/varm[key] (key defaults to "SCENE"). cells and genes
    are obs_names and var_names: equal-length sequences or a scalar string
    broadcast against the other sequence. Empty paired requests are allowed.
    Returns a DataFrame with cell, gene, distance columns in request order.
    Does not modify adata or allocate a full cell–gene distance matrix.
    """
    cell_index, gene_index, cell_values, gene_values = joint_embeddings(adata, key)
    cell_names, gene_names, rows, cols = resolve_pairs(cells, genes, cell_index, gene_index)
    distances = np.linalg.norm(cell_values[rows] - gene_values[cols], axis=1)
    return pd.DataFrame({"cell": cell_names, "gene": gene_names, "distance": distances})
