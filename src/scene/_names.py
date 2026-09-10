"""Shared name resolution for geometry and fitted predictions."""

import numpy as np
import pandas as pd


def name_index(names, kind):
    index = pd.Index(names)
    if not index.is_unique:
        raise ValueError(f"{kind} names must be unique")
    if not all(isinstance(name, str) for name in index):
        raise ValueError(f"{kind} names must be strings")
    return index


def resolve_names(names, index, kind):
    if isinstance(names, str):
        names = [names]
    else:
        try:
            names = list(names)
        except TypeError as error:
            raise ValueError(f"{kind} must be a name or sequence of names") from error
    if not all(isinstance(name, str) for name in names):
        raise ValueError(f"{kind} must contain only string names")
    positions = index.get_indexer(names)
    if (positions < 0).any():
        unknown = [name for name, position in zip(names, positions) if position < 0]
        raise KeyError(f"Unknown {kind}: {unknown[:5]}")
    return np.asarray(names, dtype=object), positions


def resolve_pairs(cells, genes, cell_index, gene_index):
    cell_names, rows = resolve_names(cells, cell_index, "cells")
    gene_names, cols = resolve_names(genes, gene_index, "genes")
    if isinstance(cells, str):
        cell_names = np.repeat(cell_names, len(cols))
        rows = np.repeat(rows, len(cols))
    elif isinstance(genes, str):
        gene_names = np.repeat(gene_names, len(rows))
        cols = np.repeat(cols, len(rows))
    if len(rows) != len(cols):
        raise ValueError("cells and genes must have equal lengths, or one must be a scalar name")
    return cell_names, gene_names, rows, cols


def embedding(adata, key, kind):
    names = name_index(adata.obs_names if kind == "cell" else adata.var_names, kind)
    slot = adata.obsm if kind == "cell" else adata.varm
    if key not in slot:
        raise KeyError(f"Missing {kind} embedding {key!r}")
    values = np.asarray(slot[key])
    if (values.ndim != 2 or values.shape[0] != len(names) or values.shape[1] == 0
            or not np.isfinite(values).all()):
        raise ValueError(f"{kind} embeddings must be finite two-dimensional arrays")
    return names, values


def joint_embeddings(adata, key):
    cells, cell_values = embedding(adata, key, "cell")
    genes, gene_values = embedding(adata, key, "gene")
    if cell_values.shape[1] != gene_values.shape[1]:
        raise ValueError("Cell and gene embeddings must have matching dimensions")
    return cells, genes, cell_values, gene_values
