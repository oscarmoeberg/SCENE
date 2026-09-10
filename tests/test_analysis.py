import numpy as np
import pytest
from anndata import AnnData

from scene import nearest_genes


def _adata():
    adata = AnnData(np.zeros((2, 4)))
    adata.obs_names = ["cell_a", "cell_b"]
    adata.var_names = ["a", "b", "c", "d"]
    adata.obsm["joint"] = np.array([[0., 0., 0.], [4., 0., 0.]])
    adata.varm["joint"] = np.array([[0., 0., 1.], [1., 0., 0.], [2., 0., 0.], [4., 0., 0.]])
    return adata


def test_cell_centroid_and_gene_rankings():
    adata = _adata()
    ranked = nearest_genes(adata, cells="cell_a", key="joint")
    assert ranked.index.name == "gene"
    assert list(ranked.columns) == ["distance"]
    assert list(ranked.index) == ["a", "b", "c", "d"]
    np.testing.assert_allclose(ranked.distance, [1, 1, 2, 4])
    centroid = nearest_genes(adata, cells=adata.obs_names, key="joint")
    assert list(centroid.index) == ["c", "b", "d", "a"]
    np.testing.assert_allclose(centroid.distance, [0, 1, 2, np.sqrt(5)])
    neighbours = nearest_genes(adata, gene="c", n_genes=2, key="joint")
    assert list(neighbours.index) == ["b", "d"]
    np.testing.assert_allclose(neighbours.distance, [1, 2])


@pytest.mark.parametrize("kwargs", [
    {}, {"cells": "cell_a", "gene": "a"}, {"cells": []},
    {"cells": "cell_a", "n_genes": 0}, {"cells": "cell_a", "n_genes": -1},
    {"cells": "cell_a", "n_genes": 1.5}, {"cells": "cell_a", "n_genes": True},
    {"gene": ["a", "b"]},
])
def test_invalid_queries(kwargs):
    with pytest.raises(ValueError):
        nearest_genes(_adata(), key="joint", **kwargs)


@pytest.mark.parametrize("kwargs", [{"cells": "missing"}, {"gene": "missing"}])
def test_unknown_names(kwargs):
    with pytest.raises(KeyError):
        nearest_genes(_adata(), key="joint", **kwargs)


def test_missing_incompatible_and_ambiguous_embeddings():
    adata = _adata()
    with pytest.raises(KeyError):
        nearest_genes(adata, cells="cell_a")
    del adata.obsm["joint"]
    with pytest.raises(KeyError):
        nearest_genes(adata, cells="cell_a", key="joint")
    adata.obsm["joint"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="matching dimensions"):
        nearest_genes(adata, cells="cell_a", key="joint")
    adata = _adata()
    adata.obs_names = ["cell_a", "cell_a"]
    with pytest.raises(ValueError, match="unique"):
        nearest_genes(adata, cells="cell_a", key="joint")
    adata.var_names = ["a", "a", "c", "d"]
    with pytest.raises(ValueError, match="unique"):
        nearest_genes(adata, gene="a", key="joint")


def test_gene_query_with_no_other_genes():
    adata = _adata()[:, ["a"]].copy()
    assert nearest_genes(adata, gene="a", key="joint").empty


def test_default_key_and_nonfinite_embeddings():
    adata = _adata()
    adata.obsm["SCENE"] = adata.obsm["joint"].copy()
    adata.varm["SCENE"] = adata.varm["joint"].copy()
    assert len(nearest_genes(adata, cells=["cell_a"], n_genes=np.int64(1))) == 1
    adata.obsm["SCENE"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        nearest_genes(adata, cells="cell_a")
    adata.varm["SCENE"][0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        nearest_genes(adata, gene="a")


from scene import nearest_cells, cell_gene_distances


def test_nearest_cells_reciprocal_and_exclusions():
    adata = _adata()
    cells = nearest_cells(adata, gene="c", key="joint")
    assert cells.index.name == "cell"
    assert cells.index.tolist() == ["cell_a", "cell_b"]
    np.testing.assert_array_equal(cells.distance, [2, 2])
    assert nearest_cells(adata, cells="cell_a", key="joint").index.tolist() == ["cell_b"]
    assert nearest_cells(adata, cells=adata.obs_names, key="joint").empty
    with pytest.raises(ValueError, match="positive"):
        nearest_cells(adata, gene="a", n_cells=0, key="joint")


def test_cell_gene_paired_distances_and_scalar_broadcast():
    adata = _adata()
    pairs = cell_gene_distances(adata, cells=["cell_b", "cell_a"], genes="a", key="joint")
    assert pairs.cell.tolist() == ["cell_b", "cell_a"]
    np.testing.assert_allclose(pairs.distance, [np.sqrt(17), 1])
    assert cell_gene_distances(adata, cells=[], genes=[], key="joint").empty
    with pytest.raises(ValueError, match="equal lengths"):
        cell_gene_distances(adata, cells=["cell_a"], genes=["a", "b"], key="joint")
    with pytest.raises(KeyError):
        cell_gene_distances(adata, cells="unknown", genes="a", key="joint")
