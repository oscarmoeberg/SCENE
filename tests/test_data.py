import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from scene.data import (
    CellBlockSampler,
    PreparedSceneData,
    SparseCellBlockDataset,
    _collect_csr_rows,
    sparse_extraction_backend_name,
)


def _adata():
    counts = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 2],
                [0, 1, 0],
                [3, 0, 1],
                [0, 2, 0],
            ],
            dtype=np.float32,
        )
    )
    adata = AnnData(X=counts)
    adata.layers["counts"] = counts.copy()
    adata.obs["batch"] = ["a", "b", "a", "b"]
    return adata


def test_preparation_does_not_mutate_batch_dtype_and_records_categories():
    adata = _adata()
    original_dtype = adata.obs["batch"].dtype
    prepared = PreparedSceneData.from_adata(
        adata,
        batch_keys="batch",
        validate=True,
        val_frac=0.25,
        validation_seed=2,
    )

    assert adata.obs["batch"].dtype == original_dtype
    assert prepared.batch_categories["batch"] == ["a", "b"]
    assert prepared.batch_size == {"batch": 2}
    assert prepared.holdout is not None

    train_pairs = set(zip(*prepared.train_matrix.nonzero()))
    heldout_pairs = set(zip(
        prepared.holdout["pos_rows"],
        prepared.holdout["pos_cols"],
    ))
    assert not train_pairs.intersection(heldout_pairs)

    full_pairs = set(zip(*sparse.csr_matrix(adata.layers["counts"]).nonzero()))
    negative_pairs = set(zip(
        prepared.holdout["neg_rows"],
        prepared.holdout["neg_cols"],
    ))
    assert not full_pairs.intersection(negative_pairs)


def test_missing_validation_exposes_block_local_missing_indices():
    prepared = PreparedSceneData.from_adata(
        _adata(),
        validate=True,
        val_frac=0.25,
        validation_seed=4,
        validation_mode="masked",
    )
    dataset = SparseCellBlockDataset(prepared)
    block = dataset[np.array([0, 2], dtype=np.int64)]

    assert prepared.missing_indices is not None
    assert prepared.observed_total < prepared.n_cells * prepared.n_genes
    assert block.missing_indices is not None
    assert block.missing_indices.shape[0] == 2
    assert block.block_weight == 2.0


def test_sampler_is_deterministic_and_covers_each_cell_once():
    cells = np.arange(7, dtype=np.int64)
    first = list(CellBlockSampler(cells, batch_size=3, seed=10))
    second = list(CellBlockSampler(cells, batch_size=3, seed=10))

    assert [block.tolist() for block in first] == [block.tolist() for block in second]
    assert sorted(np.concatenate(first).tolist()) == cells.tolist()


def test_sparse_extraction_matches_numpy_fallback():
    prepared = PreparedSceneData.from_adata(_adata())
    rows = np.array([3, 0, 2], dtype=np.int64)
    fallback_indices, fallback_values = _collect_csr_rows(
        prepared.train_matrix,
        rows,
        use_numba=False,
    )
    active_indices, active_values = _collect_csr_rows(
        prepared.train_matrix,
        rows,
        use_numba=True,
    )

    assert sparse_extraction_backend_name() in {"numpy", "numba"}
    assert np.array_equal(active_indices, fallback_indices)
    assert np.array_equal(active_values, fallback_values)


@pytest.mark.parametrize("value", [-1.0, -0.5])
def test_preparation_rejects_negative_counts(value):
    adata = _adata()
    adata.layers["counts"].data[0] = value
    with pytest.raises(ValueError, match="negative values"):
        PreparedSceneData.from_adata(adata)


def test_preparation_rejects_missing_named_layer():
    with pytest.raises(ValueError, match="misspelled_counts"):
        PreparedSceneData.from_adata(_adata(), layer="misspelled_counts")


def test_preparation_requires_counts_layer_by_default():
    adata = _adata()
    del adata.layers["counts"]
    with pytest.raises(ValueError, match="layer=None"):
        PreparedSceneData.from_adata(adata)


def test_preparation_uses_x_when_explicitly_requested():
    adata = _adata()
    adata.layers["counts"] = adata.X * 2
    prepared = PreparedSceneData.from_adata(adata, layer=None)
    np.testing.assert_array_equal(prepared.train_matrix.toarray(), adata.X.toarray())
