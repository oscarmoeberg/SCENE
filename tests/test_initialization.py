import numpy as np
from scipy import sparse

from scene import laplacian_init


def test_laplacian_init_returns_finite_cell_and_gene_embeddings():
    counts = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 2, 0, 1],
                [0, 3, 0, 1, 0],
                [2, 0, 1, 0, 1],
                [0, 1, 0, 2, 0],
            ],
            dtype=np.float32,
        )
    )

    z_cells, z_genes = laplacian_init(counts, latent_dim=2)

    assert z_cells.shape == (4, 2)
    assert z_genes.shape == (5, 2)
    assert np.isfinite(z_cells).all()
    assert np.isfinite(z_genes).all()


import pytest
import torch

from scene._decoder import _SCENEDecoder


@pytest.mark.parametrize("cell_init", ["random", "laplacian"])
@pytest.mark.parametrize("gene_init", ["random", "laplacian"])
def test_initializations_are_independent(cell_init, gene_init, monkeypatch):
    calls = []
    counts = sparse.csr_matrix(np.ones((5, 4), dtype=np.float32))
    def coordinates(matrix, latent_dim):
        calls.append(matrix)
        return np.full((5, 2), 7, dtype=np.float32), np.full((4, 2), 9, dtype=np.float32)
    monkeypatch.setattr("scene._decoder.laplacian_init", coordinates)
    model = _SCENEDecoder(5, 4, 2, counts=counts, cell_init=cell_init, gene_init=gene_init)
    assert len(calls) == int("laplacian" in (cell_init, gene_init))
    assert bool(torch.all(model.Z_cells == 7)) == (cell_init == "laplacian")
    assert bool(torch.all(model.Z_genes == 9)) == (gene_init == "laplacian")


@pytest.mark.parametrize("latent_dim", [0, -1, 2, 1.5, True])
def test_laplacian_dimension_validation(latent_dim):
    with pytest.raises(ValueError, match="latent_dim"):
        laplacian_init(sparse.csr_matrix(np.ones((4, 3))), latent_dim=latent_dim)


def test_laplacian_invalid_counts():
    with pytest.raises(ValueError, match="counts"):
        laplacian_init(None, latent_dim=1)
    with pytest.raises(ValueError, match="non-negative"):
        laplacian_init(sparse.csr_matrix(-np.ones((4, 4))), latent_dim=1)
    with pytest.raises(ValueError, match="dimensions"):
        _SCENEDecoder(4, 4, 1, counts=sparse.csr_matrix(np.ones((5, 4))), gene_init="laplacian")


def test_gene_only_laplacian_fits_through_public_api():
    from anndata import AnnData
    from scene import SCENE
    adata = AnnData(sparse.csr_matrix(np.arange(25).reshape(5, 5), dtype=np.float32))
    model = SCENE(latent_dim=2).fit(adata, gene_init="laplacian", layer=None, epochs=1, print_every=None)
    assert np.isfinite(adata.obsm["SCENE"]).all()
    assert model.config["fit"]["gene_init"] == "laplacian"
