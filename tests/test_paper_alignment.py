"""Regression checks against the manuscript's count model and graph equations."""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy import sparse

from scene import SCENE, laplacian_init
from scene._decoder import _SCENEDecoder
from scene.train import _build_optimizer, _evaluate_holdout_edges


@pytest.mark.parametrize("cell_batch_size", [None, 2])
@pytest.mark.parametrize("validation_mode", [None, "flipped", "masked"])
@pytest.mark.parametrize("cell_init,gene_init", [
    ("random", "random"), ("laplacian", "random"),
    ("random", "laplacian"), ("laplacian", "laplacian"),
])
def test_fit_retains_raw_counts_and_all_entities(monkeypatch, cell_batch_size,
                                                validation_mode, cell_init, gene_init):
    import scene.model as implementation
    counts = np.array([[5, 1, 2, 0, 0], [1, 4, 0, 1, 0], [0, 2, 7, 2, 0],
                       [2, 0, 1, 3, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    adata = AnnData(np.log1p(counts))
    adata.layers["counts"] = sparse.csr_matrix(counts)
    adata.var["highly_variable"] = [True, False, False, False, False]
    adata.obsm["X_pca"] = np.ones((5, 2))
    original_x, original_obs, original_var = adata.X.copy(), adata.obs.copy(), adata.var.copy()
    original_train = implementation._train_prepared

    def inspect_training(decoder, prepared, **options):
        expected = counts.copy()
        if validation_mode:
            holdout = prepared.holdout
            rows, cols = holdout["pos_rows"], holdout["pos_cols"]
            np.testing.assert_array_equal(holdout["values"], counts[rows, cols])
            expected[rows, cols] = 0
            assert np.all(counts[holdout["neg_rows"], holdout["neg_cols"]] == 0)
            assert (prepared.missing_indices is not None) == (validation_mode == "masked")
        np.testing.assert_array_equal(prepared.train_matrix.toarray(), expected)
        assert decoder.num_cells == 5 and decoder.num_genes == 5
        return original_train(decoder, prepared, **options)

    monkeypatch.setattr(implementation, "_train_prepared", inspect_training)
    model = SCENE(latent_dim=2).fit(
        adata, epochs=1, print_every=None, cell_batch_size=cell_batch_size,
        cell_init=cell_init, gene_init=gene_init,
        validate=validation_mode is not None, val_frac=.25,
        validation_mode=validation_mode or "flipped",
    )
    np.testing.assert_array_equal(adata.layers["counts"].toarray(), counts)
    np.testing.assert_array_equal(adata.X, original_x)
    pd.testing.assert_frame_equal(adata.obs, original_obs)
    pd.testing.assert_frame_equal(adata.var, original_var)
    np.testing.assert_array_equal(adata.obsm["X_pca"], np.ones((5, 2)))
    assert model.gene_names.equals(adata.var_names)
    assert model.cell_names.equals(adata.obs_names)
    assert adata.obsm["SCENE"].shape == adata.varm["SCENE"].shape == (5, 2)
    assert np.isfinite(model.expected_counts(cells="4", genes="4").expected_count).all()


@pytest.mark.parametrize("counts", [
    np.array([[4, 1, 0, 2], [1, 8, 2, 1], [2, 0, 3, 1], [1, 2, 1, 6], [2, 1, 4, 1]]),
    sparse.block_diag([np.array([[3, 1], [2, 4]]), np.array([[2, 1], [1, 5]])]).toarray(),
    np.diag([1, 2, 4, 8]),
    np.pad(np.array([[4, 1, 2], [1, 6, 1], [2, 1, 3]]), ((0, 1), (0, 2))),
])
def test_initialization_solves_symmetric_laplacian(counts):
    n, g = counts.shape
    adjacency = np.block([[np.zeros((n, n)), counts], [counts.T, np.zeros((g, g))]])
    degree = adjacency.sum(axis=1)
    inverse = np.divide(1., np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    laplacian = np.eye(n + g) - inverse[:, None] * adjacency * inverse[None, :]
    exact = np.linalg.eigvalsh(laplacian)
    expected = exact[exact > 1e-7][:2]
    np.random.seed(42)
    coordinates = np.vstack(laplacian_init(counts, latent_dim=2)).astype(float)
    np.testing.assert_allclose(coordinates.T @ coordinates, np.eye(2), atol=2e-7)
    np.testing.assert_allclose(laplacian @ coordinates, coordinates * expected, atol=2e-7)
    # Multiplying every count by a constant leaves the normalized graph unchanged.
    np.random.seed(42)
    rescaled = np.vstack(laplacian_init(7 * counts, latent_dim=2)).astype(float)
    # Repeated eigenvalues permit any orthonormal basis; compare the eigen-equation.
    np.testing.assert_allclose(laplacian @ rescaled, rescaled * expected, atol=2e-7)
    np.testing.assert_allclose(rescaled.T @ rescaled, np.eye(2), atol=2e-7)


def test_empty_graph_cannot_supply_laplacian_initialization():
    with pytest.raises(ValueError, match="nonzero count"):
        laplacian_init(np.zeros((4, 4)), 2)


@pytest.mark.parametrize("likelihood", ["ZIP", "Poisson"])
def test_likelihood_matches_manuscript_pmf(likelihood):
    decoder = _SCENEDecoder(2, 3, 2, likelihood=likelihood).double()
    predictor = torch.tensor([[-1., .2, .8], [.1, -.4, 1.]], dtype=torch.float64)
    counts = torch.tensor([[0., 1., 4.], [2., 0., 3.]], dtype=torch.float64)
    lam, pi = decoder._decode_from_diff(predictor)
    indices = counts.nonzero().T
    loss, mean = decoder.compute_loss(lam, pi, counts[indices[0], indices[1]], indices)
    poisson = counts * predictor - lam - torch.lgamma(counts + 1)
    if likelihood == "ZIP":
        positive = torch.log(pi) + poisson - torch.log1p(-torch.exp(-lam))
        reference = -torch.where(counts > 0, positive, torch.log1p(-pi)).sum()
    else:
        reference = -poisson.sum()
    torch.testing.assert_close(loss, reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(mean, reference / counts.numel(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("effect_type", ["none", "full", "lowrank"])
def test_adamw_regularization_matches_manuscript(effect_type):
    decoder = _SCENEDecoder(3, 4, 2, batch_size={"donor": 2}, batch_effect_type=effect_type)
    optimizer = _build_optimizer(decoder, lr=.05, weight_decay=.001)
    assert isinstance(optimizer, torch.optim.AdamW)
    exclusions = {id(decoder.re_cells), id(decoder.re_genes), id(decoder.raw_alpha)}
    decay_by_id = {id(parameter): group["weight_decay"]
                   for group in optimizer.param_groups for parameter in group["params"]}
    for parameter in decoder.parameters():
        if parameter.requires_grad:
            assert decay_by_id[id(parameter)] == (0 if id(parameter) in exclusions else .001)


def test_validation_matches_manuscript_definitions(monkeypatch):
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
    decoder = _SCENEDecoder(1, 4, 2).double()
    rates = torch.tensor([.4, 2., 1., .1], dtype=torch.float64)
    probabilities = torch.sigmoid(decoder.alpha.detach() * rates.log())
    monkeypatch.setattr(decoder, "forward_edges",
                        lambda rows, cols, **kwargs: (rates[cols], probabilities[cols]))
    observed = np.array([2., 4.])
    holdout = {"pos_rows": np.array([0, 0]), "pos_cols": np.array([0, 1]),
               "neg_rows": np.array([0, 0]), "neg_cols": np.array([2, 3]), "values": observed}
    metrics = _evaluate_holdout_edges(decoder, [], holdout, edge_batch_size=1)
    labels = np.array([1, 1, 0, 0])
    precision, recall, _ = precision_recall_curve(labels, probabilities.numpy())
    positive_mean = rates[:2].numpy() / (-np.expm1(-rates[:2].numpy()))
    assert metrics["AUC"] == roc_auc_score(labels, probabilities.numpy())
    assert metrics["PR AUC"] == auc(recall, precision)
    assert metrics["max F1"] == pytest.approx(np.max(2 * precision * recall / (precision + recall)))
    assert metrics["PoissonDeviance"] == pytest.approx(
        np.mean(2 * (observed * np.log(observed / positive_mean) - observed + positive_mean)))
