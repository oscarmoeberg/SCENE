import pytest
import torch

from scene._decoder import _SCENEDecoder


@pytest.mark.parametrize("likelihood", ["Poisson", "ZIP"])
@pytest.mark.parametrize("batch_effect_type", [None, "full", "lowrank"])
@pytest.mark.parametrize("use_random_effects, use_batch_effects", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_expected_counts_edges(likelihood, batch_effect_type, use_random_effects, use_batch_effects):
    model = _SCENEDecoder(4, 3, 2, likelihood=likelihood,
                  batch_size={"batch": 2} if batch_effect_type else None, batch_effect_type=batch_effect_type or "lowrank")
    with torch.no_grad():
        model.re_cells.fill_(0.2)
        model.re_genes.fill_(0.3)
        for parameter in model.U_levels:
            parameter.fill_(0.4)
        for parameter in model.V_levels:
            parameter.fill_(0.5)
    batch_ids = [torch.tensor([0, 1, 0, 1])] if batch_effect_type else None
    options = dict(batch_ids_per_level=batch_ids, use_random_effects=use_random_effects,
                   use_batch_effects=use_batch_effects)
    lam, pi = model(**options)
    expected = lam if pi is None else pi * lam / (-torch.expm1(-lam))
    cells = torch.tensor([3, 0, 3])
    genes = torch.tensor([1, 2, 1])
    actual = model.expected_counts(cells, genes, **options)
    torch.testing.assert_close(actual, expected[cells, genes])
    assert actual.dtype == lam.dtype
    assert actual.device == lam.device
    actual.sum().backward()
    assert torch.isfinite(model.Z_cells.grad).all()
    empty = torch.empty(0, dtype=torch.long)
    assert model.expected_counts(empty, empty, **options).shape == (0,)


def test_expected_counts_tiny_and_zero_rates(monkeypatch):
    model = _SCENEDecoder(1, 3, 2, likelihood="ZIP")
    lam = torch.tensor([0., 1e-30, 100.], requires_grad=True)
    pi = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
    monkeypatch.setattr(model, "forward_edges", lambda *args, **kwargs: (lam, pi))
    actual = model.expected_counts(torch.tensor([0, 0, 0]), torch.arange(3))
    torch.testing.assert_close(actual, torch.tensor([0.2, 0.5, 80.]))
    actual.sum().backward()
    assert torch.isfinite(lam.grad).all()
    assert torch.isfinite(pi.grad).all()


def test_poisson_forward_shapes_and_ranges():
    torch.manual_seed(0)
    model = _SCENEDecoder(num_cells=4, num_genes=3, latent_dim=2, likelihood="Poisson")

    lam, pi = model()

    assert lam.shape == (4, 3)
    assert pi is None
    assert torch.isfinite(lam).all()
    assert (lam > 0).all()


def test_zip_forward_shapes_and_ranges():
    torch.manual_seed(0)
    model = _SCENEDecoder(num_cells=4, num_genes=3, latent_dim=2, likelihood="ZIP")

    lam, pi = model()

    assert lam.shape == (4, 3)
    assert pi.shape == (4, 3)
    assert torch.isfinite(lam).all()
    assert torch.isfinite(pi).all()
    assert (lam > 0).all()
    assert ((pi >= 0) & (pi <= 1)).all()


def test_forward_edges_and_cell_block_match_dense_forward_with_batches():
    torch.manual_seed(0)
    model = _SCENEDecoder(
        num_cells=4,
        num_genes=3,
        latent_dim=2,
        likelihood="ZIP",
        batch_size={"batch": 2},
        batch_effect_type="lowrank",
        batch_rank=2,
    )
    batch_ids = [torch.tensor([0, 1, 0, 1])]
    lam, pi = model(batch_ids_per_level=batch_ids)

    cells = torch.tensor([0, 1, 3])
    genes = torch.tensor([2, 1, 0])
    edge_lam, edge_pi = model.forward_edges(
        cells,
        genes,
        batch_ids_per_level=batch_ids,
    )
    block_lam, block_pi = model.forward_cell_block(
        cells,
        batch_ids_per_level=batch_ids,
    )

    assert torch.allclose(edge_lam, lam[cells, genes])
    assert torch.allclose(edge_pi, pi[cells, genes])
    assert torch.allclose(block_lam, lam[cells])
    assert torch.allclose(block_pi, pi[cells])


def test_forward_paths_match_for_full_batch_effects():
    torch.manual_seed(1)
    model = _SCENEDecoder(
        num_cells=4,
        num_genes=3,
        latent_dim=2,
        likelihood="Poisson",
        batch_size={"batch": 2},
        batch_effect_type="full",
    )
    batch_ids = [torch.tensor([0, 1, 0, 1])]
    lam, pi = model(batch_ids_per_level=batch_ids)
    cells = torch.tensor([0, 2])
    genes = torch.tensor([1, 2])
    edge_lam, edge_pi = model.forward_edges(cells, genes, batch_ids)
    block_lam, block_pi = model.forward_cell_block(cells, batch_ids)

    assert pi is None
    assert edge_pi is None
    assert block_pi is None
    assert torch.allclose(edge_lam, lam[cells, genes])
    assert torch.allclose(block_lam, lam[cells])


def test_re_flag_disables_random_effects():
    model = _SCENEDecoder(num_cells=3, num_genes=2, latent_dim=2, use_random_effects=False)
    with torch.no_grad():
        model.re_cells.fill_(10.0)
        model.re_genes.fill_(10.0)

    with_re, _ = model(use_random_effects=True)
    without_re, _ = model(use_random_effects=False)

    assert torch.allclose(with_re, without_re)


def test_full_and_weighted_cell_block_losses_match():
    counts = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
        ]
    )
    indices = counts.nonzero(as_tuple=False).T
    values = counts[indices[0], indices[1]]

    for likelihood in ("Poisson", "ZIP"):
        torch.manual_seed(4)
        model = _SCENEDecoder(
            num_cells=counts.shape[0],
            num_genes=counts.shape[1],
            latent_dim=2,
            likelihood=likelihood,
        )
        lam, pi = model()
        full_nll, full_mean = model.compute_loss(lam, pi, values, indices)

        block_means = []
        for cells in (torch.tensor([0, 2]), torch.tensor([1, 3])):
            block_lam, block_pi = model.forward_cell_block(cells)
            block_rows = []
            block_cols = []
            for local_row, cell in enumerate(cells.tolist()):
                cell_positions = torch.nonzero(indices[0] == cell).flatten()
                block_rows.append(torch.full_like(cell_positions, local_row))
                block_cols.append(indices[1, cell_positions])
            block_indices = torch.stack([
                torch.cat(block_rows),
                torch.cat(block_cols),
            ])
            block_values = counts[cells][block_indices[0], block_indices[1]].to(
                dtype=block_lam.dtype
            )
            _, block_mean = model.compute_cell_block_loss(
                block_lam,
                block_pi,
                block_values,
                block_indices,
                n_total=counts.numel(),
                block_weight=counts.shape[0] / cells.numel(),
            )
            block_means.append(block_mean)

        assert torch.allclose(torch.stack(block_means).mean(), full_mean)
