import json
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData, read_h5ad
from scipy import sparse

from scene import SCENE, nearest_genes


def _adata():
    counts = sparse.csr_matrix(np.array([[1, 0, 2], [0, 1, 0], [3, 0, 1], [0, 2, 0]], dtype=np.float32))
    adata = AnnData(X=counts)
    adata.layers["counts"] = counts.copy()
    adata.obs["batch"] = ["a", "b", "a", "b"]
    return adata


def test_fit_is_only_entry_point_and_returns_self(capsys):
    adata = _adata()
    model = SCENE(latent_dim=2)
    assert model.fit(adata, epochs=2, print_every=None, seed=0) is model
    assert model.likelihood == "ZIP"
    assert model._decoder.likelihood == "ZIP"
    assert model.config["model"]["likelihood"] == "ZIP"
    assert capsys.readouterr().out == ""
    assert model.training_history["train"]["epoch"] == [1, 2]
    assert adata.obsm["SCENE"].shape == (4, 2)
    assert adata.varm["SCENE"].shape == (3, 2)
    assert set(adata.uns) == {"SCENE"}
    assert not hasattr(model, "train")
    assert not model._decoder.training
    first = model.expected_counts(cells="0", genes="1")
    model.fit(adata, epochs=2, print_every=None, seed=0)
    assert model.training_history["train"]["epoch"] == [1, 2]
    assert first.equals(model.expected_counts(cells="0", genes="1"))
    # AnnData output must not alias the model's CPU tensors.
    adata.obsm["SCENE"][:] = 0
    assert first.equals(model.expected_counts(cells="0", genes="1"))


def test_write_back_false_and_existing_model_survives_invalid_fit():
    adata = _adata()
    model = SCENE(latent_dim=2).fit(adata, epochs=1, write_back=False, print_every=None)
    assert not adata.obsm and not adata.varm and not adata.uns
    before = model.expected_counts(cells="0", genes="1")
    with pytest.raises(ValueError, match="epochs"):
        model.fit(adata, epochs=0)
    assert before.equals(model.expected_counts(cells="0", genes="1"))


@pytest.mark.parametrize("mode", ["flipped", "masked"])
def test_validation_labels_and_output(mode, capsys):
    model = SCENE(latent_dim=2, likelihood="zip").fit(
        _adata(), epochs=2, validate=True, val_frac=.25, validation_mode=mode, print_every=1,
    )
    assert model.likelihood == "ZIP"
    assert set(model.training_history["validation"]) == {"epoch", "AUC", "PR AUC", "max F1", "PoissonDeviance"}
    assert model.training_history["validation"]["epoch"] == [1, 2]
    assert model.config["fit"]["validation_mode"] == mode
    output = capsys.readouterr().out
    assert "PR AUC=" in output and "max F1=" in output and "PoissonDeviance=" in output


@pytest.mark.parametrize("options, match", [
    ({"cell_init": "typo"}, "cell_init"), ({"gene_init": "typo"}, "gene_init"),
    ({"epochs": 0}, "epochs"), ({"epochs": True}, "epochs"),
    ({"epochs": 1.5}, "epochs"), ({"lr": 0}, "lr"), ({"lr": np.nan}, "lr"),
    ({"lr": np.inf}, "lr"), ({"weight_decay": -1}, "weight_decay"),
    ({"val_interval": 0}, "val_interval"), ({"print_every": 0}, "print_every"),
    ({"validation_edge_batch_size": 0}, "validation_edge_batch_size"),
    ({"val_frac": 1}, "val_frac"), ({"val_frac": np.nan}, "val_frac"),
    ({"neg_ratio": np.inf}, "neg_ratio"), ({"cell_batch_size": 0}, "cell_batch_size"),
    ({"loader_workers": -1}, "loader_workers"), ({"loader_prefetch_batches": 0}, "loader_prefetch_batches"),
    ({"loader_pin_memory": "yes"}, "loader_pin_memory"), ({"seed": -1}, "seed"),
    ({"split_seed": True}, "split_seed"), ({"key": "a/b"}, "key"),
    ({"validation_mode": []}, "validation_mode"), ({"batch_keys": 42}, "batch_keys"),
    ({"validation_mode": "missing"}, "validation_mode"), ({"validate": 1}, "validate"),
    ({"write_back": "yes"}, "write_back"), ({"batch_keys": ["batch", "batch"]}, "unique"),
])
def test_fit_validation(options, match):
    with pytest.raises(ValueError, match=match):
        SCENE(latent_dim=2).fit(_adata(), **{"epochs": 1, "print_every": None, **options})


@pytest.mark.parametrize("options", [
    {"latent_dim": 0}, {"latent_dim": 1.5}, {"batch_rank": True},
    {"likelihood": "bad"}, {"use_random_effects": "yes"},
    {"batch_effect_type": "typo"}, {"batch_effect_type": ["full", "typo"]},
])
def test_model_validation(options):
    with pytest.raises(ValueError):
        SCENE(**options)


def test_names_batches_and_empty_data():
    adata = _adata()
    adata.obs_names = ["a"] * 4
    with pytest.raises(ValueError, match="unique"):
        SCENE().fit(adata)
    adata = _adata()
    adata.var_names = ["g"] * 3
    with pytest.raises(ValueError, match="unique"):
        SCENE().fit(adata)
    with pytest.raises(ValueError, match="cells and genes"):
        SCENE().fit(_adata()[:0].copy())
    with pytest.raises(ValueError, match="list length"):
        SCENE(batch_effect_type=["full", "none"]).fit(_adata(), batch_keys="batch")
    adata = _adata()
    adata.obs.loc["0", "batch"] = None
    with pytest.raises(ValueError, match="missing or unknown"):
        SCENE().fit(adata, batch_keys="batch")
    with pytest.raises(ValueError, match="view"):
        SCENE().fit(_adata()[:2])


@pytest.mark.parametrize("likelihood", ["Poisson", "ZIP"])
@pytest.mark.parametrize("batch_keys, effect", [
    (None, "lowrank"), ("batch", "none"), ("batch", "full"), ("batch", "lowrank"),
    (["batch", "donor"], ["full", "lowrank"]),
])
def test_h5ad_round_trip(tmp_path, likelihood, batch_keys, effect):
    adata = _adata()
    adata.obs["donor"] = [1, 1, 2, 2]
    model = SCENE(latent_dim=2, likelihood=likelihood, batch_effect_type=effect).fit(
        adata, batch_keys=batch_keys, epochs=1, validate=True, val_frac=.25, print_every=None,
    )
    adata.write_h5ad(tmp_path / "scene.h5ad")
    saved = read_h5ad(tmp_path / "scene.h5ad")
    np.testing.assert_array_equal(saved.obsm["SCENE"], adata.obsm["SCENE"])
    np.testing.assert_array_equal(saved.varm["SCENE"], adata.varm["SCENE"])
    info = saved.uns["SCENE"]
    assert json.loads(info["config"]) == model.config
    assert json.loads(info["batch_categories"]) == model.batch_categories
    assert info["format_version"] == 2
    assert info["package_version"]
    for section, columns in model.training_history.items():
        for name, values in columns.items():
            np.testing.assert_array_equal(info["training_history"][section][name], values)
    assert nearest_genes(saved, cells="0").equals(nearest_genes(adata, cells="0"))


# Captured from commit 76486cb before this API revision, on CPU with seed 3.
BASELINES = json.loads((Path(__file__).parent / "fixtures/pre_release_numerics.json").read_text())


@pytest.mark.parametrize("reference", BASELINES)
def test_numerical_equivalence_to_pre_release(reference):
    model = SCENE(latent_dim=2, likelihood=reference["likelihood"],
                  batch_effect_type=reference["batch_effect_type"]).fit(
        _adata(), batch_keys="batch", epochs=2, seed=3, cell_batch_size=reference["cell_batch_size"],
        validate=True, val_frac=.25, validation_mode="masked", print_every=None, loader_workers=0,
    )
    np.testing.assert_allclose(model.training_history["train"]["loss"], reference["loss"], rtol=1e-6, atol=1e-7)
    predicted = model.expected_counts(cells=["0", "3"], genes=["1", "2"])
    np.testing.assert_allclose(predicted.expected_count, reference["expected"], rtol=1e-6, atol=1e-7)



def test_public_parameter_placement():
    import inspect
    constructor = set(inspect.signature(SCENE).parameters)
    fitting = set(inspect.signature(SCENE.fit).parameters)
    assert constructor == {"latent_dim", "likelihood", "batch_effect_type", "batch_rank", "use_random_effects"}
    assert {"cell_init", "gene_init", "adata", "seed", "epochs"} <= fitting
    assert not constructor.intersection(fitting)
    with pytest.raises(TypeError, match="cell_init"):
        SCENE(cell_init="random")
    with pytest.raises(TypeError, match="likelihood"):
        SCENE().fit(_adata(), likelihood="ZIP")


@pytest.mark.parametrize("cell_init", ["random", "laplacian"])
@pytest.mark.parametrize("gene_init", ["random", "laplacian"])
def test_initialization_is_a_fit_setting(tmp_path, cell_init, gene_init):
    adata = AnnData(sparse.csr_matrix(np.arange(25).reshape(5, 5), dtype=np.float32))
    model = SCENE(latent_dim=2, likelihood="zip")
    model.fit(adata, layer=None, cell_init=cell_init, gene_init=gene_init, epochs=1, print_every=None)
    assert model.likelihood == "ZIP"
    assert model.config["model"]["likelihood"] == "ZIP"
    assert "likelihood" not in model.config["fit"]
    for name, value in (("cell_init", cell_init), ("gene_init", gene_init)):
        assert name not in model.config["model"]
        assert model.config["fit"][name] == value
        assert getattr(model._decoder, name) == value
    model.save(tmp_path / "model")
    loaded = SCENE.load(tmp_path / "model")
    assert loaded.config == model.config
    assert loaded.training_history == model.training_history
    assert model.expected_counts(cells="0", genes="1").equals(loaded.expected_counts(cells="0", genes="1"))
    original = loaded.expected_counts(cells="0", genes="1")
    with pytest.raises(ValueError, match="gene_init"):
        loaded.fit(adata, layer=None, gene_init="typo")
    assert original.equals(loaded.expected_counts(cells="0", genes="1"))
    loaded.fit(adata, layer=None, epochs=1, print_every=None)
    assert loaded.likelihood == "ZIP"
    assert loaded.config["fit"]["cell_init"] == "random"
    assert loaded.config["fit"]["gene_init"] == "random"
    assert loaded.training_history["train"]["epoch"] == [1]


def test_training_history_layout_and_persistence(tmp_path):
    adata = _adata()
    model = SCENE(latent_dim=2, likelihood="ZIP").fit(
        adata, epochs=2, print_every=None, validate=True, val_frac=.25,
    )
    assert not hasattr(model, "history")
    assert set(model.training_history) == {"train", "validation"}
    assert model.training_history["validation"]["epoch"] == [1, 2]
    assert "history" not in adata.uns["SCENE"]
    assert set(adata.uns["SCENE"]["training_history"]) == {"train", "validation"}
    model.save(tmp_path / "model")
    stored = json.loads((tmp_path / "model/metadata.json").read_text())
    assert "history" not in stored
    assert stored["training_history"] == model.training_history
    assert stored["format_version"] == 2
    loaded = SCENE.load(tmp_path / "model")
    assert not hasattr(loaded, "history")
    assert loaded.training_history == model.training_history
