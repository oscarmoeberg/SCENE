import json

import numpy as np
import pytest
import torch

from scene import SCENE
from test_train import _adata


@pytest.mark.parametrize("likelihood", ["Poisson", "ZIP"])
@pytest.mark.parametrize("effect", ["none", "full", "lowrank", ["full", "lowrank"]])
@pytest.mark.parametrize("random_effects", [True, False])
def test_checkpoint_exact_round_trip(tmp_path, likelihood, effect, random_effects):
    adata = _adata()
    adata.obs["donor"] = [2, 2, 1, 1]
    keys = ["batch", "donor"] if isinstance(effect, list) else "batch"
    model = SCENE(latent_dim=2, likelihood=likelihood, batch_effect_type=effect,
                  use_random_effects=random_effects).fit(
        adata, batch_keys=keys, epochs=2, validate=True, val_frac=.25, print_every=None,
    )
    path = tmp_path / "checkpoint"
    model.save(path)
    loaded = SCENE.load(path)
    assert loaded.config == model.config
    assert loaded.training_history == model.training_history
    assert loaded.batch_size == model.batch_size
    assert loaded.batch_categories == model.batch_categories
    for name, tensor in model._decoder.state_dict().items():
        torch.testing.assert_close(loaded._decoder.state_dict()[name], tensor, rtol=0, atol=0)
    for use_batch in (True, False):
        for use_random in (True, False):
            for method in ("expected_counts", "linear_predictor_pairs"):
                options = dict(cells=["3", "0", "3"], genes=["1", "2", "1"],
                               use_batch_effects=use_batch, use_random_effects=use_random)
                assert getattr(model, method)(**options).equals(getattr(loaded, method)(**options))
    assert not loaded._decoder.training


def test_checkpoint_without_batches_and_undefined_metrics(tmp_path):
    model = SCENE(latent_dim=2).fit(_adata(), epochs=1, validate=True, val_frac=.25,
                                   neg_ratio=0, print_every=None, write_back=False)
    model.save(tmp_path / "saved")
    loaded = SCENE.load(tmp_path / "saved")
    assert loaded.batch_size == {}
    assert np.isnan(loaded.training_history["validation"]["AUC"][0])
    assert model.expected_counts(cells="0", genes="1").equals(loaded.expected_counts(cells="0", genes="1"))
    text = (tmp_path / "saved/metadata.json").read_text()
    assert "NaN" not in text and "null" in text


def test_checkpoint_load_skips_initialization_and_preserves_dtype(tmp_path, monkeypatch):
    from anndata import AnnData
    from scipy import sparse
    adata = AnnData(sparse.csr_matrix(np.ones((5, 5), dtype=np.float32)))
    model = SCENE(latent_dim=2).fit(
        adata, cell_init="laplacian", gene_init="laplacian", layer=None, epochs=1, print_every=None,
    )
    model._decoder.double()
    model.save(tmp_path / "saved")
    def forbidden(*args, **kwargs):
        raise AssertionError("Initialization must not run during load")
    monkeypatch.setattr("scene._decoder.laplacian_init", forbidden)
    monkeypatch.setattr(torch, "randn", forbidden)
    loaded = SCENE.load(tmp_path / "saved")
    assert loaded._decoder.Z_cells.dtype == torch.float64
    assert model.expected_counts(cells="0", genes="1").equals(loaded.expected_counts(cells="0", genes="1"))


def test_checkpoint_rejection_and_overwrite(tmp_path):
    model = SCENE(latent_dim=2).fit(_adata(), epochs=1, print_every=None)
    path = tmp_path / "model"
    model.save(path)
    with pytest.raises(FileExistsError):
        model.save(path)
    model.save(path, overwrite=True)
    with pytest.raises(ValueError, match="existing SCENE"):
        model.save(tmp_path, overwrite=True)
    info_path = path / "metadata.json"
    original = json.loads(info_path.read_text())
    invalid = dict(original, format_version=100)
    info_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="format version"):
        SCENE.load(path)
    invalid = dict(original, cell_names=["0"])
    info_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="size mismatch"):
        SCENE.load(path)
    info_path.write_text(json.dumps(original))
    (path / "weights.pt").write_bytes(b"bad checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        SCENE.load(path)
    with pytest.raises(ValueError, match="Incomplete"):
        SCENE.load(tmp_path / "absent")


def test_batch_metadata_rejection(tmp_path):
    model = SCENE(latent_dim=2).fit(_adata(), batch_keys="batch", epochs=1, print_every=None)
    path = tmp_path / "model"
    model.save(path)
    info = json.loads((path / "metadata.json").read_text())
    info["batch_codes"]["batch"][0] = 999
    (path / "metadata.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="batch encoding"):
        SCENE.load(path)


def test_older_checkpoint_layout_is_rejected(tmp_path):
    model = SCENE(latent_dim=2).fit(_adata(), epochs=1, print_every=None)
    path = tmp_path / "model"
    model.save(path)
    info = json.loads((path / "metadata.json").read_text())
    info["format_version"] = 1
    (path / "metadata.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Unsupported checkpoint format version: 1"):
        SCENE.load(path)
