import numpy as np
import pytest
import torch

from scene import SCENE
from test_train import _adata


def test_names_order_broadcasting_and_automatic_batch_encoding():
    model = SCENE(latent_dim=2, likelihood="ZIP").fit(
        _adata(), batch_keys="batch", epochs=1, print_every=None,
    )
    prediction = model.expected_counts(cells=["3", "0", "3"], genes="1")
    assert prediction.cell.tolist() == ["3", "0", "3"]
    assert prediction.gene.tolist() == ["1", "1", "1"]
    decoder = model._decoder
    direct = decoder.expected_counts(torch.tensor([3, 0, 3]), torch.tensor([1, 1, 1]),
                                     batch_ids_per_level=[torch.tensor([0, 1, 0, 1])])
    np.testing.assert_array_equal(prediction.expected_count, direct.detach().numpy())
    assert len(model.expected_counts(cells="0", genes=["2", "0"])) == 2
    assert len(model.expected_counts(cells="0", genes="1")) == 1
    assert model.expected_counts(cells=[], genes=[]).empty
    assert model.expected_counts(cells="0", genes=[]).empty
    assert model.expected_counts(cells=[], genes="1").empty


@pytest.mark.parametrize("options, error", [
    ({"cells": "unknown", "genes": "0"}, KeyError),
    ({"cells": "0", "genes": "unknown"}, KeyError),
    ({"cells": ["0"], "genes": ["0", "1"]}, ValueError),
    ({"cells": [1], "genes": ["0"]}, ValueError),
    ({"cells": None, "genes": "0"}, ValueError),
])
def test_invalid_pairs(options, error):
    model = SCENE(latent_dim=2).fit(_adata(), epochs=1, print_every=None)
    with pytest.raises(error):
        model.expected_counts(**options)


def test_predictions_are_chunked(monkeypatch):
    model = SCENE(latent_dim=2).fit(_adata(), epochs=1, print_every=None)
    original = model._decoder.expected_counts
    sizes = []
    def wrapped(rows, cols, **kwargs):
        sizes.append(len(rows))
        assert not torch.is_grad_enabled()
        return original(rows, cols, **kwargs)
    monkeypatch.setattr(model._decoder, "expected_counts", wrapped)
    result = model.expected_counts(cells="0", genes=["1"] * 100_001)
    assert sizes == [100_000, 1]
    assert len(result) == 100_001
    assert result.expected_count.nunique() == 1


def test_unfitted_operations(tmp_path):
    model = SCENE()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.expected_counts(cells="0", genes="1")
    with pytest.raises(RuntimeError, match="not fitted"):
        model.save(tmp_path / "empty")
