"""Versioned fitted metadata shared by checkpoints and AnnData output."""

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ._decoder import _SCENEDecoder
from ._names import name_index

FORMAT_VERSION = 2


def json_ready(value):
    """Convert NumPy values to JSON primitives; undefined metrics become null."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {key: json_ready(entry) for key, entry in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(entry) for entry in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Metadata requires string, numeric, or boolean categories; got {type(value).__name__}")


def metadata(model):
    return json_ready({
        "format_version": FORMAT_VERSION,
        "package_version": model._package_version,
        "config": model.config,
        "cell_names": model.cell_names.tolist(),
        "gene_names": model.gene_names.tolist(),
        "batch_size": model.batch_size,
        "batch_categories": model.batch_categories,
        "batch_codes": model._batch_codes,
        "training_history": model.training_history,
    })


def write_adata(model, adata, key):
    decoder = model._decoder
    saved = metadata(model)
    # JSON strings retain nulls and category scalar types across supported
    # AnnData versions. The remaining metadata is natively H5AD-compatible.
    saved["config"] = json.dumps(saved["config"], allow_nan=False)
    saved["batch_categories"] = json.dumps(saved["batch_categories"], allow_nan=False)
    saved["training_history"] = {section: {name: np.asarray(values, dtype=np.int64 if name in
                        {"epoch", "optimizer_updates"} else np.float64)
                        for name, values in columns.items()}
                        for section, columns in model.training_history.items()}
    saved["batch_codes"] = {name: values.copy() for name, values in model._batch_codes.items()}
    batch_effects = {}
    for level, (tag, index) in enumerate(decoder.gamma_levels):
        entry = {"type": tag}
        if tag == "full":
            entry["gamma"] = decoder.U_levels[index].detach().cpu().numpy().copy()
        elif tag == "lowrank":
            entry["U"] = decoder.U_levels[index].detach().cpu().numpy().copy()
            entry["V"] = decoder.V_levels[index].detach().cpu().numpy().copy()
        batch_effects[decoder.batch_levels[level]] = entry
    saved["parameters"] = {
        "alpha": float(decoder.alpha.detach().cpu()),
        "cell_random_effects": decoder.re_cells.detach().cpu().numpy().ravel().copy(),
        "gene_random_effects": decoder.re_genes.detach().cpu().numpy().ravel().copy(),
        "batch_effects": batch_effects,
    }
    adata.obsm[key] = decoder.Z_cells.detach().cpu().numpy().copy()
    adata.varm[key] = decoder.Z_genes.detach().cpu().numpy().copy()
    adata.uns[key] = saved


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_model(model, path, *, overwrite):
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Checkpoint path already exists: {path}")
        if not path.is_dir() or not (path / "metadata.json").is_file() or not (path / "weights.pt").is_file():
            raise ValueError("overwrite requires an existing SCENE checkpoint directory")
        try:
            previous = json.loads((path / "metadata.json").read_text())
            if previous.get("format_version") != FORMAT_VERSION:
                raise ValueError("Unsupported checkpoint format")
        except (ValueError, AttributeError) as error:
            raise ValueError("overwrite requires a valid SCENE checkpoint directory") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".scene-save-", dir=path.parent) as staging:
        staging = Path(staging)
        weights = {name: tensor.detach().cpu().clone() for name, tensor in model._decoder.state_dict().items()}
        torch.save(weights, staging / "weights.pt")
        saved = metadata(model)
        saved["weights_sha256"] = _digest(staging / "weights.pt")
        (staging / "metadata.json").write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n")
        path.mkdir(exist_ok=True)
        # Write the checksum-bearing metadata last; partial writes fail load
        # explicitly instead of combining tensors and metadata from two fits.
        os.replace(staging / "weights.pt", path / "weights.pt")
        os.replace(staging / "metadata.json", path / "metadata.json")


def load_model(cls, path, *, device):
    if not (path / "metadata.json").is_file() or not (path / "weights.pt").is_file():
        raise ValueError(f"Incomplete SCENE checkpoint: {path}")
    try:
        saved = json.loads((path / "metadata.json").read_text())
        if saved["format_version"] != FORMAT_VERSION:
            raise ValueError(f"Unsupported checkpoint format version: {saved['format_version']}")
        if _digest(path / "weights.pt") != saved["weights_sha256"]:
            raise ValueError("Checkpoint weights checksum does not match metadata")
        model = cls(**saved["config"]["model"])
        model.cell_names = name_index(saved["cell_names"], "cell")
        model.gene_names = name_index(saved["gene_names"], "gene")
        if not len(model.cell_names) or not len(model.gene_names):
            raise ValueError("Checkpoint requires fitted cells and genes")
        model.batch_size = saved["batch_size"]
        model.batch_categories = saved["batch_categories"]
        levels = saved["config"]["fit"]["batch_keys"]
        if (levels != list(model.batch_size) or set(levels) != set(model.batch_categories)
                or set(levels) != set(saved["batch_codes"])):
            raise ValueError("Checkpoint batch keys are inconsistent")
        model._batch_codes = {}
        for key, size in model.batch_size.items():
            categories = model.batch_categories[key]
            codes = np.asarray(saved["batch_codes"][key])
            if (isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    or len(categories) != size or len(set(categories)) != size
                    or codes.shape != (len(model.cell_names),) or codes.dtype.kind not in "iu"
                    or (codes < 0).any() or (codes >= size).any()):
                raise ValueError(f"Invalid checkpoint batch encoding for {key}")
            model._batch_codes[key] = codes.astype(np.int64)
        weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        if not isinstance(weights, dict) or not all(isinstance(t, torch.Tensor) for t in weights.values()):
            raise ValueError("Checkpoint must contain a tensor state dictionary")
        dtype = weights["Z_cells"].dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Unsupported checkpoint tensor dtype")
        if any(t.dtype != dtype or not torch.isfinite(t).all() for t in weights.values()):
            raise ValueError("Checkpoint tensors must be finite and share a dtype")
        decoder = _SCENEDecoder(len(model.cell_names), len(model.gene_names),
                                **model._model_config(), batch_size=model.batch_size,
                                cell_init=saved["config"]["fit"]["cell_init"],
                                gene_init=saved["config"]["fit"]["gene_init"],
                                _initialize=False).to(dtype=dtype, device=device)
        decoder.load_state_dict(weights, strict=True)
        decoder.eval()
        model._decoder = decoder
        model.config = deepcopy(saved["config"])
        model.training_history = saved["training_history"]
        required = {"train": {"epoch", "loss", "optimizer_updates"},
                    "validation": {"epoch", "AUC", "PR AUC", "max F1", "PoissonDeviance"}}
        for section, columns in required.items():
            entries = model.training_history[section]
            if set(entries) != columns or len({len(values) for values in entries.values()}) != 1:
                raise ValueError("Checkpoint training history columns are inconsistent")
            for name, values in entries.items():
                entries[name] = [float("nan") if value is None else value for value in values]
        model._package_version = saved["package_version"]
        return model
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        raise ValueError(f"Invalid SCENE checkpoint: {error}") from error
