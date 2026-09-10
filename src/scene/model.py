"""The public, data-aware SCENE model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from ._decoder import _SCENEDecoder, _normalize_likelihood
from ._names import name_index, resolve_pairs
from ._validation import finite_real, positive_int, seed_value, storage_key
from .data import PreparedSceneData, resolve_loader_settings, sparse_extraction_backend_name
from .train import _train_prepared, _validate_cell_batch_size, set_seed


class SCENE:
    """Joint cell–gene model fitted directly to raw UMI counts.

    Parameters
    ----------
    latent_dim : int, default 16
        Number of shared Euclidean dimensions.
    likelihood : {"Poisson", "ZIP"}, default "ZIP"
        Count distribution; names are case-insensitive. ZIP models the
        probability of a nonzero count and uses a zero-truncated Poisson
        distribution for positive counts.
    batch_effect_type : str or sequence of str, default "lowrank"
        "none" omits batch effects; "full" fits an offset per gene and batch
        category; "lowrank" factorizes these offsets using batch_rank dimensions.
        Supply one type for all batch_keys or one per key, in the same order.
    batch_rank : int, default 2
        Factorization rank for lowrank batch effects.
    use_random_effects : bool, default True
        Fit cell and gene random effects.

    Attributes
    ----------
    training_history : dict
        After fitting: train and validation column dictionaries, suitable for
        pandas.DataFrame. Validation columns include "PR AUC" and "max F1".
    config : dict
        Fitted model, data, optimization, and effective loader settings.
    batch_size : dict
        Inferred category counts by batch key; unrelated to cell_batch_size.
    cell_names, gene_names : pandas.Index
        Fitted identifiers in decoder order.
    batch_categories : dict
        Category labels in fitted encoding order.

    Notes
    -----
    Each fit starts fresh and returns self. SCENE fits all supplied cells and
    genes and work directly on UMI counts. Quality control is performed before supplying data to SCENE.
    The model does not retain the input AnnData.
    """

    def __init__(self, *, latent_dim=16, likelihood="ZIP", batch_effect_type="lowrank",
                 batch_rank=2, use_random_effects=True):
        self.latent_dim = positive_int(latent_dim, "latent_dim")
        self.batch_rank = positive_int(batch_rank, "batch_rank")
        self.likelihood = _normalize_likelihood(likelihood)
        if not isinstance(use_random_effects, bool):
            raise ValueError("use_random_effects must be a boolean")
        self.use_random_effects = use_random_effects
        if isinstance(batch_effect_type, str):
            types = [batch_effect_type]
        elif isinstance(batch_effect_type, (list, tuple)):
            types = list(batch_effect_type)
        else:
            raise ValueError("batch_effect_type must be a string or sequence of strings")
        if any(not isinstance(t, str) or t not in {"none", "full", "lowrank"} for t in types):
            raise ValueError("batch_effect_type must contain 'none', 'full', or 'lowrank'")
        self.batch_effect_type = batch_effect_type if isinstance(batch_effect_type, str) else types
        self._decoder = None

    def _model_config(self):
        return {name: deepcopy(getattr(self, name)) for name in (
            "latent_dim", "likelihood", "batch_effect_type", "batch_rank", "use_random_effects",
        )}

    def fit(self, adata, *, layer="counts", batch_keys=None, cell_init="random", gene_init="random",
            epochs=600, lr=0.005,
            seed=42, split_seed=None, key="SCENE", validate=False, val_frac=0.10,
            val_interval=10, print_every=50, validation_mode="flipped", neg_ratio=1.0,
            cell_batch_size=1024, loader_workers=None, loader_prefetch_batches=2,
            loader_pin_memory=None, validation_edge_batch_size=100_000,
            write_back=True, weight_decay=1e-3, device="cpu"):
        """Fit from scratch, optionally write results into adata, and return self.

        Parameters
        ----------
        adata : anndata.AnnData
            Raw UMI counts with unique obs_names and var_names. All supplied
            cells and genes are retained, including zero-count rows and columns.
        layer : str or None, default "counts"
            Count layer; None explicitly selects X.
        batch_keys : str or sequence of str or None, default None
            Observation columns for batch effects, in model order.
        cell_init, gene_init : {"random", "laplacian"}, default "random"
            Independent starting coordinates for this fit. Laplacian uses
            the symmetric normalized graph Laplacian of the training counts;
            it does not transform the counts used in the likelihood. Requires
            latent_dim < min(n_cells, n_genes) - 1 and at least one nonzero count.
        epochs : int, default 600
            Number of complete training passes.
        lr : float, default 0.005
            Positive AdamW learning rate.
        seed : int, default 42
            Initialization and sampling seed in [0, 2**32).
        split_seed : int or None, default None
            Validation split seed; None uses seed.
        key : str, default "SCENE"
            Result key for obsm, varm, and uns; no '/' characters.
        validate : bool, default False
            Evaluate held-out positive counts against sampled zero entries.
        val_frac : float, default 0.10
            Fraction of positive entries held out, in [0, 1).
        val_interval : int, default 10
            Validation interval; also evaluate first and last epochs.
        print_every : int or None, default 50
            Reporting interval, including first/last epochs; None is quiet.
        validation_mode : {"flipped", "masked"}, default "flipped"
            In "flipped" mode, held-out positive counts are treated as zeros
            during training. In "masked" mode, those entries are excluded from
            the loss. Sampled zero entries remain in training. Input counts
            in adata are unchanged in both modes.
        neg_ratio : float, default 1.0
            Non-negative ratio of sampled zeros to held-out positives.
        cell_batch_size : int or None, default 1024
            Cells per optimization step, including all genes for each selected cell.
            None uses one full-matrix step per epoch, as in the manuscript.
        loader_workers : int or None, default None
            Worker count; None uses zero workers.
        loader_prefetch_batches : int, default 2
            Positive prefetch count for multiprocessing loaders.
        loader_pin_memory : bool or None, default None
            None enables pinning for CUDA cell blocks; otherwise request it
            explicitly. Effective pinning is disabled outside CUDA blocks.
        validation_edge_batch_size : int, default 100000
            Maximum paired entries per validation chunk.
        write_back : bool, default True
            Write embeddings to obsm[key]/varm[key] and metadata to uns[key].
            False leaves adata unchanged; the fitted model remains usable.
        weight_decay : float, default 0.001
            Non-negative AdamW decay, excluding random effects and ZIP alpha.
        device : str or torch.device, default "cpu"
            Training device.

        Returns
        -------
        SCENE
            This fitted object. Existing fitted state is replaced on success.
        """
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an AnnData object")
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("adata must contain cells and genes")
        if write_back and (adata.is_view or adata.isbacked):
            raise ValueError("write_back requires an in-memory AnnData, not a view; use adata.copy()")
        for name, value in (("cell_init", cell_init), ("gene_init", gene_init)):
            if not isinstance(value, str) or value not in {"random", "laplacian"}:
                raise ValueError(f"{name} must be 'random' or 'laplacian'")
        cell_names = name_index(adata.obs_names, "cell")
        gene_names = name_index(adata.var_names, "gene")
        storage_key(key)
        if layer is not None and not isinstance(layer, str):
            raise ValueError("layer must be a string or None")
        for name, value in (("validate", validate), ("write_back", write_back)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        epochs = positive_int(epochs, "epochs")
        val_interval = positive_int(val_interval, "val_interval")
        if print_every is not None:
            print_every = positive_int(print_every, "print_every")
        validation_edge_batch_size = positive_int(validation_edge_batch_size, "validation_edge_batch_size")
        lr = finite_real(lr, "lr", strict=True)
        weight_decay = finite_real(weight_decay, "weight_decay")
        val_frac = finite_real(val_frac, "val_frac")
        if val_frac >= 1:
            raise ValueError("val_frac must satisfy 0 <= val_frac < 1")
        neg_ratio = finite_real(neg_ratio, "neg_ratio")
        seed = seed_value(seed, "seed")
        split_seed = seed if split_seed is None else seed_value(split_seed, "split_seed")
        effective_batch_size = _validate_cell_batch_size(cell_batch_size, adata.n_obs)
        loader = resolve_loader_settings(device, effective_batch_size, loader_workers,
                                         loader_prefetch_batches, loader_pin_memory)
        prepared = PreparedSceneData.from_adata(
            adata, layer=layer, batch_keys=batch_keys, validate=validate, val_frac=val_frac,
            validation_seed=split_seed, neg_ratio=neg_ratio, validation_mode=validation_mode,
        )
        # Validate serializable categories before starting an expensive fit.
        from ._persistence import json_ready
        categories = json_ready(prepared.batch_categories)
        model_config = self._model_config()
        set_seed(seed)
        decoder = _SCENEDecoder(
            prepared.n_cells, prepared.n_genes, **model_config,
            batch_size=prepared.batch_size, counts=prepared.train_matrix, device=device,
            cell_init=cell_init, gene_init=gene_init,
        )
        training = dict(epochs=epochs, lr=lr, seed=seed, validate=validate,
                        val_interval=val_interval, print_every=print_every,
                        cell_batch_size=cell_batch_size, loader_workers=loader_workers,
                        loader_prefetch_batches=loader_prefetch_batches,
                        loader_pin_memory=loader_pin_memory,
                        validation_edge_batch_size=validation_edge_batch_size, weight_decay=weight_decay)
        training_history = _train_prepared(decoder, prepared, **training)
        decoder.eval()
        self._decoder = decoder
        self.cell_names, self.gene_names = cell_names.copy(), gene_names.copy()
        self.batch_size = prepared.batch_size
        self.batch_categories = categories
        self._batch_codes = {name: codes.copy() for name, codes in zip(prepared.batch_keys, prepared.batch.ids)}
        self.training_history = training_history
        self.config = json_ready({
            "model": model_config,
            "fit": {**training, "cell_init": cell_init, "gene_init": gene_init,
                    "layer": layer, "batch_keys": list(prepared.batch_keys),
                    "split_seed": split_seed, "key": key, "val_frac": val_frac,
                    "validation_mode": validation_mode, "neg_ratio": neg_ratio,
                    "write_back": write_back, "device": str(torch.device(device))},
            "effective": {"cell_batch_size": effective_batch_size, "loader": asdict(loader),
                          "sparse_extraction_backend": sparse_extraction_backend_name()
                          if effective_batch_size is not None else None},
        })
        self._package_version = version("scene-ldm")
        if write_back:
            from ._persistence import write_adata
            write_adata(self, adata, key)
        return self

    def _require_fitted(self):
        if self._decoder is None:
            raise RuntimeError("SCENE is not fitted; call fit(adata) or SCENE.load(path) first")

    def _predict_pairs(self, cells, genes, method, use_random_effects, use_batch_effects):
        self._require_fitted()
        for name, value in (("use_random_effects", use_random_effects), ("use_batch_effects", use_batch_effects)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        cell_names, gene_names, rows, cols = resolve_pairs(cells, genes, self.cell_names, self.gene_names)
        decoder = self._decoder
        batch_ids = [torch.as_tensor(self._batch_codes[key], device=decoder.device, dtype=torch.long)
                     for key in decoder.batch_levels]
        chunks = []
        with torch.no_grad():
            for start in range(0, len(rows), 100_000):
                r = torch.as_tensor(rows[start:start + 100_000], dtype=torch.long, device=decoder.device)
                c = torch.as_tensor(cols[start:start + 100_000], dtype=torch.long, device=decoder.device)
                chunks.append(getattr(decoder, method)(
                    r, c, batch_ids_per_level=batch_ids, use_random_effects=use_random_effects,
                    use_batch_effects=use_batch_effects,
                ).cpu().numpy())
        values = np.concatenate(chunks) if chunks else np.empty(0)
        column = "expected_count" if method == "expected_counts" else "linear_predictor"
        return pd.DataFrame({"cell": cell_names, "gene": gene_names, column: values})

    def expected_counts(self, *, cells, genes, use_random_effects=True, use_batch_effects=True):
        """Predict expectations for named fitted cell–gene pairs.

        Supply cell and gene names as strings or equal-length sequences.
        Two sequences are paired by position; a single string is paired with
        every name in the other sequence. Empty pairs are allowed.
        use_random_effects and use_batch_effects (both True) include fitted
        terms; batch categories are resolved automatically. Returns a DataFrame
        with cell, gene, expected_count columns in request order. Chunked at
        100,000 pairs, with no gradients or implicit full-matrix allocation.
        Poisson expectation is lambda; ZIP is pi*lambda/(1-exp(-lambda)).
        Unknown names are rejected; this does not embed new cells or genes.
        """
        return self._predict_pairs(cells, genes, "expected_counts", use_random_effects, use_batch_effects)

    def linear_predictor_pairs(self, *, cells, genes, use_random_effects=True, use_batch_effects=True):
        """Return the linear predictor for each requested cell–gene pair.

        cells, genes and both effect switches have the same meanings and
        defaults as expected_counts. Returns cell, gene, linear_predictor
        columns. The predictor is negative Euclidean distance plus enabled
        fitted random and batch terms, using all latent dimensions.
        """
        return self._predict_pairs(cells, genes, "linear_predictor_pairs", use_random_effects, use_batch_effects)

    def save(self, path, *, overwrite=False):
        """Save metadata.json and weights.pt to path; return None.

        path is a directory path (str or pathlib.Path). Existing checkpoints
        require overwrite=True; unrelated directories are never overwritten.
        Saves exact fitted tensors, identifiers, batch encoding, training history, and
        configuration. Counts and optimizer state are not saved.
        """
        self._require_fitted()
        from ._persistence import save_model
        save_model(self, Path(path), overwrite=overwrite)

    @classmethod
    def load(cls, path, *, device="cpu"):
        """Load an inference-ready SCENE from a checkpoint directory.

        path is a string or pathlib.Path pointing to a directory created by
        save. device (default CPU) selects tensor placement; tensor dtype is
        preserved. Restores fitted parameters for prediction. Counts and
        optimizer state are not restored; calling fit starts a new fit.
        Uses weights-only loading and rejects incompatible checkpoint state.
        """
        from ._persistence import load_model
        return load_model(cls, Path(path), device=device)
