"""Prepared sparse data and deterministic cell-block batching for SCENE."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader, Dataset, Sampler

try:  # Optional acceleration; the NumPy path is always available.
    from numba import njit
except Exception:  # pragma: no cover - depends on the environment
    njit = None


if njit is not None:
    @njit
    def _collect_csr_rows_numba(indptr, indices, values, row_ids, n_cols):
        total = 0
        for row_id in row_ids:
            total += indptr[row_id + 1] - indptr[row_id]

        flat_indices = np.empty(total, dtype=np.int64)
        flat_values = np.empty(total, dtype=np.float32)
        cursor = 0
        for block_row in range(row_ids.size):
            row_id = row_ids[block_row]
            start = indptr[row_id]
            end = indptr[row_id + 1]
            for offset in range(start, end):
                flat_indices[cursor] = block_row * n_cols + indices[offset]
                flat_values[cursor] = values[offset]
                cursor += 1
        return flat_indices, flat_values


    @njit
    def _collect_row_indexed_numba(indptr, cols, row_ids, n_cols):
        total = 0
        for row_id in row_ids:
            total += indptr[row_id + 1] - indptr[row_id]

        flat_indices = np.empty(total, dtype=np.int64)
        cursor = 0
        for block_row in range(row_ids.size):
            row_id = row_ids[block_row]
            start = indptr[row_id]
            end = indptr[row_id + 1]
            for offset in range(start, end):
                flat_indices[cursor] = block_row * n_cols + cols[offset]
                cursor += 1
        return flat_indices


def sparse_extraction_backend_name() -> str:
    """Return the active sparse cell-block extraction backend."""

    return "numba" if njit is not None else "numpy"


def _canonicalize_csr(matrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=True)
    else:
        matrix = sparse.csr_matrix(np.asarray(matrix))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _get_csr_from_adata(adata, layer: Optional[str] = "counts") -> sparse.csr_matrix:
    if layer is None:
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer {layer!r} is not present in adata.layers. "
                "Pass layer=None to use counts stored in adata.X."
            )
        matrix = adata.layers[layer]
    return _canonicalize_csr(matrix)


def _validate_integer_input_matrix(matrix: sparse.csr_matrix, layer) -> None:
    data = matrix.data
    if data.size == 0:
        return
    if not np.isfinite(data).all():
        raise ValueError(
            "Input matrix contains non-finite values (NaN/Inf). "
            "Please provide a finite integer count matrix."
        )

    if (data < 0).any():
        raise ValueError("Input matrix contains negative values. Provide non-negative counts.")

    rounded = np.round(data)
    is_integer = np.isclose(data, rounded, rtol=0.0, atol=1e-8)
    if is_integer.all():
        return

    bad_idx = np.flatnonzero(~is_integer)
    max_abs_diff = float(np.max(np.abs(data[bad_idx] - rounded[bad_idx])))
    source_name = "adata.X" if layer is None else f"layer='{layer}'"
    warnings.warn(
        "SCENE expects integer-valued counts. "
        f"Detected {bad_idx.size} non-integer entries in {source_name}; "
        f"max |x-round(x)|={max_abs_diff:.3e}. "
        "Training will use these values without rounding. Select a raw-count layer if the input was normalized or transformed.",
        UserWarning,
        stacklevel=2,
    )


def _mask_random_observed_entries(
    matrix: sparse.csr_matrix,
    frac: float = 0.10,
    seed: int = 0,
):
    if not 0.0 <= frac < 1.0:
        raise ValueError("val_frac must satisfy 0 <= val_frac < 1")

    matrix = _canonicalize_csr(matrix)
    coo = matrix.tocoo()
    rng = np.random.default_rng(seed)
    order = np.arange(coo.data.size, dtype=np.int64)
    rng.shuffle(order)
    n_holdout = int(frac * order.size)
    hold = order[:n_holdout]
    keep = order[n_holdout:]

    train = sparse.coo_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])),
        shape=matrix.shape,
    ).tocsr()
    train.sum_duplicates()
    train.eliminate_zeros()
    train.sort_indices()
    return train, (coo.data[hold], coo.row[hold], coo.col[hold])


def _sample_negatives(
    matrix: sparse.csr_matrix,
    n_neg: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample uniformly without replacement from zero entries."""

    matrix = _canonicalize_csr(matrix)
    n_rows, n_cols = matrix.shape
    total = int(n_rows) * int(n_cols)
    available = total - int(matrix.nnz)
    n_neg = min(max(int(n_neg), 0), available)
    if n_neg == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    # CSR entries are sorted by flattened row-major index.  Removing the
    # number of observed entries before each zero maps zero-ranks back to flat
    # matrix indices without enumerating the full matrix.
    positive_flat = np.repeat(
        np.arange(n_rows, dtype=np.int64),
        np.diff(matrix.indptr),
    )
    positive_flat *= n_cols
    positive_flat += matrix.indices.astype(np.int64, copy=False)
    positive_flat -= np.arange(matrix.nnz, dtype=np.int64)

    rng = np.random.default_rng(seed)
    zero_ranks = rng.choice(available, size=n_neg, replace=False, shuffle=False)
    flat_indices = zero_ranks + np.searchsorted(
        positive_flat,
        zero_ranks,
        side="right",
    )
    return flat_indices // n_cols, flat_indices % n_cols


def _collect_csr_rows_numpy(
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    row_ids: np.ndarray,
    n_cols: int,
) -> Tuple[np.ndarray, np.ndarray]:
    lengths = indptr[row_ids + 1] - indptr[row_ids]
    total = int(lengths.sum())
    if total == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    flat_rows = np.repeat(np.arange(row_ids.size, dtype=np.int64), lengths)
    flat_cols = np.concatenate(
        [indices[indptr[row]:indptr[row + 1]] for row in row_ids]
    ).astype(np.int64, copy=False)
    flat_indices = flat_rows * n_cols + flat_cols
    flat_values = np.concatenate(
        [values[indptr[row]:indptr[row + 1]] for row in row_ids]
    ).astype(np.float32, copy=False)
    return flat_indices, flat_values


def _collect_csr_rows(
    matrix: sparse.csr_matrix,
    row_ids: np.ndarray,
    use_numba: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if use_numba and njit is not None:
        return _collect_csr_rows_numba(
            matrix.indptr,
            matrix.indices,
            matrix.data,
            row_ids,
            matrix.shape[1],
        )
    return _collect_csr_rows_numpy(
        matrix.indptr,
        matrix.indices,
        matrix.data,
        row_ids,
        matrix.shape[1],
    )


def _collect_row_indexed_numpy(
    indptr: np.ndarray,
    cols: np.ndarray,
    row_ids: np.ndarray,
    n_cols: int,
) -> np.ndarray:
    lengths = indptr[row_ids + 1] - indptr[row_ids]
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    flat_rows = np.repeat(np.arange(row_ids.size, dtype=np.int64), lengths)
    flat_cols = np.concatenate(
        [cols[indptr[row]:indptr[row + 1]] for row in row_ids]
    ).astype(np.int64, copy=False)
    return flat_rows * n_cols + flat_cols


def _collect_row_indexed(
    indptr: np.ndarray,
    cols: np.ndarray,
    row_ids: np.ndarray,
    n_cols: int,
    use_numba: bool = True,
) -> np.ndarray:
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if use_numba and njit is not None:
        return _collect_row_indexed_numba(indptr, cols, row_ids, n_cols)
    return _collect_row_indexed_numpy(indptr, cols, row_ids, n_cols)


def _build_row_indexed_entries(
    entry_rows: np.ndarray,
    entry_cols: np.ndarray,
    n_rows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    for row in entry_rows:
        indptr[int(row) + 1] += 1
    np.cumsum(indptr, out=indptr)
    cols = np.empty(entry_cols.size, dtype=np.int64)
    cursors = indptr[:-1].copy()
    for row, col in zip(entry_rows, entry_cols):
        row = int(row)
        cols[cursors[row]] = int(col)
        cursors[row] += 1
    return indptr, cols


def _normalize_batch_keys(batch_keys) -> List[str]:
    if batch_keys is None:
        return []
    if isinstance(batch_keys, str):
        return [batch_keys]
    try:
        return list(batch_keys)
    except TypeError as error:
        raise ValueError("batch_keys must be a string or sequence of strings") from error


@dataclass
class BatchMetadata:
    keys: Tuple[str, ...]
    batch_size: Dict[str, int]
    ids: Tuple[np.ndarray, ...]
    categories: Dict[str, List[Any]]


def build_batch_metadata(
    adata,
    batch_keys=None,
    expected_categories: Optional[Dict[str, Sequence[Any]]] = None,
) -> BatchMetadata:
    keys = _normalize_batch_keys(batch_keys)
    if any(not isinstance(key, str) or not key or "/" in key for key in keys):
        raise ValueError("batch_keys must contain non-empty strings without '/'")
    if len(set(keys)) != len(keys):
        raise ValueError("batch_keys must be unique")
    batch_size: Dict[str, int] = {}
    ids: List[np.ndarray] = []
    categories: Dict[str, List[Any]] = {}
    expected_categories = expected_categories or {}

    for key in keys:
        if key not in adata.obs.columns:
            raise KeyError(f"Batch key '{key}' not found in adata.obs")

        series = adata.obs[key]
        if series.dtype.name != "category":
            series = series.astype("category")

        expected = expected_categories.get(key)
        if expected is not None:
            series = series.cat.set_categories(list(expected))
        else:
            series = series.cat.remove_unused_categories()

        codes = series.cat.codes.to_numpy(dtype=np.int64)
        if np.any(codes < 0):
            raise ValueError(
                f"Batch key '{key}' contains missing or unknown categories"
            )
        category_values = series.cat.categories.tolist()
        batch_size[key] = len(category_values)
        ids.append(codes)
        categories[key] = category_values

    return BatchMetadata(
        keys=tuple(keys),
        batch_size=batch_size,
        ids=tuple(ids),
        categories=categories,
    )


@dataclass
class PreparedSceneData:
    """Immutable-ish CPU-side representation of one SCENE training run."""

    train_matrix: sparse.csr_matrix
    n_cells: int
    n_genes: int
    batch: BatchMetadata
    validation_mode: str
    holdout: Optional[Dict[str, np.ndarray]]
    missing_indices: Optional[np.ndarray]
    missing_indptr: Optional[np.ndarray]
    missing_cols: Optional[np.ndarray]
    observed_total: int
    trainable_global_cells: np.ndarray

    @classmethod
    def from_adata(
        cls,
        adata,
        layer: Optional[str] = "counts",
        batch_keys=None,
        validate: bool = False,
        val_frac: float = 0.10,
        validation_seed: int = 0,
        neg_ratio: float = 1.0,
        validation_mode: str = "flipped",
        expected_categories: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> "PreparedSceneData":
        if not isinstance(validation_mode, str) or validation_mode not in {"flipped", "masked"}:
            raise ValueError("validation_mode must be one of {'flipped', 'masked'}")
        if neg_ratio < 0:
            raise ValueError("neg_ratio must be non-negative")

        if not 0 <= val_frac < 1:
            raise ValueError("val_frac must satisfy 0 <= val_frac < 1")
        if not np.isfinite(neg_ratio):
            raise ValueError("neg_ratio must be finite")
        matrix = _get_csr_from_adata(adata, layer=layer)
        _validate_integer_input_matrix(matrix, layer=layer)
        n_cells, n_genes = matrix.shape
        if n_cells != adata.n_obs or n_genes != adata.n_vars:
            raise ValueError("Input matrix shape must match AnnData observations and variables")

        batch = build_batch_metadata(
            adata,
            batch_keys=batch_keys,
            expected_categories=expected_categories,
        )

        train_matrix = matrix
        holdout = None
        missing_indices = None
        missing_indptr = None
        missing_cols = None
        observed_total = n_cells * n_genes

        if validate and val_frac > 0:
            train_matrix, (holdout_values, holdout_rows, holdout_cols) = (
                _mask_random_observed_entries(
                    matrix,
                    frac=val_frac,
                    seed=validation_seed,
                )
            )
            if holdout_values.size:
                negative_rows, negative_cols = _sample_negatives(
                    matrix,
                    n_neg=int(math.ceil(holdout_values.size * neg_ratio)),
                    seed=validation_seed + 1,
                )
                holdout = {
                    "values": holdout_values.astype(np.float32, copy=False),
                    "pos_rows": holdout_rows.astype(np.int64, copy=False),
                    "pos_cols": holdout_cols.astype(np.int64, copy=False),
                    "neg_rows": negative_rows.astype(np.int64, copy=False),
                    "neg_cols": negative_cols.astype(np.int64, copy=False),
                }
                if validation_mode == "masked":
                    missing_indices = np.vstack(
                        [holdout_rows.astype(np.int64), holdout_cols.astype(np.int64)]
                    )
                    missing_indptr, missing_cols = _build_row_indexed_entries(
                        holdout_rows,
                        holdout_cols,
                        n_rows=n_cells,
                    )
                    observed_total -= int(holdout_values.size)

        return cls(
            train_matrix=train_matrix,
            n_cells=n_cells,
            n_genes=n_genes,
            batch=batch,
            validation_mode=validation_mode,
            holdout=holdout,
            missing_indices=missing_indices,
            missing_indptr=missing_indptr,
            missing_cols=missing_cols,
            observed_total=observed_total,
            trainable_global_cells=np.arange(n_cells, dtype=np.int64),
        )

    @property
    def batch_keys(self) -> Tuple[str, ...]:
        return self.batch.keys

    @property
    def batch_size(self) -> Dict[str, int]:
        return self.batch.batch_size

    @property
    def batch_categories(self) -> Dict[str, List[Any]]:
        return self.batch.categories

    def batch_id_tensors(self, device) -> List[torch.Tensor]:
        return [
            torch.from_numpy(ids).to(device=device, dtype=torch.long)
            for ids in self.batch.ids
        ]

    def full_batch_tensors(self, device):
        matrix = self.train_matrix.tocoo()
        values = torch.as_tensor(matrix.data, dtype=torch.float32, device=device)
        indices = torch.as_tensor(
            np.vstack([matrix.row, matrix.col]),
            dtype=torch.long,
            device=device,
        )
        missing = None
        if self.missing_indices is not None:
            missing = torch.as_tensor(
                self.missing_indices,
                dtype=torch.long,
                device=device,
            )
        return values, indices, missing


@dataclass
class CellBlock:
    cells: torch.Tensor
    positive_indices: torch.Tensor
    positive_values: torch.Tensor
    missing_indices: Optional[torch.Tensor]
    block_weight: float

    def pin_memory(self):
        self.cells = self.cells.pin_memory()
        self.positive_indices = self.positive_indices.pin_memory()
        self.positive_values = self.positive_values.pin_memory()
        if self.missing_indices is not None:
            self.missing_indices = self.missing_indices.pin_memory()
        return self

    def to(self, device, non_blocking: bool = False):
        self.cells = self.cells.to(device=device, non_blocking=non_blocking)
        self.positive_indices = self.positive_indices.to(
            device=device,
            non_blocking=non_blocking,
        )
        self.positive_values = self.positive_values.to(
            device=device,
            non_blocking=non_blocking,
        )
        if self.missing_indices is not None:
            self.missing_indices = self.missing_indices.to(
                device=device,
                non_blocking=non_blocking,
            )
        return self


def _identity(value):
    return value


class SparseCellBlockDataset(Dataset):
    """Construct one sparse cell block for a supplied global cell array."""

    def __init__(self, prepared: PreparedSceneData):
        self.prepared = prepared

    def __len__(self) -> int:
        return int(self.prepared.trainable_global_cells.size)

    def __getitem__(self, global_block) -> CellBlock:
        if torch.is_tensor(global_block):
            global_block = global_block.detach().cpu().numpy()
        global_block = np.asarray(global_block, dtype=np.int64)
        if global_block.ndim == 0:
            global_block = global_block.reshape(1)
        if global_block.size == 0:
            raise ValueError("Cell blocks must contain at least one cell")

        flat_indices, values = _collect_csr_rows(
            self.prepared.train_matrix,
            global_block,
        )
        positive_indices = np.empty((2, flat_indices.size), dtype=np.int64)
        positive_indices[0] = flat_indices // self.prepared.n_genes
        positive_indices[1] = flat_indices % self.prepared.n_genes

        missing_indices = None
        if self.prepared.missing_indptr is not None:
            missing_flat = _collect_row_indexed(
                self.prepared.missing_indptr,
                self.prepared.missing_cols,
                global_block,
                self.prepared.n_genes,
            )
            missing_indices = np.empty((2, missing_flat.size), dtype=np.int64)
            missing_indices[0] = missing_flat // self.prepared.n_genes
            missing_indices[1] = missing_flat % self.prepared.n_genes

        return CellBlock(
            cells=torch.from_numpy(global_block.copy()),
            positive_indices=torch.from_numpy(positive_indices),
            positive_values=torch.from_numpy(values),
            missing_indices=(
                None
                if missing_indices is None
                else torch.from_numpy(missing_indices)
            ),
            block_weight=float(self.prepared.n_cells) / float(global_block.size),
        )


class CellBlockSampler(Sampler):
    """Yield deterministic shuffled cell blocks for each training epoch."""

    def __init__(self, cells: np.ndarray, batch_size: int, seed: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.cells = np.asarray(cells, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[np.ndarray]:
        order = np.random.default_rng(self.seed + self.epoch).permutation(self.cells)
        for start in range(0, order.size, self.batch_size):
            yield order[start:start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(self.cells.size / self.batch_size)


@dataclass(frozen=True)
class LoaderSettings:
    requested_workers: Optional[int]
    effective_workers: int
    prefetch_batches: int
    requested_pin_memory: Optional[bool]
    effective_pin_memory: bool


def resolve_loader_settings(
    device,
    cell_batch_size: Optional[int],
    loader_workers: Optional[int],
    loader_prefetch_batches: int,
    loader_pin_memory: Optional[bool],
) -> LoaderSettings:
    device_type = torch.device(device).type
    if loader_workers is not None and (
        isinstance(loader_workers, bool)
        or not isinstance(loader_workers, (int, np.integer))
        or loader_workers < 0
    ):
        raise ValueError("loader_workers must be a non-negative integer or None")
    if (
        isinstance(loader_prefetch_batches, bool)
        or not isinstance(loader_prefetch_batches, (int, np.integer))
        or loader_prefetch_batches <= 0
    ):
        raise ValueError("loader_prefetch_batches must be a positive integer")
    if loader_pin_memory is not None and not isinstance(loader_pin_memory, bool):
        raise ValueError("loader_pin_memory must be a boolean or None")

    effective_workers = 0 if cell_batch_size is None or loader_workers is None else int(loader_workers)
    requested_pin = (
        device_type == "cuda" if loader_pin_memory is None else loader_pin_memory
    )
    effective_pin = bool(cell_batch_size is not None and requested_pin and device_type == "cuda")
    return LoaderSettings(
        requested_workers=loader_workers,
        effective_workers=effective_workers,
        prefetch_batches=int(loader_prefetch_batches),
        requested_pin_memory=loader_pin_memory,
        effective_pin_memory=effective_pin,
    )


def build_cell_block_loader(
    prepared: PreparedSceneData,
    cell_batch_size: int,
    seed: int,
    settings: LoaderSettings,
):
    sampler = CellBlockSampler(
        prepared.trainable_global_cells,
        batch_size=cell_batch_size,
        seed=seed,
    )
    kwargs = {
        "dataset": SparseCellBlockDataset(prepared),
        "batch_size": None,
        "sampler": sampler,
        "collate_fn": _identity,
        "num_workers": settings.effective_workers,
        "pin_memory": settings.effective_pin_memory,
    }
    if settings.effective_workers > 0:
        kwargs["prefetch_factor"] = settings.prefetch_batches
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs), sampler


__all__ = [
    "BatchMetadata",
    "CellBlock",
    "CellBlockSampler",
    "LoaderSettings",
    "PreparedSceneData",
    "SparseCellBlockDataset",
    "build_batch_metadata",
    "build_cell_block_loader",
    "resolve_loader_settings",
    "sparse_extraction_backend_name",
]
