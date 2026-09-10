"""Training and AnnData integration for unimodal SCENE."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import auc, mean_poisson_deviance, precision_recall_curve, roc_auc_score

from .data import (
    PreparedSceneData,
    build_cell_block_loader,
    resolve_loader_settings,
)
from ._decoder import _SCENEDecoder

__all__ = []


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_cell_batch_size(cell_batch_size, n_cells: int) -> Optional[int]:
    if cell_batch_size is None:
        return None
    if isinstance(cell_batch_size, bool) or not isinstance(
        cell_batch_size,
        (int, np.integer),
    ):
        raise ValueError("cell_batch_size must be a positive integer or None")
    cell_batch_size = int(cell_batch_size)
    if cell_batch_size <= 0:
        raise ValueError("cell_batch_size must be a positive integer or None")
    return min(cell_batch_size, n_cells)


def _build_optimizer(model: _SCENEDecoder, lr: float, weight_decay: float):
    no_decay_ids = {
        id(model.re_cells),
        id(model.re_genes),
        id(model.raw_alpha),
    }
    no_decay = []
    decay = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in no_decay_ids:
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    parameter_groups = []
    if no_decay:
        parameter_groups.append({"params": no_decay, "weight_decay": 0.0})
    if decay:
        parameter_groups.append({"params": decay, "weight_decay": weight_decay})
    if not parameter_groups:
        raise ValueError("SCENE has no trainable parameters")
    return torch.optim.AdamW(parameter_groups, lr=lr)


def _link_score(model: _SCENEDecoder, lam: torch.Tensor, pi: Optional[torch.Tensor]) -> torch.Tensor:
    if model.likelihood == "ZIP":
        if pi is None:
            raise ValueError("ZIP validation requires pi predictions")
        return pi
    return lam


@torch.no_grad()
def _evaluate_holdout_edges(
    model: _SCENEDecoder,
    batch_ids_per_level,
    holdout: Dict[str, np.ndarray],
    edge_batch_size: int,
) -> Dict[str, float]:
    n_pos = int(holdout["pos_rows"].size)
    n_neg = int(holdout["neg_rows"].size)
    if n_pos == 0 or n_neg == 0:
        return {
            "AUC": float("nan"),
            "PR AUC": float("nan"),
            "max F1": float("nan"),
            "PoissonDeviance": float("nan"),
        }

    pos_scores = np.empty(n_pos, dtype=np.float64)
    neg_scores = np.empty(n_neg, dtype=np.float64)
    deviance_sum = 0.0

    for start in range(0, n_pos, edge_batch_size):
        end = min(start + edge_batch_size, n_pos)
        rows = torch.as_tensor(holdout["pos_rows"][start:end], dtype=torch.long, device=model.device)
        cols = torch.as_tensor(holdout["pos_cols"][start:end], dtype=torch.long, device=model.device)
        lam, pi = model.forward_edges(rows, cols, batch_ids_per_level=batch_ids_per_level)
        scores = _link_score(model, lam, pi).detach().cpu().numpy()
        pos_scores[start:end] = scores

        lam64 = lam.detach().cpu().numpy().astype(np.float64, copy=False)
        observed = holdout["values"][start:end].astype(np.float64, copy=False)
        predicted = lam64 / (-np.expm1(-lam64))
        deviance_sum += float(mean_poisson_deviance(observed, predicted) * (end - start))

    for start in range(0, n_neg, edge_batch_size):
        end = min(start + edge_batch_size, n_neg)
        rows = torch.as_tensor(holdout["neg_rows"][start:end], dtype=torch.long, device=model.device)
        cols = torch.as_tensor(holdout["neg_cols"][start:end], dtype=torch.long, device=model.device)
        lam, pi = model.forward_edges(rows, cols, batch_ids_per_level=batch_ids_per_level)
        neg_scores[start:end] = _link_score(model, lam, pi).detach().cpu().numpy()

    y_true = np.concatenate([
        np.ones(n_pos, dtype=np.int8),
        np.zeros(n_neg, dtype=np.int8),
    ])
    y_score = np.concatenate([pos_scores, neg_scores])
    auc_roc = float(roc_auc_score(y_true, y_score))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc(recall, precision))
    f1_values = 2.0 * precision * recall / (precision + recall + 1e-12)

    return {
        "AUC": auc_roc,
        "PR AUC": pr_auc,
        "max F1": float(np.max(f1_values)),
        "PoissonDeviance": deviance_sum / n_pos,
    }


def _train_prepared(
    model: _SCENEDecoder,
    prepared: PreparedSceneData,
    *,
    epochs: int,
    lr: float,
    seed: int,
    validate: bool,
    val_interval: int,
    print_every: Optional[int],
    cell_batch_size: Optional[int],
    loader_workers: Optional[int],
    loader_prefetch_batches: int,
    loader_pin_memory: Optional[bool],
    validation_edge_batch_size: int,
    weight_decay: float,
) -> Dict[str, Any]:
    device = model.device
    batch_ids_per_level = prepared.batch_id_tensors(device)
    cell_batch_size = _validate_cell_batch_size(cell_batch_size, prepared.n_cells)
    loader_settings = resolve_loader_settings(
        device=device,
        cell_batch_size=cell_batch_size,
        loader_workers=loader_workers,
        loader_prefetch_batches=loader_prefetch_batches,
        loader_pin_memory=loader_pin_memory,
    )

    opt = _build_optimizer(model, lr=lr, weight_decay=weight_decay)
    full_batch = None
    cell_loader = None
    cell_sampler = None
    if cell_batch_size is None:
        full_batch = prepared.full_batch_tensors(device)
    else:
        cell_loader, cell_sampler = build_cell_block_loader(
            prepared,
            cell_batch_size=cell_batch_size,
            seed=seed,
            settings=loader_settings,
        )

    training_history: Dict[str, Any] = {
        "train": {
            "epoch": [],
            "loss": [],
            "optimizer_updates": [],
        },
        "validation": {name: [] for name in ("epoch", "AUC", "PR AUC", "max F1", "PoissonDeviance")},
    }

    optimizer_updates = 0
    for epoch in range(epochs):
        model.train()
        batch_losses = []

        if cell_batch_size is None:
            values, indices, missing_indices = full_batch
            opt.zero_grad(set_to_none=True)
            lam, pi = model(batch_ids_per_level=batch_ids_per_level)
            loss, mean_loss = model.compute_loss(
                lam,
                pi,
                count_values=values,
                count_indices=indices,
                missing_indices=missing_indices,
            )
            loss.backward()
            opt.step()
            optimizer_updates += 1
        else:
            cell_sampler.set_epoch(epoch)
            for block in cell_loader:
                block.to(device=device, non_blocking=loader_settings.effective_pin_memory)
                opt.zero_grad(set_to_none=True)
                lam, pi = model.forward_cell_block(
                    block.cells,
                    batch_ids_per_level=batch_ids_per_level,
                )
                loss, mean_loss = model.compute_cell_block_loss(
                    lam,
                    pi,
                    count_values=block.positive_values,
                    count_indices=block.positive_indices,
                    n_total=prepared.observed_total,
                    block_weight=block.block_weight,
                    missing_indices=block.missing_indices,
                )
                loss.backward()
                opt.step()
                optimizer_updates += 1
                batch_losses.append(mean_loss.detach())

            if not batch_losses:
                raise ValueError("Cell batching produced no trainable blocks")
            mean_loss = torch.stack(batch_losses).mean()

        training_history["train"]["epoch"].append(epoch + 1)
        training_history["train"]["loss"].append(float(mean_loss.detach().cpu()))
        training_history["train"]["optimizer_updates"].append(optimizer_updates)

        should_validate = (
            validate
            and prepared.holdout is not None
            and (epoch == 0 or (epoch + 1) % val_interval == 0 or epoch == epochs - 1)
        )
        validation_metrics = None
        if should_validate:
            model.eval()
            validation_metrics = _evaluate_holdout_edges(
                model,
                batch_ids_per_level=batch_ids_per_level,
                holdout=prepared.holdout,
                edge_batch_size=validation_edge_batch_size,
            )
            for name, value in {"epoch": epoch + 1, **validation_metrics}.items():
                training_history["validation"][name].append(value)

        if print_every is not None and (epoch == 0 or (epoch + 1) % print_every == 0 or epoch == epochs - 1):
            message = (
                f"[{epoch + 1:04d}/{epochs}] "
                f"loss={float(mean_loss.detach().cpu()):.4f} "
                f"likelihood={model.likelihood}"
            )
            if validation_metrics is not None:
                message += (
                    f" AUC={validation_metrics['AUC']:.3f}"
                    f" PR AUC={validation_metrics['PR AUC']:.3f}"
                    f" max F1={validation_metrics['max F1']:.3f}"
                    f" PoissonDeviance={validation_metrics['PoissonDeviance']:.3f}"
                )
            print(message)

    return training_history
