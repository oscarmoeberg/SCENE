"""Unimodal Euclidean _SCENEDecoder model."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .initialization import laplacian_init

__all__ = []


def _normalize_likelihood(likelihood: str) -> str:
    normalized = str(likelihood).lower()
    if normalized == "poisson":
        return "Poisson"
    if normalized == "zip":
        return "ZIP"
    raise ValueError("likelihood must be one of {'Poisson', 'ZIP'}")


class _SCENEDecoder(nn.Module):
    """Internal single-modality Euclidean SCENE decoder.

    The model represents cells and genes as points in a shared Euclidean
    space.  Random effects and optional hierarchical batch effects modify the
    geometric linear predictor before it is decoded with either a Poisson or
    ZIP likelihood with a positive-count gate and zero-truncated Poisson counts.
    """

    def __init__(
        self,
        num_cells: int,
        num_genes: int,
        latent_dim: int,
        likelihood: str = "ZIP",
        batch_size: Optional[Dict[str, int]] = None,
        batch_effect_type: Union[str, Sequence[str]] = "lowrank",
        batch_rank: int = 2,
        cell_init: str = "random",
        gene_init: str = "random",
        counts=None,
        use_random_effects: bool = True,
        device: Union[str, torch.device] = "cpu",
        _initialize: bool = True,
    ):
        super().__init__()

        if num_cells <= 0 or num_genes <= 0 or latent_dim <= 0:
            raise ValueError("num_cells, num_genes, and latent_dim must be positive")
        if batch_rank <= 0:
            raise ValueError("batch_rank must be positive")

        self.num_cells = int(num_cells)
        self.num_genes = int(num_genes)
        self.latent_dim = int(latent_dim)
        self.likelihood = _normalize_likelihood(likelihood)
        self.use_random_effects = bool(use_random_effects)
        self.batch_rank = int(batch_rank)
        self.batch_effect_type = batch_effect_type
        self.cell_init = cell_init
        self.gene_init = gene_init
        self.batch_size = dict(batch_size or {})
        self.batch_levels = list(self.batch_size.keys())
        self._initial_device = torch.device(device)
        dtype = torch.get_default_dtype()

        if isinstance(batch_effect_type, (list, tuple)):
            if len(batch_effect_type) != len(self.batch_levels):
                raise ValueError(
                    f"batch_effect_type list length ({len(batch_effect_type)}) must match "
                    f"number of batch levels ({len(self.batch_levels)})"
                )
            self.batch_effect_types = list(batch_effect_type)
        else:
            self.batch_effect_types = [batch_effect_type] * len(self.batch_levels)

        for name, value in (("cell_init", cell_init), ("gene_init", gene_init)):
            if value not in {"random", "laplacian"}:
                raise ValueError(f"{name} must be 'random' or 'laplacian'")
        coordinates = None
        if _initialize and "laplacian" in (cell_init, gene_init):
            if counts is None or counts.shape != (num_cells, num_genes):
                raise ValueError("Initialization counts must match model dimensions")
            coordinates = laplacian_init(counts, latent_dim)
        random_coordinates = torch.randn if _initialize else torch.zeros
        self.Z_cells = nn.Parameter(
            torch.as_tensor(coordinates[0], dtype=dtype, device=self._initial_device)
            if coordinates is not None and cell_init == "laplacian" else
            random_coordinates(num_cells, latent_dim, dtype=dtype, device=self._initial_device)
        )
        self.Z_genes = nn.Parameter(
            torch.as_tensor(coordinates[1], dtype=dtype, device=self._initial_device)
            if coordinates is not None and gene_init == "laplacian" else
            random_coordinates(num_genes, latent_dim, dtype=dtype, device=self._initial_device)
        )

        initial_alpha = torch.log(
            torch.expm1(torch.tensor(1.0, dtype=dtype, device=self._initial_device))
        )
        self.raw_alpha = nn.Parameter(
            initial_alpha,
            requires_grad=self.likelihood == "ZIP",
        )

        if self.use_random_effects:
            self.re_cells = nn.Parameter(
                random_coordinates(
                    num_cells,
                    1,
                    dtype=dtype,
                    device=self._initial_device,
                )
                * 0.01
            )
            self.re_genes = nn.Parameter(
                random_coordinates(
                    num_genes,
                    1,
                    dtype=dtype,
                    device=self._initial_device,
                )
                * 0.01
            )
        else:
            self.re_cells = nn.Parameter(
                torch.zeros(
                    num_cells,
                    1,
                    dtype=dtype,
                    device=self._initial_device,
                ),
                requires_grad=False,
            )
            self.re_genes = nn.Parameter(
                torch.zeros(
                    num_genes,
                    1,
                    dtype=dtype,
                    device=self._initial_device,
                ),
                requires_grad=False,
            )

        self.gamma_levels = []
        self.U_levels = nn.ParameterList()
        self.V_levels = nn.ParameterList()

        for level, n_batches in enumerate(self.batch_size.values()):
            if n_batches <= 0:
                raise ValueError("batch levels must contain at least one category")
            effect_type = self.batch_effect_types[level]
            placeholder = nn.Parameter(
                torch.empty(0, dtype=dtype, device=self._initial_device),
                requires_grad=False,
            )

            if effect_type == "full":
                gamma = nn.Parameter(
                    torch.zeros(
                        num_genes,
                        n_batches,
                        dtype=dtype,
                        device=self._initial_device,
                    )
                )
                self.gamma_levels.append(("full", level))
                self.U_levels.append(gamma)
                self.V_levels.append(placeholder)
            elif effect_type == "lowrank":
                U = nn.Parameter(
                    random_coordinates(
                        n_batches,
                        self.batch_rank,
                        dtype=dtype,
                        device=self._initial_device,
                    )
                    * 0.01
                )
                V = nn.Parameter(
                    random_coordinates(
                        num_genes,
                        self.batch_rank,
                        dtype=dtype,
                        device=self._initial_device,
                    )
                    * 0.01
                )
                self.gamma_levels.append(("lowrank", level))
                self.U_levels.append(U)
                self.V_levels.append(V)
            elif effect_type == "none":
                self.gamma_levels.append(("none", None))
                self.U_levels.append(placeholder)
                self.V_levels.append(placeholder)
            else:
                raise ValueError(
                    f"Unknown batch interaction batch_effect_type at level {level}: "
                    f"{effect_type}"
                )

    @property
    def alpha(self) -> torch.Tensor:
        """Positive ZIP gate sensitivity."""

        return F.softplus(self.raw_alpha)

    @property
    def device(self) -> torch.device:
        return self.Z_cells.device

    def _prepare_rows(self, rows: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if rows is None:
            return None
        return rows.to(device=self.device, dtype=torch.long)

    def _prepare_edges(
        self,
        cell_indices: torch.Tensor,
        gene_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_indices = cell_indices.to(device=self.device, dtype=torch.long)
        gene_indices = gene_indices.to(device=self.device, dtype=torch.long)
        if cell_indices.shape != gene_indices.shape:
            raise ValueError("cell_indices and gene_indices must have the same shape")
        return cell_indices, gene_indices

    def build_re_mat(
        self,
        batch_ids_per_level=None,
        rows: Optional[torch.Tensor] = None,
        use_random_effects: bool = True,
        use_batch_effects: bool = True,
    ) -> torch.Tensor:
        """Build random and batch effects for all genes of selected cells."""

        rows = self._prepare_rows(rows)
        n_rows = self.num_cells if rows is None else rows.numel()
        re_mat = torch.zeros(
            (n_rows, self.num_genes),
            dtype=self.Z_cells.dtype,
            device=self.device,
        )

        if use_random_effects and self.use_random_effects:
            cell_effects = self.re_cells if rows is None else self.re_cells[rows]
            re_mat = re_mat + cell_effects + self.re_genes.T

        if use_batch_effects and batch_ids_per_level is not None:
            for level, (tag, index) in enumerate(self.gamma_levels):
                if level >= len(batch_ids_per_level) or tag == "none":
                    continue
                ids = batch_ids_per_level[level].to(device=self.device, dtype=torch.long)
                if rows is not None:
                    ids = ids[rows]

                if tag == "full":
                    re_mat = re_mat + self.U_levels[index][:, ids].T
                elif tag == "lowrank":
                    re_mat = re_mat + self.U_levels[index][ids] @ self.V_levels[index].T
                else:
                    raise ValueError(f"Unknown batch interaction tag: {tag}")

        return re_mat

    def _build_re_values(
        self,
        cell_indices: torch.Tensor,
        gene_indices: torch.Tensor,
        batch_ids_per_level=None,
        use_random_effects: bool = True,
        use_batch_effects: bool = True,
    ) -> torch.Tensor:
        values = torch.zeros(
            cell_indices.shape,
            dtype=self.Z_cells.dtype,
            device=self.device,
        )

        if use_random_effects and self.use_random_effects:
            values = values + self.re_cells[cell_indices, 0] + self.re_genes[gene_indices, 0]

        if use_batch_effects and batch_ids_per_level is not None:
            for level, (tag, index) in enumerate(self.gamma_levels):
                if level >= len(batch_ids_per_level) or tag == "none":
                    continue
                ids = batch_ids_per_level[level].to(device=self.device, dtype=torch.long)
                edge_ids = ids[cell_indices]
                if tag == "full":
                    values = values + self.U_levels[index][gene_indices, edge_ids]
                elif tag == "lowrank":
                    values = values + (
                        self.U_levels[index][edge_ids] * self.V_levels[index][gene_indices]
                    ).sum(dim=1)
                else:
                    raise ValueError(f"Unknown batch interaction tag: {tag}")

        return values

    def _decode_from_diff(
        self,
        diff: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        lam = torch.exp(diff)
        if self.likelihood == "ZIP":
            return lam, torch.sigmoid(self.alpha * diff)
        return lam, None

    def _full_diff(
        self,
        batch_ids_per_level=None,
        use_random_effects: bool = True,
        use_batch_effects: bool = True,
    ) -> torch.Tensor:
        geometry = -torch.cdist(self.Z_cells, self.Z_genes, p=2)
        return geometry + self.build_re_mat(
            batch_ids_per_level=batch_ids_per_level,
            use_random_effects=use_random_effects,
            use_batch_effects=use_batch_effects,
        )

    def forward(
        self,
        batch_ids_per_level=None,
        use_random_effects: bool = True,
        use_batch_effects: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode all cell-gene pairs."""

        return self._decode_from_diff(
            self._full_diff(
                batch_ids_per_level=batch_ids_per_level,
                use_random_effects=use_random_effects,
                use_batch_effects=use_batch_effects,
            )
        )

    def linear_predictor_pairs(
        self,
        cell_indices: torch.Tensor,
        gene_indices: torch.Tensor,
        batch_ids_per_level=None,
        use_random_effects: bool = True,
        use_batch_effects: bool = True,
    ) -> torch.Tensor:
        """Return the linear predictor for requested cell-gene edges."""

        cell_indices, gene_indices = self._prepare_edges(cell_indices, gene_indices)
        geometry = -torch.linalg.vector_norm(
            self.Z_cells[cell_indices] - self.Z_genes[gene_indices],
            dim=1,
        )
        return geometry + self._build_re_values(
            cell_indices,
            gene_indices,
            batch_ids_per_level=batch_ids_per_level,
            use_random_effects=use_random_effects,
            use_batch_effects=use_batch_effects,
        )

    def forward_edges(
        self,
        cell_indices: torch.Tensor,
        gene_indices: torch.Tensor,
        batch_ids_per_level=None,
        use_random_effects: bool = True,
        use_batch_effects: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._decode_from_diff(
            self.linear_predictor_pairs(
                cell_indices,
                gene_indices,
                batch_ids_per_level=batch_ids_per_level,
                use_random_effects=use_random_effects,
                use_batch_effects=use_batch_effects,
            )
        )

    def expected_counts(
        self,
        cell_indices: torch.Tensor,
        gene_indices: torch.Tensor,
        batch_ids_per_level=None,
        use_random_effects: bool = True,
        use_batch_effects: bool = True,
    ) -> torch.Tensor:
        """Return model expectations, not observed counts, for paired edges.

        Poisson expectations equal lambda. For ZIP, they equal
        pi * lambda / (1 - exp(-lambda)), with ratio 1 at lambda=0.
        Batch IDs must follow the fitted category order for each batch level,
        with one ID per fitted cell, as in forward_edges.
        """
        lam, pi = self.forward_edges(
            cell_indices,
            gene_indices,
            batch_ids_per_level=batch_ids_per_level,
            use_random_effects=use_random_effects,
            use_batch_effects=use_batch_effects,
        )
        if pi is None:
            return lam
        denominator = -torch.expm1(-lam)
        denominator = torch.where(lam == 0, torch.ones_like(lam), denominator)
        positive_mean = torch.where(lam == 0, torch.ones_like(lam), lam / denominator)
        return pi * positive_mean

    def forward_cell_block_diff(
        self,
        cell_indices: torch.Tensor,
        batch_ids_per_level=None,
        use_random_effects: bool = True,
        use_batch_effects: bool = True,
    ) -> torch.Tensor:
        """Return the linear predictor for selected cells and all genes."""

        cell_indices = self._prepare_rows(cell_indices)
        geometry = -torch.cdist(self.Z_cells[cell_indices], self.Z_genes, p=2)
        return geometry + self.build_re_mat(
            batch_ids_per_level=batch_ids_per_level,
            rows=cell_indices,
            use_random_effects=use_random_effects,
            use_batch_effects=use_batch_effects,
        )

    def forward_cell_block(
        self,
        cell_indices: torch.Tensor,
        batch_ids_per_level=None,
        use_random_effects: bool = True,
        use_batch_effects: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._decode_from_diff(
            self.forward_cell_block_diff(
                cell_indices,
                batch_ids_per_level=batch_ids_per_level,
                use_random_effects=use_random_effects,
                use_batch_effects=use_batch_effects,
            )
        )

    @staticmethod
    def _select_values(matrix: torch.Tensor, indices: Optional[torch.Tensor]) -> torch.Tensor:
        if indices is None or indices.numel() == 0:
            return matrix.new_empty((0,))
        indices = indices.to(device=matrix.device, dtype=torch.long)
        return matrix[indices[0], indices[1]]

    @staticmethod
    def _observed_count(
        matrix: torch.Tensor,
        missing_indices: Optional[torch.Tensor],
    ) -> int:
        n_missing = 0 if missing_indices is None else int(missing_indices.shape[1])
        n_observed = matrix.numel() - n_missing
        if n_observed <= 0:
            raise ValueError("Missing entries must leave at least one observed matrix entry")
        return n_observed

    def compute_loss(
        self,
        lambda_matrix: torch.Tensor,
        pi_matrix: Optional[torch.Tensor],
        count_values: torch.Tensor,
        count_indices: torch.Tensor,
        missing_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the full-matrix negative log likelihood."""

        if self.likelihood == "ZIP":
            return self._zip_nll(
                lambda_matrix,
                pi_matrix,
                count_values,
                count_indices,
                missing_indices=missing_indices,
            )
        return self._poisson_nll(
            lambda_matrix,
            count_values,
            count_indices,
            missing_indices=missing_indices,
        )

    def compute_cell_block_loss(
        self,
        lambda_block: torch.Tensor,
        pi_block: Optional[torch.Tensor],
        count_values: torch.Tensor,
        count_indices: torch.Tensor,
        n_total: int,
        block_weight: float = 1.0,
        missing_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a weighted cell-block negative log likelihood."""

        if self.likelihood == "ZIP":
            return self._zip_block_nll(
                lambda_block,
                pi_block,
                count_values,
                count_indices,
                n_total=n_total,
                block_weight=block_weight,
                missing_indices=missing_indices,
            )
        return self._poisson_block_nll(
            lambda_block,
            count_values,
            count_indices,
            n_total=n_total,
            block_weight=block_weight,
            missing_indices=missing_indices,
        )

    def _zip_nll(
        self,
        lambda_matrix: torch.Tensor,
        pi_matrix: Optional[torch.Tensor],
        count_values: torch.Tensor,
        count_indices: torch.Tensor,
        missing_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pi_matrix is None:
            raise ValueError("ZIP likelihood requires pi_matrix")
        epsilon = torch.finfo(lambda_matrix.dtype).eps
        count_values = count_values.to(device=lambda_matrix.device, dtype=lambda_matrix.dtype)
        lambda_values = self._select_values(lambda_matrix, count_indices)
        pi_values = self._select_values(pi_matrix, count_indices)

        positive_ll = lambda_matrix.new_tensor(0.0)
        positive_zero_ll = lambda_matrix.new_tensor(0.0)
        if count_values.numel() > 0:
            log_expm1 = lambda_values + torch.log(
                -torch.expm1(-lambda_values) + epsilon
            )
            positive_ll = (
                torch.log(pi_values + epsilon)
                + count_values * torch.log(lambda_values + epsilon)
                - torch.lgamma(count_values + 1.0)
                - log_expm1
            ).sum()
            positive_zero_ll = torch.log(1.0 - pi_values + epsilon).sum()

        zero_ll = torch.log(1.0 - pi_matrix + epsilon).sum() - positive_zero_ll
        if missing_indices is not None and missing_indices.numel() > 0:
            missing_pi = self._select_values(pi_matrix, missing_indices)
            zero_ll = zero_ll - torch.log(1.0 - missing_pi + epsilon).sum()

        nll = -(positive_ll + zero_ll)
        return nll, nll / self._observed_count(lambda_matrix, missing_indices)

    def _zip_block_nll(
        self,
        lambda_block: torch.Tensor,
        pi_block: Optional[torch.Tensor],
        count_values: torch.Tensor,
        count_indices: torch.Tensor,
        n_total: int,
        block_weight: float,
        missing_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pi_block is None:
            raise ValueError("ZIP likelihood requires pi_block")
        if n_total <= 0:
            raise ValueError("n_total must be positive")
        epsilon = torch.finfo(lambda_block.dtype).eps
        count_values = count_values.to(device=lambda_block.device, dtype=lambda_block.dtype)
        lambda_values = self._select_values(lambda_block, count_indices)
        pi_values = self._select_values(pi_block, count_indices)

        positive_ll = lambda_block.new_tensor(0.0)
        positive_zero_ll = lambda_block.new_tensor(0.0)
        if count_values.numel() > 0:
            log_expm1 = lambda_values + torch.log(
                -torch.expm1(-lambda_values) + epsilon
            )
            positive_ll = (
                torch.log(pi_values + epsilon)
                + count_values * torch.log(lambda_values + epsilon)
                - torch.lgamma(count_values + 1.0)
                - log_expm1
            ).sum()
            positive_zero_ll = torch.log(1.0 - pi_values + epsilon).sum()

        zero_ll = torch.log(1.0 - pi_block + epsilon).sum() - positive_zero_ll
        if missing_indices is not None and missing_indices.numel() > 0:
            missing_pi = self._select_values(pi_block, missing_indices)
            zero_ll = zero_ll - torch.log(1.0 - missing_pi + epsilon).sum()

        nll = -(positive_ll + zero_ll) * float(block_weight)
        return nll, nll / float(n_total)

    def _poisson_nll(
        self,
        lambda_matrix: torch.Tensor,
        count_values: torch.Tensor,
        count_indices: torch.Tensor,
        missing_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        epsilon = torch.finfo(lambda_matrix.dtype).eps
        count_values = count_values.to(device=lambda_matrix.device, dtype=lambda_matrix.dtype)
        lambda_values = self._select_values(lambda_matrix, count_indices).clamp_min(epsilon)
        positive_ll = (
            count_values * torch.log(lambda_values)
            - torch.lgamma(count_values + 1.0)
        ).sum()
        total_lambda = lambda_matrix.sum()
        if missing_indices is not None and missing_indices.numel() > 0:
            total_lambda = total_lambda - self._select_values(
                lambda_matrix, missing_indices
            ).sum()

        nll = -(positive_ll - total_lambda)
        return nll, nll / self._observed_count(lambda_matrix, missing_indices)

    def _poisson_block_nll(
        self,
        lambda_block: torch.Tensor,
        count_values: torch.Tensor,
        count_indices: torch.Tensor,
        n_total: int,
        block_weight: float,
        missing_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if n_total <= 0:
            raise ValueError("n_total must be positive")
        epsilon = torch.finfo(lambda_block.dtype).eps
        count_values = count_values.to(device=lambda_block.device, dtype=lambda_block.dtype)
        lambda_values = self._select_values(lambda_block, count_indices).clamp_min(epsilon)
        positive_ll = (
            count_values * torch.log(lambda_values)
            - torch.lgamma(count_values + 1.0)
        ).sum()
        total_lambda = lambda_block.sum()
        if missing_indices is not None and missing_indices.numel() > 0:
            total_lambda = total_lambda - self._select_values(
                lambda_block, missing_indices
            ).sum()

        nll = -(positive_ll - total_lambda) * float(block_weight)
        return nll, nll / float(n_total)
