"""Symmetric normalized Laplacian initialization of cell–gene coordinates."""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, eigsh

__all__ = ["laplacian_init"]


def laplacian_init(counts: sparse.csr_matrix, latent_dim: int):
    """Return (cell_coordinates, gene_coordinates) from raw counts.

    counts is a finite, non-negative matrix convertible to CSR with at least
    one nonzero entry. latent_dim is a positive integer below
    min(counts.shape) - 1. Each returned array has latent_dim columns and
    float32 dtype; rows follow input order.

    Uses orthonormal eigenvectors for the smallest positive eigenvalues of
    L = I - D^(-1/2) A D^(-1/2), where A is the weighted bipartite adjacency.
    Zero modes from every connected component are excluded. Isolated cells
    and genes are retained with inverse degree zero (L has diagonal 1 there).
    Counts are not transformed for fitting. fit's seed controls the solver's
    starting vector; direct calls use NumPy's current random state.
    """
    if counts is None:
        raise ValueError("counts are required for Laplacian initialization")
    counts = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    if (isinstance(latent_dim, (bool, np.bool_)) or
            not isinstance(latent_dim, (int, np.integer)) or
            not 0 < latent_dim < min(counts.shape) - 1):
        raise ValueError("latent_dim must be positive and less than min(counts.shape) - 1")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("counts must be finite and non-negative")
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if counts.nnz == 0:
        raise ValueError("Laplacian initialization requires at least one nonzero count")

    adjacency = sparse.bmat([[None, counts], [counts.T, None]], format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros_like(degree)
    np.divide(1., np.sqrt(degree), out=inverse_sqrt, where=degree > 0)
    scaling = sparse.diags(inverse_sqrt)
    laplacian = sparse.eye(len(degree), format="csr") - scaling @ adjacency @ scaling

    # Each non-isolated component has one zero mode proportional to sqrt(degree).
    # Shift those modes above the Laplacian spectrum [0, 2]. This avoids
    # allocating an eigenvector for every component of a disconnected graph.
    n_components, labels = connected_components(adjacency, directed=False)
    component_degree = np.bincount(labels, weights=degree)
    active = np.flatnonzero(degree > 0)
    zero_modes = sparse.csr_matrix(
        (np.sqrt(degree[active] / component_degree[labels[active]]),
         (active, labels[active])), shape=(len(degree), n_components),
    )

    def multiply(values):
        return laplacian @ values + 3 * (zero_modes @ (zero_modes.T @ values))

    operator = LinearOperator(laplacian.shape, matvec=multiply, matmat=multiply,
                              dtype=np.float64)
    eigenvalues, coordinates = eigsh(operator, k=int(latent_dim), which="SA",
                                     v0=np.random.normal(size=len(degree)), tol=1e-8)
    coordinates = coordinates[:, np.argsort(eigenvalues)]
    n_cells = counts.shape[0]
    return (coordinates[:n_cells].astype(np.float32),
            coordinates[n_cells:].astype(np.float32))
