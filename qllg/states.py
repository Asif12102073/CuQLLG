#LR denisty matrix factorization container initial generaotrs
"""Low-Rank density matrix factorization container and initial state generators."""

from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class LowRankState:
    """Factorized state: rho = V @ diag(lambda_vec) @ V^H
    V: shape (N, r), with V^H @ V = I_r
    lambda_vec: shape (r,), positive spectrum summing to <= 1.0
    """
    V: Any
    lambda_vec: Any

    @property
    def rank(self) -> int:
        return len(self.lambda_vec)

    @property
    def hilbert_dim(self) -> int:
        return self.V.shape[0]

class StateBuilder:
    @staticmethod
    def pure_state(psi: np.ndarray) -> LowRankState:
        """Builds a rank-1 pure state from a normalized state vector."""
        psi = np.asarray(psi, dtype=np.complex128)
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi = psi / norm
        V = psi[:, np.newaxis]
        lambdas = np.array([1.0], dtype=np.float64)
        return LowRankState(V=V, lambda_vec=lambdas)

    @staticmethod
    def antiferromagnetic(n_spins: int, rank: int = 2) -> LowRankState:
        """Constructs an initial low-rank mixed Neel/antiferromagnetic state."""
        N = 2 ** n_spins
        v1 = np.zeros(N, dtype=np.complex128)
        v2 = np.zeros(N, dtype=np.complex128)

        # Bitmasks for 0101... and 1010...
        mask1 = sum(1 << b for b in range(0, n_spins, 2))
        mask2 = sum(1 << b for b in range(1, n_spins, 2))

        v1[mask1] = 1.0
        v2[mask2] = 1.0

        if rank == 1:
            return LowRankState(V=v1[:, np.newaxis], lambda_vec=np.array([1.0]))

        V = np.column_stack([v1, v2])
        lambdas = np.array([0.7, 0.3], dtype=np.float64)
        return LowRankState(V=V, lambda_vec=lambdas)
