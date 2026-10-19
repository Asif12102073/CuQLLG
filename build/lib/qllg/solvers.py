# Unified CPU/GPU solver 
"""Unified Low-Rank Eigenmode Integration (LREI) Solver. Supports both q-LLG and q-LL equations with transparent CPU and GPU execution."""

from typing import List, Optional, Tuple, Callable
import time
import numpy as np
from .backend import Backend
from .states import LowRankState

class QSpinSolver:
    """Structure-preserving Runge-Kutta integrator for quantum spin equations."""

    def __init__(self, H: Any, equation: str = "qllg", kappa: float = 0.1, hbar: float = 1.0, backend: str = "cpu"):
        self.equation = equation.lower()
        if self.equation not in ("qllg", "qll"):
            raise ValueError(f"Unknown equation '{equation}'. Choose 'qllg' or 'qll'.")
        self.kappa = kappa
        self.hbar = hbar
        self.backend = Backend(use_gpu=(backend.lower() == "gpu"))
        # Move Hamiltonian to target backend
        self.H = self.backend.to_device(H)

    def _diagonalize_stage(
        self, A_list: List[Any], V_list: List[Any], B_list: List[Optional[Any]], N: int, r: int
    ) -> Any:
        """Subspace Rayleigh-Ritz stage projection."""
        xp = self.backend.xp
        blocks = []
        for A, V, B in zip(A_list, V_list, B_list):
            blocks.append(A)
            blocks.append(V)
            if B is not None:
                blocks.append(B)
        U = xp.hstack(blocks)
        Q, _ = self.backend.reduced_qr(U)
        # Matrix-free evaluation of K @ Q
        KQ = xp.zeros_like(Q)
        for A, V, B in zip(A_list, V_list, B_list):
            KQ += A @ (V.conj().T @ Q)
            if B is not None:
                KQ += V @ (B.conj().T @ Q)
        # Subspace Hamiltonian H_sub = Q^H (K Q)
        H_sub = 0.5 * (Q.conj().T @ KQ + (KQ).conj().T @ Q)
        vals, vecs = self.backend.eigh(H_sub)
        idx = xp.argsort(vals)[::-1][:r]
        return Q @ vecs[:, idx]

    def _compute_rhs(self, V_tilde: Any, lambda_vec: Any) -> Tuple[Any, Any]:
        """Computes factored derivative blocks (Z, W) such that \\dot{\\rho} = Z V^H + V W^H."""
        xp = self.backend.xp
        Lambda_tilde = xp.diag(lambda_vec)
        HV = self.H @ V_tilde
        Vt_HV = V_tilde.conj().T @ HV

        if self.equation == "qllg":
            # Sylvester decoupled division
            comm_11 = Lambda_tilde @ Vt_HV - Vt_HV @ Lambda_tilde
            D11 = (1j / self.hbar) * comm_11
            s = -1j * self.kappa * lambda_vec
            denom_11 = 1.0 + s[:, xp.newaxis] - s[xp.newaxis, :]
            X11 = D11 / denom_11

            # Complement coupling block
            W_scale_vec = -1j / (self.hbar * (1.0 + 1j * self.kappa * lambda_vec))
            M = HV @ (Lambda_tilde * W_scale_vec[xp.newaxis, :])
            W = M - V_tilde @ (V_tilde.conj().T @ M)

        else:
            # "qll"
            # Explicit double commutator blocks
            comm_11 = (1j / self.hbar) * (Lambda_tilde @ Vt_HV - Vt_HV @ Lambda_tilde)
            double_11 = -(self.kappa / self.hbar) * (
                (Lambda_tilde @ Lambda_tilde) @ Vt_HV + Vt_HV @ (Lambda_tilde @ Lambda_tilde) - 2.0 * Lambda_tilde @ Vt_HV @ Lambda_tilde
            )
            X11 = comm_11 + double_11

            # Complement coupling block
            W_coeff = (-1j * xp.ones_like(lambda_vec) - self.kappa * lambda_vec) / self.hbar
            M = HV @ (Lambda_tilde * W_coeff[xp.newaxis, :])
            W = M - V_tilde @ (V_tilde.conj().T @ M)

        Z = V_tilde @ X11 + W
        return Z, W

    def step(self, state: LowRankState, dt: float) -> LowRankState:
        """Advances state by one time step dt using Conservative LREI-RK4."""
        xp = self.backend.xp
        N, r = state.V.shape
        Lambda_0 = xp.diag(state.lambda_vec)

        # Stage 1
        V1 = state.V
        Z1, W1 = self._compute_rhs(V1, state.lambda_vec)

        # Stage 2
        A1_s2 = V1 @ Lambda_0 + 0.5 * dt * Z1
        B1_s2 = 0.5 * dt * W1
        V2 = self._diagonalize_stage([A1_s2], [V1], [B1_s2], N, r)
        Z2, W2 = self._compute_rhs(V2, state.lambda_vec)

        # Stage 3
        A1_s3 = V1 @ Lambda_0
        A2_s3 = 0.5 * dt * Z2
        B2_s3 = 0.5 * dt * W2
        V3 = self._diagonalize_stage([A1_s3, A2_s3], [V1, V2], [None, B2_s3], N, r)
        Z3, W3 = self._compute_rhs(V3, state.lambda_vec)

        # Stage 4
        A1_s4 = V1 @ Lambda_0
        A3_s4 = dt * Z3
        B3_s4 = dt * W3
        V4 = self._diagonalize_stage([A1_s4, A3_s4], [V1, V3], [None, B3_s4], N, r)
        Z4, W4 = self._compute_rhs(V4, state.lambda_vec)

        # Final combination
        A1_fin = V1 @ Lambda_0 + (dt / 6.0) * Z1
        B1_fin = (dt / 6.0) * W1
        A2_fin = (dt / 3.0) * Z2
        B2_fin = (dt / 3.0) * W2
        A3_fin = (dt / 3.0) * Z3
        B3_fin = (dt / 3.0) * W3
        A4_fin = (dt / 6.0) * Z4
        B4_fin = (dt / 6.0) * W4
        V_next = self._diagonalize_stage(
            [A1_fin, A2_fin, A3_fin, A4_fin], [V1, V2, V3, V4], [B1_fin, B2_fin, B3_fin, B4_fin], N, r
        )
        return LowRankState(V=V_next, lambda_vec=state.lambda_vec)

    def solve(
        self, initial_state: LowRankState, t_span: Tuple[float, float], dt: float, callback: Optional[Callable[[float, LowRankState], None]] = None
    ) -> LowRankState:
        """Solves the system over t_span = (t0, tf)."""
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        # Upload initial state to active backend
        curr_state = LowRankState(
            V=self.backend.to_device(initial_state.V),
            lambda_vec=self.backend.to_device(initial_state.lambda_vec)
        )
        if callback is not None:
            callback(t0, curr_state)
        t = t0
        for _ in range(n_steps):
            curr_state = self.step(curr_state, dt)
            t += dt
            if callback is not None:
                callback(t, curr_state)
        self.backend.synchronize()
        return curr_state
