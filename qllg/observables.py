"""Observables, energy, magnetization, and quantum correlation calculators."""

from typing import Any, Tuple
import numpy as np
from .states import LowRankState
from .backend import Backend



def expectation_value(op: Any, state: LowRankState, backend: Backend) -> float:
    r"""Evaluates \\langle A \\rangle = \\text{Tr}[\\Lambda\_0 (V^\\dagger A V)] in O(N r) operations."""
    xp = backend.xp
    op_dev = backend.to_device(op)
    V_dev = backend.to_device(state.V)
    lambda_dev = backend.to_device(state.lambda_vec)
    AV = op_dev @ V_dev
    Vt_AV = V_dev.conj().T @ AV
    exp_val = xp.sum(lambda_dev * xp.diag(Vt_AV))
    return float(backend.to_host(xp.real(exp_val)))

def spin_covariance_matrix(
    Sx: Any, Sy: Any, Sz: Any, state: LowRankState, backend: Backend
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes mean spin vector <s> and 3x3 covariance matrix C_jk."""
    ops = [Sx, Sy, Sz]
    s_mean = np.array([expectation_value(op, state, backend) for op in ops])
    C = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        for k in range(3):
            # C\_jk = 0.5 \* &lt;{S\_j, S\_k}&gt; + <S\_j> <S\_k>
            comm = expectation_value(ops[j] @ ops[k] + ops[k] @ ops[j], state, backend)
            C[j, k] = 0.5 * comm - s_mean[j] * s_mean[k]
    return s_mean, C

def wootters_concurrence_dimer(state: LowRankState, backend: Backend) -> float:
    """Computes Wootters concurrence C(rho) for a 2-spin-1/2 dimer in O(r^2) operations."""
    xp = backend.xp
    V = backend.to_host(state.V)  # (4, r)
    lambdas = backend.to_host(state.lambda_vec)  # Full 4x4 density matrix reconstruction
    rho = (V * lambdas) @ V.conj().T  # Spin-flip matrix sigma_y ⊗ sigma_y
    sy = np.array([[0, -1j], [1j, 0]])
    sy_sy = np.kron(sy, sy)  # Spin-flipped density matrix
    R = rho @ (sy_sy @ rho.conj() @ sy_sy)
    rho_tilde = sy_sy @ rho.conj() @ sy_sy
    R = rho @ rho_tilde
    evals_raw = np.linalg.eigvals(R)
    evals = np.sort(np.sqrt(np.maximum(0.0, np.real(evals_raw))))[::-1]
    concurrence = max(0.0, evals[0] - evals[1] - evals[2] - evals[3])
    return float(concurrence)

def chsh_bell_parameter_dimer(state: LowRankState, backend: Backend) -> float:
    """
    Computes Horodecki CHSH Bell parameter S_CHSH(rho) for a 2-spin-1/2 dimer (N = 4).
    Local realism implies S <= 2. Maximal quantum violation achieves S = 2*sqrt(2) ~ 2.8284.
    """
    if state.V.shape[0] != 4:
        raise ValueError("chsh_bell_parameter_dimer requires a 2-spin-1/2 system (N = 4).")
    
    V_host = backend.to_host(backend.to_device(state.V))
    lambda_host = backend.to_host(backend.to_device(state.lambda_vec))
    rho = (V_host * lambda_host) @ V_host.conj().T
    
    paulis = [
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    ]
    
    T = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            op = np.kron(paulis[i], paulis[j])
            T[i, j] = np.real(np.trace(rho @ op))
            
    tt_evals = np.linalg.eigvalsh(T.T @ T)[::-1]
    u1, u2 = tt_evals[0], tt_evals[1]
    return 2.0 * float(np.sqrt(max(0.0, u1 + u2)))
