#Observables, energy, magnetization, and quantum correlation calculators
"""Observables, energy, magnetization, and quantum correlation calculators."""

from typing import Any, Tuple
import numpy as np
from .states import LowRankState
from .backend import Backend

def expectation_value(op: Any, state: LowRankState, backend: Backend) -> float:
    r"""Evaluates \\langle A \\rangle = \\text{Tr}[\\Lambda\_0 (V^\\dagger A V)] in O(N r) operations."""
    xp = backend.xp
    op_dev = backend.to_device(op)
    AV = op_dev @ state.V
    Vt_AV = state.V.conj().T @ AV
    exp_val = xp.sum(state.lambda_vec * xp.diag(Vt_AV))
    return float(backend.to_host(xp.real(exp_val)))

def spin_covariance_matrix(
    Sx: Any, Sy: Any, Sz: Any, state: LowRankState, backend: Backend
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes mean spin vector <s> and 3x3 covariance matrix C\_jk."""
    ops = [Sx, Sy, Sz]
    s_mean = np.array([expectation_value(op, state, backend) for op in ops])
    C = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        for k in range(3):
            # C\_jk = 0.5 \* &lt;{S\_j, S\_k}&gt; - <S\_j> <S\_k>
            comm = expectation_value(ops[j] @ ops[k] - ops[k] @ ops[j], state, backend)
            C[j, k] = 0.5 * comm - s_mean[j] * s_mean[k]
    return s_mean, C
