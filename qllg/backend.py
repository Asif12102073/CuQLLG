"""Hardware backend abstraction for CPU (NumPy/SciPy) and GPU (CuPy)."""

from typing import Any, Tuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None
    cp_sp = None

class Backend:
    def __init__(self, use_gpu: bool = False):
        if use_gpu and not HAS_CUPY:
            raise RuntimeError("CuPy is not installed or no CUDA device is available.")
        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np
        self.sp = cp_sp if use_gpu else sp

    def to_device(self, arr: Any) -> Any:
        """Transfers array or sparse matrix to the target backend."""
        if self.use_gpu:
            if isinstance(arr, (cp.ndarray, cp_sp.spmatrix)):
                return arr
            if sp.issparse(arr):
                return cp_sp.csr_matrix(arr)
            return cp.asarray(arr)
        else:
            if HAS_CUPY and isinstance(arr, (cp.ndarray, cp_sp.spmatrix)):
                return arr.get()
            if sp.issparse(arr):
                return sp.csr_matrix(arr)
            return np.asarray(arr)

    def to_host(self, arr: Any) -> np.ndarray:
        """Brings array back to CPU NumPy format."""
        if self.use_gpu and hasattr(arr, "get"):
            return arr.get()
        if sp.issparse(arr):
            return arr.toarray()
        return np.asarray(arr)

    def synchronize(self) -> None:
        """Synchronizes GPU stream if running on CUDA."""
        if self.use_gpu:
            cp.cuda.Device().synchronize()

    def reduced_qr(self, U: Any) -> Tuple[Any, Any]:
        """Computes reduced QR decomposition: U = Q @ R."""
        return self.xp.linalg.qr(U, mode="reduced")

    def eigh(self, A: Any) -> Tuple[Any, Any]:
        """Computes eigendecomposition of a Hermitian matrix."""
        if self.use_gpu:
            return cp.linalg.eigh(A)
        return la.eigh(A)
