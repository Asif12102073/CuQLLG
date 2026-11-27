# Operator definition needed for spinor computations
"""Spin operators and many-body Hamiltonian generators."""

from typing import List, Tuple
import numpy as np
import scipy.sparse as sp

#PRL QLLG papers convention-conversion factors
PAPER_J_FACTOR = 4.0
PAPER_D_FACTOR = 4.0

def spin_half_operators() -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Generates Pauli spin-1/2 operators in CSR format."""
    sx = sp.csr_matrix([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)
    sy = sp.csr_matrix([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex128)
    sz = sp.csr_matrix([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
    return sx, sy, sz

def spin_one_operators() -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """ Generates spin-1 angular momentum operators in CSR format."""
    sx = (1.0 / np.sqrt(2.0)) * sp.csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    sy = (1.0 / (np.sqrt(2.0) * 1j)) * sp.csr_matrix([[0.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.complex128)
    sz = sp.csr_matrix(np.diag([1.0, 0.0, -1.0]), dtype=np.complex128)
    return sx, sy, sz

class SpinSystem:
    """ Builder for spin lattice Hamiltonians."""

    def __init__(self, n_spins: int, spin: float = 0.5):
        self.n_spins = n_spins
        self.spin = spin
        self.dim_single = int(2 * spin + 1)
        self.N = self.dim_single ** n_spins

        if spin == 0.5:
            sx, sy, sz = spin_half_operators()
        elif spin == 1.0:
            sx, sy, sz = spin_one_operators()
        else:
            raise NotImplementedError(f"Spin s={spin} not yet implemented.")

        id_single = sp.identity(self.dim_single, dtype=np.complex128, format="csr")

        # Build individual many-body operators via Kronecker product
        self.Sx: List[sp.csr_matrix] = []
        self.Sy: List[sp.csr_matrix] = []
        self.Sz: List[sp.csr_matrix] = []

        for i in range(n_spins):
            ox, oy, oz = 1.0, 1.0, 1.0
            for j in range(n_spins):
                cx = sx if j == i else id_single
                cy = sy if j == i else id_single
                cz = sz if j == i else id_single
                ox = sp.kron(ox, cx, format="csr")
                oy = sp.kron(oy, cy, format="csr")
                oz = sp.kron(oz, cz, format="csr")
            self.Sx.append(ox)
            self.Sy.append(oy)
            self.Sz.append(oz)

    def build_heisenberg_hamiltonian(self, J=1.0, D_z=0.0, B=(0.0, 0.0, 1.0),
                                     periodic=True, convention="native"):
        #Constructs Heisenberg exchange, DMI, and Zeeman Hamiltonian. (For example heisneberg rings)
        if convention == "native":
            j_scale = d_scale = 1.0
        elif convention in ("paper", "prl"):
            j_scale, d_scale = PAPER_J_FACTOR, PAPER_D_FACTOR
        else:
            raise ValueError(f"unknown convention {convention!r}; use 'native' or 'paper'")
        J, D_z = j_scale * J, d_scale * D_z

        H = sp.csr_matrix((self.N, self.N), dtype=np.complex128)
        n_couplings = self.n_spins if periodic else self.n_spins - 1
        for i in range(n_couplings):
            j = (i + 1) % self.n_spins
            H += J * (self.Sx[i] @ self.Sx[j] + self.Sy[i] @ self.Sy[j] + self.Sz[i] @ self.Sz[j])
            if D_z != 0.0:
                H += D_z * (self.Sx[i] @ self.Sy[j] - self.Sy[i] @ self.Sx[j])

        bx, by, bz = B   # Zeeman, applied once
        for k in range(self.n_spins):
            if bx != 0.0: H -= bx * self.Sx[k]
            if by != 0.0: H -= by * self.Sy[k]
            if bz != 0.0: H -= bz * self.Sz[k]
        return H.tocsr()


# previous Heisenberg Hamiltonian builder
"""
    def build_heisenberg_hamiltonian(
        self,
        J: float = 1.0,
        D_z: float = 0.0,
        B: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        periodic: bool = True
    ) -> sp.csr_matrix:
        #Constructs Heisenberg exchange, DMI, and Zeeman Hamiltonian.
        H = sp.csr_matrix((self.N, self.N), dtype=np.complex128)
        n_couplings = self.n_spins if periodic else self.n_spins - 1
        
        # 1. Bond interaction loop (Exchange + DMI)
        for idx in range(n_couplings):
            j = (idx + 1) % self.n_spins
            H += J * (self.Sx[idx] @ self.Sx[j] + self.Sy[idx] @ self.Sy[j] + self.Sz[idx] @ self.Sz[j])
            if D_z != 0.0:
                H += D_z * (self.Sx[idx] @ self.Sy[j] - self.Sy[idx] @ self.Sx[j])
                
        # 2. Zeeman interaction loop (Dedented OUTSIDE the bond loop)
        bx, by, bz = B
        for idx in range(self.n_spins):
            if bx != 0.0:
                H -= bx * self.Sx[idx]
            if by != 0.0:
                H -= by * self.Sy[idx]
            if bz != 0.0:
                H -= bz * self.Sz[idx]
                
        return H

"""
