#Operator's defifnition needed for spinor computations
"""Spin operators and many-body Hamiltonian generators."""

from typing import List, Tuple
import numpy as np
import scipy.sparse as sp

def spin_half_operators() -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Generates Pauli spin-1/2 operators in CSR format."""
    sx = sp.csr_matrix([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)
    sy = sp.csr_matrix([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex128)
    sz = sp.csr_matrix([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
    return sx, sy, sz

def spin_one_operators() -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Generates spin-1 angular momentum operators in CSR format."""
    sx = (1.0 / np.sqrt(2.0)) * sp.csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    sy = (1.0 / (np.sqrt(2.0) * 1j)) * sp.csr_matrix([[0.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.complex128)
    sz = sp.csr_matrix(np.diag([1.0, 0.0, -1.0]), dtype=np.complex128)
    return sx, sy, sz

class SpinSystem:
    """Builder for spin lattice Hamiltonians."""

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

    def build_heisenberg_hamiltonian(
        self,
        J: float = 1.0,
        D_z: float = 0.0,
        B: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        periodic: bool = True
    ) -> sp.csr_matrix:
        """Constructs Heisenberg exchange, DMI, and Zeeman Hamiltonian."""
        H = sp.csr_matrix((self.N, self.N), dtype=np.complex128)
        n_couplings = self.n_spins if periodic else self.n_spins - 1

        for i in range(n_couplings):
            j = (i + 1) % self.n_spins  # Heisenberg exchange: J * (S_i . S_j)
            H += J * (self.Sx[i] @ self.Sx[j] + self.Sy[i] @ self.Sy[j] + self.Sz[i] @ self.Sz[j])

            # Dzyaloshinskii-Moriya Interaction (z-axis DMI)
            if D_z != 0.0:
                H += D_z * (self.Sx[i] @ self.Sy[j] - self.Sy[i] @ self.Sx[j])

        # Zeeman interaction: - mu * sum_i B . S_i
        bx, by, bz = B
        for i in range(self.n_spins):
            if bx != 0.0:
                H -= bx * self.Sx[i]
            if by != 0.0:
                H -= by * self.Sy[i]
            if bz != 0.0:
                H -= bz * self.Sz[i]

        return H
