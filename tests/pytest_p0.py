"""P0 test 1: Hamiltonian regression tests for qllg.SpinSystem.build_heisenberg_hamiltonian.

H = J sum_bonds S_i.S_j + D_z sum_bonds (Sx_i Sy_j - Sy_i Sx_j) - B . sum_i S_i   (S = sigma/2, hbar = 1)
"""
import numpy as np
import pytest
import scipy.sparse as sp
from qllg import SpinSystem, QSpinSolver, StateBuilder, expectation_value

SX = np.array([[0, 1], [1, 0]], dtype=complex) / 2
SY = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
SZ = np.array([[1, 0], [0, -1]], dtype=complex) / 2


def site_op(o, i, n):
    """Independent construction: dense Kronecker product, site 0 most significant."""
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, o if k == i else np.eye(2))
    return out


def reference_H(n, J, D_z, B, periodic):
    X, Y, Z = ([site_op(o, i, n) for i in range(n)] for o in (SX, SY, SZ))
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n if periodic else n - 1):
        j = (i + 1) % n
        H += J * (X[i] @ X[j] + Y[i] @ Y[j] + Z[i] @ Z[j]) + D_z * (X[i] @ Y[j] - Y[i] @ X[j])
    for i in range(n):
        H -= B[0] * X[i] + B[1] * Y[i] + B[2] * Z[i]
    return H


def lib_H(n, **kw):
    return SpinSystem(n_spins=n, spin=0.5).build_heisenberg_hamiltonian(**kw)


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("n", [3, 4, 5])
def test_matches_independent_kron_construction(n, periodic):
    kw = dict(J=1.3, D_z=0.3, B=(0.7, -0.4, 0.9), periodic=periodic)
    H = lib_H(n, **kw).toarray()
    assert np.abs(H - reference_H(n, **kw)).max() < 1e-12


def test_zeeman_applied_once():
    """The original bug: field was added once per bond (B_eff = n * B)."""
    n = 4
    s = SpinSystem(n_spins=n, spin=0.5)
    H0 = s.build_heisenberg_hamiltonian(J=1.0, D_z=0.3, B=(0, 0, 0))
    H1 = s.build_heisenberg_hamiltonian(J=1.0, D_z=0.3, B=(1.0, 0, 0))
    SXtot = sum(s.Sx)
    assert abs((H1 - H0) + SXtot).max() < 1e-12


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_zeeman_linear_per_component(axis):
    n = 4
    s = SpinSystem(n_spins=n, spin=0.5)
    ops = [sum(s.Sx), sum(s.Sy), sum(s.Sz)]
    B = [0.0, 0.0, 0.0]; B[axis] = 2.5
    H0 = s.build_heisenberg_hamiltonian(J=1.0, D_z=0.3, B=(0, 0, 0))
    H1 = s.build_heisenberg_hamiltonian(J=1.0, D_z=0.3, B=tuple(B))
    assert abs((H1 - H0) + 2.5 * ops[axis]).max() < 1e-12


def test_open_chain_has_n_minus_1_bonds():
    n = 5
    s = SpinSystem(n_spins=n, spin=0.5)
    Hp = s.build_heisenberg_hamiltonian(J=1.0, D_z=0.3, B=(0, 0, 0), periodic=True)
    Ho = s.build_heisenberg_hamiltonian(J=1.0, D_z=0.3, B=(0, 0, 0), periodic=False)
    i, j = n - 1, 0  # closing bond
    closing = (s.Sx[i] @ s.Sx[j] + s.Sy[i] @ s.Sy[j] + s.Sz[i] @ s.Sz[j]
               + 0.3 * (s.Sx[i] @ s.Sy[j] - s.Sy[i] @ s.Sx[j]))
    assert abs((Hp - Ho) - closing).max() < 1e-12


def test_hermitian():
    H = lib_H(4, J=1.0, D_z=0.3, B=(0.5, 0.2, -0.1))
    assert abs(H - H.conj().T).max() < 1e-12


@pytest.mark.parametrize("n,E0", [(4, -2.0), (6, -2.8027756377)])
def test_heisenberg_ring_ground_state_energy(n, E0):
    """Known spin-1/2 Heisenberg ring energies for H = sum S_i.S_j (fixes the J normalisation)."""
    H = lib_H(n, J=1.0, D_z=0.0, B=(0, 0, 0)).toarray()
    assert abs(np.linalg.eigvalsh(H)[0] - E0) < 1e-6


@pytest.mark.parametrize("B", [(1.0, 0.0, 0.0), (0.8, 0.3, 0.5)])
def test_energy_stays_within_spectrum(B):
    """E_min <= <H>(t) <= E_max along a q-LLG trajectory (any valid state obeys this)."""
    n = 6
    H = lib_H(n, J=1.0, D_z=0.3, B=B).tocsr()
    ev = np.linalg.eigvalsh(H.toarray())
    st = StateBuilder.antiferromagnetic(n_spins=n, rank=2)
    sol = QSpinSolver(H=H, equation="qllg", kappa=0.1, backend="cpu")
    E = []
    sol.solve(st, t_span=(0.0, 1.0), dt=0.01, callback=lambda t, s: E.append(expectation_value(H, s, sol.backend)))
    E = np.real(np.array(E))
    assert E.min() >= ev[0] - 1e-9 and E.max() <= ev[-1] + 1e-9
