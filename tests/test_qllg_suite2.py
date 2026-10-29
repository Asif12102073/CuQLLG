"""
Automated Test Suite for the qllg Library.
Runnable via `python test_qllg_suite.py` or standard `unittest`.

Covers:
1. Hamiltonian Construction & Spectral Bounds (Zeeman term single-application check)
2. Trace, Purity, and Spectrum Preservation during Time Integration
3. Spin Covariance Matrix Properties (Anticommutator check, non-negative variances, uncertainty trace bounds)
4. Hermiticity and Positive Semi-definiteness
5. Pure-State Equivalence of q-LL and q-LLG under Time Rescaling
6. Bipartite Entanglement Generation, Revivals, and CHSH Bell Nonlocality Violation
"""

import unittest
import numpy as np
import scipy.linalg as la

from qllg import (
    SpinSystem,
    StateBuilder,
    QSpinSolver,
    LowRankState,
    expectation_value,
    spin_covariance_matrix,
    wootters_concurrence_dimer,
    chsh_bell_parameter_dimer,
    spin_one_operators,
)
from qllg.backend import Backend


class TestQLLGLibrary(unittest.TestCase):

    # ==========================================================================
    # 1. Hamiltonian Spectral Bounds & Zeeman Term Unit Test
    # ==========================================================================
    def test_hamiltonian_zeeman_single_application(self):
        """
        Verifies that the Zeeman term in build_heisenberg_hamiltonian is applied
        exactly once (not n_spins times) and that spectral bounds match theory.
        """
        n_spins = 10
        sys = SpinSystem(n_spins=n_spins, spin=0.5)
        
        # 10-spin ring with J=1.0, D_z=0.3, B=(1.0, 0.0, 0.0)
        H = sys.build_heisenberg_hamiltonian(J=1.0, D_z=0.3, B=(1.0, 0.0, 0.0), periodic=True)
        H_dense = H.toarray()
        
        # Compute extreme eigenvalues
        evals_min = la.eigh(H_dense, eigvals_only=True, subset_by_index=[0, 0])[0]
        evals_max = la.eigh(H_dense, eigvals_only=True, subset_by_index=[1023, 1023])[0]
        
        # Bounds for single-applied field B=(1,0,0) on 10 spins:
        # E_min approx -5.193, E_max approx +7.500
        self.assertGreater(evals_min, -10.0, "Zeeman field was over-applied!")
        np.testing.assert_allclose(evals_min, -5.193366, atol=1e-3)
        np.testing.assert_allclose(evals_max, 7.500000, atol=1e-3)

    # ==========================================================================
    # 2. Trace, Purity, and Spectrum Preservation Tests
    # ==========================================================================
    def test_trace_and_purity_preservation_qllg(self):
        self._check_trace_and_purity("qllg")

    def test_trace_and_purity_preservation_qll(self):
        self._check_trace_and_purity("qll")

    def _check_trace_and_purity(self, equation):
        """
        Tests that time integration under QSpinSolver preserves unit trace (Tr(rho) = 1)
        and initial spectrum/purity (Tr(rho^2) = const) down to machine precision.
        """
        n_spins = 6
        sys = SpinSystem(n_spins=n_spins, spin=0.5)
        H = sys.build_heisenberg_hamiltonian(J=1.0, D_z=0.4, B=(1.0, 0.0, 1.0))
        
        # Rank-2 initial state: 70% |AF1> + 30% |AF2>
        state0 = StateBuilder.antiferromagnetic(n_spins=n_spins, rank=2)
        
        solver = QSpinSolver(H=H, equation=equation, kappa=0.2, backend="cpu")
        
        dt = 0.01
        curr_state = state0
        purity_initial = float(np.sum(state0.lambda_vec ** 2))
        
        for _ in range(20):
            curr_state = solver.step(curr_state, dt)
            
            # Check orthogonality of factor V
            overlap = curr_state.V.conj().T @ curr_state.V
            np.testing.assert_allclose(overlap, np.eye(2), atol=1e-12)
            
            # Check trace conservation: sum(lambdas) == 1.0
            trace_val = float(np.sum(curr_state.lambda_vec))
            np.testing.assert_allclose(trace_val, 1.0, atol=1e-12)
            
            # Check purity conservation: sum(lambdas^2) == constant
            purity_val = float(np.sum(curr_state.lambda_vec ** 2))
            np.testing.assert_allclose(purity_val, purity_initial, atol=1e-12)

    # ==========================================================================
    # 3. Spin Covariance Matrix & Quantum Uncertainty Bounds
    # ==========================================================================
    def test_spin_covariance_matrix_anticommutator_and_bounds(self):
        """
        Tests spin_covariance_matrix on a spin-1 particle to confirm:
        1. Diagonal elements C_jj are non-negative quantum variances.
        2. Anticommutator logic is used (not commutator).
        3. Tr(C) for spin-type state with m0=1 satisfies 1 <= Tr(C) <= 2 (specifically 14/9 ~ 1.556).
        """
        backend = Backend(use_gpu=False)
        Sx, Sy, Sz = spin_one_operators()
        
        # Spin-1 spin-type state with m0 = 1: rho0 = (1/3)*(I + Sz) = diag(2/3, 1/3, 0)
        rho0 = (1.0 / 3.0) * (np.eye(3, dtype=np.complex128) + Sz.toarray())
        w, V = la.eigh(rho0)
        idx = np.argsort(w)[::-1]
        
        state0 = LowRankState(V=V[:, idx], lambda_vec=w[idx])
        
        s_mean, C = spin_covariance_matrix(Sx, Sy, Sz, state0, backend)
        
        # Mean magnetization <S> = (0, 0, 2/3)
        np.testing.assert_allclose(s_mean, [0.0, 0.0, 2.0 / 3.0], atol=1e-12)
        
        # Diagonal variances must be strictly non-negative
        diag_C = np.diag(C)
        self.assertTrue(np.all(diag_C >= 0.0), f"Negative variance in C: {diag_C}")
        
        # Theoretical values: C_xx = 2/3 (~0.667), C_yy = 2/3 (~0.667), C_zz = 2/9 (~0.222)
        expected_diag = np.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 9.0])
        np.testing.assert_allclose(diag_C, expected_diag, atol=1e-12)
        
        # Total variance trace: Tr(C) = 14/9 (~1.556)
        trace_C = np.trace(C)
        self.assertTrue(1.0 <= trace_C <= 2.0, f"Tr(C) = {trace_C} violates 1 <= Tr(C) <= 2 bound!")
        np.testing.assert_allclose(trace_C, 14.0 / 9.0, atol=1e-12)

    # ==========================================================================
    # 4. Hermiticity and Positivity Safeguards
    # ==========================================================================
    def test_hermiticity_and_positivity(self):
        """
        Checks that density matrices constructed via LowRankState remain Hermitian
        and positive semi-definite during solver updates.
        """
        n_spins = 4
        sys = SpinSystem(n_spins=n_spins, spin=0.5)
        H = sys.build_heisenberg_hamiltonian(J=1.0, B=(0.5, 0.5, 0.0))
        
        psi0 = np.ones(2**n_spins, dtype=np.complex128)
        psi0 /= np.linalg.norm(psi0)
        state0 = StateBuilder.pure_state(psi0)
        
        solver = QSpinSolver(H=H, equation="qllg", kappa=0.3, backend="cpu")
        state = solver.solve(state0, t_span=(0.0, 0.5), dt=0.01)
        
        # Reconstruct density matrix rho = V @ diag(lambdas) @ V^H
        rho_dense = state.V @ np.diag(state.lambda_vec) @ state.V.conj().T
        
        # Hermiticity check
        np.testing.assert_allclose(rho_dense, rho_dense.conj().T, atol=1e-12)
        
        # Positivity check
        evals = la.eigvalsh(rho_dense)
        self.assertTrue(np.all(evals >= -1e-14), f"Negative eigenvalue in rho: {np.min(evals)}")

    # ==========================================================================
    # 5. Pure-State Equivalence Test
    # ==========================================================================
    def test_pure_state_qll_qllg_equivalence(self):
        """
        Verifies that for pure states, q-LL and q-LLG yield identical trajectories
        under time rescaling t -> t / (1 + kappa^2).
        """
        n_spins = 4
        sys = SpinSystem(n_spins=n_spins, spin=0.5)
        H = sys.build_heisenberg_hamiltonian(J=1.0, D_z=0.2, B=(1.0, 0.0, 0.0))
        
        psi0 = np.zeros(2**n_spins, dtype=np.complex128)
        psi0[0] = 1.0
        state0 = StateBuilder.pure_state(psi0)
        
        kappa = 0.5
        dt = 0.01
        
        solver_qllg = QSpinSolver(H=H, equation="qllg", kappa=kappa, backend="cpu")
        solver_qll = QSpinSolver(H=H, equation="qll", kappa=kappa, backend="cpu")
        
        # q-LLG evolved to t = 1.0
        state_qllg = solver_qllg.solve(state0, t_span=(0.0, 1.0), dt=dt)
        
        # q-LL evolved to rescaled time t_rescaled = 1.0 / (1 + kappa^2) = 1.0 / 1.25 = 0.8
        t_rescaled = 1.0 / (1.0 + kappa ** 2)
        state_qll = solver_qll.solve(state0, t_span=(0.0, t_rescaled), dt=dt * t_rescaled)
        
        # Reconstruct density matrices
        rho_qllg = state_qllg.V @ np.diag(state_qllg.lambda_vec) @ state_qllg.V.conj().T
        rho_qll = state_qll.V @ np.diag(state_qll.lambda_vec) @ state_qll.V.conj().T
        
        diff = la.norm(rho_qllg - rho_qll, 'fro')
        self.assertLess(diff, 1e-4, f"Pure state q-LL vs q-LLG discrepancy: {diff}")

    # ==========================================================================
    # 6. Bipartite Entanglement Revival & CHSH Violation in Spin Dimers
    # ==========================================================================
    def test_entanglement_revival_and_chsh_violation(self):
        """
        Simulates an anisotropic 2-spin-1/2 dimer under q-LLG dynamics starting
        from an unentangled product state |up, down>. Verifies:
        1. Dynamic generation of bipartite entanglement (Concurrence C > 0.9).
        2. Violation of the CHSH Bell inequality (S_CHSH > 2.5, reaching ~ 2.828).
        3. Entanglement collapse and subsequent revival oscillations.
        """
        sys = SpinSystem(n_spins=2, spin=0.5)
        # Anisotropic dimer Hamiltonian with DMI
        H = sys.build_heisenberg_hamiltonian(J=1.0, D_z=0.5, B=(0.0, 0.0, 0.5), periodic=False)
        
        # Unentangled product state |1, 0> = |up, down>
        psi0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
        state0 = StateBuilder.pure_state(psi0)
        
        backend = Backend(use_gpu=False)
        solver = QSpinSolver(H=H, equation="qllg", kappa=0.05, backend="cpu")
        
        # At t=0: product state has C=0 and S=2.0
        c0 = wootters_concurrence_dimer(state0, backend)
        s0 = chsh_bell_parameter_dimer(state0, backend)
        self.assertAlmostEqual(c0, 0.0, places=6)
        self.assertAlmostEqual(s0, 2.0, places=6)
        
        dt = 0.02
        concurrences = []
        chsh_values = []
        
        curr_state = state0
        for _ in range(300):
            curr_state = solver.step(curr_state, dt)
            c_val = wootters_concurrence_dimer(curr_state, backend)
            s_val = chsh_bell_parameter_dimer(curr_state, backend)
            concurrences.append(c_val)
            chsh_values.append(s_val)
            
        max_c = max(concurrences)
        max_s = max(chsh_values)
        
        # Verify dynamic entanglement generation & Bell violation
        self.assertGreater(max_c, 0.9, f"Peak concurrence too low: {max_c}")
        self.assertGreater(max_s, 2.5, f"Peak CHSH parameter too low: {max_s}")
        
        # Verify revival peaks
        peaks = 0
        for i in range(1, len(concurrences)-1):
            if concurrences[i] > concurrences[i-1] and concurrences[i] > concurrences[i+1] and concurrences[i] > 0.1:
                peaks += 1
        self.assertGreaterEqual(peaks, 2, f"Expected at least 2 entanglement revivals, got {peaks}")


if __name__ == "__main__":
    unittest.main()
