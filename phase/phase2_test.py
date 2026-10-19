import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from lrei_solver import lrei_rk4_step

def build_spin_half_operators(n_spins):
    sx = sp.csr_matrix([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)
    sy = sp.csr_matrix([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex128)
    sz = sp.csr_matrix([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
    id2 = sp.identity(2, dtype=np.complex128, format='csr')

    Sx_list, Sy_list, Sz_list = [], [], []
    for i in range(n_spins):
        op_x, op_y, op_z = 1.0, 1.0, 1.0
        for j in range(n_spins):
            curr_x = sx if j == i else id2
            curr_y = sy if j == i else id2
            curr_z = sz if j == i else id2
            op_x = sp.kron(op_x, curr_x, format='csr')
            op_y = sp.kron(op_y, curr_y, format='csr')
            op_z = sp.kron(op_z, curr_z, format='csr')
        Sx_list.append(op_x)
        Sy_list.append(op_y)
        Sz_list.append(op_z)
    return Sx_list, Sy_list, Sz_list

def build_heisenberg_hamiltonian(n_spins, J=1.0, D_z=0.4, B_x=1.0):
    Sx, Sy, Sz = build_spin_half_operators(n_spins)
    N = 2**n_spins
    H = sp.csr_matrix((N, N), dtype=np.complex128)
    for i in range(n_spins):
        j = (i + 1) % n_spins
        H += J * (Sx[i] @ Sx[j] + Sy[i] @ Sy[j] + Sz[i] @ Sz[j])
        if D_z != 0.0:
            H += D_z * (Sx[i] @ Sy[j] - Sy[i] @ Sx[j])
        H -= B_x * Sx[i]
    return H

def test_phase2():
    print("==============================================================")
    print("TEST 1: Analytical Pure-State Accuracy Benchmark (N=2, r=1)")
    print("==============================================================")
    hbar, kappa, dt, t_total = 1.0, 0.2, 0.01, 5.0
    
    # Properly shaped 2x2 Pauli matrices
    sigma_x = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = sp.csr_matrix([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sigma_z = sp.csr_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    H_single = -0.5 * hbar * (sigma_x + sigma_z)
    
    psi0 = (1.0 / np.sqrt(2)) * np.array([1.0, 1.0], dtype=np.complex128)
    V_curr = psi0[:, np.newaxis]
    lambda_vec = np.array([1.0])
    
    n_steps = int(t_total / dt)
    errors = []
    for step in range(n_steps):
        t = step * dt
        H_tilde = ((1.0 - 1j * kappa) / (1.0 + kappa**2)) * H_single.toarray()
        U = la.expm(-1j * H_tilde * t / hbar)
        psi_t = U @ psi0
        rho_exact = np.outer(psi_t, psi_t.conj()) / np.real(np.vdot(psi_t, psi_t))
        
        rho_lrei = V_curr @ np.diag(lambda_vec) @ V_curr.conj().T
        errors.append(la.norm(rho_lrei - rho_exact, 'fro'))
        V_curr = lrei_rk4_step(V_curr, lambda_vec, H_single, kappa, dt, hbar)
        
    print(f"Max Error vs Analytical Solution: {np.max(errors):.2e}")
    assert np.max(errors) < 1e-9, "Analytical test failed!"
    print("Test 1 passed successfully.\n")

    print("==============================================================")
    print("TEST 2: Multi-Spin Chain with Matrix-Free ARPACK (N=64, r=2)")
    print("==============================================================")
    n_spins = 6
    N = 2**n_spins
    r = 2
    H_chain = build_heisenberg_hamiltonian(n_spins)
    
    # Rank-2 mixed initial state: 70% |AF1> + 30% |AF2>
    v1 = np.zeros(N, dtype=np.complex128); v1[int("010101", 2)] = 1.0
    v2 = np.zeros(N, dtype=np.complex128); v2[int("101010", 2)] = 1.0
    V_chain = np.column_stack([v1, v2])
    lambda_chain = np.array([0.7, 0.3])
    
    t0 = time.time()
    for _ in range(50):
        V_chain = lrei_rk4_step(V_chain, lambda_chain, H_chain, kappa, dt)
    elapsed = time.time() - t0
    
    overlap = V_chain.conj().T @ V_chain
    ortho_err = la.norm(overlap - np.eye(r))
    HV = H_chain @ V_chain
    energy = np.real(np.sum(lambda_chain * np.diag(V_chain.conj().T @ HV)))
    
    print(f"Time for 50 steps: {elapsed:.3f} s ({elapsed/50*1000:.2f} ms/step)")
    print(f"Orthonormality Error ||V^H V - I||: {ortho_err:.2e}")
    print(f"Expectation Value <H>: {energy:.6f}")
    assert ortho_err < 1e-10, "Orthonormality failed!"
    print("Test 2 passed successfully.\n")
    print("Phase 2 CPU LREI verification complete!")

if __name__ == "__main__":
    test_phase2()