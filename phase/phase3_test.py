import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as cp_sp

from cupy_lrei_solver import lrei_rk4_step_gpu

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


def test_phase3_gpu():
    dev = cp.cuda.Device()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()
    print("==============================================================s")
    print(f"CUDA DEVICE: {dev_name} (Device ID: {dev.id})")
    print("==============================================================")
    
    # ------------------------------------------------------------------
    # TEST 1: Pure-State Analytical Benchmark on GPU (N=2, r=1)
    # ------------------------------------------------------------------
    print("\nTEST 1: Pure-State Analytical Accuracy on GPU")
    hbar, kappa, dt, t_total = 1.0, 0.2, 0.01, 5.0
    sigma_x = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_z = sp.csr_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    H_single = -0.5 * hbar * (sigma_x + sigma_z)
    
    # Transfer H to GPU once
    H_single_gpu = cp_sp.csr_matrix(H_single)
    
    psi0 = (1.0 / np.sqrt(2)) * np.array([1.0, 1.0], dtype=np.complex128)
    V_gpu = cp.array(psi0[:, np.newaxis])
    lambda_gpu = cp.array([1.0], dtype=cp.float64)
    
    n_steps = int(t_total / dt)
    for _ in range(n_steps):
        V_gpu = lrei_rk4_step_gpu(V_gpu, lambda_gpu, H_single_gpu, kappa, dt, hbar)
        
    dev.synchronize()
    
    # Compare with exact analytical solution
    H_tilde = ((1.0 - 1j * kappa) / (1.0 + kappa**2)) * H_single.toarray()
    U = la.expm(-1j * H_tilde * t_total / hbar)
    psi_t = U @ psi0
    rho_exact = np.outer(psi_t, psi_t.conj()) / np.real(np.vdot(psi_t, psi_t))
    
    V_final = V_gpu.get()
    rho_gpu = V_final @ np.diag(lambda_gpu.get()) @ V_final.conj().T
    err = la.norm(rho_gpu - rho_exact, 'fro')
    print(f"Max Error vs Analytical Solution on GPU: {err:.2e}")
    assert err < 1e-9, "GPU analytical test failed!"
    print("Test 1 passed successfully.")

    # ------------------------------------------------------------------
    # TEST 2: Multi-Spin Scaling Benchmarks (n=6, n=10, n=12)
    # ------------------------------------------------------------------
    print("\nTEST 2: GPU Multi-Spin Performance & Scaling Benchmarks")
    spin_counts = [6, 8,10, 12, 14, 16,18,20] # 20 for 8GB GPU 22 for 10GB GPU 
    r = 2
    steps_bench = 50
    
    for n_spins in spin_counts:
        N = 2**n_spins
        print(f"\n--- Simulating n = {n_spins} spins (Hilbert Space N = {N:,}, Rank r = {r}) ---")
        
        # Build Hamiltonian on CPU and transfer to GPU
        H_cpu = build_heisenberg_hamiltonian(n_spins)
        H_gpu = cp_sp.csr_matrix(H_cpu)
        
        # Initial rank-2 antiferromagnetic states
        v1 = np.zeros(N, dtype=np.complex128); v1[0] = 1.0
        v2 = np.zeros(N, dtype=np.complex128); v2[-1] = 1.0
        V_gpu = cp.array(np.column_stack([v1, v2]))
        lambda_gpu = cp.array([0.7, 0.3], dtype=cp.float64)
        
        # Warmup GPU
        V_gpu = lrei_rk4_step_gpu(V_gpu, lambda_gpu, H_gpu, kappa, dt)
        dev.synchronize()
        
        # Timed benchmark
        t0 = time.time()
        for _ in range(steps_bench):
            V_gpu = lrei_rk4_step_gpu(V_gpu, lambda_gpu, H_gpu, kappa, dt)
        dev.synchronize()
        elapsed = time.time() - t0
        
        # Validation in GPU memory
        overlap = V_gpu.conj().T @ V_gpu
        ortho_err = float(cp.linalg.norm(overlap - cp.eye(r)).get())
        
        HV = H_gpu @ V_gpu
        energy = float(cp.real(cp.sum(lambda_gpu * cp.diag(V_gpu.conj().T @ HV))).get())
        
        step_ms = (elapsed / steps_bench) * 1000
        print(f"GPU Runtime for {steps_bench} steps : {elapsed:.3f} s ({step_ms:.2f} ms/step)")
        print(f"Orthonormality Error ||V^H V - I|| : {ortho_err:.2e}")
        print(f"Expectation Value <H>             : {energy:.6f}")
        assert ortho_err < 1e-10, "Orthonormality check failed on GPU!"
        
    print("\nPhase 3 CuPy GPU validation and benchmarking complete!")


if __name__ == "__main__":
    test_phase3_gpu()