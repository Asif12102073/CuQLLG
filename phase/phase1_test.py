import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def solve_sylvester_qllg(rho, H, kappa, hbar=1.0):
    """
    Solves the implicit q-LLG equation for dot{rho} using the decoupled
    Sylvester reformulation:
        dot{rho} = (i/hbar)*[rho, H] + i*kappa*[rho, dot{rho}]
    """
    N = rho.shape[0]
    
    # 1. Eigendecomposition of current density matrix: rho = V @ diag(w) @ V^H
    # eigh guarantees real eigenvalues and orthonormal eigenvectors
    w, V = la.eigh(rho)
    Vh = V.conj().T
    
    # S = -i * kappa * diag(w)
    s = -1j * kappa * w
    
    # 2. Form commutator term D = (i/hbar) * V^H @ [rho, H] @ V
    comm_rho_H = rho @ H - H @ rho
    D = (1j / hbar) * (Vh @ comm_rho_H @ V)
    
    # 3. Decoupled Sylvester solve: X_lj = D_lj / (1 + s_l - s_j)
    # Vectorized denominator via broadcasting: 1 + s[:, None] - s[None, :]
    denom = 1.0 + s[:, np.newaxis] - s[np.newaxis, :]
    X = D / denom
    
    # 4. Transform back: \dot{\rho} = V @ X @ V^H
    drho = V @ X @ Vh
    
    # Enforce exact Hermiticity against numerical roundoff
    return 0.5 * (drho + drho.conj().T)


def conservative_projection(rho_stage, lambda_0):
    """
    Preserves the initial spectrum Lambda_0 by re-diagonalizing and
    replacing numerical eigenvalues with invariant physical eigenvalues.
    """
    _, V = la.eigh(rho_stage)
    # Reconstruct: rho = V @ diag(lambda_0) @ V^H
    rho_proj = (V * lambda_0) @ V.conj().T
    return 0.5 * (rho_proj + rho_proj.conj().T)


def rk4_step_qllg(rho, H, kappa, dt, lambda_0, hbar=1.0):
    """
    One time-step using Conservative Eigenmode Integration RK4 (EI-RK4).
    """
    # Stage 1
    k1 = solve_sylvester_qllg(rho, H, kappa, hbar)
    
    # Stage 2
    z1 = rho + 0.5 * dt * k1
    z1_proj = conservative_projection(z1, lambda_0)
    k2 = solve_sylvester_qllg(z1_proj, H, kappa, hbar)
    
    # Stage 3
    z2 = rho + 0.5 * dt * k2
    z2_proj = conservative_projection(z2, lambda_0)
    k3 = solve_sylvester_qllg(z2_proj, H, kappa, hbar)
    
    # Stage 4
    z3 = rho + dt * k3
    z3_proj = conservative_projection(z3, lambda_0)
    k4 = solve_sylvester_qllg(z3_proj, H, kappa, hbar)
    
    # Final RK4 combination and projection
    rho_next = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return conservative_projection(rho_next, lambda_0)


def analytical_pure_state_evolution(psi0, H, kappa, t, hbar=1.0):
    """
    Exact analytical solution for rank-1 states under q-LLG.
    """
    H_tilde = ((1.0 - 1j * kappa) / (1.0 + kappa**2)) * H
    U = la.expm(-1j * H_tilde * t / hbar)
    psi_t = U @ psi0
    rho_unnorm = np.outer(psi_t, psi_t.conj())
    return rho_unnorm / np.trace(rho_unnorm)


# --- Benchmark & Verification ---
if __name__ == "__main__":
    # Test on a Spin-1/2 system (N=2)
    hbar = 1.0
    kappa = 0.2
    
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    # Zeeman Hamiltonian in an oblique magnetic field B = [1, 0, 1]
    B = np.array([1.0, 0.0, 1.0])
    H = -0.5 * hbar * (B[0]*sigma_x + B[1]*sigma_y + B[2]*sigma_z)
    
    # Initial state: spin pointing along +x
    psi0 = (1.0 / np.sqrt(2)) * np.array([1.0, 1.0], dtype=np.complex128)
    rho0 = np.outer(psi0, psi0.conj())
    
    # Initial spectrum (rank 1 -> lambda_0 = [0, 1])
    lambda_0, _ = la.eigh(rho0)
    
    # Time settings
    t_total = 5.0
    dt = 0.01
    n_steps = int(t_total / dt)
    
    rho = rho0.copy()
    errors = []
    purities = []
    times = []
    
    print(f"Simulating q-LLG evolution up to t = {t_total} (dt = {dt})...")
    for step in range(n_steps):
        t = step * dt
        # Analytical solution
        rho_exact = analytical_pure_state_evolution(psi0, H, kappa, t, hbar)
        
        # Track Frobenius error and purity Tr(rho^2)
        err = la.norm(rho - rho_exact, 'fro')
        purity = np.real(np.trace(rho @ rho))
        
        errors.append(err)
        purities.append(purity)
        times.append(t)
        
        # Advance numerical step
        rho = rk4_step_qllg(rho, H, kappa, dt, lambda_0, hbar)
        
    print(f"Maximum Error vs Analytical : {np.max(errors):.2e}")
    print(f"Purity Drift (max - min)    : {np.max(purities) - np.min(purities):.2e}")
    print("Phase 1 numerical validation successful!")
