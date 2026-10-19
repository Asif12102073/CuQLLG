"""
Phase 4: Comparative Simulation of q-LL vs. q-LLG Dynamics.
Demonstrating dynamical inequivalence for a single spin-1 system with crystal anisotropy.
"""

import numpy as np
import scipy.linalg as la
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# 1. Operators & Hamiltonian for Spin-1 System (s = 1)
# ---------------------------------------------------------------------------
def build_spin_one_system(gamma_g_B0=0.1, B0=1.0, kappa=0.5):
    # Spin-1 representation in |1>, |0>, |-1> basis
    Sx = (1.0 / np.sqrt(2.0)) * np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.complex128)

    Sy = (1.0 / (np.sqrt(2.0) * 1j)) * np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0],
        [0.0, -1.0, 0.0]
    ], dtype=np.complex128)

    Sz = np.diag([1.0, 0.0, -1.0]).astype(np.complex128)

    # Oblique magnetic field at 45 degrees in x-z plane: B = (Bx, By, Bz)
    Bx = B0 / np.sqrt(2.0)
    By = 0.0
    Bz = B0 / np.sqrt(2.0)
    
    # Anisotropy parameters: K_perp / (gamma_g * B0) = 0.3, K_para / (gamma_g * B0) = -0.1
    K_perp = 0.3 * gamma_g_B0
    K_para = -0.1 * gamma_g_B0

    # Zeeman term: -gamma_g * (B . S)
    H_zeeman = - (gamma_g_B0 / B0) * (Bx * Sx + By * Sy + Bz * Sz)
    
    # Crystal-field anisotropy term: K_perp * Sx^2 + K_para * Sz^2
    H_aniso = K_perp * (Sx @ Sx) + K_para * (Sz @ Sz)
    
    H = H_zeeman + H_aniso
    return (Sx, Sy, Sz), H


# ---------------------------------------------------------------------------
# 2. Right-Hand Side Functions for q-LLG and q-LL
# ---------------------------------------------------------------------------
def compute_qllg_rhs(rho, H, kappa, hbar=1.0):
    w, V = la.eigh(rho)
    Vh = V.conj().T
    s = -1j * kappa * w
    comm = rho @ H - H @ rho
    D = (1j / hbar) * (Vh @ comm @ V)
    denom = 1.0 + s[:, np.newaxis] - s[np.newaxis, :]
    X = D / denom
    drho = V @ X @ Vh
    return 0.5 * (drho + drho.conj().T)


def compute_qll_rhs(rho, H, kappa, hbar=1.0):
    comm = rho @ H - H @ rho
    # Double-bracket dissipative term: -kappa/hbar * [rho, [rho, H]]
    double_comm = rho @ comm - comm @ rho
    drho = (1j / hbar) * comm - (kappa / hbar) * double_comm
    return 0.5 * (drho + drho.conj().T)


def conservative_projection(rho_stage, lambda_0):
    _, V = la.eigh(rho_stage)
    rho_proj = (V * lambda_0) @ V.conj().T
    return 0.5 * (rho_proj + rho_proj.conj().T)


def rk4_step(rho, H, kappa, dt, lambda_0, rhs_func, hbar=1.0):
    k1 = rhs_func(rho, H, kappa, hbar)
    z1 = conservative_projection(rho + 0.5 * dt * k1, lambda_0)
    k2 = rhs_func(z1, H, kappa, hbar)
    z2 = conservative_projection(rho + 0.5 * dt * k2, lambda_0)
    k3 = rhs_func(z2, H, kappa, hbar)
    z3 = conservative_projection(rho + dt * k3, lambda_0)
    k4 = rhs_func(z3, H, kappa, hbar)
    rho_next = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return conservative_projection(rho_next, lambda_0)


# ---------------------------------------------------------------------------
# 3. Observables Computation
# ---------------------------------------------------------------------------
def compute_observables(rho, S_tuple):
    Sx, Sy, Sz = S_tuple
    S_ops = [Sx, Sy, Sz]
    S_exp = np.array([np.real(np.trace(rho @ op)) for op in S_ops])
    return S_exp


# ---------------------------------------------------------------------------
# 4. Simulation & Rescaling Misfit Analysis
# ---------------------------------------------------------------------------
def run_phase4_benchmark():
    print("==============================================================")
    print("PHASE 4: q-LL vs. q-LLG Dynamical Inequivalence Benchmark")
    print("==============================================================")
    
    gamma_g_B0 = 0.1  # timescale ~ 10 ps
    B0 = 1.0
    kappa = 0.5
    dt = 0.1
    t_total = 300.0   # ps
    n_steps = int(t_total / dt)
    
    S_ops, H = build_spin_one_system(gamma_g_B0, B0, kappa)
    Sx, Sy, Sz = S_ops
    
    # Spin-type initial state: rho0 = (1/3)*(I + m0*Sz), m0 = 1 -> rho0 = diag(2/3, 1/3, 0)
    rho0 = (1.0 / 3.0) * (np.eye(3, dtype=np.complex128) + Sz)
    lambda_0, _ = la.eigh(rho0)
    
    print(f"Simulating q-LLG and q-LL trajectories up to t = {t_total} ps...")
    rho_llg = rho0.copy()
    rho_ll = rho0.copy()
    
    times = np.zeros(n_steps)
    S_llg_traj = np.zeros((n_steps, 3))
    S_ll_traj = np.zeros((n_steps, 3))
    
    for step in range(n_steps):
        t = step * dt
        times[step] = t
        
        S_llg_traj[step] = compute_observables(rho_llg, S_ops)
        S_ll_traj[step] = compute_observables(rho_ll, S_ops)
        
        rho_llg = rk4_step(rho_llg, H, kappa, dt, lambda_0, compute_qllg_rhs)
        rho_ll = rk4_step(rho_ll, H, kappa, dt, lambda_0, compute_qll_rhs)
        
    print("Evolution complete. Evaluating temporal rescaling misfits R_j(zeta)...")
    
    # Interpolators for q-LLG to evaluate S_j(zeta * t)
    interp_llg = [
        interp1d(times, S_llg_traj[:, j], kind='cubic', fill_value='extrapolate')
        for j in range(3)
    ]
    
    # Evaluation grid t in [0, 250] ps
    eval_times = np.linspace(0.0, 250.0, 2500)
    S_ll_eval = [
        interp1d(times, S_ll_traj[:, j], kind='cubic')(eval_times)
        for j in range(3)
    ]
    
    zetas = np.linspace(1.020, 1.032, 121)
    R_misfits = { 'x': [], 'y': [], 'z': [] }
    keys = ['x', 'y', 'z']
    
    for z in zetas:
        for j, k in enumerate(keys):
            scaled_llg = interp_llg[j](z * eval_times)
            misfit = np.mean((scaled_llg - S_ll_eval[j])**2)
            R_misfits[k].append(misfit)
            
    # Locate minima for each spin component
    opt_zetas = {}
    min_misfits = {}
    for k in keys:
        min_idx = np.argmin(R_misfits[k])
        opt_zetas[k] = zetas[min_idx]
        min_misfits[k] = R_misfits[k][min_idx]
        print(f"Optimal zeta_{k}: {opt_zetas[k]:.4f} (Minimum Misfit: {min_misfits[k]:.2e})")
        
    delta_zeta = abs(opt_zetas['z'] - opt_zetas['x'])
    print(f"\nRescaling Factor Spread |zeta_z - zeta_x|: {delta_zeta:.4f}")
    assert delta_zeta > 5e-4, "Misfit minima should be split by crystal anisotropy!"
    print("Inequivalence successfully demonstrated: no common temporal rescaling exists.")

if __name__ == "__main__":
    run_phase4_benchmark()