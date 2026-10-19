"""
CuPy-Accelerated Low-Rank Eigenmode Integration (LREI) Solver for q-LLG.
Designed for execution on NVIDIA CUDA GPUs.
"""

import cupy as cp
import cupyx.scipy.sparse as cp_sp

def diagonalize_stage_gpu(A_list, V_list, B_list, N, r):
    """
    Subspace Rayleigh-Ritz method on GPU:
    Computes exact dominant r eigenvectors of K = sum_l [ A_l V_l^H + V_l B_l^H ]
    using cuSOLVER QR and tiny-matrix eigh without iterative Lanczos overhead.
    """
    # 1. Assemble generating column blocks in VRAM
    blocks = []
    for A, V, B in zip(A_list, V_list, B_list):
        blocks.append(A)
        blocks.append(V)
        if B is not None:
            blocks.append(B)
            
    U = cp.hstack(blocks)  # Shape (N, d) with d <= 12*r
    
    # 2. Reduced QR on GPU: O(N * d^2)
    Q, _ = cp.linalg.qr(U, mode='reduced')
    
    # 3. Matrix-free evaluation of K @ Q on GPU
    KQ = cp.zeros_like(Q)
    for A, V, B in zip(A_list, V_list, B_list):
        KQ += A @ (V.conj().T @ Q)
        if B is not None:
            KQ += V @ (B.conj().T @ Q)
            
    # 4. Project into subspace: H_sub in C^{d x d}
    H_sub = 0.5 * (Q.conj().T @ KQ + (KQ).conj().T @ Q)
    
    # 5. Diagonalize tiny matrix on GPU
    vals, vecs = cp.linalg.eigh(H_sub)
    idx = cp.argsort(vals)[::-1][:r]
    
    # 6. Reconstruct dominant eigenvectors on GPU
    return Q @ vecs[:, idx]


def compute_lrei_rhs_gpu(V_tilde, lambda_vec, H, kappa, hbar=1.0):
    """
    Evaluates the factored time derivative \dot{\rho} = Z \tilde{V}^* + \tilde{V} W^* on GPU.
    Returns Z, W in C^{N x r}.
    """
    N, r = V_tilde.shape
    Lambda_tilde = cp.diag(lambda_vec)
    
    # 1. cuSPARSE SpMM: H @ V_tilde
    HV = H @ V_tilde
    
    # 2. X11 block evaluation
    Vt_HV = V_tilde.conj().T @ HV
    comm_11 = Lambda_tilde @ Vt_HV - Vt_HV @ Lambda_tilde
    D11 = (1j / hbar) * comm_11
    
    s = -1j * kappa * lambda_vec
    denom_11 = 1.0 + s[:, cp.newaxis] - s[cp.newaxis, :]
    X11 = D11 / denom_11
    
    # 3. Orthogonal complement coupling block W = (I - V V^H) M
    scale_vec = -1j / (hbar * (1.0 + 1j * kappa * lambda_vec))
    M = HV @ (Lambda_tilde * scale_vec[cp.newaxis, :])
    W = M - V_tilde @ (V_tilde.conj().T @ M)
    
    Z = V_tilde @ X11 + W
    return Z, W


def lrei_rk4_step_gpu(V_tilde, lambda_vec, H, kappa, dt, hbar=1.0):
    """
    Executes one time-step of the Conservative LREI-RK4 entirely in GPU memory.
    """
    N, r = V_tilde.shape
    Lambda_0 = cp.diag(lambda_vec)
    
    # Stage 1
    V1 = V_tilde
    Z1, W1 = compute_lrei_rhs_gpu(V1, lambda_vec, H, kappa, hbar)
    
    # Stage 2
    A1_s2 = V1 @ Lambda_0 + 0.5 * dt * Z1
    B1_s2 = 0.5 * dt * W1
    V2 = diagonalize_stage_gpu([A1_s2], [V1], [B1_s2], N, r)
    Z2, W2 = compute_lrei_rhs_gpu(V2, lambda_vec, H, kappa, hbar)
    
    # Stage 3
    A1_s3 = V1 @ Lambda_0
    A2_s3 = 0.5 * dt * Z2
    B2_s3 = 0.5 * dt * W2
    V3 = diagonalize_stage_gpu([A1_s3, A2_s3], [V1, V2], [None, B2_s3], N, r)
    Z3, W3 = compute_lrei_rhs_gpu(V3, lambda_vec, H, kappa, hbar)
    
    # Stage 4
    A1_s4 = V1 @ Lambda_0
    A3_s4 = dt * Z3
    B3_s4 = dt * W3
    V4 = diagonalize_stage_gpu([A1_s4, A3_s4], [V1, V3], [None, B3_s4], N, r)
    Z4, W4 = compute_lrei_rhs_gpu(V4, lambda_vec, H, kappa, hbar)
    
    # Final Combination
    A1_fin = V1 @ Lambda_0 + (dt / 6.0) * Z1
    B1_fin = (dt / 6.0) * W1
    A2_fin = (dt / 3.0) * Z2
    B2_fin = (dt / 3.0) * W2
    A3_fin = (dt / 3.0) * Z3
    B3_fin = (dt / 3.0) * W3
    A4_fin = (dt / 6.0) * Z4
    B4_fin = (dt / 6.0) * W4
    
    V_next = diagonalize_stage_gpu(
        [A1_fin, A2_fin, A3_fin, A4_fin],
        [V1, V2, V3, V4],
        [B1_fin, B2_fin, B3_fin, B4_fin],
        N, r
    )
    return V_next
