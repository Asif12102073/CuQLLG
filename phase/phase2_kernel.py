import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numba

# ---------------------------------------------------------------------------
# 1. Numba-Accelerated Householder Kernels
# ---------------------------------------------------------------------------
@numba.njit(fastmath=True)
def householder_qr_numba(V_tilde):
    """
    Computes Householder vectors u_k and factors beta_k for V_tilde in C^{N x r}.
    Cost: ~ 2*N*r^2 flops.
    """
    N, r = V_tilde.shape
    V_work = V_tilde.copy()
    U_house = np.zeros((N, r), dtype=np.complex128)
    betas = np.zeros(r, dtype=np.float64)
    
    for k in range(r):
        x = V_work[k:, k].copy()
        norm_x = np.linalg.norm(x)
        if norm_x > 1e-15:
            phase = x / np.abs(x) if np.abs(x) > 1e-15 else 1.0 + 0.0j
            alpha = -phase * norm_x
            u = x.copy()
            u -= alpha
            norm_u_sq = np.real(np.vdot(u, u))
            beta = 2.0 / norm_u_sq
        else:
            u = np.zeros_like(x)
            beta = 0.0
            
        U_house[k:, k] = u
        betas[k] = beta
        
        # Apply P_k to remaining columns of V_work: V = V - beta * u * (u^H V)
        for j in range(k, r):
            s = 0.0 + 0.0j
            for i in range(k, N):
                s += np.conj(U_house[i, k]) * V_work[i, j]
            s_beta = beta * s
            for i in range(k, N):
                V_work[i, j] -= s_beta * U_house[i, k]
                
    return U_house, betas

@numba.njit(fastmath=True)
def apply_mat_V_numba(A, U_house, betas):
    """
    Computes A @ V = A @ P_1 ... P_r for A in C^{m x N}.
    Cost: ~ 4*m*N*r flops.
    """
    m, N = A.shape
    r = U_house.shape[1]
    res = A.copy()
    
    for k in range(r):
        beta = betas[k]
        if beta == 0.0:
            continue
        w = np.zeros(m, dtype=np.complex128)
        for i in range(m):
            s = 0.0 + 0.0j
            for j in range(k, N):
                s += res[i, j] * U_house[j, k]
            w[i] = s
            
        for i in range(m):
            w_beta = beta * w[i]
            for j in range(k, N):
                res[i, j] -= w_beta * np.conj(U_house[j, k])
                
    return res

@numba.njit(fastmath=True)
def apply_V_mat_numba(U_house, betas, B):
    """
    Computes V @ B = P_1 ... P_r @ B for B in C^{N x m}.
    Cost: ~ 4*m*N*r flops.
    """
    N, m = B.shape
    r = U_house.shape[1]
    res = B.copy()
    
    for k in range(r - 1, -1, -1):
        beta = betas[k]
        if beta == 0.0:
            continue
        w = np.zeros(m, dtype=np.complex128)
        for j in range(m):
            s = 0.0 + 0.0j
            for i in range(k, N):
                s += np.conj(U_house[i, k]) * res[i, j]
            w[j] = s
            
        for j in range(m):
            w_val = w[j]
            for i in range(k, N):
                res[i, j] -= beta * U_house[i, k] * w_val
                
    return res

# ---------------------------------------------------------------------------
# 2. Low-Rank Right-Hand Side (Z, W) Assembly
# ---------------------------------------------------------------------------
def compute_lrei_rhs(V_tilde, lambda_vec, H, kappa, hbar=1.0, method="projector"):
    """
    Evaluates the factored time derivative: \dot{\rho} = Z \tilde{V}^* + \tilde{V} W^*
    Returns Z, W \in C^{N x r}.
    """
    N, r = V_tilde.shape
    Lambda_tilde = np.diag(lambda_vec)
    
    # 1. Sparse matrix-vector product H @ V_tilde: O(N * r * nnz_per_row)
    HV = H @ V_tilde
    
    # 2. X11 block evaluation: O(N * r^2)
    Vt_HV = V_tilde.conj().T @ HV
    comm_11 = Lambda_tilde @ Vt_HV - Vt_HV @ Lambda_tilde
    D11 = (1j / hbar) * comm_11
    
    s = -1j * kappa * lambda_vec
    denom_11 = 1.0 + s[:, np.newaxis] - s[np.newaxis, :]
    X11 = D11 / denom_11
    
    # 3. Complement coupling block W = \hat{V} X_{12}^*
    if method == "projector":
        # Direct projection: W = (I - V_tilde V_tilde^H) M
        scale_vec = -1j / (hbar * (1.0 + 1j * kappa * lambda_vec))
        M = HV @ (Lambda_tilde * scale_vec[np.newaxis, :])
        W = M - V_tilde @ (V_tilde.conj().T @ M)
    elif method == "householder":
        # Householder QR evaluation
        U_house, betas = householder_qr_numba(V_tilde)
        A = (1j / hbar) * (Lambda_tilde @ (V_tilde.conj().T @ H))
        AV = apply_mat_V_numba(A, U_house, betas)
        D12 = AV[:, r:]
        denom_12 = 1.0 - 1j * kappa * lambda_vec
        X12 = D12 / denom_12[:, np.newaxis]
        B0 = np.zeros((N, r), dtype=np.complex128)
        B0[r:, :] = X12.conj().T
        W = apply_V_mat_numba(U_house, betas, B0)
    else:
        raise ValueError(f"Unknown method '{method}'")
        
    Z = V_tilde @ X11 + W
    return Z, W

# ---------------------------------------------------------------------------
# 3. Matrix-Free Krylov Eigensolver for RK Stage Diagonalization
# ---------------------------------------------------------------------------
def diagonalize_stage_operator(A_list, V_list, B_list, N, r, V_guess=None):
    """
    Computes dominant r eigenmodes of K = \sum_l [ A_l V_l^H + V_l B_l^H ]
    using SciPy's ARPACK IRLM (eigsh) without materializing N x N matrices.
    """
    def matvec(x):
        res = np.zeros(N, dtype=np.complex128)
        for A_l, V_l, B_l in zip(A_list, V_list, B_list):
            c1 = V_l.conj().T @ x
            c2 = B_l.conj().T @ x
            res += A_l @ c1 + V_l @ c2
        return res

    op = spla.LinearOperator((N, N), matvec=matvec, dtype=np.complex128)
    
    ncv = max(2 * r + 1, 8)
    if ncv >= N:
        # Fallback to dense eigh for small test systems (N <= 8)
        K_dense = np.zeros((N, N), dtype=np.complex128)
        for A_l, V_l, B_l in zip(A_list, V_list, B_list):
            K_dense += A_l @ V_l.conj().T + V_l @ B_l.conj().T
        K_dense = 0.5 * (K_dense + K_dense.conj().T)
        vals, vecs = la.eigh(K_dense)
        idx = np.argsort(vals)[-r:]
        return vecs[:, idx]
        
    v0 = V_guess[:, 0] if V_guess is not None else None
    vals, vecs = spla.eigsh(op, k=r, which='LA', ncv=ncv, v0=v0, tol=1e-12)
    # Sort descending to align with sorted Lambda_0
    idx = np.argsort(vals)[::-1]
    return vecs[:, idx]

# ---------------------------------------------------------------------------
# 4. Conservative LREI-RK4 Step
# ---------------------------------------------------------------------------
def lrei_rk4_step(V_tilde, lambda_vec, H, kappa, dt, hbar=1.0, method="projector"):
    """
    Executes one time-step of the Conservative Low-Rank Eigenmode Integration RK4.
    """
    N, r = V_tilde.shape
    Lambda_0 = np.diag(lambda_vec)
    
    # Stage 1
    V1 = V_tilde
    Z1, W1 = compute_lrei_rhs(V1, lambda_vec, H, kappa, hbar, method=method)
    
    # Stage 2: K2 = V1 Lambda_0 V1^H + 0.5*dt*(Z1 V1^H + V1 W1^H)
    A1_s2 = V1 @ Lambda_0 + 0.5 * dt * Z1
    B1_s2 = 0.5 * dt * W1
    V2 = diagonalize_stage_operator([A1_s2], [V1], [B1_s2], N, r, V_guess=V1)
    Z2, W2 = compute_lrei_rhs(V2, lambda_vec, H, kappa, hbar, method=method)
    
    # Stage 3: K3 = V1 Lambda_0 V1^H + 0.5*dt*(Z2 V2^H + V2 W2^H)
    A1_s3 = V1 @ Lambda_0
    B1_s3 = np.zeros_like(V1)
    A2_s3 = 0.5 * dt * Z2
    B2_s3 = 0.5 * dt * W2
    V3 = diagonalize_stage_operator([A1_s3, A2_s3], [V1, V2], [B1_s3, B2_s3], N, r, V_guess=V2)
    Z3, W3 = compute_lrei_rhs(V3, lambda_vec, H, kappa, hbar, method=method)
    
    # Stage 4: K4 = V1 Lambda_0 V1^H + dt*(Z3 V3^H + V3 W3^H)
    A1_s4 = V1 @ Lambda_0
    B1_s4 = np.zeros_like(V1)
    A3_s4 = dt * Z3
    B3_s4 = dt * W3
    V4 = diagonalize_stage_operator([A1_s4, A3_s4], [V1, V3], [B1_s4, B3_s4], N, r, V_guess=V3)
    Z4, W4 = compute_lrei_rhs(V4, lambda_vec, H, kappa, hbar, method=method)
    
    # Assembly & Final Eigendecomposition
    A1_fin = V1 @ Lambda_0 + (dt / 6.0) * Z1
    B1_fin = (dt / 6.0) * W1
    A2_fin = (dt / 3.0) * Z2
    B2_fin = (dt / 3.0) * W2
    A3_fin = (dt / 3.0) * Z3
    B3_fin = (dt / 3.0) * W3
    A4_fin = (dt / 6.0) * Z4
    B4_fin = (dt / 6.0) * W4
    
    V_next = diagonalize_stage_operator(
        [A1_fin, A2_fin, A3_fin, A4_fin],
        [V1, V2, V3, V4],
        [B1_fin, B2_fin, B3_fin, B4_fin],
        N, r, V_guess=V4
    )
    return V_next
