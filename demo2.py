"""
Benchmark qllg (CuQLLG, LREI-RK4) on the 10-spin Heisenberg ring from demo_qllg.py.

Findings baked in (verified against the repo source):
  * Requires the fixed SpinSystem.build_heisenberg_hamiltonian (Zeeman applied once).
    The script asserts qllg's H equals the reference H before comparing.
  * Cases: B = 1 (the demo) and B = 10 (a stiffer test of the LREI-RK4 step size).

Reference methods
  1. Exact unitary dynamics (energy conserved)
  2. Exact diagonalisation: E_min / E_max bounds
  3. Exact imaginary-time evolution from the same rank-2 Neel state
  4. Dense full-rho RK4 of the SAME q-LLG equation (tiny dt, no low-rank/Rayleigh-Ritz
     truncation) -> checks the LREI integrator, not the equation
  5. Classical atomistic LLG (10 macrospins, Gilbert alpha = kappa)
  6. qllg itself (GPU if available, else CPU)
"""
import sys, time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

N, J, D, KAPPA, DT, T = 10, 1.0, 0.3, 0.1, 0.01, 1.0
S = 0.5
sx = sp.csr_matrix([[0, .5], [.5, 0]], dtype=complex)
sy = sp.csr_matrix([[0, -.5j], [.5j, 0]], dtype=complex)
sz = sp.csr_matrix([[.5, 0], [0, -.5]], dtype=complex)


def op(o, i):
    return sp.kron(sp.kron(sp.identity(2 ** i), o), sp.identity(2 ** (N - i - 1)), format="csr")


X, Y, Z = ([op(o, i) for i in range(N)] for o in (sx, sy, sz))


def build_H(B_eff):
    H = sp.csr_matrix((2 ** N, 2 ** N), dtype=complex)
    for i in range(N):
        j = (i + 1) % N
        H += J * (X[i] @ X[j] + Y[i] @ Y[j] + Z[i] @ Z[j]) + D * (X[i] @ Y[j] - Y[i] @ X[j])
        H -= B_eff * X[i]
    return H.tocsr()


def neel_rho():
    m1 = sum(1 << b for b in range(0, N, 2))
    m2 = sum(1 << b for b in range(1, N, 2))
    r = np.zeros((2 ** N, 2 ** N), dtype=complex)
    r[m1, m1], r[m2, m2] = 0.7, 0.3          # same weights as StateBuilder.antiferromagnetic
    return r, m1, m2


def energy(H, rho):
    return float(np.real(H.multiply(rho.T).sum()))


# ---- dense reference of the q-LLG equation (same equation as solvers.py, hbar = 1)
def qllg_rhs(rho, H, kappa, Om):
    # top-r eigenpairs of rho by 2-step subspace iteration (Om has r + 6 random columns)
    r = 2
    Q, _ = np.linalg.qr(rho @ (rho @ Om))
    sm = Q.conj().T @ rho @ Q
    lam, u = np.linalg.eigh(0.5 * (sm + sm.conj().T))
    lam, V = lam[-r:], (Q @ u)[:, -r:]
    HV = H @ V
    M11 = V.conj().T @ HV
    L = np.diag(lam)
    D11 = 1j * (L @ M11 - M11 @ L)
    s = -1j * kappa * lam
    X11 = D11 / (1 + s[:, None] - s[None, :])
    M = HV @ np.diag(lam * (-1j / (1 + 1j * kappa * lam)))
    W = M - V @ (V.conj().T @ M)
    return (V @ X11) @ V.conj().T + W @ V.conj().T + V @ W.conj().T


def dense_qllg(H, rho0, dt, nsteps, every):
    Om = np.random.default_rng(0).standard_normal((2 ** N, 8)) + 0j
    rho, E, tr = rho0.copy(), [energy(H, rho0)], []
    for n in range(nsteps):
        k1 = qllg_rhs(rho, H, KAPPA, Om)
        k2 = qllg_rhs(rho + .5 * dt * k1, H, KAPPA, Om)
        k3 = qllg_rhs(rho + .5 * dt * k2, H, KAPPA, Om)
        k4 = qllg_rhs(rho + dt * k3, H, KAPPA, Om)
        rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (n + 1) % every == 0:
            E.append(energy(H, rho))
    return np.array(E), float(np.real(np.trace(rho)))


# ---- classical LLG
def llg(B_eff, nsteps, dt, alpha):
    def grad(m):
        p, q = np.roll(m, -1, 0), np.roll(m, 1, 0)
        g = S ** 2 * J * (p + q)
        g[:, 0] += S ** 2 * D * (p[:, 1] - q[:, 1])
        g[:, 1] += S ** 2 * D * (-p[:, 0] + q[:, 0])
        g[:, 0] -= S * B_eff
        return g

    def f(m):
        Be = -grad(m) / S
        c = np.cross(m, Be)
        return (c - alpha * np.cross(m, c)) / (1 + alpha ** 2)

    def E(m):
        p = np.roll(m, -1, 0)
        return (S ** 2 * J * np.sum(m * p) + S ** 2 * D * np.sum(m[:, 0] * p[:, 1] - m[:, 1] * p[:, 0])
                - S * B_eff * m[:, 0].sum())

    m = np.zeros((N, 3)); m[:, 2] = [(-1) ** i for i in range(N)]
    out = [E(m)]
    for _ in range(nsteps):
        k1 = f(m); k2 = f(m + .5 * dt * k1); k3 = f(m + .5 * dt * k2); k4 = f(m + dt * k3)
        m = m + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        out.append(E(m))
    return np.array(out)


# ---- qllg
try:
    from qllg import SpinSystem, StateBuilder, QSpinSolver, expectation_value
    HAVE_Q = True
except ImportError:
    HAVE_Q = False
    print("qllg not importable: plotting references only.")


def run_qllg(H, backend):
    st0 = StateBuilder.antiferromagnetic(n_spins=N, rank=2)
    sol = QSpinSolver(H=H, equation="qllg", kappa=KAPPA, backend=backend)
    hist = []
    t0 = time.time()
    sol.solve(st0, t_span=(0.0, T), dt=DT, callback=lambda t, s: hist.append(expectation_value(H, s, sol.backend)))
    return np.array(hist), time.time() - t0


nsteps = int(round(T / DT))
ts = np.linspace(0, T, nsteps + 1)
rho0, ka, kb = neel_rho()
cases = {"B = 1 (demo)": 1.0, "B = 10 (stiff test)": 10.0}
res = {}
for name, Beff in cases.items():
    H = build_H(Beff)
    ev, vec = np.linalg.eigh(H.toarray())
    w = 0.7 * np.abs(vec[ka]) ** 2 + 0.3 * np.abs(vec[kb]) ** 2
    taus = np.linspace(0, 3, 301)
    ite = np.array([(w * np.exp(-2 * ev * t) * ev).sum() / (w * np.exp(-2 * ev * t)).sum() for t in taus])
    Eref, tr = dense_qllg(H, rho0, DT / 4, 4 * nsteps, 4)
    r = dict(H=H, Emin=ev[0], Emax=ev[-1], E0=energy(H, rho0), taus=taus, ite=ite,
             ref=Eref, tr=tr, llg=llg(Beff, nsteps, DT, KAPPA))
    if HAVE_Q:
        Hq = SpinSystem(N, 0.5).build_heisenberg_hamiltonian(J=J, D_z=D, B=(Beff, 0, 0))
        dH = abs(Hq - H).max()
        print(f"{name}: max|H_qllg - H_reference| = {dH:.1e}")
        assert dH < 1e-12, "qllg Hamiltonian differs from the reference (check build_heisenberg_hamiltonian)"
        try:
            r["q"], r["tq"] = run_qllg(Hq, "gpu")
        except Exception:
            r["q"], r["tq"] = run_qllg(Hq, "cpu")
    res[name] = r
    print(f"{name}: E in [{ev[0]:.3f},{ev[-1]:.3f}]  dense qLLG ref final={Eref[-1]:.4f} Tr={tr:.6f}"
          + (f"  qllg final={r['q'][-1]:.4f} max|dE|={np.abs(r['q']-Eref).max():.2e}" if HAVE_Q else ""))

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for k, (name, r) in enumerate(res.items()):
    a = ax[0, k]
    a.plot(ts, r["ref"], "C0", lw=4, alpha=.4, label="Dense full-rho RK4 (same eq., dt/4)")
    if HAVE_Q:
        a.plot(ts, r["q"], "k--", label="qllg LREI-RK4")
    a.plot(ts, r["llg"], "C2", label="Classical atomistic LLG")
    a.axhline(r["E0"], color="gray", ls=":", label="Exact unitary (conserved)")
    a.axhline(r["Emin"], color="r", ls="--", label=f"Exact E0 = {r['Emin']:.2f}")
    a.set(title=name, xlabel="t", ylabel="<H>"); a.legend(fontsize=7)
a = ax[1, 0]
for name, r in res.items():
    a.plot(r["taus"], r["ite"], label=name.split(" (")[0])
    a.axhline(r["Emin"], ls="--", color="r", alpha=.5)
a.set(title="Exact imaginary-time relaxation", xlabel=r"$\tau$", ylabel="<H>"); a.legend(fontsize=8)
a = ax[1, 1]
if HAVE_Q:
    for name, r in res.items():
        a.semilogy(ts, np.abs(r["q"] - r["ref"]) + 1e-18, label=name.split(" (")[0])
    a.set(title="|E_qllg - E_dense reference|", xlabel="t", ylabel="abs. error"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("qllg_comparison3.png", dpi=150); print("saved qllg_comparison.png")
