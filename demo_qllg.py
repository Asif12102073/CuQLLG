#test of library functionality
import time
from qllg import SpinSystem, StateBuilder, QSpinSolver, expectation_value

def run_demo():
    print("==============================================================")
    print("qllg Library: End-to-End Multi-Spin GPU Simulation Demo (10 spin Heisenberg ring on minimum rank 2)")
    print("==============================================================")
    n_spins = 10  # Hilbert space
    N = 1024
    print(f"Initializing {n_spins}-spin system (Hilbert Space N = {2**n_spins:,})...")
    sys = SpinSystem(n_spins=n_spins, spin=0.5)
    H = sys.build_heisenberg_hamiltonian(J=1.0, D_z=0.3, B=(1.0, 0.0, 0.0))

    # Initial state: rank-2 mixed antiferromagnet state
    state0 = StateBuilder.antiferromagnetic(n_spins=n_spins, rank=2)

    # Initialize solver on GPU ( CUDA Compute Capability 7.0 + Needed )
    solver = QSpinSolver(H=H, equation="qllg", kappa=0.1, backend="gpu")

    # Track observables during evolution
    energies = []

    def monitor(t, state):
        e = expectation_value(H, state, solver.backend)
        energies.append((t, e))

    print("Running simulation on GPU (dt = 0.01, t_span = (0.0, 1.0))...")
    t0 = time.time()
    final_state = solver.solve(state0, t_span=(0.0, 1.0), dt=0.01, callback=monitor)
    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.3f} s ({elapsed/100*1000:.2f} ms/step).")

    # Print initial and final energies
    initial_energy = expectation_value(H, state0, solver.backend)
    print(f"Initial Energy <H_i> : {energies[0][1]:.6f}")
    print(f"Final Energy <H_f> : {energies[-1][1]: .6f})")

if __name__ == "__main__":
    run_demo()