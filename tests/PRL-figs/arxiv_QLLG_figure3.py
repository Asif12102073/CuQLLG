#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Yuefei et. al., arXiv preprint arXiv:2403.09255

        Compare DMI strength effect on non-locality.
        
        This is the one produced the Figure 3 plot in preprint draft. 
        
        
"""


import numpy as np
import matplotlib.pyplot as plt
import qutip as qt


"""
Pauli matrices
"""
sigmaX = np.array([[0, 1], [1, 0]])
sigmaY = np.array([[0, -1j], [1j, 0]])
sigmaZ = np.array([[1, 0], [0, -1]])
identity = np.eye(2)


# Pauli matrices
Sigmas = [sigmaX, sigmaY, sigmaZ]

# Bell states
phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
phi_minus = np.array([1, 0, 0, -1]) / np.sqrt(2)
psi_plus = np.array([0, 1, 1, 0]) / np.sqrt(2)
psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)


def compute_density_matrix(psi):
    """
    Compute the density matrix for a given pure state.
    """
    return np.outer(psi, np.conj(psi))

# Corresponding density matrices
rho_phi_plus = compute_density_matrix(phi_plus)
rho_phi_minus = compute_density_matrix(phi_minus)
rho_psi_plus = compute_density_matrix(psi_plus)
rho_psi_minus = compute_density_matrix(psi_minus)

# rho_phi_plus, rho_phi_minus, rho_psi_plus, rho_psi_minus


def Werner_state(p):
    """
    Werner state (density operator)
    """
    rho_w = (1-p)*np.eye(4)/4 + p*rho_psi_plus
    return rho_w


def H_2s_B(B, B_theta, B_phi):
    """
    Returns the Hamiltonian for a two-qubit system in a magnetic field defined by spherical coordinates.

    :param gamma: Gyromagnetic ratio
    :param B: Magnitude of the magnetic field
    :param theta: Polar angle (radians) [0, pi]
    :param phi: Azimuthal angle (radians) [0, 2*pi]
    :return: 4x4 numpy array representing the Hamiltonian
    """
    # Calculate the components of the magnetic field
    Bx = B * np.sin(B_theta) * np.cos(B_phi)
    By = B * np.sin(B_theta) * np.sin(B_phi)
    Bz = B * np.cos(B_theta)

    # Single-qubit Hamiltonians
    gamma = 1
    H1 = -gamma / 2 * (Bx * sigmaX + By * sigmaY + Bz * sigmaZ)
    H2 = -gamma / 2 * (Bx * sigmaX + By * sigmaY + Bz * sigmaZ)

    # Kronecker product for two-qubit system
    H1_full = np.kron(H1, identity)
    H2_full = np.kron(identity, H2)

    # Total Hamiltonian
    H = H1_full + H2_full

    return H


def H_2s_JD(J_ij, D_z):
    """
    Define the Hamiltonian H_2s for two-qubit chain
    :param J_ij: the exchange coupling of two spins
    :param D_z: the Dzyaloshinskii–Moriya interaction
    """
    term1 = J_ij * (np.kron(sigmaX, sigmaX) + np.kron(sigmaY, sigmaY) + np.kron(sigmaZ, sigmaZ))
    term2 = D_z * (np.kron(sigmaX, sigmaY) - np.kron(sigmaY, sigmaX))
    return term1 + term2


def commutator(A, B):
    """
    Calculate commutators
    """
    return np.dot(A, B) - np.dot(B, A)


def drho_dt_our(rho, H, kappa, mix_param):
    """
    Parameters
    ----------
    rho : Matrix 
        The density matrix of system
    H : Matrix
        The Hamiltonian of system
    lambda_ : A Number
        The damping rate of the dynamics

    Returns
    -------
    new_rho_dot : TYPE
        DESCRIPTION.

    """
    max_iter = 10000 # Iterations for the guess
    tol = 1e-8 # Tolerance, defined to have a stable solution
    # print("Max iterations:", max_iter)

    # Use drho_dt_w as the initial guess for rho_dot
    rho_dot_guess = np.zeros((4, 4))
    # drho_dt_w(rho, H, lambda_)
    
    # hbar = 6.582119569*(10**(-4)) ## eV.ps 

    for _ in range(max_iter):
        term1 = (1j / hbar) * commutator(rho, H)
        term2 =  1j * kappa * commutator(rho, rho_dot_guess)
        rho_dot = term1 + term2

        # Mix the new and old
        new_rho_dot = mix_param * rho_dot + (1 - mix_param) * rho_dot_guess
        
        # Check for convergence
        if np.linalg.norm(new_rho_dot - rho_dot_guess) < tol:
            rho_dot_next = rho + dt * new_rho_dot
            return rho_dot_next

        rho_dot_guess = new_rho_dot
        
    # If we reach here, it means we didn't converge in max_iter iterations
    raise ValueError("The iterative method did not converge in {} iterations.".format(max_iter))

    

def euler_method(H, lambda_, rho0, t_span, dt, mix_param):
    """
    Solve differential equations using the Euler method with two different derivative functions.

    Parameters:
    - H: Hamiltonian of the system.
    - lambda_: Parameter used in the derivative functions.
    - rho0: Initial condition.
    - t_span: Tuple (t_start, t_end) specifying the time interval.
    - dt: Time step.

    Returns:
    - times: Array of time points.
    - solutions_our: Array of solutions at each time point using drho_dt_our.
    - solutions_w: Array of solutions at each time point using drho_dt_w.
    """

    t_start, t_end = t_span
    times = np.arange(t_start, t_end, dt)
    solutions_our = [rho0]
    
    counter = 0
    
    for t in times[:-1]:
        counter += 1
        # For the first set of solutions using drho_dt_our
        rho_current_our = solutions_our[-1]
        rho_next_our = drho_dt_our(rho_current_our, H, lambda_, mix_param)
        solutions_our.append(rho_next_our)
        progress = ((counter + 1) / len(times)) * 100
        print(f'\rProgress quantum: {progress:.1f}%', end='', flush=True)

    return times, solutions_our
# Example usage:
# times, solutions = euler_method(drho_dt_our, rho0, (0, 10), 0.01)


# Extract Bloch vectors and matrix elements from rho
def extract_values_from_rho_2s(rho):
    # Lists to store Bloch vectors and matrix elements for all matrices
    r_values = []
    s_values = []
    Corr_values = []
    
    # Loop over each 4x4 matrix in rho
    for matrix in rho:
        # Compute Bloch vectors for the current matrix
        r = [np.trace(np.dot(matrix, np.kron(sig, identity))).real for sig in Sigmas]
        s = [np.trace(np.dot(matrix, np.kron(identity, sig))).real for sig in Sigmas]
        
        # Compute matrix elements T_cor for the current matrix
        Corr = [[np.trace(np.dot(matrix, np.kron(sigma_i, sigma_j))).real for sigma_j in Sigmas] for sigma_i in Sigmas]
        
        r_values.append(r)
        s_values.append(s)
        Corr_values.append(Corr)
    
    # Convert the lists to numpy arrays
    r_values = np.array(r_values)
    s_values = np.array(s_values)
    Corr_values = np.array(Corr_values)
    
    return r_values, s_values, Corr_values    



def compute_non_locality(Corr_qobj):
    non_localities = {}
    for i in range(3):
        for j in range(i+1, 3):
            U = Corr_qobj.dag() * Corr_qobj  # Properly handle complex numbers using QuTiP
            eigenvalues = U.eigenenergies()
            M_rho = max([eigenvalues[k] + eigenvalues[l] for k in range(len(eigenvalues)) for l in range(k+1, len(eigenvalues))])
            non_locality = max(0, M_rho - 1)
            non_localities[(i, j)] = non_locality
    return non_localities


"""
Plots
"""

# Define parameters
dpi = 300  # dots per inch for resolution
fig_width_pt = 308.0  # Get this from your LaTeX template or journal guidelines (one column width)
inches_per_pt = 1.0/72.27  # Convert pt to inch
golden_mean = 0.6 #(np.sqrt(5)-1.0)/2.0  # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean  # height in inches
fig_size = [fig_width, fig_height]

# Update matplotlib parameters
plt.rcParams.update({
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': True,  # Use LaTeX for all text
    'figure.figsize': fig_size,
    'axes.linewidth': 1,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'savefig.dpi': dpi,
})


# Parameters
hbar = 1            
# Planck constant

lambda_ = 0.5       
# Damping rate

# D_z = 0.4 # J_ij / 10     
# Dzyaloshinskii–Moriya interaction

p = 0.9 # /1.2 #np.sqrt(2)
# for Werner state

J_ij = 1 

B = 1 

B_theta = 1/2 * np.pi

B_phi = 0


# Solve for rho(t) over the time span (0, 10)
t_span = (0.00001, 6)
dt = 1e-5



colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Define parameter sets
parameter_sets = [
    {'D_z': 0, 'color': colors[0], 'linewidth': 2, 'alpha':1},
    {'D_z': 0.2, 'color': colors[1], 'linewidth': 2, 'alpha':1},
    {'D_z': 0.4, 'color': colors[3], 'linewidth': 2, 'alpha':1}, 
    {'D_z': 0.6, 'color': colors[4],'linewidth': 2, 'alpha':1} 
]


# Plotting setup
plt.figure()

for params in parameter_sets:
    # Extract parameters
    D_z = params['D_z']
    # line_style = params['line_style']
    color = params['color']
    linewidth = params['linewidth']
    alpha = params['alpha']
    # Mixing parameters for the solvers
    mix_param_q = 0.1 # For the quantum case

    
    # Define the Hamiltonian
    H = H_2s_JD(J_ij, D_z) + H_2s_B(B, B_theta, B_phi)
    
    # Define the initial state
    rho0 = Werner_state(p)
    
    # Integrate over time using Euler's method (or another method as appropriate)
    times_2s, rho_Solutions = euler_method(H, lambda_, rho0, t_span, dt, mix_param_q)
    
    times_2s = 10*times_2s
    
    # Get data
    r_2s_values, s_values, Corr_values = extract_values_from_rho_2s(rho_Solutions)

    # Compute non-locality
    non_localities = [compute_non_locality(qt.Qobj(Corr, dims=[[3, 3], [3, 3]])) for Corr in Corr_values]
    non_locality_measure = [nl[(0, 1)] for nl in non_localities]

    # Plot the non-locality
    plt.plot(times_2s, non_locality_measure, label="$D/J=$"+f"{D_z}", linestyle= '-', color=color,linewidth=linewidth, alpha=alpha)


# Plot formatting
plt.xlabel("$ t \ {\\rm (ps) }$")
# plt.ylabel("$ \\rm Non-locality$")
plt.legend()
plt.tight_layout()
plt.savefig('concurrence_nonlocality_D_z.pdf', format='pdf', bbox_inches='tight')
plt.show()



