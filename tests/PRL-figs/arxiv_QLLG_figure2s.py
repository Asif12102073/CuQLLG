#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Yuefei et. al., arXiv preprint arXiv:2403.09255

        Compare DMI strength effect on correlation matrix of two-qubit density operator.
        
        This is the one produced the Figure 2 plots in preprint draft. 
        
        
"""

import numpy as np
import matplotlib.pyplot as plt 



"""
Pauli matrices
"""
sigmaX = np.array([[0, 1], [1, 0]])
sigmaY = np.array([[0, -1j], [1j, 0]])
sigmaZ = np.array([[1, 0], [0, -1]])
identity = np.eye(2)


# Pauli matrices
Sigmas = [sigmaX, sigmaY, sigmaZ]

Sigmas_half = [0.5*sigmaX, 0.5*sigmaY, 0.5*sigmaZ] 

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



def Werner_state(p):
    """
    Werner state (density operator)
    """
    rho_w = (1-p)*np.eye(4)/4 + p*rho_phi_minus
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

    # Single-qubit Hamiltonians + / - 29Jan 12:22
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

    # Use drho_dt_w as the initial guess for rho_dot
    rho_dot_guess = np.zeros((4, 4))

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
fig_width_pt = 308.0  # Get this from your LaTeX template or journal guidelines (one column width) 308pt / 246pt
inches_per_pt = 1.0/72.27  # Convert pt to inch
golden_mean = 1.2 #(np.sqrt(5)-1.0)/2.0  # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean  # height in inches
fig_size = [fig_width, fig_height]


# Update matplotlib parameters
plt.rcParams.update({
    'axes.labelsize': 19,
    'font.size': 19,
    'legend.fontsize': 13,
    'xtick.labelsize': 19,
    'ytick.labelsize': 19,
    'text.usetex': True,  # Use LaTeX for all text
    'figure.figsize': fig_size,
    'axes.linewidth': 1,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'savefig.dpi': dpi,
})



# Parameters
hbar = 1            
# Planck constant

lambda_ = 0.5       
# Damping rate

B = 1           
# External magnetic field strength

B_theta = np.pi/2
# theta: Polar angle (radians) [0, pi]

B_phi = 0
# phi: Azimuthal angle (radians) [0, 2*pi]

J_ij = 1            
# Exchange coupling 

D_z = 0 # J_ij / 10     
# Dzyaloshinskii–Moriya interaction

# Solve for rho(t) over the time span (0, 10)
t_span = (0.00001, 6)
dt = 1e-5


# Define the Hamiltonian
H_0 = H_2s_JD(J_ij, D_z) + H_2s_B(B, B_theta, B_phi)


# Mixing parameters for the solvers
mix_param_q = 0.1 # For the quantum case

"""
QLLG

CHECK THIS RESCALE PART
"""
# The Hamiltonian for QLLG needs to be rescaled by a factor (1 + lambda_**2 /4 ) * 
H = H_0 

# rho0 Product state
q = 1


rho0 = np.array([[0, 0, 0, 0],
                    [0, q, np.sqrt(q*(1-q)), 0],
                    [0, np.sqrt(q*(1-q)), 1-q, 0],
                    [0, 0, 0, 0]])


# Integrate over time using Euler's method (or another method as appropriate)
times_2s, rho_Solutions = euler_method(H, lambda_, rho0, t_span, dt, mix_param_q)

times_2s = 10*times_2s
 
# Get data
r_2s_values, s_values, Corr_values = extract_values_from_rho_2s(rho_Solutions)




# Define labels for the elements in the 3x3 matrices
labels = ['T_xx', 'T_xy', 'T_xz', 'T_yx', 'T_yy', 'T_yz', 'T_zx', 'T_zy', 'T_zz']

# Open a text file to write the labeled data
with open('labeled_corr_values.txt', 'w') as file:
    # Write a header line with labels
    file.write('MatrixIndex ' + ' '.join(labels) + '\n')
    
    # Iterate over each 3x3 matrix in the array
    for index, matrix in enumerate(Corr_values):
        # Flatten the matrix to a 1D array and create a string for each element with its label
        labeled_elements = [f'{label}={value:.6f}' for label, value in zip(labels, matrix.flatten())]
        
        # Write the index and the labeled elements of the current matrix to the file
        file.write(f'Matrix_{index+1} ' + ' '.join(labeled_elements) + '\n')


# Plot T_cor_values on the same plot
T_labels = [["$T_{xx}$", "$T_{xy}$", "$T_{xz}$"], 
            ["$T_{yx}$", "$T_{yy}$", "$T_{yz}$"], 
            ["$T_{zx}$", "$T_{zy}$", "$T_{zz}$"]]

plt.figure()
for i in range(3):
    for j in range(3):
        plt.plot(times_2s, Corr_values[:, i, j], label=T_labels[i][j])

plt.xlabel("$ t \ {\\rm (ps) }$")

plt.legend(ncol=2, loc='upper right', bbox_to_anchor=(1.02, 1)) # 

plt.xticks([0, 20, 40, 60])
# plt.xlim(-0.1, 60)  # Setting y-axis limits
plt.ylim(-1.08, 1.01)  # Setting y-axis limits
plt.grid(False)
plt.tight_layout()
plt.savefig('T_cor_values_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()
