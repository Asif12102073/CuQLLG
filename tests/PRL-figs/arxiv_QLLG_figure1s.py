#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Yuefei et. al., arXiv preprint arXiv:2403.09255

        Compare the QLLG and LLG in one figure
        
        This is the one produced the Figure 1 plots in preprint draft. 
        
        
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


"""
QLLG
"""
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
    mix_param: A number
        Mix the new guess and previous rho_dot to exam the stability of convergence

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

# Extract expectation values of two spins (Bloch vectors) and T correlation matrix from rho
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
    S_1_values = np.array(r_values)
    S_2_values = np.array(s_values)
    Corr_values = np.array(Corr_values)
    
    return S_1_values, S_2_values, Corr_values    


"""
LLG: do the calculation one spin by one spin...
"""
def hamiltonian_gradient(m1, m2, B, B_theta, B_phi):
    """
    Compute the Hamiltonian and its gradient for a two-spin system.

    Parameters:
    - m1, m2: numpy arrays
        Magnetization vectors of the two spins.
    - B: Magnitude of the magnetic field, float
    - B_theta: Polar angle (radians) [0, pi] of the magnetic field
    - B_phi: Azimuthal angle (radians) [0, 2*pi] of the magnetic field

    Returns:
    - H: float
        The Hamiltonian value.
    - grad_H_m1, grad_H_m2: tuple of numpy arrays
        Gradients of the Hamiltonian with respect to m1 and m2.
    """
    
    # Bohr magneton
    # bohr_m = 9.2740100783*(10**(-24))*(6.241509074)*(10**(18)) # eV/T
    bohr_m=1
    
    # External field
    Bx = B * np.sin(B_theta) * np.cos(B_phi)
    By = B * np.sin(B_theta) * np.sin(B_phi)
    Bz = B * np.cos(B_theta)
    
    # Gradients

    grad_H_m1 = -J_ij * m2 * (1/bohr_m) + D_z * np.array([-m2[1], m2[0], 0]) * (1/bohr_m) + np.array([Bx, By, Bz])
    grad_H_m2 = -J_ij * m1 * (1/bohr_m) + D_z * np.array([m1[1], -m1[0], 0]) * (1/bohr_m) + np.array([Bx, By, Bz])

    return grad_H_m1, grad_H_m2


def llg(m, B, alpha, dt, mix_param):
    """
    Iterative Landau-Lifshitz-Gilbert equation solver.

    Parameters:
    - m: numpy array
        The magnetization vector.
    - B: numpy array
        The magnetic field vector.
    - alpha: float
        The damping coefficient.
    - dt: float
        Time step for the Euler method.

    Returns:
    - m_next: numpy array
        The magnetization vector at the next time step.
    """
    max_iter = 10000
    tol = 1e-8
    
    # gamma = 1.76085963023*(10**(-1)) ## ps^(-1)*T^(-1)
    gamma =1

    m_dot_guess = np.zeros_like(m)  # Initial guess for m_dot

    for _ in range(max_iter):
        m_cross_B = np.cross(m, B)
        m_dot = gamma * m_cross_B - alpha * np.cross(m, m_dot_guess)
        
        # Use mixing parameter to refine m_dot_guess
        m_dot_guess_new = mix_param * m_dot + (1 - mix_param) * m_dot_guess

        if np.linalg.norm(m_dot_guess_new - m_dot_guess) < tol:
            m_next = m + dt * m_dot_guess_new
            m_next /= np.linalg.norm(m_next)  # Normalize m_next
            return m_next

        m_dot_guess = m_dot_guess_new  # Update m_dot_guess for the next iteration

    # If convergence not achieved, raise an error
    raise ValueError("The iterative method did not converge in {} iterations.".format(max_iter))


def integrate_llg(m1_0, m2_0, alpha, B, B_theta, B_phi, t_span, dt, mix_param):
    """
    Integrate the LLG equation for a two-spin system.
    
    - B: Magnitude of the magnetic field, float
    - B_theta: Polar angle (radians) [0, pi] of the magnetic field
    - B_phi: Azimuthal angle (radians) [0, 2*pi] of the magnetic field
    """
    t_start, t_end = t_span
    times = np.arange(t_start, t_end, dt)
    m1_values = [m1_0]
    m2_values = [m2_0]
    
    counter = 0

    for _ in times[:-1]:
        counter += 1
        m1_current, m2_current = m1_values[-1], m2_values[-1]

        # Compute Hamiltonian and its gradients
        grad_H_m1, grad_H_m2 = hamiltonian_gradient(m1_current, m2_current, B, B_theta, B_phi)
        B1, B2 = grad_H_m1, grad_H_m2

        # Update magnetizations
        m1_next = llg(m1_current, B1, alpha, dt, mix_param)
        m2_next = llg(m2_current, B2, alpha, dt, mix_param)

        m1_values.append(m1_next)
        m2_values.append(m2_next)
        
        progress = ((counter + 1) / len(times)) * 100
        print(f'\rProgress classical: {progress:.1f}%', end='', flush=True)

    return times, m1_values, m2_values




"""
Plots
"""

# Define parameters
dpi = 300  # dots per inch for resolution
fig_width_pt = 308.0  # Get this from your LaTeX template or journal guidelines (one column width)
inches_per_pt = 1.0/72.27  # Convert pt to inch
golden_mean = 1.2 #(np.sqrt(5)-1.0)/2.0  # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean  # height in inches
fig_size = [fig_width, fig_height]

# Update matplotlib parameters
plt.rcParams.update({
    'axes.labelsize': 19,
    'font.size': 19,
    'legend.fontsize': 14,
    'xtick.labelsize': 19,
    'ytick.labelsize': 19,
    'text.usetex': True,  # Use LaTeX for all text
    'figure.figsize': fig_size,
    'axes.linewidth': 1,
    'lines.linewidth': 2,
    'lines.markersize': 0.2,
    'savefig.dpi': dpi,
})



"""
- sign in the q-LLG gives the same trajactories between q-LLG and LLG. 
Now the setting:
    (FM coupling + |00> )
    LLG: - B field +J coupling term ; - term1 + term2 
    q-LLG: + B field +J coupling term ; term1 - term2 
    
Note: the damping terms in q-LLG and LLG should be opposite signs,
      if we are looking for the same behaviors of q-LLG and LLG under "FM+|00>".

"""


# Parameters
hbar = 1            
# Planck constant

alpha = 0.5  # classical

kappa =  alpha    #quantum  

D_z = 0.6 # J_ij / 10     7March 0.6
# Dzyaloshinskii–Moriya interaction

J_ij = 1 

B = 1 

B_theta = np.pi/2

B_phi = 0

# Define the Hamiltonian
H_0 = H_2s_JD(J_ij, D_z) + H_2s_B(B, B_theta, B_phi)
# Notice: ${\bf r} = {\bf m}/2$

# Solve for rho(t) over the time span (0, 10)
t_span = (0.00001, 8)
dt = 1e-5

"""
QLLG
"""
# The Hamiltonian for QLLG needs to be rescaled by a factor (1 + lambda_**2 /4 ) * 
H = H_0 
    
# Define the initial state
q = 1

rho0 = np.array([[0, 0, 0, 0],
                    [0, q, np.sqrt(q*(1-q)), 0],
                    [0, np.sqrt(q*(1-q)), 1-q, 0],
                    [0, 0, 0, 0]])

# Mixing parameters for the solvers
mix_param_q = 0.1 # For the quantum case
mix_param_c = 1.0 # For the classical case

# Integrate over time using Euler's method (or another method as appropriate)
times, rho_Solutions = euler_method(H, kappa, rho0, t_span, dt, mix_param_q)

times = 10*times

# Get data
r1_values, r2_values, Corr_values = extract_values_from_rho_2s(rho_Solutions)


"""
LLG
"""
# Initial conditions and parameters
m1_0 = (2*q-1)*np.array([0, 0, 1])
m2_0 = (2*q-1)*np.array([0, 0, -1])


# Integrate the LLG equation
_, m1_values, m2_values = integrate_llg(m1_0, m2_0, alpha, B, B_theta, B_phi, t_span, dt, mix_param_c)

# Convert lists to numpy arrays for easier slicing
m1_values_array = np.array(m1_values)
m2_values_array = np.array(m2_values)


"""
Calculate magnitudes
"""
r1_magnitude = np.linalg.norm(r1_values, axis=1)
m1_magnitude = np.linalg.norm(m1_values, axis=1)
r2_magnitude = np.linalg.norm(r2_values, axis=1)
m2_magnitude = np.linalg.norm(m2_values, axis=1)


# Get the default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Function to create simplified legend handles for r1 and m1
def create_legend_handles_r1_m1():
    return [plt.Line2D([0], [0], color='gray', lw=2, label="${\\rm Quantum}$"),
            plt.Line2D([0], [0], color='gray', lw=2, linestyle='--', label="${\\rm Classical}$"),
            plt.Line2D([0], [0], color='none', lw=0, label=" \ $|\\! \\! \\uparrow \\downarrow \\rangle$"), # f"$q = {q}$"+", \ $|\psi_{\\rm AFM}\\rangle$"
            plt.Line2D([0], [0], color='none', lw=0, label="$J>0, {\\rm AFM}$")]

# Function to create simplified legend handles for r2 and m2
def create_legend_handles_r2_m2():
    return [plt.Line2D([0], [0], color='gray', lw=2, label="${\\rm Quantum}$"),
            plt.Line2D([0], [0], color='gray', lw=2, linestyle='--', label="${\\rm Classical}$"),
            plt.Line2D([0], [0], color='none', lw=0, label=" \ $|\\! \\! \\uparrow \\downarrow \\rangle$"),
            plt.Line2D([0], [0], color='none', lw=0, label="$J>0, {\\rm AFM}$")]


def annotate_components(ax, times, data, label, arrow_configs):
    # Pick a time for the annotation somewhere in the middle
    time_index = 9*len(times) // 100
    num_components = 1 if data.ndim == 1 else data.shape[1]  # Check if data is 1D or 2D

    for i in range(num_components):
        # Ensure there's an arrow configuration for each component or magnitude
        if i < len(arrow_configs):
            config = arrow_configs[i]
        else:
            config = {}

        # Set default values if not specified in config
        xytext_offsets = config.get('xytext_offsets', (40, np.sign(i-1)*25 + 30))
        arrowstyle = config.get('arrowstyle', '->')
        alpha = config.get('alpha', 0.5)
        color = config.get('color', 'black')  # Default color set to 'black'

        # Adjust how data is accessed based on its dimensionality
        if data.ndim == 1:  # For 1D arrays, like magnitude
            data_value = data[time_index]
            annotation_text = f"${label}$"  # Use the label as is for magnitude
        else:  # For 2D arrays, like component vectors
            data_value = data[time_index, i]
            annotation_text = f"${['x', 'y', 'z'][i]}$"  # Use just the component label for 'x', 'y', 'z'

        # Annotate the line with individual arrow properties
        ax.annotate(annotation_text,
                    xy=(times[time_index], data_value),
                    xytext=xytext_offsets,
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle=arrowstyle, alpha=alpha),
                    color=color)


# Define arrow configurations for each component
arrow_configs = [
    {'xytext_offsets': (55, 25), 'arrowstyle': '->', 'alpha': 0.6, 'color': colors[0]},
    {'xytext_offsets': (45, 0), 'arrowstyle': '->', 'alpha': 0.6, 'color': colors[1]},
    {'xytext_offsets': (45, -10), 'arrowstyle': '->', 'alpha': 0.6, 'color': colors[2]},
]

arrow_configs_mag = [
    {'xytext_offsets': (50, -5), 'arrowstyle': '->', 'alpha': 0.6, 'color': colors[4]},
                ]


# Plot for r1 and m1
fig, ax = plt.subplots()
for i, component in enumerate(['x', 'y', 'z']):
    ax.plot(times, r1_values[:, i], color=colors[i])
    ax.plot(times, m1_values_array[:, i], linestyle='--', alpha=0.6, color=colors[i])

# Plot the magnitudes
ax.plot(times, r1_magnitude, linestyle='-', alpha=0.6, color=colors[4])
# ax.plot(times, m1_magnitude, linestyle='--', alpha=0.6, color=colors[3])

ax.set_xlabel("$ t \ {\\rm (ps) }$") 
# Call the function with the defined arrow configurations
annotate_components(ax, times, r1_values, 'r1', arrow_configs)
annotate_components(ax, times, r1_magnitude, "${\\rm magnitude}$" , arrow_configs_mag) 

plt.ylim(-1.1, 1.1)  # Setting y-axis limits
plt.xlim(0, 80)  # Setting x-axis limits
plt.xticks([0, 20, 40, 60, 80])  # Setting specific x-axis ticks
ax.legend(handles=create_legend_handles_r1_m1(), loc='lower right')
plt.tight_layout()
plt.savefig('r1_m1_plot.pdf')  # Save the figure as a PDF