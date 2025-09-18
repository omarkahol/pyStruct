"""
Newton-Raphson Solver for Nonlinear Finite Element Analysis
---------------------------------------------------
- Iterative solution of nonlinear systems
- Handles large deformations and nonlinear material behavior
- Modular design for easy integration with different element types and material models
- Convergence criteria based on residual norms and displacement increments
- Optional verbosity for monitoring convergence process


Author: Original: Omar Kahol
"""

import numpy as np
from FEM.fem import KM_global
from scipy.sparse.linalg import spsolve

def solve_system(nodes, elems, free_dofs, F, material_law, rho, thickness, itmax=1000, tol=1e-3, w=0.1, verbose = True):
    
    # Number of degrees of freedom
    ndof = nodes.shape[0] * 2
    
    # Assume no initial displacement
    u = np.zeros(ndof)

    # Iterative fixed-point scheme
    converged = False
    for it in range(itmax):

        # Assemble the tangent matrices
        K, _, Fint = KM_global(elems, nodes, material_law, rho, u, thickness)
        K_reduced = K[free_dofs, :][:, free_dofs].tocsc()

        # Compute the residual vector
        R = F - Fint.flatten()
        R_reduced = R[free_dofs]

        # Solve for the increment
        delta_u_reduced = spsolve(K_reduced, R_reduced)

        # Check convergence (norm of displacement increment)
        norm_delta_u = np.linalg.norm(delta_u_reduced)/np.linalg.norm(u)
        if norm_delta_u < 1e-3:
            converged = True
            break
        # Update displacements
        u[free_dofs] += w*delta_u_reduced

        if verbose:
            print(f"--- Iteration {it+1} ---")
            print(f"\t Norm of displacement increment: {norm_delta_u:.6e}")
            print(f"\t Max displacement: {np.max(np.abs(u)):.6e}")

    if not converged and verbose:
        print("*** Warning: Newton-Raphson did not converge within the maximum number of iterations ***")
    
    return u