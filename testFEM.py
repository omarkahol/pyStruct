# Import necessary modules from the FEM package
from FEM.mesh import generate_quad_mesh
from FEM.fem import KM_global

# Other necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

# Import the Solver
from FEM.newtonSolver import solve_system

# Import the material models
from material import linear_material_law, bilinear_material

# Seed for reproducibility
np.random.seed(42)

# Geometrical properties of the beam (in SI units)
L = 1.0         # Length                     [m]
h = 0.1         # Height                     [m]
thickness = 0.1 # Thickness                  [m]

# Material properties (in SI units)
E = 210e9      # Young's modulus            [Pa]
nu = 0.3       # Poisson's ratio
rho = 7800     # Density                    [kg/m^3]

# Loads
F_tip = 1e3    # Point load at the free end [N]


# Make the mesh
nelemx = 20
nelemy = 5
nodes, elems = generate_quad_mesh(L, h, nelemx, nelemy)
ndof = nodes.shape[0] * 2

# Plot the mesh
plt.figure()
for e in elems:
    x = nodes[e, 0]
    y = nodes[e, 1]
    plt.fill(x, y, edgecolor='k', fill=False)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mesh')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Find nodes at x=0 in order to constrain them
tol = 1e-8
fixed_nodes = np.where(np.abs(nodes[:, 0]) < tol)[0]
fixed_dofs = []
for n in fixed_nodes:
    fixed_dofs.append(2 * n)     # u_x
    fixed_dofs.append(2 * n + 1) # u_y
fixed_dofs = np.array(fixed_dofs)


all_dofs = np.arange(ndof)
free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

# Apply a tip load at the right edge (downward in y)
# Find nodes at x=L and y=0
tip_nodes = np.where((np.abs(nodes[:, 0] - L) < tol) & (np.abs(nodes[:, 1]) < tol))[0]
assert len(tip_nodes) == 1, "There should be exactly one tip node"
tip_node = tip_nodes[0]


F = np.zeros(ndof)
F[tip_node * 2 + 1] = -F_tip  # downward force at the tip in y-direction
F_reduced = F[free_dofs]

# Solve the bilinear system using the Newton-Raphson solver
u_bil = solve_system(nodes, elems, free_dofs, F, bilinear_material, rho, thickness, itmax=1000, tol=1e-3, verbose = True, w=0.1)

# Solve the linear system, it is just one step
K, _, _ = KM_global(elems, nodes, linear_material_law, rho, None, thickness)

K_reduced = K[free_dofs, :][:, free_dofs].tocsc()
u_lin_reduced = spsolve(K_reduced, F_reduced)
u_lin = np.zeros(ndof)
u_lin[free_dofs] = u_lin_reduced

# Plot the deformed shape
scale = 1e3  # Scale factor for visualization
plt.figure()
for e in elems:
    x = nodes[e, 0]
    y = nodes[e, 1]
    plt.fill(x, y, edgecolor='k', fill=False)
    
    x_def_bil = x + scale * u_bil[2 * e]
    y_def_bil = y + scale * u_bil[2 * e + 1]
    plt.fill(x_def_bil, y_def_bil, edgecolor='r', fill=False)

    x_def_lin = x + scale * u_lin[2 * e]
    y_def_lin = y + scale * u_lin[2 * e + 1]
    plt.fill(x_def_lin, y_def_lin, edgecolor='b', fill=False, linestyle='--')


plt.xlabel('x')
plt.ylabel('y')
plt.title('Deformed shape (scaled)')
plt.axis('equal')
plt.tight_layout()
plt.show()