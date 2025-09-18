# pyStruct

**pyStruct** is a Python library for discrete structural analysis, featuring solvers for beams, finite elements, and semi-monocoque structures. It was developed as a project for the "Aerospace Structures" course during an MSc in Aeronautical Engineering at Politecnico di Milano and later refactored to include additional tools.

## Features

**Beam Solver**: 2D Euler-Bernoulli finite element solver for axial and bending analysis. Includes clear class design (`Node`, `Beam`, `Structure`), constraint handling, and plotting utilities ([BEAM/BeamSolver.py](BEAM/BeamSolver.py)).

**FEM Module**: Mesh generation ([FEM/mesh.py](FEM/mesh.py)), element matrix assembly ([FEM/fem.py](FEM/fem.py)), and a Newton-Raphson nonlinear solver ([FEM/newtonSolver.py](FEM/newtonSolver.py)) for quadrilateral elements. Supports large deformations and nonlinear material behavior.

**Material Models**: Linear and bilinear material laws for 2D stress analysis ([material.py](material.py)).

**Monocoque Solver**: Discrete solver for semi-monocoque structures, with classes for longerons, panels, and overall monocoque assembly ([MONOCOQUE/MonocoqueSolver.py](MONOCOQUE/MonocoqueSolver.py)).

## Project Structure

```
pyStruct/
├── BEAM/
│   ├── BeamSolver.py      # Beam finite element solver
│   └── __init__.py
├── FEM/
│   ├── fem.py             # FEM element matrices
│   ├── mesh.py            # Mesh generation utilities
│   ├── newtonSolver.py    # Nonlinear Newton-Raphson solver
│   └── __init__.py
├── MONOCOQUE/
│   └── MonocoqueSolver.py # Semi-monocoque solver
├── material.py            # Material models (linear, bilinear)
├── testBeam.py            # Example usage of BeamSolver
├── testFEM.py             # Example usage of FEM solver
└── README.md
```

## Installation

Clone the repository and ensure you have Python 3.8+ installed. Required packages:

- `numpy`
- `matplotlib`

Install dependencies with:

```bash
pip install numpy matplotlib
```

## Usage Example


### Beam Solver Example
See [`testBeam.py`](testBeam.py) for a sample script using the BeamSolver:

```python
# Import BeamSolver and supporting classes
from BEAM.BeamSolver import Node, Beam, Structure

# 1. Define nodes with coordinates
n1 = Node(x1, y1)
n2 = Node(x2, y2)
n3 = Node(x3, y3)
# ... more nodes as needed

# 2. Apply constraints to nodes (e.g., clamped, free)
n1.constrain(True, True, True)  # Fully fixed
n3.constrain(False, True, False)  # Partially fixed

# 3. Apply loads to nodes
n2.load(Fx, Fy, M)
# ... more loads as needed

# 4. Create beams between nodes with material properties
b1 = Beam(n1, n2, E, I, A, rho)
b2 = Beam(n2, n3, E, I, A, rho)
# ... more beams as needed

# 5. Assemble structure and solve
structure = Structure([n1, n2, n3], [b1, b2])
structure.assemble()
structure.solve()

# 6. Post-process and plot results
structure.plot_deformed_shape()
structure.plot_internal_forces()
```

### Finite Element Example
See [`testFEM.py`](testFEM.py) for a sample script using the FEM solver:

```python
# Import mesh generator, FEM routines, nonlinear solver, and material models
from FEM.mesh import generate_quad_mesh
from FEM.fem import KM_global
from FEM.newtonSolver import solve_system
from material import linear_material_law, bilinear_material

# 1. Define geometry and mesh parameters
Lx, Ly = ...  # Domain size
nelemx, nelemy = ...  # Number of elements

# 2. Generate mesh (nodes and element connectivity)
nodes, elems = generate_quad_mesh(Lx, Ly, nelemx, nelemy)

# 3. Define material law (linear or bilinear)
material_law = linear_material_law  # or bilinear_material

# 4. Assemble global stiffness/mass matrices and force vector
K, M, Fint = KM_global(elems, nodes, material_law, rho, u_elem, thickness)

# 5. Apply boundary conditions and external loads
free_dofs = ...  # Indices of unconstrained DOFs
F = ...          # External force vector

# 6. Solve nonlinear system using Newton-Raphson
u = solve_system(nodes, elems, free_dofs, F, material_law, rho, thickness)

# 7. Post-process results (e.g., plot deformed mesh, stress distribution)
plot_deformed_mesh(nodes, u)
plot_stress_distribution(elems, u)
```

### Material Models
Material models are defined in [`material.py`](material.py):
- `linear_material_law`: Linear elastic
- `bilinear_material`: Bilinear with strain hardening/softening

### Nonlinear Solver
The Newton-Raphson solver in [`FEM/newtonSolver.py`](FEM/newtonSolver.py) handles nonlinear finite element problems, including large deformations and nonlinear materials.

## Authors


- Omar Kahol (original author)

## License


Public Domain

---
For more details, see the docstrings in each module.
