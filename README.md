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
from BEAM.BeamSolver import *
# Define nodes, constraints, loads, and beams
# Solve and plot results
```

### Finite Element Example
See [`testFEM.py`](testFEM.py) for a sample script using the FEM solver:
```python
from FEM.mesh import generate_quad_mesh
from FEM.fem import KM_global
from FEM.newtonSolver import solve_system
from material import linear_material_law, bilinear_material
# Generate mesh, assemble matrices, solve nonlinear system
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
