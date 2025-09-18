"""
Beam Solver
----------------------
A 2D Euler-Bernoulli beam finite-element solver (axial + bending).

Features:
- Clear class design: Node, Beam, Structure
- Type hints and docstrings
- PEP8-style naming
- Separate assembly and constraint application
- Solve reduced system for numerical stability
- Optional plotting utility

Author: Original: Omar Kahol
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

# Constants
DOF_PER_NODE = 3  # [u, v, theta]


@dataclass
class Node:
    """A structural node with 3 degrees of freedom: axial (u), transverse (v), rotation (theta).

    Attributes
    ----------
    x, y: float
        Coordinates in 2D space.
    global_dof: tuple[int, int, int]
        Global DOF indices assigned by the Structure (initialized as (-1,-1,-1)).
    constraints: tuple[bool, bool, bool]
        Boolean flags for constraints (x, y, rotation).
    loads: tuple[float, float, float]
        Nodal loads [Fx, Fy, Mz].
    displacement: Optional[np.ndarray]
        Local displacement vector for the node after solve (u, v, theta).
    """

    x: float
    y: float
    global_dof: tuple[int, int, int] = field(default_factory=lambda: (-1, -1, -1))
    constraints: tuple[bool, bool, bool] = (False, False, False)
    loads: tuple[float, float, float] = (0.0, 0.0, 0.0)
    displacement: Optional[np.ndarray] = None

    def constrain(self, ux: bool = False, uy: bool = False, rz: bool = False) -> None:
        """Apply displacement constraints to the node.

        Parameters
        ----------
        ux, uy, rz: bool
            Whether the corresponding DOF is constrained.
        """
        self.constraints = (ux, uy, rz)

    def load(self, fx: float = 0.0, fy: float = 0.0, mz: float = 0.0) -> None:
        """Apply nodal loads (forces and moment).

        Parameters
        ----------
        fx, fy, mz: float
            Nodal loads for axial, transverse and rotational DOFs.
        """
        self.loads = (fx, fy, mz)


class Beam:
    """A 2-node Euler-Bernoulli beam element in 2D (axial + bending).

    Ordering of DOFs for the element-local vectors and matrices:
    [u1, v1, theta1, u2, v2, theta2]
    """

    def __init__(self, node1: Node, node2: Node, E: float, I: float, A: float, rho: float):

        # Two nodes
        self.n1 = node1
        self.n2 = node2

        # Variable for the distributed load
        self.qx = 0.0  # Axial distributed load (N/m)
        self.qy = 0.0  # Transverse distributed load (N/m)
        self.mz = 0.0  # Distributed moment (N*m/m)
        
        # Material and section properties
        self.E = E      # Young's modulus (in Pa)
        self.I = I      # Second moment of area (in m^4)
        self.A = A      # Cross-sectional area (in m^2)
        self.rho = rho  # Density (in kg/m^3)

        # Computed attributes
        dx = self.n2.x - self.n1.x
        dy = self.n2.y - self.n1.y
        self.length = math.hypot(dx, dy)
        if self.length <= 0:
            raise ValueError("Beam length must be positive")

        self.alpha = math.atan2(dy, dx)

        # Make the local mass and stiffness matrices
        self.transformation = self._make_transformation()
        self.K_local = self._make_local_stiffness()
        self.M_local = self._make_local_mass()

        # Placeholder for element displacement and internal force in local coords
        self.axial_strain: Optional[float] = None
        self.axial_stress: Optional[float] = None
    
    def load_distributed(self, qx: float = 0.0, qy: float = 0.0, mz: float = 0.0, isGlobalCoordinates: bool = True) -> None:
        """Apply distributed loads to the beam element.

        Parameters
        ----------
        qx, qy, mz: float
            Distributed loads per unit length (N/m for forces, N*m/m for moment).
        """

        if isGlobalCoordinates:
            # Transform global distributed loads to local coordinates
            c = math.cos(self.alpha)
            s = math.sin(self.alpha)
            qx_local = c * qx + s * qy
            qy_local = -s * qx + c * qy
            self.qx = qx_local
            self.qy = qy_local
            
            # Note: mz is invariant under rotation in 2D
            self.mz = mz
        else:
            self.qx = qx
            self.qy = qy
            self.mz = mz


    def _make_transformation(self) -> np.ndarray:
        """Return 6x6 transformation matrix from local to global DOFs."""
        c = math.cos(self.alpha)
        s = math.sin(self.alpha)
        T = np.zeros((6, 6))
        # Node 1
        T[0, 0] = c
        T[0, 1] = s
        T[1, 0] = -s
        T[1, 1] = c
        T[2, 2] = 1.0
        # Node 2
        T[3, 3] = c
        T[3, 4] = s
        T[4, 3] = -s
        T[4, 4] = c
        T[5, 5] = 1.0
        return T

    def _make_local_stiffness(self) -> np.ndarray:
        L = self.length
        E = self.E
        I = self.I
        A = self.A
        k = np.zeros((6, 6))
        k[0, 0] = E * A / L
        k[0, 3] = -E * A / L
        k[3, 0] = -E * A / L
        k[3, 3] = E * A / L

        k[1, 1] = 12 * E * I / L ** 3
        k[1, 2] = 6 * E * I / L ** 2
        k[1, 4] = -12 * E * I / L ** 3
        k[1, 5] = 6 * E * I / L ** 2

        k[2, 1] = 6 * E * I / L ** 2
        k[2, 2] = 4 * E * I / L
        k[2, 4] = -6 * E * I / L ** 2
        k[2, 5] = 2 * E * I / L

        k[4, 1] = -12 * E * I / L ** 3
        k[4, 2] = -6 * E * I / L ** 2
        k[4, 4] = 12 * E * I / L ** 3
        k[4, 5] = -6 * E * I / L ** 2

        k[5, 1] = 6 * E * I / L ** 2
        k[5, 2] = 2 * E * I / L
        k[5, 4] = -6 * E * I / L ** 2
        k[5, 5] = 4 * E * I / L

        # Ensure symmetry
        k = (k + k.T) / 2.0
        return k

    def _make_local_mass(self) -> np.ndarray:
        L = self.length
        A = self.A
        rho = self.rho
        # consistent mass matrix for a 2-node beam element (6x6)
        m = np.array([
            [140, 0, 0, 70, 0, 0],
            [0, 156, 22 * L, 0, 54, -13 * L],
            [0, 22 * L, 4 * L ** 2, 0, 13 * L, -3 * L ** 2],
            [70, 0, 0, 140, 0, 0],
            [0, 54, 13 * L, 0, 156, -22 * L],
            [0, -13 * L, -3 * L ** 2, 0, -22 * L, 4 * L ** 2],
        ], dtype=float)
        m = m * (rho * A * L / 420.0)
        # symmetry
        m = (m + m.T) / 2.0
        return m

    def stiffness_global(self) -> np.ndarray:
        """Return the element stiffness matrix expressed in global coordinates."""
        return self.transformation.T @ self.K_local @ self.transformation

    def mass_global(self) -> np.ndarray:
        """Return the element mass matrix expressed in global coordinates."""
        return self.transformation.T @ self.M_local @ self.transformation

    def dof_indices(self) -> List[int]:
        """Return the 6 global DOF indices associated with the element."""
        i1, i2, i3 = self.n1.global_dof
        j1, j2, j3 = self.n2.global_dof
        return [i1, i2, i3, j1, j2, j3]
    

    def _compute_axial_strain(self) -> None:
        """Compute the axial strain in the beam element."""
        u1, _, _ = self.n1.displacement
        u2, _, _ = self.n2.displacement
        L = self.length
        
        eps_xx = (u2 - u1) / L
        self.axial_strain = eps_xx
        

    # Compute stress based on strain
    def _compute_axial_strain_and_stress(self) -> None:
        """Compute the axial stress in the beam element."""
        if self.axial_strain is None:
            self._compute_axial_strain()
        self.axial_stress = self.E * self.axial_strain


class Structure:
    """Finite-element structure composed of nodes and beam elements.

    Methods
    -------
    distribute_dofs()
        Assign global DOF indices to nodes.
    assemble_system()
        Assemble global stiffness and mass matrices and the global load vector.
    apply_constraints()
        Apply boundary conditions and return reduced system indices.
    solve()
        Solve the system for displacements.
    plot()
        Simple visualization of undeformed and deformed shapes.
    """

    def __init__(self, nodes: Sequence[Node], beams: Sequence[Beam]):
        self.nodes = list(nodes)
        self.beams = list(beams)
        self.n_dofs = 0
        self.K: Optional[np.ndarray] = None
        self.M: Optional[np.ndarray] = None
        self.RHS: Optional[np.ndarray] = None
        self.solution: Optional[np.ndarray] = None

        # Distribute the DOFs to nodes
        self._distribute_dofs()

    def _distribute_dofs(self) -> None:
        for idx, node in enumerate(self.nodes):
            base = DOF_PER_NODE * idx
            node.global_dof = (base, base + 1, base + 2)
        self.n_dofs = DOF_PER_NODE * len(self.nodes)

    def assemble_system(self) -> None:

        self.K = np.zeros((self.n_dofs, self.n_dofs), dtype=float)
        self.M = np.zeros((self.n_dofs, self.n_dofs), dtype=float)
        self.RHS = np.zeros((self.n_dofs,), dtype=float)

        # assemble element contributions
        for beam in self.beams:
            idx = beam.dof_indices()
            Kg = beam.stiffness_global()
            Mg = beam.mass_global()
            for a in range(6):
                A_idx = idx[a]
                for b in range(6):
                    B_idx = idx[b]
                    self.K[A_idx, B_idx] += Kg[a, b]
                    self.M[A_idx, B_idx] += Mg[a, b]

        # assemble nodal loads
        for node in self.nodes:
            i, j, k = node.global_dof
            fx, fy, mz = node.loads
            self.RHS[i] += fx
            self.RHS[j] += fy
            self.RHS[k] += mz

        

    def _apply_constraints(self) -> np.ndarray:
        """Return array of free DOF indices after applying nodal constraints.

        This method does NOT modify K/M/RHS in-place; instead it computes a boolean mask
        of free DOFs which can be used to extract the reduced system.
        """
        if self.K is None or self.RHS is None:
            raise RuntimeError("System not assembled. Call assemble_system() first")

        constrained = np.zeros((self.n_dofs,), dtype=bool)
        for node in self.nodes:
            i, j, k = node.global_dof
            cx, cy, cz = node.constraints
            if cx:
                constrained[i] = True
            if cy:
                constrained[j] = True
            if cz:
                constrained[k] = True

        free_mask = ~constrained
        return np.where(free_mask)[0]

    def solve(self) -> np.ndarray:
        """Solve the static equilibrium K u = RHS with boundary conditions.

        Returns
        -------
        u : ndarray
            Global displacement vector (length n_dofs).
        """
        if self.K is None or self.RHS is None:
            raise RuntimeError("System not assembled. Call assemble_system() first")

        free_idx = self._apply_constraints()

        Kff = self.K[np.ix_(free_idx, free_idx)]
        Rf = self.RHS[free_idx]

        if Kff.size == 0:
            raise RuntimeError("No free degrees of freedom to solve for")

        u_free = np.linalg.solve(Kff, Rf)
        u = np.zeros((self.n_dofs,))
        u[free_idx] = u_free
        self.solution = u

        # store nodal local displacements for convenience
        for node in self.nodes:
            i, j, k = node.global_dof
            node.displacement = np.array([u[i], u[j], u[k]])

        # compute elemental internal forces (local)
        for beam in self.beams:
            # Extract element displacement in global coords
            u_e = np.array([self.solution[idx] for idx in beam.dof_indices()])
            # element internal force in local coords
            beam.F_local = beam.K_local @ (beam.transformation @ u_e)

            # compute strain and stress
            beam._compute_axial_strain_and_stress()

        return self.solution

    def plot(self, ax: Optional[plt.Axes] = None, scale: float = 1.0) -> plt.Axes:
        """Plot the undeformed and deformed structure.

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure and axes are created.
        scale: float
            Amplification factor for displacements when plotting.
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")

        # undeformed
        for beam in self.beams:
            ax.plot([beam.n1.x, beam.n2.x], [beam.n1.y, beam.n2.y], "k-", lw=2)

        # deformed (if available)
        if self.solution is not None:
            for beam in self.beams:
                ux1, uy1, _ = beam.n1.displacement
                ux2, uy2, _ = beam.n2.displacement
                ax.plot([beam.n1.x + scale * ux1, beam.n2.x + scale * ux2],
                        [beam.n1.y + scale * uy1, beam.n2.y + scale * uy2], "r--", lw=1.5)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax


# Example usage / small test
if __name__ == "__main__":
    # simple cantilever: node0 fixed at origin, node1 at x=1
    E = 210e9
    I = 1e-6
    A = 1e-3
    rho = 7850

    n0 = Node(0.0, 0.0)
    n1 = Node(1.0, 0.0)

    n0.constrain(True, True, True)  # fully fixed
    n1.load(0.0, -1000.0, 0.0)  # downward point load at node 1

    beam = Beam(n0, n1, E=E, I=I, A=A, rho=rho)
    struct = Structure([n0, n1], [beam])
    struct.assemble_system()
    disp = struct.solve()

    print("Displacements (global)")
    print(disp)

    ax = struct.plot(scale=1000.0)
    ax.set_title("Cantilever: undeformed (black) and deformed (red, amplified)")
    plt.show()
