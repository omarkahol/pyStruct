from BEAM.BeamSolver import * # Import the BeamSolver class

# Import other necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Geometrical properties of the beams (in SI units)
t = 0.1              # Thickness                  [m]
h = 0.1              # Height                     [m]
A = t * h            # Cross-section              [m^2]
L = 1.0              # Length                     [m]
I = (h * t**3) / 12  # Moment of inertia          [m^4]

# Material properties (in SI units)
E = 210e9    # Young's modulus            [Pa]
rho = 7850   # Density                    [kg/m^3]
g = 9.81     # Gravity                    [m/s^2]

# Loads
F = 1e3  # Point load at the free end [N]

# Create the nodes
n1 = Node(0,0)
n2 = Node(0,1)
n3 = Node(L, 1)
n4 = Node(L+0.5,0)

# Apply the constraints (clamped at node 1 and node 4)
n1.constrain(True, True, True)  
n4.constrain(True, True, True)

# Apply the loads
n2.load(F, 0, 0)

# Create the 3 beams
b1 = Beam(n1,n2,E,I,A, rho)
b2 = Beam(n2,n3,E,I,A, rho)
b3 = Beam(n3,n4,E,I,A, rho)

# Add distributed load due to self-weight
for beam in [b1, b2, b3]:
    beam.load_distributed(0, -beam.rho * beam.A * g, 0)

# Define the structure
s = Structure([n1,n2,n3,n4], [b1,b2,b3])

# Assemble the system and solve
s.assemble_system()

# Solution (vector of nodal displacements and rotations)
solution = s.solve()

# Draw the results
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
s.plot(ax=ax, scale=1e+3)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
plt.show()



