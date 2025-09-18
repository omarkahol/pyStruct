"""
Material Models
----------------------
A collection of material models for 2D stress analysis.

Features:
- Linear elastic material model
- Bilinear material model with strain hardening (or softening)


Author: Original: Omar Kahol
"""

import numpy as np

def linear_material_law(eps, E=210e9, nu=0.3, plain_stress=True):

    C_lin = None
    if plain_stress:
        C_lin = (E / (1 - nu**2)) * np.array([
            [1,    nu,           0],
            [nu,   1,            0],
            [0,    0, (1 - nu)/2]
        ])
    else:
        C_lin = (E*(1-nu)) / ((1+nu)*(1-2*nu)) * np.array([
            [1,    nu/(1-nu),           0],
            [nu/(1-nu),   1,            0],
            [0,    0, (1-2*nu)/(2*(1-nu))]
        ])
    sigma = C_lin @ eps

    return C_lin, sigma

def bilinear_material(eps, E1=210e9, E2=500e9, eps_y=1e-5, nu=0.33, k=1e+6, plain_stress=True):

    # Equivalent strain for switching (von Mises-like for 2D)
    eps_eq = np.sqrt(eps[0]**2 + eps[1]**2 - eps[0]*eps[1] + 0.5*eps[2]**2)

    # Compute the effective Young's modulus using a sigmoid transition
    E = E1 + (E2 - E1) / (1 + np.exp(-k * (eps_eq - eps_y)))

    C_lin = None
    if plain_stress:
        C_lin = (E / (1 - nu**2)) * np.array([
            [1,    nu,           0],
            [nu,   1,            0],
            [0,    0, (1 - nu)/2]
        ])
    else:
        C_lin = (E*(1-nu)) / ((1+nu)*(1-2*nu)) * np.array([
            [1,    nu/(1-nu),           0],
            [nu/(1-nu),   1,            0],
            [0,    0, (1-2*nu)/(2*(1-nu))]
        ])

    # Stress
    sigma = C_lin @ eps

    return C_lin, sigma