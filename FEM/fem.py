import numpy as np

def KM_local(X, Y, material_law, rho , u_elem):
    '''
    Construct the local stiffness and mass matrices for a 4-node quadrilateral element.
    
    Parameters
    ----------
    X : array-like, shape (4,)
        x-coordinates of the element nodes
    Y : array-like, shape (4,)
        y-coordinates of the element nodes
    C : ndarray, shape (3,3)
        Constitutive matrix (elasticity matrix)
    rho : float
        Density
    u_elem : array-like, shape (8,)
        Element displacement vector
    
    Returns
    -------
    Kl : ndarray, shape (8,8)
        Element stiffness matrix
    Ml : ndarray, shape (8,8)
        Element mass matrix
    '''

    ir = 3   # number of Gauss points per direction
    if ir == 1:
        g1, w1 = 0.0, 2.0
        gp = np.array([[g1, g1]])
        w  = np.array([[w1, w1]])
    elif ir == 2:
        g1, w1 = 0.577350269189626, 1.0
        gp = np.array([[-g1,-g1], [ g1,-g1], [-g1, g1], [ g1, g1]])
        w  = np.ones_like(gp)
    elif ir == 3:
        g1, g2 = 0.774596669241483, 0.0
        w1, w2 = 0.555555555555555, 0.888888888888888
        gp = np.array([
            [-g1,-g1], [-g2,-g1], [ g1,-g1],
            [-g1, g2], [-g2, g2], [ g1, g2],
            [-g1, g1], [-g2, g1], [ g1, g1]
        ])
        w = np.array([
            [w1,w1], [w2,w1], [w1,w1],
            [w1,w2], [w2,w2], [w1,w2],
            [w1,w1], [w2,w1], [w1,w1]
        ])
    else:
        raise ValueError("Unsupported number of Gauss points")

    wp = w[:,0] * w[:,1]
    xsi, eta = gp[:,0], gp[:,1]
    ngp = len(wp)

    # Shape functions
    N = np.zeros((ngp, 4))
    N[:,0] = (1 - xsi) * (1 - eta) / 4
    N[:,1] = (1 + xsi) * (1 - eta) / 4
    N[:,2] = (1 + xsi) * (1 + eta) / 4
    N[:,3] = (1 - xsi) * (1 + eta) / 4

    # Derivatives wrt xsi, eta
    dNr = np.zeros((ngp, 4))
    dNs = np.zeros((ngp, 4))

    dNr[:,0] = -(1 - eta) / 4
    dNr[:,1] =  (1 - eta) / 4
    dNr[:,2] =  (1 + eta) / 4
    dNr[:,3] = -(1 + eta) / 4

    dNs[:,0] = -(1 - xsi) / 4
    dNs[:,1] = -(1 + xsi) / 4
    dNs[:,2] =  (1 + xsi) / 4
    dNs[:,3] =  (1 - xsi) / 4

    Kl = np.zeros((8,8))
    Ml = np.zeros((8,8))

    # Ceate a vector containing the internal forces
    Fint = np.zeros((8,1))

    
    for i in range(ngp):
        # Jacobian matrix
        F = np.array([
            [np.dot(dNr[i,:], X), np.dot(dNs[i,:], X)],
            [np.dot(dNr[i,:], Y), np.dot(dNs[i,:], Y)]
        ])
        detF = np.linalg.det(F)
        if detF < 10 * np.finfo(float).eps:
            raise ValueError("Jacobian determinant equal or less than zero!")

        # Shape function derivatives wrt global coords
        DD = np.vstack((dNr[i,:], dNs[i,:])).T
        DN = np.linalg.solve(F.T, DD.T).T   # DN(j,1:2)

        # Construct B matrix
        B = []
        for j in range(4):
            B_j = np.array([
                [DN[j,0],       0.0],
                [0.0,           DN[j,1]],
                [DN[j,1], DN[j,0]]
            ])
            B.append(B_j)
        B = np.hstack(B)

        # Construct Ne matrix
        Ne = []
        for j in range(4):
            Ne_j = np.array([
                [N[i,j], 0.0],
                [0.0, N[i,j]]
            ])
            Ne.append(Ne_j)
        Ne = np.hstack(Ne)

        # Compute the strain at this element
        eps = B @ u_elem if u_elem is not None else np.zeros((3,1))
        
        # Get the stress and the tangent component
        C_tan, sigma = material_law(eps)

        # Integration
        Kl += B.T @ C_tan @ B * detF * wp[i]
        Ml += Ne.T @ (rho * Ne) * detF * wp[i]
        Fint += B.T @ sigma.reshape(3, 1) * detF * wp[i]

    return Kl, Ml, Fint

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix

def KM_global(elems, nodes, C, rho, u, thickness):
    """
    Assemble global stiffness and mass matrices from local contributions.

    Parameters
    ----------
    elems : ndarray, shape (nelem, 4)
        Connectivity of quad4 elements (node indices, 0-based!)
    nodes : ndarray, shape (nnodes, 2)
        Node coordinates
    C : ndarray, shape (3,3)
        Constitutive (elasticity) matrix
    rho : float
        Density

    Returns
    -------
    K : scipy.sparse.csc_matrix, shape (ndof, ndof)
        Global stiffness matrix
    M : scipy.sparse.csc_matrix, shape (ndof, ndof)
        Global mass matrix
    """

    nelem = elems.shape[0]
    nnodes = nodes.shape[0]
    ndof = 2 * nnodes

    # use sparse matrices in LIL format for efficient assembly
    K = lil_matrix((ndof, ndof))
    M = lil_matrix((ndof, ndof))
    Fint_global = np.zeros((ndof,1))  # Global internal force vector

    for i in range(nelem):
        idnod = elems[i, :]           # element node ids (0-based)
        dofx = 2 * idnod              # x dofs
        dofy = 2 * idnod + 1          # y dofs
        dofelems = np.empty(8, dtype=int)
        dofelems[0::2] = dofx
        dofelems[1::2] = dofy

        X = nodes[idnod, 0]
        Y = nodes[idnod, 1]

        u_elem = u[dofelems] if u is not None else None

        Kl, Ml, Fint = KM_local(X, Y, C, rho, u_elem)

        # assembly
        for a in range(8):
            for b in range(8):
                K[dofelems[a], dofelems[b]] += Kl[a, b]
                M[dofelems[a], dofelems[b]] += Ml[a, b]
        
        # Assemble internal force vector if needed
        Fint_global[dofelems, 0] += Fint[:, 0]

    # convert to CSC for efficient linear algebra
    return csc_matrix(K*thickness), csc_matrix(M*thickness), Fint_global*thickness
