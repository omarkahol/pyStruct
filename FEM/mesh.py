import numpy as np

def generate_quad_mesh(Lx, Ly, nelemx, nelemy):
    '''
    Generate a 2D mesh with quad4 elements.

    Parameters
    ----------
    Lx : float
        Length in x-direction
    Ly : float
        Length in y-direction
    nelemx : int
        Number of elements along x
    nelemy : int
        Number of elements along y

    Returns
    -------
    nodes : ndarray of shape (Nnodes, 2)
        Node coordinates
    elems : ndarray of shape (Nelems, 4)
        Connectivity of quad elements
    '''
    x = np.linspace(0, Lx, nelemx + 1)
    y = np.linspace(0, Ly, nelemy + 1)

    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack((X.ravel(), Y.ravel()))

    elems = np.zeros((nelemx * nelemy, 4), dtype=int)

    for j in range(nelemx):
        for i in range(nelemy):
            n1 = j * (nelemy + 1) + i
            n2 = n1 + 1
            n3 = n1 + nelemy + 1
            n4 = n3 + 1

            elems[i * nelemx + j, :] = [n1, n3, n4, n2]

    return nodes, elems