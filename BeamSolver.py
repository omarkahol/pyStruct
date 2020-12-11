__author__ = "Omar Kahol"
__email__ = "omar.kahol@mail.polimi.it"
__copyright__ = "Copyright (C) 2020 Omar Kahol"
__license__ = "Public Domain"
__title__ = "Beam Solver"
__description__='''
BEAM SOLVER
Discrete solver for DSV beams.
Classes ==> Node, Beam, Structure
'''

from math import *
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y, mass = 1, rot_mass = 1):
        self.x = x
        self.y = y
        self.globalDoFx = None
        self.globalDoFy = None
        self.globalDoFz = None
        self.xConstrain = (False, 0)
        self.yConstrain = (False, 0)
        self.zConstrain = (False, 0)
        self.xLoad = 0
        self.yLoad = 0
        self.zLoad = 0

        self.mass = mass
        self.rot_mass = rot_mass

    def constrain(self, x, y, z):
        self.xConstrain = (x, 0)
        self.yConstrain = (y, 0)
        self.zConstrain = (z, 0)

    def load(self, x, y, z):
        self.xLoad = x
        self.yLoad = y
        self.zLoad = z

class Beam:
    def __init__(self, node1, node2, E, J, A, rho):
        self.E = E
        self.J = J
        self.A = A

        self.n1 = node1
        self.n2 = node2

        self.x1 = node1.x
        self.y1 = node1.y
        self.x2 = node2.x
        self.y2 = node2.y

        self.rho = rho

        self.L = sqrt((self.y2 - self.y1) ** 2 + (self.x2 - self.x1) ** 2)

        self.alfa = atan2(self.y2 - self.y1, self.x2 - self.x1)
        self.T = np.array(
            [[cos(self.alfa), sin(self.alfa), 0, 0, 0, 0],
             [-sin(self.alfa), cos(self.alfa), 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, cos(self.alfa), sin(self.alfa), 0],
             [0, 0, 0, -sin(self.alfa), cos(self.alfa), 0],
             [0, 0, 0, 0, 0, 1]
             ])

        self.M = np.array([
            [140,0,0,70,0,0],
            [0,156,22*self.L,0,54,-13*self.L],
            [0,22*self.L,4*self.L**2,0,13*self.L,-3*self.L**2],
            [70,0,0,140,0,0],
            [0,54,13*self.L,0,156,-22*self.L],
            [0,-13*self.L,-3*self.L**2,0,-22*self.L,4*self.L**2]
        ])*self.rho*self.A*self.L/420
        self.sparsityMatrix = None
        self.K = np.array(
            [
                [self.E * self.A / self.L, 0, 0, -self.E * self.A / self.L, 0, 0],
                [0, 12 * self.E * self.J / self.L ** 3, 6 * self.E * self.J / self.L ** 2, 0,
                 -12 * self.E * self.J / self.L ** 3, 6 * self.E * self.J / self.L ** 2],
                [0, 6 * self.E * self.J / self.L ** 2, 4 * self.E * self.J / self.L, 0,
                 -6 * self.E * self.J / self.L ** 2, 2 * self.E * self.J / self.L],

                [-self.E * self.A / self.L, 0, 0, self.E * self.A / self.L, 0, 0],
                [0, -12 * self.E * self.J / self.L ** 3, -6 * self.E * self.J / self.L ** 2, 0,
                 12 * self.E * self.J / self.L ** 3, -6 * self.E * self.J / self.L ** 2],
                [0, 6 * self.E * self.J / self.L ** 2, 2 * self.E * self.J / self.L, 0,
                 -6 * self.E * self.J / self.L ** 2, 4 * self.E * self.J / self.L]
            ])

        self.U = None
        self.F = None
        return

    def buildSparsityMatrix(self, nnodes):
        self.sparsityMatrix = np.zeros((6, nnodes))
        dofs = [self.n1.globalDoFx, self.n1.globalDoFy, self.n1.globalDoFz, self.n2.globalDoFx, self.n2.globalDoFy,
                self.n2.globalDoFz]
        for i, j in enumerate(dofs):
            self.sparsityMatrix[i, j] = 1
        return


class Structure:
    def __init__(self, nodes, beams):
        self.nodes = nodes
        self.beams = beams
        self.nDoFs = None
        self.K = None
        self.M = None
        self.solution = None
        return

    def distributeDoF(self):
        for i, node in enumerate(self.nodes):
            node.globalDoFx = 3 * i
            node.globalDoFy = 3 * i + 1
            node.globalDoFz = 3 * i + 2

        self.nDoFs = len(self.nodes) * 3
        for beam in self.beams:
            beam.buildSparsityMatrix(self.nDoFs)
        return

    def defineSystem(self):
        self.K = 0
        self.M = 0

        for beam in self.beams:
            self.K += beam.sparsityMatrix.T.dot(beam.T.T.dot(beam.K.dot(beam.T.dot(beam.sparsityMatrix))))
            self.M += beam.sparsityMatrix.T.dot(beam.T.T.dot(beam.M.dot(beam.T.dot(beam.sparsityMatrix))))

        self.RHS = np.zeros(self.nDoFs)
        for i, node in enumerate(self.nodes):
            self.RHS[3 * i] = node.xLoad
            self.RHS[3 * i + 1] = node.yLoad
            self.RHS[3 * i + 2] = node.zLoad

            if node.xConstrain[0]:
                self.K[node.globalDoFx, :] = 0
                self.K[:, node.globalDoFx] = 0
                self.K[node.globalDoFx, node.globalDoFx] = 1
                self.M[node.globalDoFx, :] = 0
                self.M[:, node.globalDoFx] = 0
                self.M[node.globalDoFx, node.globalDoFx] = 1
                self.RHS[node.globalDoFx]=0
            if node.yConstrain[0]:
                self.K[node.globalDoFy, :] = 0
                self.K[:, node.globalDoFy] = 0
                self.K[node.globalDoFy, node.globalDoFy] = 1.
                self.M[node.globalDoFy, :] = 0
                self.M[:, node.globalDoFy] = 0
                self.M[node.globalDoFy, node.globalDoFy] = 1.
                self.RHS[node.globalDoFy] = 0
            if node.zConstrain[0]:
                self.K[node.globalDoFz, :] = 0
                self.K[:, node.globalDoFz] = 0
                self.K[node.globalDoFz, node.globalDoFz] = 1.
                self.M[node.globalDoFz, :] = 0
                self.M[:, node.globalDoFz] = 0
                self.M[node.globalDoFz, node.globalDoFz] = 1.
                self.RHS[node.globalDoFz] = 0
        return

    def solveSystem(self):
        print('Solver initialized...')
        self.solution = np.linalg.solve(self.K, self.p)
        print('Solution found...')
        for beam in self.beams:
            beam.U = beam.sparsityMatrix.dot(self.solution)
            beam.F = np.round(beam.T.T.dot(beam.K.dot(beam.T.dot(beam.U))), 3)
            beam.n1.ux = beam.U[0]
            beam.n1.uy = beam.U[1]
            beam.n2.ux = beam.U[3]
            beam.n2.uy = beam.U[4]
        return

    def structDraw(self, ax):
        for beam in self.beams:
            ax.plot([beam.x1,beam.x2],[beam.y1,beam.y2],'k-', lw=3)
            ax.plot([beam.x1+beam.U[0], beam.x2 + beam.U[3]], [beam.y1+beam.U[1], beam.y2+beam.U[4]], 'r--', lw=2)
        for node in self.nodes:

            ax.arrow(node.x, node.y,
                     (node.xLoad/abs(node.xLoad) if node.xLoad != 0 else 0) * 0.1 * np.mean([beam.L for beam in self.beams]), 0, width=0.01,
                     shape='full',
                     lw=0, length_includes_head=True, color='red')

            ax.arrow(node.x, node.y,0,
                     (node.yLoad / abs(node.yLoad) if node.yLoad != 0 else 0) * 0.1 * np.mean(
                         [beam.L for beam in self.beams]), width=0.01,
                     shape='full',
                     lw=0, length_includes_head=True, color='red')
            if node.zLoad != 0:
                if node.zLoad/abs(node.zLoad)>0:
                    ax.plot([node.x], [node.y], 'r-', marker=r'$\circlearrowleft$', ms=25)
                else:
                    ax.plot([node.x], [node.y], 'r-', marker=r'$\circlearrowright$', ms=25)

            if node.xConstrain[0] and node.yConstrain[0] and node.zConstrain[0]:
                ax.plot(node.x, node.y, 'bs-')
            elif node.xConstrain[0] and node.yConstrain[0] and ~node.zConstrain[0]:
                ax.plot(node.x, node.y,'bo-')
            elif node.xConstrain[0]:
                ax.plot(node.x,node.y,'b<')
            elif node.yConstrain[0]:
                ax.plot(node.x,node.y,'b^')
            elif node.zConstrain[0]:
                ax.plot(node.x,node.y, 'bX')
            else:
                ax.plot(node.x, node.y, 'ks-')
        return


