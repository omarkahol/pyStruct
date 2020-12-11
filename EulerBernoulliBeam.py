__author__ = "Omar Kahol"
__email__ = "omar.kahol@mail.polimi.it"
__copyright__ = "Copyright (C) 2020 Omar Kahol"
__license__ = "Public Domain"
__title__ = "Euler Bernoulli Solver"
__description__='''
EULER BERNOULLI BEAM SOLVER
Discrete solver for Euler-Bernoulli beams.
Classes ==> Node, Beam, Structure
'''

from math import *
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.globalDoFy = None
        self.globalDoFtheta = None
        self.yConstrain = False
        self.thetaConstrain = False
        self.yLoad = 0
        self.thetaLoad = 0

    def constrain(self, y, theta):
        self.yConstrain = y
        self.thetaConstrain = theta

    def load(self, y, theta):
        self.yLoad = y
        self.thetaLoad = theta

class Beam:
    def __init__(self, node1, node2, E, J, A, rho):
        self.E = E
        self.J = J
        self.A = A
        self.rho = rho

        self.n1 = node1
        self.n2 = node2

        self.x1 = node1.x
        self.y1 = node1.y
        self.x2 = node2.x
        self.y2 = node2.y

        self.L = sqrt((self.y2 - self.y1) ** 2 + (self.x2 - self.x1) ** 2)

        self.alfa = atan2(self.y2 - self.y1, self.x2 - self.x1)
        self.T = np.array(
            [[cos(self.alfa), sin(self.alfa),0,0],
             [-sin(self.alfa), cos(self.alfa),0,0],
             [0, 0, cos(self.alfa), sin(self.alfa)],
             [0, 0, -sin(self.alfa), cos(self.alfa)],
             ])
        self.sparsityMatrix = None
        self.K = np.array(
            [
                [12,6*self.L,-12,6*self.L],
                [6*self.L,4*self.L**2,-6*self.L,2*self.L**2],
                [-12, -6 * self.L, 12, -6 * self.L],
                [6 * self.L, 2 * self.L ** 2, -6 * self.L, 4 * self.L ** 2]
            ]) * self.E*self.J / self.L**3
        self.M = np.array(
            [
                [156, 22*self.L,54,-13*self.L],
                [22*self.L,4*self.L**2, 13*self.L,-3*self.L**2],
                [54,13*self.L,156,-22*self.L],
                [-13*self.L, -3*self.L**2, -22*self.L,4*self.L**2]
            ]) * self.L*self.A*self.rho / 420
        return

    def buildSparsityMatrix(self, nnodes):
        self.sparsityMatrix = np.zeros((4, nnodes))
        dofs = [self.n1.globalDoFy, self.n1.globalDoFtheta, self.n2.globalDoFy, self.n2.globalDoFtheta]
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
        self.RHS = None
        return

    def distributeDoF(self):
        for i, node in enumerate(self.nodes):
            node.globalDoFy = 2 * i
            node.globalDoFtheta = 2 * i + 1

        self.nDoFs = len(self.nodes) * 2
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
            self.RHS[2 * i] = node.yLoad
            self.RHS[2 * i + 1] = node.thetaLoad

            if node.yConstrain:
                self.K[node.globalDoFy, :] = 0
                self.K[:, node.globalDoFy] = 0
                self.K[node.globalDoFy, node.globalDoFy] = 1
                self.M[node.globalDoFy, :] = 0
                self.M[:, node.globalDoFy] = 0
                self.M[node.globalDoFy, node.globalDoFy] = 1
                self.RHS[node.globalDoFy]=0
            if node.thetaConstrain:
                self.K[node.globalDoFtheta, :] = 0
                self.K[:, node.globalDoFtheta] = 0
                self.K[node.globalDoFtheta, node.globalDoFtheta] = 1
                self.M[node.globalDoFtheta, :] = 0
                self.M[:, node.globalDoFtheta] = 0
                self.M[node.globalDoFtheta, node.globalDoFtheta] = 1
                self.RHS[node.globalDoFtheta] = 0
        return