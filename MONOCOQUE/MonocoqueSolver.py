__author__ = "Omar Kahol"
__email__ = "omar.kahol@mail.polimi.it"
__copyright__ = "Copyright (C) 2020 Omar Kahol"
__license__ = "Public Domain"
__title__ = "Monocoque Solver"
__description__='''
MONOCOQUE SOLVER
Discrete solver for semi-monocoque structures.
Classes ==> Longeron, Panel, Monocoque
'''

from math import *
import numpy as np
import matplotlib.pyplot as plt


class Longeron:
    def __init__(self, x, y, A):
        self.x = x
        self.y = y
        self.globalDoF = None
        self.A = A
        self.Sx = None
        self.Sy = None


class Panel:
    def __init__(self, n1, n2, G, t):
        self.n1 = n1
        self.n2 = n2
        self.G = G
        self.t = t
        self.length = sqrt((n1.y - n2.y) ** 2 + (n1.x - n2.x) ** 2)
        self.sparsityMatrix = None
        self.K = None
        self.Omega = None
        self.Q = None
        self.M = None

class Monocoque:
    def __init__(self, longerons, panels):
        self.panels = panels
        self.longerons = longerons
        self.xg = None
        self.yg = None
        self.Jx = None
        self.Jy = None
        self.SxArray = np.zeros(len(self.longerons)+1)
        self.SyArray = np.zeros(len(self.longerons)+1)
        self.rhs=None
        self.K=None
        self.Tx=0
        self.Ty=0
        self.Mz=0
        self.solution=None
        return

    def buildReferenceFrame(self):
        div = 0
        self.Jx = 0
        self.Jy = 0
        self.xg = 0
        self.yg = 0
        for longeron in self.longerons:
            div += longeron.A
            self.xg += longeron.A * longeron.x
            self.yg += longeron.A * longeron.y
        self.xg = self.xg / div
        self.yg = self.yg / div
        Jxy = 0
        for longeron in self.longerons:
            self.Jx += longeron.A * (longeron.y - self.yg) ** 2
            self.Jy += longeron.A * (longeron.x - self.xg) ** 2
            Jxy += longeron.A * (longeron.y - self.yg) * (longeron.x - self.xg)
            longeron.Sx = longeron.A * (longeron.y - self.yg)
            longeron.Sy = longeron.A * (longeron.x - self.xg)

        Jmatrix = np.array([[self.Jx, Jxy], [Jxy, self.Jy]])
        W, V = np.linalg.eig(Jmatrix)
        self.Jx = W[0]
        self.Jy = W[1]
        Vinv = np.linalg.inv(V)
        for i, longeron in enumerate(self.longerons):
            longeron.Sx, longeron.Sy = Vinv.dot([longeron.Sx, longeron.Sy])
            self.SxArray[i] = longeron.Sx
            self.SyArray[i] = longeron.Sy
            longeron.globalDoF=i
        return

    def load(self,Tx,Ty,Mz):
        self.Tx=Tx
        self.Ty=Ty
        self.Mz=Mz
        return

    def buildSystem(self):
        self.K = 0
        for panel in self.panels:
            length1 = sqrt((panel.n1.x-self.xg)**2 + (panel.n1.y-self.yg)**2)
            length2 = sqrt((panel.n2.x - self.xg) ** 2 + (panel.n2.y - self.yg) ** 2)
            length3 = panel.length
            p = 0.5*(length1+length2+length3)
            panel.Omega=sqrt(p*(p-length1)*(p-length2)*(p-length3))
            panel.K = np.array([
                [1, -1, -2*panel.Omega],
                [-1, 1, 2*panel.Omega],
                [-2*panel.Omega, 2*panel.Omega, 4*panel.Omega**2]
            ])*panel.G*panel.t/panel.length
            panel.sparsityMatrix=np.zeros((3,len(self.longerons)+1))
            panel.sparsityMatrix[0,panel.n1.globalDoF]=1
            panel.sparsityMatrix[1,panel.n2.globalDoF]=1
            panel.sparsityMatrix[2,-1]=1
            self.K += panel.sparsityMatrix.T.dot(panel.K.dot(panel.sparsityMatrix))
        self.rhs = (self.Tx/self.Jy) * self.SyArray + (self.Ty/self.Jx)*self.SxArray
        self.rhs[-1] += self.Mz
        self.K[1:,0]=0
        self.K[0,1:]=0
        self.rhs[0]=0
        return

    def solveSystem(self):
        self.solution=np.linalg.solve(self.K,self.rhs)
        for i,panel in enumerate(self.panels):
            qj = panel.K.dot(panel.sparsityMatrix.dot(self.solution))
            panel.q = -qj[0]
            panel.M = qj[2]
        return

    def structDraw(self,ax):
        ax.arrow(self.xg, self.yg, (1 if self.Tx !=0 else 0)*0.1*np.mean([panel.length for panel in self.panels]), 0, width=0.01, shape='full',
                 lw=0, length_includes_head=True, color='red')
        ax.arrow(self.xg, self.yg,0, (1 if self.Ty !=0 else 0) * 0.1 * np.mean([panel.length for panel in self.panels]), width=0.01,
                 shape='full',
                 lw=0, length_includes_head=True, color='red')
        if self.Mz != 0:
            ax.plot([self.xg],[self.yg],'r-',marker=r'$\circlearrowleft$',ms=25)
        for (i, longeron) in enumerate(self.longerons):
            ax.plot(longeron.x, longeron.y,'rs',lw=3)
            ax.annotate(str(i),(longeron.x, longeron.y))
        for (i, panel) in enumerate(self.panels):
            ax.arrow(panel.n1.x, panel.n1.y, panel.n2.x-panel.n1.x, panel.n2.y-panel.n1.y, width=0.01*np.mean([panel.length for panel in self.panels]),shape='full', lw=0, length_includes_head=True)
            ax.annotate('Q = {:.3f}'.format(panel.q),
                        (
                        0.5*(panel.n1.x+panel.n2.x),#+0.01*(panel.n1.x+panel.n2.x),
                        0.5*(panel.n1.y+ panel.n2.y)#+0.01*(panel.n1.y+ panel.n2.y)
                        ))


