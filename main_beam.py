from BeamSolver import *

########################################################################################################################
#STRUCTURE DEFINITION
########################################################################################################################

#DEFINE THE NODES, INPUTS ARE X AND Y COORDINATES
n1 = Node(-1,0)
n2 = Node(0,1)
n3 = Node(2,1)
n4 = Node(3,0)
n5 = Node(4,1)
n6 = Node(1,0)

#ADD CONSTRAINS
# INPUT1 = CONSTRAIN X DISPLACEMENT
# INPUT2 = CONSTRAIN Y DISPLACEMENT
# INPUT3 = CONSTRAIN Z DISPLACEMENT
n6.constrain(False, True, False)
n5.constrain(True, True, True)
n4.constrain(True,False, False)

#ADD LOADS TO NODES
# INPUT1 = FX
# INPUT2 = FY
# INPUT3 = MX
n2.load(0,-.1,0)
n3.load(0,-.1,0)
n1.load(0,0,-.1)

#DEFINE BEAMS ==> FROM NODE N1 TO NODE N2
E=1
J=1
A=1
b1 = Beam(n2,n6,E,J,A)
b2 = Beam(n1,n2,E,J,A)
b3 = Beam(n1,n6,E,J,A)
b4 = Beam(n6,n4,E,J,A)
b5 = Beam(n6,n3,E,J,A)
b6 = Beam(n3,n4,E,J,A)
b7 = Beam(n2,n3,E,J,A)
b8 = Beam(n3,n5,E,J,A)
b9 = Beam(n1,n3,E,J,A)
b10 = Beam(n2,n4,E,J,A)

#DEFINE THE STRUCTURE
s = Structure([n1,n2,n3,n4,n5,n6],
              [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10])

s.distributeDoF() #DISTRIBUTE THE DOFS
s.defineSystem() #INITIALIZE THE LINEAR SYSTEM
s.solveSystem() #SOLVE

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Structure and displacements')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
s.structDraw(ax) #DRAW
plt.show()

