from MonocoqueSolver import *

########################################################################################################################
#MONOCOCQUE DEFINITION
########################################################################################################################

#DEFINE LONGERONS
# INPUT1 = X
# INPUT2 = Y
# INPUT3 = AREA
A=200
d = 100
n1 = Longeron(0,2*d,A)
n2 = Longeron(d,0,A)
n3 = Longeron(0,0,A)
n4 = Longeron(-d,0,A)

#DEFINE PANELS
G=28
t=1.5
p1 = Panel(n1,n2,G,t)
p2 = Panel(n2,n3,G,t)
p3 = Panel(n3,n4,G,t)
p4 = Panel(n4,n1,G,t)
p5 = Panel(n1,n3,G,t)


#DEFINE MONOCOQUE
m = Monocoque([n1,n2,n3,n4],[p1,p2,p3,p4, p5])

m.load(0,8000,0) #LOAD WITH TX, TY, AND MZ
m.buildReferenceFrame() #COMPUTE THE REFERENCE FRAME
m.buildSystem() #BUILD THE LINEAR SYSTEM
m.solveSystem() #SOLVE

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('SOLUTION')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
m.structDraw(ax) #DRAW
plt.show()