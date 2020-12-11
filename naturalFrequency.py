import matplotlib.pyplot as plt
import numpy as np
from math import *

#CARICO IL SOLUTORE DI TRAVI
from EulerBernoulliBeam import *

#DATI DEL PROBLEMA
E = 2e+11
rho=8000
l = 0.18
b=0.0015
a=0.020
A=a*b
I=a*(b**3)/12
m=rho*A*l
m_acc = 0.016

#DISCRETIZZAZIONE
nPoints = 4 #discretizzo la trave in 10 punti
delta_z = l/(nPoints+1) #clacolo la lunghezza del segmentino di trave
points = np.arange(0,l,delta_z) #creo un vettore con le coordinate dei punti della trave
dm = m / nPoints #massa di ogni nodo

#ADESSO DEFINIAMO LA STRUTTURA BASE
nodes = [] #vettore che contine tutte le coordinate dei nodi
beams = [] #vettore che contiene tutte le travi in cui ho discretizzato il problema
for i in range(nPoints):
        nodes.append(Node(points[i],0)) #credo un nodo di coordinata x e y=0
for i in range(1,nPoints):
    beams.append(Beam(nodes[i-1],nodes[i],E,I,A,dm)) #creo una singola trave e gli assegno una massa dm :)

#VINCOLO IL NODO 1 CON UN INCASTRO ...
nodes[0].constrain(True, True)

nodes[-1].load(-1,0)
#STRUTTURA DUMMY CHE USERÒ PER CALCOLARE LA MATRICE DI MASSA INVERSA UNA VOLTA SOLA
s = Structure(nodes, beams)
s.distributeDoF()
s.defineSystem()
Minv = np.linalg.inv(s.M)

#DEFINISCO IL MODELLO DA INTEGRARE
def model(state, t):
    U = state[0:2*nPoints] #vettore spostamenti nodali
    dU = state[2*nPoints:] #vettore velocità nodali

    U[0] = 0
    dU[0] = 0

    #CARICO IL NODO FINALE CON UNA FORZA NON NULLA SOLO SE t < 1 secondo
    if t<0.005:
        nodes[-1].load(-1,0) #FORZA FY=-1 e MOMENTO 0
    else:
        nodes[-1].load(0, 0) #FORZA FY= 0 e MOMENTO 0

    #DEFINISCO LA STRUTTURA
    s = Structure(nodes, beams)
    s.distributeDoF() # distribuisco i gradi di libertà
    s.defineSystem() #calcolo matrice di massa, di rigidezza e termine noto

    ddU = Minv.dot(s.RHS- s.K.dot(U))

    ddU[0] = 0
    ddU[1] = 0

    return [*dU, *ddU]

#CONDIZIONI INIZIALI
U0 = np.zeros(2*nPoints) #spostamenti nodali nulli
dU0 = np.zeros(2*nPoints) #velocità nodali nulle

from scipy.integrate import odeint #funzione per integrare un modello
time = np.linspace(0,0.05,1000) #integriamo per 10 secondi
solution = odeint(model,[*U0,*dU0],time) #risolviamo il sistema

#ADESSO ABBIAMO LA SOLUZIONE
lastElementDisplacement = [el[-2] for el in solution] #movimento dell'ultimo trattino di trave

#PLOT DELLA SOLUZIONE
plt.title('OSCILLAZIONE DELL'' ULTIMO ELEMENTO DI TRAVE')
plt.xlabel('t [s]')
plt.ylabel('y [m]')
plt.plot(time, lastElementDisplacement, 'k-')
plt.show()

#TRASFORMATA DI FOURIER
dft = np.fft.rfft(lastElementDisplacement)
PSD = np.sqrt(np.real(dft)**2 + np.imag(dft)**2)
freqs = np.fft.rfftfreq(len(time),time[1]-time[0])

#PLOT DELLA TRASFORMATA DI FOURIER
plt.title('PSD')
plt.xlabel('f [Hz]')
plt.ylabel('X(f)')
plt.plot(freqs, PSD, 'k-')
plt.show()
