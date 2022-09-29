from matplotlib.lines import _LineStyle
import numpy as np
from numpy.core.fromnumeric import argmax
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fmin

data = np.loadtxt('full_ogtt.txt')

## Calibration du model.

    # 1) Visualisation des données

plt.subplot(211)
for k in range(1,np.shape(data)[1]-1):
    plt.plot(data[:,0],data[:,k],'--',label=f'Souris n°{k}')
plt.plot(data[:,0],data[:,-1],'k',label= 'Moyenne')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(data[:,0],data[:,-1],'k',label= 'Moyenne')
plt.axvline(x=0,color='blue',linestyle='--',label = "Phase d'absorption par l'intestin")
plt.axvline(x=data[np.where(data[:,-1]==max(data[:,-1]))[0][0],0],color='blue',linestyle='--')
plt.axvline(x=data[np.where(data[:,-1]==max(data[:,-1]))[0][0]-3,0],color='red',linestyle='--')
plt.axvline(x=100,color='red',linestyle='--',label = "Effet de l'insuline")
plt.legend()
plt.grid()

plt.show()

    # 2)

def Za(Z,t):
    a = 1.2
    res = [
        -Z[0]*a ,
        Z[0]*a
    ]
    return res

t = np.linspace(0,10,100)
y0 = [5.0,1.0]

y = odeint(Za,y0,t)

plt.plot(t,y)
plt.show()

    #3) #4)

# Il y a 9 paramètres libre : a, b, c, d, e, n, Gs_ini, I_ini
def Za(Z,t,a,b,c,d,e,n,G0):
    Gi, Gs, I = Z
    res = [
        -Gi*a ,
        Gi*a - (b*I*Gs + c*Gs*(Gs>G0)),
        d*(Gs**n) - e*I
    ]
    return res

t = np.linspace(0,10,100)
G0 = 100
y0 = [4500.0,100,10]

# Réutilisation des paramètres optimaux du TD4
arg = (0.8,0.0138,0.0004,0.0316,3.9176,1.6032,G0)

y = odeint(Za,y0,t,args=arg)

plt.plot(t,y[:,0],color='blue',label ='Glucose Intestinal')
plt.plot(t,y[:,1],color='red',label ='Glucose Sanguin')
plt.plot(t,y[:,2],color='orange',label ='Insuline')
plt.plot(t,np.ones(len(t),)*G0,color='gray',linestyle='--',label = 'Go')
plt.grid()
plt.legend()
plt.axis((0,max(t),0,np.max(y[:,1])+G0))
plt.show()