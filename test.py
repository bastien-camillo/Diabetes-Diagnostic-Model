import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fmin

plt.rcParams["figure.figsize"] = [10, 5]

dctrl = np.loadtxt('control.txt')
x, y = dctrl[:,0], dctrl[:,2]

def PdL(x,y,n) :
    X, Y = np.linspace( min(x), max(x), n), []
    for k in X :
        b = []
        for i in range(len(x)):
            a = 1
            for j in range(len(x)):
                if i == j :
                    continue
                else :
                    a *= (k-x[j])/(x[i]-x[j])
            b.append(a*y[i])
        Y.append(sum(b))
    return X, Y

A, B = PdL(x,y,100)

plt.figure()
plt.plot(A,B,'k', label = 'Model')
plt.plot(x,y,'r', label = 'Données souris Ko')
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title("Comparaison de la fonction d'approximation par les polynômes de Lagrange et les données")


plt.figure()
x1 = np.linspace(0,10,100)
y1 = [ n**2 for n in x1]
A1, B1 = PdL(x1,y1,100)
plt.plot( A1, B1, 'k', label = 'Model')
plt.plot( x1, y1, 'r', label = 'Courbe x^2')
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title("Comparaison de la fonction d'approximation par les polynômes de Lagrange et d'une fonction carré contrôle")

plt.show()