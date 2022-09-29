import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fmin

## -------- Insuline/Glucose apres injection en IV

#Chargement des donnees 
data = np.loadtxt('IG_1.txt')
# tracer les donnees
plt.figure()
plt.plot(data[:,0], data[:, 1], label = 'Glucose (mg/dl)')
plt.plot(data[:,0], data[:, 2], label = 'Insuline (mU/ml)')
plt.xlabel('temps (min)')
plt.legend()

# injection en IV :
G0 = max(data[:, 1])
# a t = 0 on a un niveau basal et au max on suppose que tout le glucose est arrivé
print('G0 = ', G0)

## modele le plus simple
def model_IG(X, t, s, b, r, m) :
    G, I = X
    Gb = 100
    
    dG = - s*I*G - b*max(0, G-Gb) # max(0, G-Gb) : quand G = Gb, il n'y a plus de degradation normal ;
    dI = r*G - m*I
    return [dG, dI]

## modele le plus simple avec glucogénèse hépatique
def model_IG2(X, t, s, b, r, m, a) :
    G, I = X
    Gb = 100
    gs = 0.5
    
    dG = a*max(gs-G, 0)/(gs-G) - s*I*G - b*max(0, G-Gb) # max(gs-G, 0) : quand il n'y a plus de glucose, le foie en relargue une quantite a
    dI = r*G - m*I
    return [dG, dI]

## parametres
s = 0.001 # couplage
b = 0.05
r = 0.01 # couplage
m = 0.1
a = 0.02

Tf = data[-1, 0]# on prend le temps final des donnees 
t = np.linspace(0, Tf, int(Tf*10))
x0 = [G0, data[0, 2]]
IG = odeint(model_IG, x0, t, args = (s, b, r, m,))
IG2 = odeint(model_IG2, x0, t, args = (s, b, r, m, a))

plt.plot(t, IG[:, 0], label = 'glucose')
plt.plot(t, IG[:, 1], label = 'insuline')
plt.plot(t, IG2[:, 0],'--')
plt.plot(t, IG2[:, 1], '--')
plt.legend()

# distance a minimiser pour "ajuster" le modele aux donnees - vous pouvez aussi essayer avec le modèle IG2, mais il faut ajouter le paramètre a
def dist(param, data, G0, I0):
    s, b, r, m = param
    
    model = odeint(model_IG, [G0, I0], data[:, 0], args = (s, b, r, m,))
    # poids de chaque donnee Glucose / Insuline dans la distance
    alpha = 1.0
    beta = 1.0
    dist = np.sum( alpha*(model[:, 0] - data[:, 1])**2 + beta * (model[:, 1] - data[:, 2])**2 ) + (s<0) * 1e8 + (b<0) * 1e8 + (r<0) * 1e8 + (m<0) * 1e8 # on penalise la distance si les paramètres sont négatifs
    return dist

datafit = data[1:, :]
datafit[:, 0] = datafit[:, 0] - datafit[0, 0]    
p_opt = fmin(dist, [s,b,r,m], args = (datafit, datafit[0, 1], datafit[0, 2]))
print('optimal parameter', p_opt)

IG_opt = odeint(model_IG, [ datafit[0, 1], datafit[0, 2] ], t, args = (p_opt[0], p_opt[1], p_opt[2], p_opt[3],))

plt.figure()
plt.plot(datafit[:, 0], datafit[:, 1], 'o', label = 'glucose')
plt.plot(datafit[:, 0], datafit[:, 2], 'o', label = 'insuline')
plt.plot(t, IG_opt[:, 0], '--', label = 'fit glucose')
plt.plot(t, IG_opt[:, 1], '--', label = 'fit insuline')
plt.legend()

# pour aller plus loin - dependance non lineaire dans la production d'insuline
def model_IGavance(X, t, s, b, r, m, a, n) :
    G, I = X
    Gb = 100
    gs = 0.5
    
    dG = a*max(gs-G, 0)/(gs-G) - s*I*G - b*max(0, G-Gb)
    dI = r*G**n - m*I # on peut aussi essayer r1*G + r2*G**n, est ce que c'est mieux?
    return [dG, dI]

def dist2(param, data, G0, I0):
    s, b, r, m, a, n= param
    
    model = odeint(model_IGavance, [G0, I0], data[:, 0], args = (s, b, r, m, a, n))
    # poids de chaque donnee Glucose / Insuline dans la distance
    alpha = 1.0
    beta = 5.0
    dist = np.sum( alpha*(model[:, 0] - data[:, 1])**2 + beta * (model[:, 1] - data[:, 2])**2 ) + (s<0) * 1e8 + (b<0) * 1e8 + (r<0) * 1e8 + (m<0) * 1e8 + (a<0) * 1e8 + (n<0)*1e8 # on penalise la distance si les paramètres sont négatifs
    return dist

n0 = 2
a = 0.01
p_opt1 = fmin(dist2, [s,b,r,m, a, n0], args = (datafit, datafit[0, 1], datafit[0, 2]))
print('optimal parameter', p_opt1)

IG_opt1 = odeint(model_IGavance, [ datafit[0, 1], datafit[0, 2] ], t, args = (p_opt1[0], p_opt1[1], p_opt1[2], p_opt1[3], p_opt1[4], p_opt1[5]))

plt.figure()
plt.plot(datafit[:, 0], datafit[:, 1], 'o', label = 'glucose')
plt.plot(datafit[:, 0], datafit[:, 2], 'o', label = 'insuline')
plt.plot(t, IG_opt1[:, 0], '--', label = 'fit avance glucose')
plt.plot(t, IG_opt1[:, 1], '--', label = 'fit avance insuline')
plt.legend()


## --------  mise en place du diabete
## 3 variables :
## G : concentration glocose 
## I : concentration insuline
## B : nombre de cellules beta

def mod(X, t, param):
    G, I, B = X
    G0, s, dg0, rho, di0, alpha1, alpha2, alpha0 = param
    
    dG = G0 - (s*I + dg0)*G
    # G0 production de glucose , s sensibilite a l'insuline, dg0 degradation naturelle
    dI = (rho *B*G) - di0 * I
    #rho*B*G production prop aux cellules beta et au glucose, di0 degradation naturelle de l'insuline
    dB = alpha1 * G * B - alpha2 * G**2 * B - alpha0 * B
    #  alpha1 * G : prodution prop au glucose,  alpha2 * G**2 : trop de glucose tue les cellules beta, alpha0 : degradation naturelle

    return [dG, dI, dB]

# parametres jouets - on peut aller voir le papier Topp et al JTB 2000 pour avoir des valeurs plus réalistes
G0 = 10
s = 10.0
dg0 = 0.05
rho = 0.01
di0 = 3.0

alpha1 = 0.3
alpha2 = 0.1
alpha0 = 0.02
parametres = (G0,s, dg0, rho, di0, alpha1, alpha2, alpha0)

X0 = [G0, 0, 20]
t = np.linspace(0, 50, 5000)
X = odeint(mod,X0, t, args = (parametres,))

plt.figure()
plt.subplot(211)
plt.plot(t, X[:, 0], label = 'glucose')
plt.plot(t, X[:, 1], label = 'insuline')
plt.ylabel('concentration')
plt.xlabel('temps')
plt.legend()

plt.subplot(212)
plt.plot(t, X[:, 2], label = 'cellule beta')
plt.ylabel('nombre de cellule')
plt.xlabel('temps')
plt.legend()

#impact de l'apport en glucose
fig = plt.figure(figsize = (8,10))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, sharex=ax1)
ax3 = fig.add_subplot(313, sharex =ax1)

X0 = [10, 0, 20]
Tf = 400
t = np.linspace(0, Tf, Tf*10)
listG0 = np.arange(1, 50, 5)

for G0 in listG0 :    
    parametres = (G0, s, dg0, rho, di0, alpha1, alpha2, alpha0)

    X = odeint(mod,X0, t, args = (parametres,))

    ax1.plot(t, X[:, 0])
    ax2.plot(t, X[:, 1])
    ax3.plot(t, X[:, 2])

ax1.set_title('Glucose')
ax1.set_ylabel('concentration')

ax2.set_title('Insuline')
ax2.set_ylabel('concentration')

ax3.set_title('Cellules beta')
ax3.set_xlabel('time')
ax3.set_ylabel('# cellules')

#impact de la sensibilite a l'insuline

fig1 = plt.figure(figsize = (8,10))
bx1 = fig1.add_subplot(311)
bx2 = fig1.add_subplot(312, sharex=bx1)
bx3 = fig1.add_subplot(313, sharex =bx1)

G0 = 10
X0 = [G0, 0, 20]
Tf = 20
t = np.linspace(0, Tf, Tf*10)
listS = np.arange(0.01, 50, 1)

for s in listS :
    parametres = (G0, s, dg0, rho, di0, alpha1, alpha2, alpha0)

    X = odeint(mod, X0, t, args = (parametres,))

    bx1.plot(t, X[:, 0], label = 's ='+str(s))
    bx2.plot(t, X[:, 1])
    bx3.plot(t, X[:, 2])

#bx1.legend()
bx1.set_title('Glucose')
bx1.set_ylabel('concentration')
bx2.set_title('Insuline')
bx2.set_ylabel('concentration')
bx3.set_title('Cellules beta')
bx3.set_xlabel('time')
bx3.set_ylabel('# cellules')

## Dans les cas présent, on observe 2 états d'equilibres stables pour le glucose, l'un presque nul et l'autre grand
## Si on regarde les equations :
## a l'equi (alpha1 * G - alpha2 * G**2 - alpha0) * B = 0
## donc 2 options : Bstar = 0 ou Gstar racine du polynome P = -alpha2 X^2 + alpha1 X - alpha0
## 1. a l'equi si Bstar = 0 alors Istar = 0 et Gstar = G0 / dg0
## 2. a l'equi si Gstar racine du poly P, alors Istar = (1/s)*(G0/Gstar - dg0) et Bstar = di0*Istar/(rho*Gstar)

# lorsque le diabete est installé il y a destruction des cellules beta, et glycémie importante, on retrouve l'état d'equilibre 1. - il est stable
# une facon d'etre sur d'etre dans ce cas est de s'assurer que le polynome P n'a pas de racine reelle :
# alpha1^2 - 4*alpha2*alpla0 < 0
# exemple :
G0 = 10
s = 10.0
dg0 = 0.05
rho = 0.1
di0 = 3.0
alpha1 = 0.03
alpha2 = 0.1
alpha0 = 0.02
# ici  alpha1^2 - 4*alpha2*alpla0 < 0
parametres = (G0,s, dg0, rho, di0, alpha1, alpha2, alpha0)
Gstar = G0/dg0

X0 = [G0, 0, 20]
Tf = 400
t = np.linspace(0, Tf, Tf*10)
X = odeint(mod,X0, t, args = (parametres,))

plt.figure()
plt.subplot(311)
plt.plot(t, X[:, 0], label = 'glucose')
plt.plot(t, Gstar*np.ones(len(t)), '--')
plt.legend()

plt.subplot(312)
plt.plot(t, X[:, 1], label = 'insuline')
plt.ylabel('concentration')
plt.xlabel('temps')
plt.legend()

plt.subplot(313)
plt.plot(t, X[:, 2], label = 'cellule beta')
plt.ylabel('nombre de cellule')
plt.xlabel('temps')
plt.legend()

## - Insulino - resistance 

def mod_resist(X, t, param):
    G, I, B = X
    G0, s0, dg0, rho, di0, alpha1, alpha2, alpha0, k = param

    s = s0 * np.exp(-k*t) # decroissance de la sensibilité a vitesse k
    
    dG = G0 - (s*I + dg0)*G
    # G0 production de glucose , s sensibilite a l'insuline, dg0 degradation naturelle
    dI = rho *B*G - di0 * I
    #rho*B*G production prop aux cellules beta et au glucose, di0 degradation naturelle de l'insuline
    dB = alpha1 * G * B - alpha2 * G**2 * B - alpha0 * B
    #  alpha1 * G : prodution prop au glucose,  alpha2 * G**2 : trop de glucose tue les cellules beta, alpha0 : degradation naturelle

    return [dG, dI, dB]

# parametres test
G0 = 10
s0 = 60 # on commence avec une sensibilite a l'insuline elevee
dg0 = 0.05
rho = 0.1
di0 = 3.0
alpha1 = 0.3
alpha2 = 0.1
alpha0 = 0.02
k = 0.50 # lente degradation de la sensibilité s

parametres_s = (G0, s0, dg0, rho, di0, alpha1, alpha2, alpha0, 0.0) #k = 0, pas de dégradation de la sensibilite s
parametres_d = (G0, s0, dg0, rho, di0, alpha1, alpha2, alpha0, k)


X0 = [G0, 0, 20]
Tf = 100
t = np.linspace(0, Tf, Tf*10)
X_sain = odeint(mod_resist,X0, t, args = (parametres_s,))
X_diab = odeint(mod_resist,X0, t, args = (parametres_d,))

plt.figure() # decroissance de la sensibilité
plt.plot(t, s0*np.exp(-k*t))
plt.ylabel('parametre s')
plt.xlabel('temps')

plt.figure()
plt.subplot(311)
plt.title('Glucose')
plt.plot(t, X_sain[:, 0], label = 'sain')
plt.plot(t, X_diab[:, 0], label = 'diabete')
plt.ylabel('concentration')
plt.legend()

plt.subplot(312)
plt.title('Insuline')
plt.plot(t, X_sain[:, 1], label = 'sain')
plt.plot(t, X_diab[:, 1], label = 'diabete')
plt.ylabel('concentration')
plt.legend()

plt.subplot(313)
plt.title('Cellules beta')
plt.plot(t, X_sain[:, 2], label = 'sain')
plt.plot(t, X_diab[:, 2], label = 'diabete')
plt.ylabel('nombre de cellule')
plt.xlabel('temps')
plt.legend()

# en jouant sur le paramètre de vitesse de degradation de la sensibilite on remarque que dans ce modèle il faut une degradation suffisemment rapide pour voir apparaitre le diabete.
plt.tight_layout()
plt.show()
