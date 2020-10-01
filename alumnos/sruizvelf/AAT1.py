#python 2.7.15
import numpy as np
from numpy import linalg as LA

def f(x):
    g = sum(x**2 - 3*x + 1)
    return g
def grad(f, x0, h):
    n = x0.size
    G = np.zeros(n)
    for i in range(n):
        z = np.zeros(n)
        z[i] += h
        xi = x0 + z
        G[i] = (f(xi) - f(x0))/h
    return G
    
def hess(f,x0,h):
    n = x0.size
    s= (n,n)
    H = np.zeros(s)
    for i in range(n):
        for j in range(n):
            z1 = np.zeros(n)
            z2 = z1
            z1[i] += h
            z2[j] += h
            x1 = x0 + z1 + z2
            x2 = x0 + z1 - z2
            x3 = x0 - z1 + z2
            x4 = x0 - z1 - z2
            H[i,j] = (f(x1) - f(x2) - f(x3) + f(x4))/(4*(h**2))
    return H

def optim(f,x0,h):
    Cond1 = grad(f,x0,h) 
    Cond2 = LA.eig(hess(f,x0,h))
    CO1 = np.all(abs(Cond1) < h)
    CO2 = np.all(Cond2[0] >= h)    
    return CO1 and CO2
    
def mk(f,x0,h):
    G = grad(f, x0, h)
    H = hess(f, x0, h)
    norma = np.linalg.norm(G)
    p1 = -G/norma
    p = np.transpose(p1)
    mk = f(x0) + p*G + 0.5*p*H*p1
    return mk
x0 = np.array([1.5,1.5])
print("El gradiente es:")
print(grad(f,x0,0.00001))
print("El hessiano es:")
print(hess(f,x0,0.00001))
if(optim(f,x0,0.00001)):
    print("Se cumplen las condiciones de optimalidad")
else:
    print("NO se cumplen las condiciones de optimalidad")
print("mk=")
print(mk(f,x0,0.00001))