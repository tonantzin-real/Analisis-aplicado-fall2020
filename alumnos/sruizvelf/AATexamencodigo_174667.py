import numpy as np
from numpy import linalg as LA


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

def mk(f,x0,h):
    G = grad(f, x0, h)
    H = hess(f, x0, h)
    norma = np.linalg.norm(G)
    p1 = -G/norma
    p = np.transpose(p1)
    mk = f(x0) + p*G + 0.5*p*H*p1
    return mk
    
    
def RC(f,x0,r,h):    
    mk = mk(f,x0,h)
    s= (2,7999)
    H = np.zeros(s)
    for i in range(7999):
      alpha = 2*r
      p = -grad(f,x0,h)/np.linalg.norm(grad(f,x0,h))
      rho = (f(0)-f(x0 + p))/(f(x0) - mk)
      if rho < 0.25:
        alpha = 0.75*r
      else:
        if rho > 0.75 and np.linalg.norm(p) == alpha:
          alpha = 5*r
      if np.linalg.norm(p) > alpha or np.linalg.norm(p) < r:
        H[i,] = x0
        xk = x0
      else:
        H[i,] = x0 + p
        xk = x0 + p
      x0 = xk
    return H
    
    