import numpy as np
def derivada_parcial(f, xk, pos):
    eps= 0.0001
    n = xk.size
    h = np.zeros(n)
    h[pos]+=eps
    return (f(xk + h) - f(xk))/eps

def gradiente(f,xk):
    n = xk.size
    res = np.zeros(n)
    for i in range(n):
        res[i] = derivada_parcial(f,xk,i)
    return res

def segunda_derivada(f, xk, pos1, pos2):
    eps = 0.0001
    n = xk.size
    h = np.zeros(n)
    h[pos2] += eps
    def f_prima(x):
        return derivada_parcial(f,x,pos1)
    return derivada_parcial(f_prima,xk,pos2)

def hessiana(f,xk):
    n = xk.size
    res = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            res[i][j] = segunda_derivada(f,xk,i,j)
            res[j][i] = res[i][j]
    return np.matrix(res)
def is_pos_def(H):
    eps = 0.0001
    return np.all(np.linalg.eigvals(H) > eps)
def condiciones_optimalidad(f,xk):
    eps = 0.0001
    if(np.all(gradiente(f,xk) >= eps)):
        return is_pos_def(hessiana(f,xk))
    return False

def mk(f, xk, p):
    pt= p.transpose()
    return f(xk) + pt.dot(gradiente(f,xk)) + .5*(pt.dot((hessiana(f,xk)).dot(p)))

