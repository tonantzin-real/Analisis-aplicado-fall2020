import numpy as np

x0 = np.array([1.0, 2, 3, 4])


def is_pos_semi_def(m):
    w,v=np.linalg.eig(m)
    return np.all(w <= 0)


def check_optimality(grad, hes):
    if all(grad == 0):
        return is_pos_semi_def(hes)
    return False


def hess(f, x0, h=1e-7):
    n = x0.size
    fx = f(x0)
    H = np.zeros((n, n))
    fxt = np.zeros(n)
    for i in range(0, n):
        aux = np.copy(x0)
        aux[i] += h
        fxt[i] = f(aux)
    # endFor
    for i in range(0, n):
        for j in range(0, i + 1):
            H[i, j] = fxt[i] + fxt[j]
            xt = np.copy(x0)
            xt[i] += h
            xt[j] += h
            H[i, j] += f(xt) + fx
            if H[i, j] < h:
                H[i, j] = 0
            H[i, j] /= h ** 2
            if i != j:
                H[j, i] = H[i, j]
        # endFor
    # endFor
    return H


def grad(f, x0, h=1e-7):
    n = x0.size
    res = np.zeros(n)
    for i in range(0, n):
        xt1 = np.copy(x0)
        xt1[i] -= h
        xt2 = np.copy(x0)
        xt2[i] += h
        res[i] = (f(xt1) - f(xt2))
        res[i] /= 2 * h
    # endFor
    return res


def f(x0):
    return np.sqrt(sum(x0 ** 2))

def mk(f,xk):
    g = grad(f,xk)
    H = hess(f,xk)
    fk = f(xk)
    mks = lambda p : fk+g.T*p+(1/2)*p.T*H*p;
    return mks
