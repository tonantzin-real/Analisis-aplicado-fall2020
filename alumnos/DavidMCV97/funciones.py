import numpy as np

def gradiente(f,x):
    
    """
    Aproximacion numerica para el gradiente de una funcion f de R^n a R.
    Se usa el metodo de diferenciacion central.
    
    entradas:
        - f funcion de R^n a R.
        - x vector en R^n.
    salidas : 
        -grad aproximacion al gradiente de f en x.
    
    Antes de usar esta funcion, asegurese de que f sea derivable en x.
    """
    x = x.astype(np.float64)
    h = np.float64(1e-4)
    k = 1/(2*h)
    n = x.shape[0]
    grad = np.zeros(n).astype(np.float64)
    for i in range(n):
        aux1 = np.copy(x)
        aux2 = np.copy(x)
        aux1[i] = aux1[i]+h
        aux2[i] = aux2[i]-h
        grad[i] = f(aux1)-f(aux2)
        grad[i] = grad[i]*k
    return grad

def hessiana(f,x):
    
    """
    Aproximacion numerica para la matriz Hessiana de una funcion f de R^n a R.
    Se usa el metodo de segunda derivada por Taylor para los elementos de la diagonal
    y la segunda derivada parcial por Taylor para el resto.
    
    entradas:
        - f funcion de R^n a R.
        - x vector en R^n.
    salidas : 
        -hess aproximacion a la matriz  Hessiana de f en x.
    
    Antes de usar esta funcion, asegurese de que la Hessiana exista para f en x.
    """
    x = x.astype(np.float64)
    h = np.float64(1e-2)
    k = 1/(h**2)
    n = x.shape[0]
    hess = np.zeros((n,n)).astype(np.float64)
    for i in range(n):
        for j in range(i+1):
            if i == j:
                aux1 = np.copy(x)
                aux2 = np.copy(x)
                aux1[i] = aux1[i]+h
                aux2[i] = aux2[i]-h
                hess[i,j] = f(aux1)+f(aux2)-2*f(x)
                hess[i,j] = hess[i,j]*k
            else:
                aux1 = np.copy(x)
                aux2 = np.copy(x)
                aux3 = np.copy(x)
                aux1[i] = aux1[i]+h
                aux1[j] = aux1[j]+h
                aux2[i] = aux2[i]+h
                aux3[j] = aux3[j]+h
                hess[i,j] = f(aux1)-f(aux2)-f(aux3)+f(x)
                hess[i,j] = hess[i,j]*k
                hess[j,i] = hess[i,j]   #por simetria de la matriz hessiana
    return hess

def es_optimo(f,x):
    
    """
    Comprobación si un vector x es minimo local de una función f.
    Si la función evaluada en el punto tiene gradiente cero y si 
    su matriz hessiana es positiva, entonces devuelve verdadero.
    Entradas:
        - f funcion de R^n a R.
        - x vector en R^n.
    salidas : 
        -optimo valor binario indicador de optimalidad.
    
    Antes de usar esta funcion, asegurese de que el gradiente y la 
    Hessiana existan para f en x, y que su Hessiana sea simetrica.
    """
    
    n = x.shape[0] 
    grad = abs(gradiente(f,x))
    eps = np.float64(1e-10) * np.ones(n)
    if all(grad < eps):
        hess = hessiana(f,x)
        if all (np.linalg.eigh(hess)[0] >= 0):    #esta funcion devuelve los eigenvalores de una matriz hermitiana
            optimo = True
        else:
            optimo = False
    else:
        optimo = False
    
    return optimo

def mk(f,x):
    
    """
    Aproximación a una función f en un punto x por medio de un polinomio de taylor de segundo grado.
    Entradas:
        - f funcion de R^n a R.
        - x vector en R^n.
    salidas : 
        - "aproximacion" es una funcion que aproxima f en x.
    """
    
    def aproximacion(y):
        grad = gradiente(f,x)
        hess = hessiana(f,x)
        val = f(x) + y.dot(grad) + 0.5*y.dot(hess).dot(y)    #dot es el producto matricial y producto punto
        return val
    
    return aproximacion

def wolfe(f,x,p,alpha,c1,c2):
    """
    revisa las condiciones de Wolfe.
    
    entradas:
        f funcion de R^n a R.
        x vector central en R^n.
        p vector de direccion en R^n.
        alpha constante a evaluar en R.
        c1 constante de la condicion de Armijo.
        c2 constante de la condicion de curvatura.
    salidas:
        bueno variable binaria.
        
    Si bueno es verdadero, las condiciones se cumplen.
    """
    aux = x + alpha*p
    producto = gradiente(f,x).dot(p)
    bueno = False
    
    if f(aux) <= f(x) + c1*alpha*producto:
        if gradiente(f,aux).dot(p) >= c2*producto:
            bueno = True
            
    return bueno

def wolfe_fuerte(f,x,p,alpha,c1,c2):
    """
    revisa las condiciones fuertes de Wolfe.
    
    entradas:
        f funcion de R^n a R.
        x vector central en R^n.
        p vector de direccion en R^n.
        alpha constante a evaluar en R.
        c1 constante de la condicion de Armijo.
        c2 constante de la condicion de curvatura.
    salidas:
        bueno variable binaria.
        
    Si bueno es verdadero, las condiciones se cumplen.
    """
    aux = x + alpha*p
    producto = gradiente(f,x).dot(p)
    bueno = False
    
    if f(aux) <= f(x) + c1*alpha*producto:
        if abs(gradiente(f,aux).dot(p)) <= abs(c2*producto):
            bueno = True
            
    return bueno

def volver_pd(x):
    if np.all(np.linalg.eigvals(x) > 0) == True:
        return x #la matriz es positiva definida
    else:
        e=abs(np.linalg.eigvals(x)) #eigenvalor más chico en abs
        l=min(e)+3*np.finfo(float).eps #lambda es e más el épsilon de la 
        #computadora (3 veces porque si no se enoja)
        E=np.identity(len(x))
        x=x+(l*E) #a x le sumo lambda veces la identidad
    return x
