import numpy as np

def check_optimality(gk,Hk,tolT): 
#   Revisa que se cumplan las condiciones de optimalidad: Grad=0, Hess positiva semidefinida

#   gk - Vector Gradiente de f en xk
#   Hk - Matriz Hessiana de f en xk
#   tolT - Tolerancia de el gradiente

    return es_pos_def(Hk) and all(abs(gk)<tolT)


def Grad(f, xk, h):
#   Calcula el gradiende de f en xk, con diferencia h

#   f - Función
#   xk - Punto a evaluar
#   h - Diferencia de aproximación

    n = xk.size
    g = np.zeros(n)
    for i in range(0, n):
        b = np.copy(xk)
        b[i] += h
        g[i] = (f(b) - f(xk))/(h)
    return g

def Hess(f,xk,h):
#   Calcula el Hessiano en xk, con diferencia h

#   f - Función
#   xk - Punto a evaluar
#   h - Diferencia de aproximación

    n=xk.size
    H=np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            ff=np.copy(xk)
            ff[i]+=h
            ff[j]+=h
            
            fb=np.copy(xk)
            fb[i]+=h
            fb[j]-=h
            
            bf=np.copy(xk)
            bf[i]-=h
            bf[j]+=h
            
            bb=np.copy(xk)
            bb[i]-=h
            bb[j]-=h
            
            H[i,j]=(f(ff)-f(fb)-f(bf)+f(bb))/(4*h**2)
    return H

def es_pos_def(A):
#   Prueba si la matriz A es positiva semidefinida si A es simétrica

#   A - Matriz a probar

        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
        return False
    

def PruebaMod_Hess(Hk, Beta ) :
#   Modifica la Hessiana para que sea positiva semidefinida, según el algoritmo 3.3 de Nocedal Ed. 2

#   Hk - Matrizz hessiana a modificar
#   Beta - Valor arbitrario a aumentar

    n=int(np.sqrt(np.size(Hk)))
    tk=0
    v=np.matrix.diagonal(Hk)
    
    if min(v)<0:
        tk=-min(v)+Beta    
    

    while es_pos_def(Hk+np.eye(n)*tk)==False:
           tk= max(2*tk, Beta);
    
    return Hk+np.eye(n)*tk

def BactrackSearch(f,xk,pk,gk,alpha0,c,rho):
#   Busca el paso según el algoritmo de 3.1 de Nocedal Ed. 2

#   f - Función a evaluar
#   xk - punto alrededor del que se evalúa
#   pk -Dirección elegida previamente
#   alpha0 - Valor de paso máximo
#   c - peso sobre la derivada direccional
#   rho - factor de disminución del paso

    alpha_k=alpha0
   

    while f(xk+alpha_k*pk)> f(xk)+c*alpha_k*np.dot(gk,pk):
        
        alpha_k=alpha_k*rho
    return alpha_k


def Min_LineSearchNewton(f , x0 , tolT , h1 ,h2 , alpha0, c, rho,maxit):  
#   Busca el mínimo de una función dada una aproximación inicial x0. Utiliza el método de búsqueda lineal de Newton
#   con una aproximación de la Hessiana adaptada

#   f-Función
#   x0 - Punto inicial
#   tolT -  Tolerancia sobre condición de optimalidad
#   h1 -Diferencias para gradiente
#   h2 -Diferencias para Hessiana
#   c - peso sobre el gradiente para búsqueda lineal Bactracking
#   rho - factor de disminución del paso para búsqueda lineal Bactracking
#   maxit - Máximo número de Iteraciones

    xk=x0
    n=np.size(x0)
    for k in range(0,maxit):
    
    
        gk=Grad(f,xk,h1)                                       
        Hk=Hess(f,xk,h2)         
        
        if check_optimality(gk,Hk,tolT) :
            break
        
        HessMod=PruebaMod_Hess( Hk, 1e-3 ) 

        pk= -np.linalg.solve(HessMod,gk)
        
        
        alpha_k=BactrackSearch(f,xk, pk, gk, alpha0 ,c,  rho) 
        
        xk_1=xk
        xk=xk+alpha_k*pk
        
        
        print(xk)
        print(f(xk))
            
    return [xk,k]

f = open('crime_data.csv')
f.readline()
X = np.loadtxt(f, delimiter=',', converters={
    0: lambda s: 0,
    1: lambda s: 0,
    2: lambda s: 0,
    3: lambda s: float(s),
    4: lambda s: float(s)
})

X = X[:,3:5]

def C_Camaras(C):
    T = 0;
    
    for i in range(0,7999):
        for j in range(i+1,8000):
            T += 1/(np.linalg.norm(C[i,:]-C[j,:]))
    return T 

def C_Crimenes(C): 
    T = 0;
    for i in range(0,np.size(X[:,0])):    
        for j in range(0,8000):
            T += np.linalg.norm(C[j,:]-X[i,:])
    return T 


def F_objetivo(C):
    
    return C_Camaras(C) + C_Crimenes(C)

x0 = np.random.rand(8000,2)+(19,-99)

Min_LineSearchNewton(F_objetivo,x0,.00001,.00000001,.00001,1, 0.8, 0.9,100)
