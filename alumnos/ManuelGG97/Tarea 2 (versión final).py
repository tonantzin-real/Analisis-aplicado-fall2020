#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[35]:


#Función para ejemplos
def g(x0):
    return sum(x0**2+2*x0-5)

x0 = np.array([4,2,5,2])


# # Tarea 1
# 10.9.20 
# 
# 173199

# ### Gradiente

# In[3]:


def grad(f, x0):
    n = x0.size
    eps = 0.00001
    res = np.zeros(n)
    for i in range(n):
        aux = np.zeros(n)
        aux[i] = eps
        x1 = x0 + aux
        res[i] = (f(x1) - f(x0))/eps        
    return res


# In[4]:


grad(g,x0)


# ### Hessiana
# En hess1 intenté hacer lo del siguiente link, pero no me quedaba: http://www2.math.umd.edu/~dlevy/classes/amsc466/lecture-notes/differentiation-chap.pdf.
# 
# En hess, usé diferenciación como en: https://neos-guide.org/content/difference-approximations#:~:text=One%20method%20for%20approximating%20the,evaluated%20at%20two%20nearby%20points.

# In[5]:


def hess1(f,x0):
    n = x0.size
    eps = 0.00001
    res = np.zeros([n,n])
    for i in range(n):
        for j in range(i+1): #porque es simétrica, y entonces mejor res[j][i] = res[i][j]
            aux1 = np.zeros(n)
            aux2 = np.zeros(n)
            #aux1 = aux2 no jala en numpy; mejor np.copy
            aux1[i] = eps
            aux2[j] = eps
            xij = x0 + aux1 + aux2
            xi = x0 + aux1
            xj = x0 + aux2
            res[i][j] = (f(xij) - f(xi) - f(xj) + f(x0))/(eps**2)
            res[j][i] = res[i][j]
    return res


# In[6]:


hess1(g,x0)


# In[7]:


def hess(f,x0):
    n = x0.size
    eps = 0.00001
    res = np.zeros((n,n))
    for i in range(n):
        aux = np.zeros(n)
        aux[i] = eps
        x1 = x0 + aux
        res[:,i] = (grad(f,x1)-grad(f,x0))/eps
    return res


# In[8]:


hess(g,x0)


# ### Condiciones de optimalidad
# Queremos checar los second orden necessary conditions, es decir,
# 
# $∇f(x^*) = 0$
# 
# $∇^2f(x^*)$ semidefinida positiva (para esto, lo más fácil es usar los eigenvalores)

# In[9]:


def condiciones_optimalidad(f,x0):
    res = ""
    if np.all(grad(f,x0) == 0):
        res += "Cumple con tener gradiente 0. "
    else:
        res += "No cumple con tener gradiente 0. "
    
    eigs = np.linalg.eigvals(hess(f,x0))
    if np.all(eigs > 0):
        res += "Cumple con tener Hessiana semidefinida."
    else:
        res += "No cumple con tener Hessiana semidefinida."
        
    return res


# In[10]:


condiciones_optimalidad(g,x0)


# ### Función de aproximación

# In[11]:


def mk (f,x0,p):
    H = hess(f,x0)
    G = grad(f,x0)
    aux = np.dot(p.T,H)
    return f(x0) + np.dot(p.T,G) + 0.5*np.dot(aux,p)


# ### Ejemplo

# In[12]:


print(grad(g,x0))
print(hess(g,x0))
p = np.array([1,2,3,4])
p = p.T
print(condiciones_optimalidad(g,x0))
print(mk(g,x0,p))


# In[ ]:





# # Tarea 2
# 11.10.20
# 
# 173199 (Alejandro Chávez) en equipo con: 162136 (Manuel García), 149427 (Héctor Vela), 174144 (Karla Alva)

# ### Para encontrar $\alpha$ (Algoritmo 3.1)

# In[13]:


def alfa(f,x0,p):
    a_gorro = 1
    ro = 0.8
    c = 0.0001
    alpha = a_gorro
    while f(x0+alpha*p) > f(x0)+c*alpha*(grad(f,x0).T).dot(p):
        alpha = ro*alpha
        
    return alpha


# ### Cholesky with Added Multiple of the Identity (Algoritmo 3.3)

# In[14]:


def calcula_gamma(A):
    k = 100
    n = A.shape[0]
    beta = 0.001
    
    diagonal = np.diagonal(A)
    minimo = np.amin(diagonal)
    if minimo > 0:
        gamma = 0
    else:
        gamma = -minimo + beta
    #end if
    
    for i in range(k):     
        try:
            np.linalg.cholesky(A+gamma*np.identity(n))
        except np.linalg.LinAlgError:
            gamma = max([2*gamma, beta])
        else:
            break
            
        
    return gamma


# In[15]:


A = np.matrix([[-8,-1,0], [-1,2,-1],[0,-1,2]])
A


# In[16]:


print(calcula_gamma(A))


# ### Método de Newton con modificación a la Hessiana (Algoritmo 3.2)

# In[30]:


#nuestro primer intento, sin saber de la existencia del algoritmo 3.3
def newton_mod1(f,x0):
    k = 1000
    gamma = 0.01
    
    B0 = hess(f,x0)
    eigs = np.linalg.eigvals(B0)  
    
    for i in range(k):
        while not np.all(eigs > 0): #inicia while
            B0 = hess(f,x0)+gamma
            gamma = gamma + 0.01
            eigs = np.linalg.eigvals(B0)
        #terminó while
        
        p0 = -np.linalg.inv(B0).dot(grad(f,x0))        
        alpha = alfa(f,x0,p0)
        x0 = x0+alpha*p0
        #print(f(x0)) (para ver si la función realmente decrece)
        
    return x0


# In[31]:


#nuestro segundo intento, sabiendo de la existencia del 3.3
def newton_mod(f,x0):
    k = 1000
    n = x0.size
    
    for i in range(k):
        gamma = calcula_gamma(hess(f,x0))
        B0 = hess(f,x0)+gamma*np.identity(n)
        
        p0 = -np.linalg.inv(B0).dot(grad(f,x0))        
        alpha = alfa(f,x0,p0)
        x0 = x0+alpha*p0
        #print(f(x0)) (para ver si la función realmente decrece)
        
    return x0


# ### Rosenbrock

# In[34]:


def rosenbrock(x0):
    a = 1
    b = 100
    return (a-x0[0])**2+b*(x0[1]-x0[0]**2)**2


x0 = np.array([1,1])
rosenbrock(x0) #como es el mínimo debería de dar cero


# In[20]:


x0 = np.array([1,1])
p = np.array([1,2])
p = p.T
alpha = alfa(rosenbrock,x0,p)
print (alpha)


# In[32]:


#primer intento
x0 = np.array([22,50])
respuesta = newton_mod1(rosenbrock,x0)
print(respuesta)


# In[33]:


#segundo intento
x0 = np.array([22,50])
respuesta = newton_mod(rosenbrock,x0)
print(respuesta)


# In[ ]:




