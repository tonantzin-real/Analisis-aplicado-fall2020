# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:24:52 2020

@author: Bernardo
"""

import numpy as np
import funciones as fn

a = 1
b = 100
f = lambda x: (a-x[0])**2 + b*(x[1]-x[0]**2)**2

x = np.array([5,20])

c1= 0.1
c2= 0.8
rho = 0.9

#Algoritmo 3.1

def alpha(f,x,p,c1,c2,rho):
    a = 1
    i = 1
    M=5000
    wolfe=fn.wolfe(f,x,p,a,c1,c2)
    while i<M and wolfe==False:
        a = rho*a
        i = i+1
        wolfe=fn.wolfe(f,x,p,a,c1,c2)
    if i==M:
        print("i máxima alcanzada")
    return a

#Algoritmo 3.2

def busqueda_lneal_modif_hessiana(f,x):
    
    optimo=fn.es_optimo(f,x)
    k=0
    while k<300 and optimo==False:
        B = fn.hessiana(f,x) #Aproximación de la Hessiana
        B=fn.volver_pd(B) #Si no es positiva definada la forzamos a que lo sea
        g = -fn.gradiente(f,x)
        p = np.linalg.solve(B,g)
        a = alpha(f,x,p,c1,c2,rho)
        x = x+a*p
        optimo = fn.es_optimo(f,x)
        k=k+1
    if k==300:
        print ("K máxima alcanzada")
    return x
    
#    x = x + a*p
#    B = fn.hessiana(f,x)
#    g = -fn.gradiente(f,x)
#    p = np.linalg.solve(B,g)
#    a = alpha(f,g,B)


z=busqueda_lneal_modif_hessiana(f,x)
print(z)


