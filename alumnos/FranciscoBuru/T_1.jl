#Para la Hessiana y el gradiente
using ForwardDiff
#Para la func. isposdef(). (Es positiva definida)
using LinearAlgebra

# Revisa optimalidad-------------------------------------------------------
function checkOptimality(f, xk, eps = 1e-6)
    #Revisa optimalidad de segundo orden, esto es:
        #1.-El gradiente = 0 en todas las entradas.
        #2.-La hessiana sea Positiva definida.
        gr = grad(f,xk,eps)
        if( gr == zeros(length(gr)))
            return isposdef(hess(f,xk,eps))
        else
            return false
        end
    end


#--------------------------------------------------------------------------

#Métodos con paquetes de Julia---------------------------------------------
#Los uso para verificar que mis métodos den algo decente.
function gradPrueba(f, xk)
    return ForwardDiff.gradient(f,xk);
end

function hessPrueba(f, xk)
    return ForwardDiff.hessian(f,xk);
end

#---------------------------------------------------------------------------

#Métodos para la clase------------------------------------------------------
#Gradiente
function grad(f, xk, eps = 1e-6)
    len = length(xk);
    res = zeros(len)
    for i in 1:len
        z = zeros(len)
        z[i] = eps
        res[i] = deriva(f, xk + z, xk - z, eps )
    end
    return res
end

#Hessiana
function hess(f, xk, eps = 1e-6)
    len = length(xk);
    res = zeros(len, len)
    for i in 1:len
        for j in 1:len;
            res[i,j] = deriva2(f, xk, eps, i, j)
        end
    end
    return res;
end

function mk(f,xk,eps = 1e-6)
    gra = grad(f,xk,eps)
    hes = hess(f,xk,eps)
    p = -gra/normaDos(gra)
    print(p)
    return f(xk) + transpose(p)*gra + 0.5*transpose(p)*hes*p
end

#Funciones auxiliares para derivar
#Derivada de primer orden
function deriva(f, x00, x01, eps)
    return (f(x00) - f(x01))/(2*eps)
end

#Derivada de 2o orden, 1a cra indice1a y 2a cra indice2a
function deriva2(f, x00, eps, indice1a, indice2a)
    z1 = zeros(length(x00))
    z2 = zeros(length(x00))
    z1[indice1a] = eps
    z2[indice2a] = eps
    return (f(x00 + z1 + z2)- f(x00 + z1 - z2) - f(x00 - z1 + z2) + f(x00 - z1 - z2))/(4*eps.^2)
end

function normaDos(x)
    return sqrt(sum(x.^2))
end

#---------------------------------------------------------------------------


#----------------------------PRUEBAS----------------------------------------
#Función a evaluar

function f(x)
    return 10(x[2]-x[1].^2).^2 + (1-x[1]).^2
end

#Pruebas -------------------------------------------------------------------
arre = [0,1]
ep = 1e-6

mk(f,arre)

#Librerías
gradi = grad(f,arre)
hessi = hess(f,arre)

r = checkOptimality(f,arre)
