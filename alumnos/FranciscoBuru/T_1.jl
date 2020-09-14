#Para la Hessiana y el gradiente
using ForwardDiff
#Para la func. isposdef(). (Es positiva definida)
using LinearAlgebra

# Revisa optimalidad-------------------------------------------------------
function checkOptimality(grad, hess)
    #Revisa optimalidad de segundo orden, esto es:
        #1.-El gradiente = 0 en todas las entradas.
        #2.-La hessiana sea Positiva definida.
        if(grad == zeros(length(grad)))
            return isposdef(hess)
        else
            return false
        end
    end
#--------------------------------------------------------------------------

#Métodos con paquetes de Julia---------------------------------------------
function grad(f, x0)
    return ForwardDiff.gradient(f,x0);
end

function hess(f, x0)
    return ForwardDiff.hessian(f,x0);
end

function solve(f, x0, eps)
    return checkOptimality(grad(f,x0,eps), hess(f,x0))
end

#---------------------------------------------------------------------------

#Métodos a manopla----------------------------------------------------------
function grad2(f, x0, eps)
    len = length(x0);
    res = zeros(len)
    for i in 1:len
        z = zeros(len)
        z[i] = eps
        res[i] = (f(x0 + z)-f(x0))/eps
    end
    return res
end
#No pude implementarla por problemas con las derivadas parciales.
#function hess2(f, x0, eps)
#    gra = grad2(f, x0, eps);
#    len = length(x0);
#    res = zeros(len, len)
#        for j in 1:len
#            z = zeros(len)
#            z[i] = eps
#            res[i,j] = (f(gra + z)-f(gra))/eps
#            print(z)
#        end
#    end
#    return res
#end
#---------------------------------------------------------------------------


#----------------------------PRUEBAS----------------------------------------
#Variables a usar y ejecución

function f(x)

    return sum(x.^2 + 2*x)
end
f(arre)
arre = [4,2,5,2]
ep = 1e-6

checkOptimality(grad(f,arre), hess(f,arre))

#Pruebas

gradi = grad(f,arre)
gradi2 = grad2(f,arre,ep)
hessi = hess(f,arre)
#hessi2 = hess2(f, arre, ep)
