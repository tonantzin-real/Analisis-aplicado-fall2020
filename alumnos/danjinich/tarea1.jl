using LinearAlgebra
using ForwardDiff

function check_optimality(grad::Array{Float64}, hess::Array{Float64})
    #Checa optimalidad
    if all(x->x==0,grad)
        return isposdef(hess)
    end
    return false
end

function grad(f::Function, x0::Array{Real}, h::Float64=0.000001)
    #Encuentra el gradiente de matrices si no son Float64
    x0=convert(Array{Float64},x0)
    return grad(f,x0)
end

function grad(f::Function, x0::Array{Float64}, h::Float64=0.000001)
    #Encuentra el gradiente de matrices
    res=Matrix{Float64}(undef, size(x0)[1], size(x0)[2])

    for i in 1:size(x0)[1]
        for j in 1:size(x0)[2]
            aux=copy(x0)
            aux[i,j]-=h
            res[i,j]=(f(x0)-f(aux))/h
        end
    end
    return res
end

function altGrad(f::Function, x0::Array{Float64})
    #Como se deberia de implementar el gradiente
    ForwardDiff.gradient(f, x0)
end

function f(x0::Array{Float64})
    #La funcion que usamos
    return sqrt(sum(x0.^2))
end

function altF(x0)
    #Como se deberia de implementar
    return norm(x0,2)
end

function hess(f::Function, x0::Array{Float64})
    #La hessiana de una funcion f en un punto x0
    return ForwardDiff.hessian(f,x0)
end
