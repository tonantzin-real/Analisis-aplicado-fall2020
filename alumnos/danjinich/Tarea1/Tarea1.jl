#=
Todo este archivo esta basado en el
Nocedal capitulo 2 y capitulo 8
=#
#=
En este programa se usan multi-threading
Para revisar cuantos threads pueden correr al mismo tiempo ver
    Threads.nthreads()
Para cambiar ese valor, desde la terminal linux debes de empezar julia con
    julia --threads 4
o se puede correr
    export JULIA_NUM_THREADS=4
Para mas info ver https://docs.julialang.org/en/v1/manual/multi-threading/
=#
using LinearAlgebra #Permite el uso de matrices
using ForwardDiff #Hace todo lo que hace este programa pero mejor

function is_pos_semi_def(hess::Array{Float64, 2})::Bool
    return all(x->x>=0,eigvals(hess))
end

function check_optimality(grad::Array{Float64,1}, hess::Array{Float64, 2})::Bool
    #Checa optimalidad
    if all(x->x==0,grad)
        return is_pos_semi_def(hess)
    end
    return false
end

function hess(f::Function, x0::Array{Float64,1}, h::Float64=1e-7)::Array{Float64, 2}
    n = length(x0);
    fx=f(x0);
    H=zeros(n,n);
    fxt=Array{Float64}(undef, n);
    for i in 1:n
        xt=copy(x0); xt[i]+=h;
        fxt[i]=f(xt);
    end;

    Threads.@threads for i in 1:n
        #sleep(rand())
        Threads.@threads for j in 1:i
            H[i,j]-=(fxt[i]+fxt[j]);
            xt=copy(x0); xt[i]+=h; xt[j]+=h;
            H[i,j]+=f(xt)+fx;
            H[i,j]/=h^2;
            if i!=j
                H[j,i]=H[i,j];
            end # if
            #sleep(rand())
            #println("done",i,j)
        end # for
        #println("DONE",i)
    end # for
    return H
end # function

function altHess(f::Function, x0::Array{Float64,1})::Array{Float64,2}
    # La hessiana de una funcion f en un punto x0
    # Asi se hace usando paquetes de Julia
    return ForwardDiff.hessian(f,x0)
end

function grad(f::Function, x0::Array{Float64,1}, h::Float64=1e-6)::Array{Float64,1}
    #Encuentra el gradiente de matrices
    n=length(x0)
    res=Array{Float64}(undef, n)

    Threads.@threads for i in 1:n
        xt1=copy(x0)
        xt1[i]+=h
        xt2=copy(x0)
        xt2[i]-=h
        res[i]=(f(xt1)-f(xt2))
        res[i]/=2*h
    end
    return res
end

function altGrad(f::Function, x0::Array{Float64,1})::Array{Float64,1}
    #Como se deberia de implementar el gradiente
    # Asi se hace usando paquetes de Julia
    ForwardDiff.gradient(f, x0)
end

function f(x0::Array{Float64,1})::Float64
    #La funcion que usamos
    return sqrt(sum(x0.^2))
end

function altF(x0::Array{Float64,1})::Float64
    # Como se deberia de implementar
    return norm(x0,2)
end

function mk(f::Function, xk::Array{Float64,1})
    n=length(xk);
    H=Array{Float64}(undef, n,n);
    g=Array{Float64}(undef, n);
    t1=@async g=grad(f, xk);
    t2=@async H=hess(f, xk);
    fk=f(xk);
    wait(t1); wait(t2);
    g(p::Array{Float64,1})::Float64=begin;
        return fk+transpose(g)*p+(1/2)*transpose(p)*H*p;
    end
    return g;
end
