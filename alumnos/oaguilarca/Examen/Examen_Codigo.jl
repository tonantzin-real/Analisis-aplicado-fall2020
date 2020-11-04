#include("path/to/Proyecto1.jl")
using LinearAlgebra 
#using Pkg
#Pkg.add("CSV")
#Pkg.add("DataFrames")
using CSV
using DataFrames

#Traigo las funciones del Proyecto 1

function is_pos_semi_def(A::Array{Float64, 2})::Bool
	#Checa si es semidefinida positiva calculando los eigenvalores y checando que sean >= 0
	return all(x->x>=0,eigvals(A))
end

function is_pos_def(A::Array{Float64, 2})::Bool
	#Checa si es definida positiva calculando los eigenvalores y checando que sean > 0
	return all(x->x>0,eigvals(A))
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

function check_optimality(grad::Array{Float64,1}, hess::Array{Float64, 2}; tol::Float64=1e-20)::Bool
    #Checa optimalidad
    if all(x->abs(x)<=tol,grad) #El gradiente es menor a la tolerancia
        return is_pos_semi_def(hess) #La Hessiana es semidefinida positiva
    end
    return false
end

function backtracking_line_search(f::Function, x::Array, d::Array; a::Float64=1.0, p::Float64=0.5, c::Float64=1e-4)::Float64
	#Habia que hacerlo
	y = f(x)
	g = grad(f,x)
	while f(x + a * d) > y+c*a*dot(g,d)
		a *= p
	end
	return a
end

function add_identity(A::Array{Float64,2}; b=1e-4)::Array{Float64,2}
	#Metodo para convertir en definida positiva una matriz
	#Encuentra una t tal que A+I*t es definida positiva
	n=size(A)[1]; Bk=copy(A); i=Matrix{Float64}(I, n, n)
	if (minimum(diag(A))>0)
		t=0
	else
		t=-minimum(diag(A))+b
	end
	while (!is_pos_def(Bk))
		t=max(2*t,b) #Si no funciona crecemos y volvemos a tratar
		Bk=A+i*t
	end
	return Bk
end


function line_search_newton_modification(f::Function, x0::Array; tol::Float64=1e-4, maxit::Int=10000)::Array{Float64,1}
	xk=copy(x0); ak=1.0; n=length(x0)
	Hk=Array{Float64}(undef, n,n);
	j=0
	for i in 1:maxit
		t=@async Hk=hess(f,xk) #Calcula la Hessiana concurrentemente a la derivada
		g=grad(f, xk)
		wait(t)
		if (check_optimality(g, Hk; tol=tol))
			break
		end
		Bk=add_identity(Hk)
		pk=Bk \ (grad(f,xk)*(-1))
		ak=backtracking_line_search(f,xk,pk; a=ak)
		xk += ak*pk
		j=i
		println("Numero de iteraciones:\t",j)
	end
	println("Numero de iteraciones:\t",j)
	return xk
end

#Respecto al problema a resolver:

#Importo los datos y fijo el número de cámaras
crime_data = CSV.read("crime_data.csv");
num_cam=8000

#Calculo la distancia más corta al punto inicial
function dist(x0, lat, long)
	n=floor(Int,length(x0)/2)
	res=Array{Float64}(undef, n)
	for i in 1:n
		res[n]=(x0[i]-lat)^2+(x0[i+n]-long)^2
	end
	return minimum(res)
end

#Calculo la función que busco minimizar "Función de costos en términos de la distancia"
function dist_cost(x0, crime_data)
	n=size(crime_data)[1]
	res=Array{Float64}(undef, n)
	Threads.@threads for i in 1:n
		res[i]=dist(x0,crime_data["lat"][i],crime_data["long"][i])
	end
	return sum(res)
end


# Minimizo la función usando el método de búsqueda lineal modificado de Newton
function max_cobert(num_cam, crime_data; tol::Float64=1e-3, maxit::Int=10000)
  #Elijo un x0 para empezar que esté en la frontera de la zona de la base de datos.
	x0=(minimum(crime_data["lat"]),maximum(crime_data["long"])) 
	f(x)=dist_cost(x,crime_data)
	x=line_search_newton_modification(f, x0; tol=tol, maxit=maxit)
	return cat(x[1:num_cam],x[num_cam+1:2*num_cam], dims=2)
end

#Resuelvo para los datos
max_cobert(num_cam,crime_data;maxit=10000, tol=1e-3)