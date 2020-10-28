#include("path/to/Proyecto1.jl")
using LinearAlgebra #Permite el uso de matrices

function is_pos_semi_def(A::Array{Float64, 2})::Bool
	#Checa si es semidefinida positiva calculando los eigenvalores y checando que sean >= 0	
	return all(x->x>=0,eigvals(A))
end

function is_pos_def(A::Array{Float64, 2})::Bool
	#Checa si es definida positiva calculando los eigenvalores y checando que sean > 0
	return all(x->x>0,eigvals(A))
end

function grad(f::Function, x0::Array; h::Float64=1e-20)::Array{Float64,1}
    #Encuentra el gradiente de vectores haciendo un paso complejo
    n=length(x0)
    res=Array{Float64}(undef, n) #Arreglo vacio

    Threads.@threads for i in 1:n
        xt1=convert(Array{ComplexF64,1}, x0) #Hacemos una copia y convertimos en arreglo de numeros complejos
        xt1[i]+=h*im #Hacemos el paso complejo (im es i)
        res[i]=imag(f(xt1)) #Extraemos la parte imaginaria de la funcion con paso complejo
        res[i]/=h #Dividimos entre el tamaño del paso
    end
    return res
end

function hess(f::Function, x0::Array; h::Float64=1e-7)::Array{Float64, 2}
	#Calcula la hessiana de una funcion en un vector, usando paso complejo y paso centrado
	#Algoritmo sacado de:
	# Yi Cao (2020). Complex step Hessian (https://www.mathworks.com/matlabcentral/fileexchange/18177-complex-step-hessian), MATLAB Central File Exchange. Retrieved October 2, 2020. 
	n = length(x0)
	H=zeros(n,n)
	h2=h^2
	Threads.@threads for i in 1:n
	    	x1=convert(Array{ComplexF64,1}, copy(x0))
	    	x1[i] += h*im #Se hace el paso complejo en el valor i
	    	Threads.@threads for j=i:n
	    		x2=copy(x1)
	    		x2[j] += h #Se hace un paso real hacia delante en el valor j
	    		u1=f(x2)
	    		x2[j] = x1[j] - h #Se hace un paso real hacia atras en el valor j
	    		u2 = f(x2)
	    		H[i,j] = imag(u1-u2)/(2*h2) #Se extrae la diferencia de la parte imaginaria y se divide entre 2*h^2
	    		H[j,i]=H[i,j] #Ya que es simetrica la matriz
    		end #for
   	 end # for
    return H
end

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
	end
	println("Numero de iteraciones:\t",j)
	return xk
end

function rosenbrock(x0::Array; a::Number=1.0, b::Number=100.0)::Number
	return (a-x0[1])^2+b*(x0[2]-x0[1]^2)^2
end

function min_rosenbrock(x0::Array, a::Number, b::Number; tol::Float64=1e-10, maxit::Int=10000)
	f(x)=(a-x[1])^2+b*(x[2]-x[1]^2)^2
	println("El minimo real es:\t\tf([",a,",",a^2,"])=0")
	x= line_search_newton_modification(f,x0;tol=tol,maxit=maxit)
	println("El minimo que se encontro es:\tf([", x[1], ",", x[2], "])=", f(x), "\n\n")
	println("El error absoluto en x es:\t",abs(x[1]-a))
	println("El error relativo en x es:\t",abs((x[1]-a)/a),"\n")
	println("El error absoluto en y es:\t",abs(x[2]-a^2))
	println("El error relativo en y es:\t",abs((x[2]-a^2)/a^2))
end
#min_rosenbrock([rand(1:10),rand(1:100)], rand(1:10), rand(1:1000); tol=1e-10, maxit=10000)
