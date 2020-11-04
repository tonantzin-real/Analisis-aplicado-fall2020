using CSV
using DataFrames
#Para la Hessiana y el gradiente
using ForwardDiff
#Para la func. isposdef(). (Es positiva definida)
using LinearAlgebra


data = CSV.read("/home/fran/Downloads/crime_data.csv");

x = data["long"]
y = data["lat"]

x0 = zeros(31056)
y0 = zeros(31056)

for i in 1:31056
    x0[i] = data["long"][i]
    y0[i] = data["lat"][i]
end

xmin = minimum(x0)
xmax = maximum(x0)
ymin = minimum(y0)
ymax = maximum(y0)

#pasamos todo al intervalo [0,1]
for i in 1:31056
    x0[i] = (1/(xmax - xmin))*(x0[i] - xmin)
    y0[i] = (1/(ymax - ymin))*(y0[i] - ymin)
end


#Medimos distancias de cada punto del mallado a cada punto del dataset.
#Es la funcion que vamos a minimizar.
function mide_distancias(pos)
    dist = 0;
	len = length(x0)
    for i in 1:len
        dist = dist + sqrt((pos[1]-x0[i])^2 + (pos[2] - y0[i])^2)
    end
    return dist
end

#Primera vuelta, si pongo un error menor no jala, se queda corriendo
#horas.
z = line_search_newton_modification(mide_distancias,[0,0],3e-3)
println("El minimo que se encontro es:\tf([", z[1], ",", z[2], "])=", mide_distancias(z), "\n\n")

divide_y_venceras(mide_distancias,z[1],z[2],x0,y0,1)

#Dividimos en 4 el plano con centro en el optimo anterior y repetimos el proceso
#para cada cuadrante.
m0 = x0
n0 = y0

x0 = m0
y0 = n0

function divide_y_venceras(f,posx,posy,x0,y0, eval)
	#Dividimos x0 y y0 en 4 subconjuntos
	a1 = dividir(1,posx,posy,x0,y0)
	a2 = dividir(2,posx,posy,x0,y0)
	a3 = dividir(3,posx,posy,x0,y0)
	a4 = dividir(4,posx,posy,x0,y0)
	##Le mandamos al algoritmo los 4 nuevos cuadrantes a evaluar.
	##Aguas, a falta de imaginacion dependemos del scope de las variables.
	x0 = a1[1]
	y0 = a1[2]
	z1 = line_search_newton_modification(mide_distancias,[x0[1],y0[1]],3e-3)
	x0 = a2[1]
	y0 = a2[2]
	z2 = line_search_newton_modification(mide_distancias,[x0[1],y0[1]],3e-3)
	x0 = a3[1]
	y0 = a3[2]
	z3 = line_search_newton_modification(mide_distancias,[x0[1],y0[1]],3e-3)
	x0 = a4[1]
	y0 = a4[2]
	z4 = line_search_newton_modification(mide_distancias,[x0[1],y0[1]],3e-3)
	println("El minimo que se encontro en 1.",eval," es:\tf([", z1[1], ",", z1[2], "])=", mide_distancias(z), "\n\n")
	println("El minimo que se encontro en 2.",eval,"  es:\tf([", z2[1], ",", z2[2], "])=", mide_distancias(z), "\n\n")
	println("El minimo que se encontro en 3.",eval,"  es:\tf([", z3[1], ",", z3[2], "])=", mide_distancias(z), "\n\n")
	println("El minimo que se encontro en 4.",eval,"  es:\tf([", z4[1], ",", z4[2], "])=", mide_distancias(z), "\n\n")
	if(eval < 8000)
		divide_y_venceras(mide_distancias,z1[1],z1[2],a1[1],a1[2],4*eval)
		divide_y_venceras(mide_distancias,z2[1],z2[2],a2[1],a2[2],4*eval)
		divide_y_venceras(mide_distancias,z3[1],z3[2],a3[1],a3[2],4*eval)
		divide_y_venceras(mide_distancias,z4[1],z4[2],a4[1],a4[2],4*eval)
	else
		return 1;
	end end


function dividir(cuad,x,y,x0,y0)
	resx = Float64[];
	resy = Float64[];
	len = length(x0)
	if(cuad == 1)
		for i in 1:len
			if(x0[i]>=x && y0[i]>=y)
				push!(resx,x0[i])
				push!(resy,y0[i])
			end
		end
	elseif cuad == 2
		for i in 1:len
			if(x0[i]<=x && y0[i]>=y)
				push!(resx,x0[i])
				push!(resy,y0[i])
			end
		end
	elseif cuad == 3
		for i in 1:len
			if(x0[i]<=x && y0[i]<=y)
				push!(resx,x0[i])
				push!(resy,y0[i])
			end
		end
	else
		for i in 1:len
			if(x0[i]>=x && y0[i]<=y)
				push!(resx,x0[i])
				push!(resy,y0[i])
			end
		end
	end
	return resx, resy end






#------------------------------------------------------------------------------
#Metodos del proyecto----------------------------------------------------------

function line_search_newton_modification(f::Function, x0::Array, tol::Float64=1e-4, maxit::Int=10000)::Array{Float64,1}
	print(x0)
	xk=copy(x0); ak=1.0; n=length(x0)
	Hk=Array{Float64}(undef, n,n);
	j=0
	for i in 1:maxit
		Hk=hess(f,xk)
		g=grad(f, xk)
		if (check_optimality(g, Hk; tol=tol))
			break
		end
		Bk=add_identity(Hk)
		pk=Bk \ (grad(f,xk)*(-1))
		ak=backtracking_line_search(f,xk,pk; a=ak)
		xk += ak*pk
		print(xk)
		print("\n")
		j=i
	end
	println("Numero de iteraciones:\t",j)
	return xk
end

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

function gradPruebas(f, xk)
    return ForwardDiff.gradient(f,xk);
end

function hessPruebas(f, xk)
    return ForwardDiff.hessian(f,xk);
end

function is_pos_semi_def(A::Array{Float64, 2})::Bool
	#Checa si es semidefinida positiva calculando los eigenvalores y checando que sean >= 0
	return all(x->x>=0,eigvals(A))
end

function is_pos_def(A::Array{Float64, 2})::Bool
	#Checa si es definida positiva calculando los eigenvalores y checando que sean > 0
	return isposdef(A)
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

function check_optimality(grad::Array{Float64,1}, hess::Array{Float64, 2}; tol::Float64=1e-2)::Bool
    #Checa optimalidad
    if all(x->abs(x)<=tol,grad) #El gradiente es menor a la tolerancia
        return is_pos_semi_def(hess) #La Hessiana es semidefinida positiva
    end
    return false
end
