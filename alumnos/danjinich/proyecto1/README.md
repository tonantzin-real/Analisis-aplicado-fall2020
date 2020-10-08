# Proyecto 1 Análisis Aplicado
Este es un proyecto en Julia, para poder correrlo hay que seguir las siguientes instrucciones:
1. Correr en la terminal `julia`
  * Se recomienda empezar usando `julia --threads 2` o algún otro número en lugar de `2`. Este programa utiliza varios threads y mientras mayor sea el número mayor va a ser la velocidad, esto está acotado por el tamaño de la matriz más grande.
2. Dentro de Julia correr en la terminal `include("path/to/Proyecto1.jl")`, utilizado el path al proyecto.
## Guia basica de Julia
Julia como lenguaje de programacion es relativamente sencillo, con sintaxis similar a lenguajes como Python, R y Matlab, una guia de sus diferencias basicas se puede ver [aqui](https://docs.julialang.org/en/v1/manual/noteworthy-differences/). Aquí están presentadas algunas de las características del lenguaje importantes para este proyecto.
### Tipos
El lenguaje Julia puede ser estricto con los tipos utilizados. Si la función pide un `Float64` es importante pasar algo que el compilador entiende como `Float64`, es decir si se quiere pasar un entero es importante agregar un `.0` después del número.
### Vectores
Los vectores son arreglos unidimensionales, que se pueden utilizar como vectores después de importar el paquete `LinearAlgebra`. Al correr este programa se importa el paquete, pero la instrucción para importarlo es `using LinearAlgebra`. Un vector o arreglo se define como
```
x=[1.0, 2, 3] #Arreglo de Float64
x=[1, 2, 3] #Arreglo de Int64
x=Array{Float64}(undef, 3) #Arreglo vacío de longitud 3
x=zeros(3) #Arreglo de ceros de longitud 3
```
### Matrices
Funcionan con los mismos principios que el vector, siendo arreglos bidimensionales.
```
x=[1.0 2 3; 4 5 6; 7 8 9] #Arreglo de Float64
x=[1 2 3; 4 5 6; 7 8 9] #Arreglo de Int64
x=Array{Float64}(undef, 3, 3) #Arreglo vacío de tamaño 3x3
x=zeros(3, 3) #Arreglo de ceros de tamaño 3x3
x=Matrix{Float64}(I, 3, 3) #Matriz identidad de 3x3
```
### Funciones
Para definir una función anónima se puede hacer:
```
f(x)=x+1
```
Las funciones tienen dos tipos de parámetros, los obligatorios y los opcionales. Los parámetros obligatorios deben de pasar en el mismo orden que en la definición de la función separados por comas. Si se quiere pasar parámetros opcionales, se pone un `;` después de los parámetros obligatorios y se asignan con nombre al parámetro:
```
f(x;y=1)=x+y
f(5) #Regresa 6
f(5;y=1) #Regresa 6
f(5; y=7) #Regresa 12
```
## Documentación
### is_pos_semi_def
Esta función recibe un arreglo bidimensional de `Float64` y regresa un Booleano que es `true` si la matriz es semidefinida positiva y `false` si no lo es.
#### Ejemplo:
```
julia> is_pos_semi_def([1.0 0 0; 0 1 0; 0 0 1])
true
```
### is_pos_def
Esta función recibe un arreglo bidimensional de `Float64` y regresa un Booleano que es `true` si la matriz es definida positiva y `false` si no lo es.
#### Ejemplo:
```
julia> is_pos_def([1.0 0 0; 0 1 0; 0 0 1])
true
```
### grad
Esta función recibe una función, un arreglo numérico de 1 dimensión y opcionalmente una `h` opcional con valor default `1e-20`, y regresa un arreglo de `Float64` de 1 dimensión. Está calculada utilizando un paso complejo, lo que limita las funciones utilizables a todas las funciones que utilizan valores absolutos, pero aumentan significativamente la precisión. El metodo esta basado en [este paper](https://www.researchgate.net/publication/222112601_The_Complex-Step_Derivative_Approximation).
#### Ejemplo:
```
julia> f(x)=x[1]*x[2]
f (generic function with 1 method)

julia> grad(f, [1,3]; h=1e-10)
2-element Array{Float64,1}:
 3.0
 1.0
```
### hess
Esta función recibe una función, un arreglo numérico de 1 dimensión y opcionalmente una `h` opcional con valor default `1e-7`, y regresa un arreglo de `Float64` de 2 dimensiones. Está la calcula utilizando un paso compleja y diferencia centrada, lo que limita las funciones utilizables a todas las funciones que utilizan valores absolutos, pero aumenta significativamente la precisión. El metodo esta basado en [este programa en Matlab](https://www.mathworks.com/matlabcentral/fileexchange/18177-complex-step-hessian).
#### Ejemplo:
```
julia> f(x)=(x[1]^2)*(x[2]^2)
f (generic function with 1 method)

julia> hess(f, [1,3]; h=1e-5)
2×2 Array{Float64,2}:
 18.0  12.0
 12.0   2.0
```
### check_optimality
Esta función recibe una gradiente, una hessiana y opcionalmente recibe una tolerancia `tol` con default `1e-20`. Regresa un booleano que es `true` si es óptimo y `false` si no es óptimo.
#### Ejemplo:
```
julia> f(x)=x[1]^2+x[2]^2
f (generic function with 1 method)

julia> x=[0.0, 0]
2-element Array{Float64,1}:
 0.0
 0.0

julia> check_optimality(grad(f,x),hess(f,x);tol=0.0)
true
```
### backtracking_line_search
Es el algoritmo 3.1 del Nocedal.  Recibe una función, un arreglo numérico, otro arreglo numérico y opcionalmente recibe tres variables `Float64` una `a` con default `1`, una `c` con default `1e-4` y una `p` (rho) con default `0.5`. Regresa un `Float64`.

### add_identity
Busca iterativamente un número `t` tal que la matriz A+I*t sea definida positiva, regresa la matriz A+I*t. Recibe una matriz de `Float64` y opcionalmente un `Float64` `b` con default `1e-4`. Regresa una matriz definida positiva.

### line_search_newton_modification
Es el algoritmo 3.2 del Nocedal. Recibe una función y un punto inicial y opcionalmente recibe una tolerancia `tol` con default `1e-4` y un máximo de iteraciones `maxit` cond default `10000`. Regresa las coordenadas de la mejor aproximación que se logró e imprime el número de iteraciones.
#### Ejemplo
```
julia> f(x)=x[1]^2+x[2]^2
f (generic function with 1 method)

julia> x=[10.0, 100]
2-element Array{Float64,1}:
  10.0
 100.0

julia> line_search_newton_modification(f,x;tol=0.0,maxit=1000000)
Numero de iteraciones:    4
2-element Array{Float64,1}:
 0.0
 0.0
```
### rosenbrock
Es la función de Rosenbrock. Recibe un arreglo numérico y opcionalmente números `a` con default `1` y `b` y regresa el resultado de la función de rosenbrock
#### Ejemplo
```
julia> rosenbrock([1,2]; a=2, b=200)
201
```
### min_rosenbrock
Esta función sirve para demostrar el funcionamiento del código. Recibe un punto inicial, un valor de a y un valor de b y opcionalmente recibe una tolerancia `tol` con default `1e-10` y un máximo de iteraciones `maxit` con default `10000`. Imprime un reporte con el resultado exacto, el número de iteraciones, el resultado exacto y el error.
#### Ejemplo
```
julia> min_rosenbrock([rand(1:10),rand(1:100)], rand(1:10), rand(1:1000); tol=0.0, maxit=10000)
El minimo real es:        f([2,4])=0
Numero de iteraciones:    10000
El minimo que se encontro es:    f([2.000000000000077,4.000000000000307])=5.9024545080899156e-27


El error absoluto en x es:    7.682743330406083e-14
El error relativo en x es:    3.8413716652030416e-14

El error absoluto en y es:    3.0730973321624333e-13
El error relativo en y es:    7.682743330406083e-14
```

## Integrantes
* Dan Jinich
* Maria Jose Sedano
* Oscar Aguilar
* Francisco Aranburu


