# Proyecto 1 Analisis Aplicado
Este es un proyecto en Julia, para poder correrlo hay que seguir las siguientes instrucciones:
1. Correr en la terminal `julia`
  * Se recomienda empezar usando `julia --threads 2` o algun otro numero en lugar de `2`. Este programa utiliza varios threads y mientras mayor sea el numero mayor va a ser la velocidad, esto esta acotado por el tamaño de la matriz mas grande.
2. Dentro de Julia correr en la terminal `include("path/to/Proyecto1.jl")`, utilizado el path al proyecto.
## Funciones
### is_pos_semi_def
Esta funcion recibe un arreglo bidimensional de `Float64` y regresa un Booleano que es `true` si la matriz es semidefinida positiva y `false` si no lo es.
#### Ejemplo:
```
julia> is_pos_semi_def([1.0 0 0; 0 1 0; 0 0 1])
true
```
### is_pos_def
Esta funcion recibe un arreglo bidimensional de `Float64` y regresa un Booleano que es `true` si la matriz es definida positiva y `false` si no lo es.
#### Ejemplo:
```
julia> is_pos_def([1.0 0 0; 0 1 0; 0 0 1])
true
```
### grad
Esta funcion recibe una funcion, un arreglo numerico de 1 dimensión y opcionalmente una `h` opcional con valor default `1e-20`, y regresa un arreglo de `Float64` de 1 dimensión. Esta la calcula utilizando un paso complejo, lo que limita las funciones utilizables a todas las funciones que utilizan valores absolutos, pero aumenta significativamente la precision. El metodo esta basado en [este paper](https://www.researchgate.net/publication/222112601_The_Complex-Step_Derivative_Approximation).
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
Esta funcion recibe una funcion, un arreglo numerico de 1 dimensión y opcionalmente una `h` opcional con valor default `1e-7`, y regresa un arreglo de `Float64` de 2 dimensionwa. Esta la calcula utilizando un paso compleja y diferencia centrada, lo que limita las funciones utilizables a todas las funciones que utilizan valores absolutos, pero aumenta significativamente la precision. El metodo esta basado en [este programa en Matlab](https://www.mathworks.com/matlabcentral/fileexchange/18177-complex-step-hessian).
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
Esta función recibe una gradiente, una hessiana y opcionalmente recibe una tolerancia `tol` con default `1e-20`. Regresa un booleano que es `true` si es optimo y `false` si no es optimo.
#### Ejemplo: 
```
julia> f(x)=(x[1]^2)*(x[2]^2)
f (generic function with 1 method)

julia> hess(f, [1,3]; h=1e-5)
2×2 Array{Float64,2}:
 18.0  12.0
 12.0   2.0
```

Proyecto elaborado por:
* Dan Jinich
* Maria Jose Sedano 
* Oscar Aguilar
* Francisco Aranburu
