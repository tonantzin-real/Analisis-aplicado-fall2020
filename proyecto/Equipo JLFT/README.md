### **Proyecto Final Análisis Aplicado**
##### Otoño 2020

Integrantes del equipo
- José Luis Cordero Rodríguez
- Tonantzin Real Rojas
- Francisco Velasco Medina

### **Métodos y algoritmos**
Para este proyecto lo que se hizo fue, en una clase de Python llamada MetodosPF, se programaron 4 métodos y 4 algoritmos para presentar el trabajo final. Los métodos y algoritmos implementados fueron:

Métodos utilizados:
1.   **grad:** método para calcular el gradiente de una función
2.   **hess:** método para calcular la hessiana de una función
3.   **norma:** método para calcular la norma 2 de un vector
4.   **cholesky:** método para volver suficientemente positiva definida a la hessiana y poder aplicarle la descomposición Cholesky

Algoritmos requeridos en el proyecto:
1.   **backS:** algoritmo que corresponde a la búsqueda lineal base
2.   **newton:** algoritmo que corresponde al algoritmo de Newton
3.   **newtonMod:** algoritmo que corresponde al algoritmo de Newton modificado; es decir, con modificación a la hessiana
4.   **bfgs:** algoritmo que corresponde al algoritmo BFGS

### **Funciones de prueba**
Para probar que efectivamente nuestros algoritmos funcionaron, estos los probamos con tres funciones en total. La primera fue un polinomio de tercer grado 
$$x^2 + 3y^3$$
Posteriormente con la función de Rosenbrock
$$f(x,y)=(a-x)^2 + b(y-x^2)^2$$
con $a=1$ y $b=100$ y otros parámetros como $a=2,3,4$ y $b=25,50,75,100$. \\
Por último utilizamos la función del examen dada por 
$$f(cams) = \sum_i \sum_j ||cam_i - cam_j ||_2^2 + \sum_{i \neq j}\frac{1}{||cam_i - cam_j ||_2^2}$$

Todas las pruebas que realizamos, funcionaron y arrojaron los resultados correctos; sin embargo, para la última función el tiempo de ejecución fue muy largo. 

### **Crímenes y costos**
Para la última función que probamos, $f(cams)$, el archivo csv original constaba de más de 31 mil datos de coordenadas de crímenes y la fecha y hora de cuando sucedieron. De las 7 columnas en total del archivo, únicamente nos interesaban las referentes a la ubicación de los crímenes; es decir, la latitud y la longitud. Por ello, lo primero que hicimos fue tomar estas dos columnas y normalizar todos los valores para poder trabajar con ellos más fácilmente y dar una interpretación. En vez de definir una función de $\mathbb{R}^n \times \mathbb{R}^2 \rightarrow \mathbb{R}$, transformamos los datos para que la función fuera de $\mathbb{R}^{2n} \rightarrow \mathbb{R}$ donde $n$ corresponde a la cantidad de cámaras a ubicar. \\
Debido al elevado costo computacional que representa este problema, no probamos la función con las 8000 cámaras sugeridas; de igual forma, puesto que la cantidad de coordenadas de crímenes era muy grande, decidimos quedarnos únicamente con una parte de ellas. En el ejemplo que utilizamos en nuestro código, nos quedamos con 8 cámaras y 24 crímenes; sin embargo, probamos con más valores pero por el tiempo de ejecución, consideramos que esa combinación era la más adecuada.