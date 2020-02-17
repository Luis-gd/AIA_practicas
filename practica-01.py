
# AIA
# 
# Dpto. de C. de la Computacion e I.A. (Univ. de Sevilla)

# ===================================================================

# En esta prctica vamos a programar el algoritmo de backtracking
# combinado con consistencia de arcos AC3 y la heurstica MRV. 

import random, copy

# ===================================================================
# Representaci�n de problemas de satisfacci�n de restricciones
# ===================================================================

#   Definimos la clase PSR que servir� para representar problemas de
# satisfacci�n de restricciones.

# La clase tiene cuatro atributos:
# - variables: una lista con las variables del problema.
# - dominios: un diccionario que asocia a cada variable su dominio,
#      una lista con los valores posibles.
# - restricciones: un diccionario que asigna a cada tupla de
#      variables la restricci�n que relaciona a esas variables.
# - vecinos: un diccionario que asigna a cada variables una lista con
#      las variables con las que tiene una restricci�n asociada.

# El constructor de la clase recibe los valores de los atributos
# "dominios" y "restricciones". Los otros dos atributos se definen a
# partir de �stos valores.

# NOTA IMPORTANTE: Supondremos en adelante que todas las
# restricciones son binarias y que existe a lo sumo una restricci�n
# por cada par de variables.

class PSR:
    """Clase que describe un problema de satisfacci�n de
    restricciones, con los siguientes atributos:
       variables     Lista de las variables del problema
       dominios      Diccionario que asigna a cada variable su dominio
                     (una lista con los valores posibles)
       restricciones Diccionario que asocia a cada tupla de variables
                     involucrada en una restricci�n, una funci�n que,
                     dados valores de los dominios de esas variables,
                     determina si cumplen o no la restricci�n.
                     IMPORTANTE: Supondremos que para cada combinaci�n
                     de variables hay a lo sumo una restricci�n (por
                     ejemplo, si hubiera dos restricciones binarias
                     sobre el mismo par de variables, considerar�amos
                     la conjunci�n de ambas). 
                     Tambi�n supondremos que todas las restricciones
                     son binarias
        vecinos      Diccionario que representa el grafo del PSR,
                     asociando a cada variable, una lista de las
                     variables con las que comparte restricci�n.

    El constructor recibe los valores de los atributos dominios y
    restricciones; los otros dos atributos ser�n calculados al
    construir la instancia."""

    def __init__(self, dominios, restricciones):
        """Constructor de PSRs."""

        self.dominios = dominios
        self.restricciones = restricciones
        self.variables = list(dominios.keys())

        vecinos = {v: [] for v in self.variables}
        for v1, v2 in restricciones:
            vecinos[v1].append(v2)
            vecinos[v2].append(v1)
        self.vecinos = vecinos

# ===================================================================
# Ejercicio 1
# ===================================================================

#   Definir una funci�n n_reinas(n), que recibiendo como entrada un
# n�mero natural n, devuelva una instancia de la clase PSR,
# correspondiente al problema de las n-reinas.

# Ejemplos:

# >>> psr_n4 = n_reinas(4)
# >>> psr_n4.variables
# [1, 2, 3, 4]
# >>> psr_n4.dominios
# {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2, 3, 4]}
# >>> psr_n4.restricciones
# {(1, 2): <function <lambda> at ...>,
#  (1, 3): <function <lambda> at ...>,
#  (1, 4): <function <lambda> at ...>,
#  (2, 3): <function <lambda> at ...>,
#  (3, 4): <function <lambda> at ...>,
#  (2, 4): <function <lambda> at ...>}
# >>> psr_n4.vecinos
# {1: [2, 3, 4], 2: [1, 3, 4], 3: [1, 2, 4], 4: [1, 2, 3]}
# >>> psr_n4.restricciones[(1,4)](2,3)
# True
# >>> psr_n4.restricciones[(1,4)](4,1)
# False

def n_reinas(n):
	
	def n_reinas_restr(x,y):
		return=(lambda vx,vy: vx != vy and
					abs(x-y) != abs(vx-vy) ) 
	
	doms = {i+1 :[range(1,n+1)] for i in range(n)}
	restrs = dict()
	for x in range(1,n):
		for y un range(x+1,n+1):
			restrs[(x,y)]=n_reinas_restr(x,y) 
	return PSR(doms,restrs)









# ===================================================================
# Parte II: Algoritmo de consistencia de arcos AC3
# ===================================================================

#   En esta parte vamos a definir el algoritmo de consistencia de arcos
# AC3 que, dado un problema de satisfacci�n de restricciones,
# devuelve una representaci�n equivalente que cumple la propiedad de
# ser arco consistente (y que usualmente tiene dominios m�s
# reducidos.)

#   Dado un PSR, un arco es una restricci�n cualquiera del problema,
# asociada con una de las variables implicadas en la misma, a la que
# llamaremos variable distinguida.





# ===================================================================
# Ejercicio 2
# ===================================================================

#   Definir una funci�n restriccion_arco que, dado un PSR, la
# variable distinguida de un arco y la variable asociada; devuelva
# una funci�n que, dado un elemento del dominio de la variable
# distinguida y otro de la variable asociada, determine si verifican
# la restricci�n asociada al arco.

# Ejemplos:

# >>> restriccion_arco(psr_n4, 1, 2)
# <function n_reinas.<locals>.n_reinas_restriccion.<locals>.<lambda>
# at 0x7fdfa13d30d0>
# >>> restriccion_arco(psr_n4, 1, 2)(1, 4)
# True
# >>> restriccion_arco(psr_n4, 1, 2)(3, 2)
# False






       
# ===================================================================
# Ejercicio 3
# ===================================================================

#   Definir un m�todo arcos para la clase PSR que construya un
# conjunto con todos los arcos asociados al conjunto de restricciones
# del problema. Utilizaremos las tuplas para representar a los
# arcos. El primer elemento ser� la variable distinguida y el segundo
# la variable asociada.

# Ejemplo:

# >>> psr_n4 = n_reinas(4)
# >>> arcos_n4 = psr_n4.arcos()
# >>> arcos_n4
# [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (3, 4), (4, 3),
#  (2, 4), (4, 2), (1, 4), (4, 1)]
# >>> psr_n4.restriccion_arco(1, 2)(4, 1)
# True
# >>> psr_n4.restriccion_arco(1, 2)(2, 3)
# False









# ===================================================================
# Ejercicio 4
# ===================================================================

#   Definir la funci�n AC3(psr,doms) que, recibiendo como entrada una
# instancia de la clase PSR y diccionario doms que a cada variable
# del problema le asigna un dominio, aplica el algoritmo de
# consistencia de arcos AC3 a los dominios recibidos (ver tema 1).

# NOTA: La funci�n AC3 debe actualizar los dominios de forma
# destructiva (es decir, despu�s de ejecutar la llamada "AC3(psr,
# doms)", en el diccionario doms debe quedar actualizados.

# Ejemplos:

# >>> psr_n4=n_reinas(4)
# >>> dominios = {1:[2,4],2:[1,2,3,4],3:[1,2,3,4],4:[1,2,3,4]}
# >>> AC3(psr_n4, dominios)
# >>> dominios
# {1: [2, 4], 2: [1, 4], 3: [1, 3], 4: [1, 3, 4]}

# >>> dominios = {1:[1],2:[1,2,3,4],3:[1,2,3,4],4:[1,2,3,4]}
# >>> AC3(psr_n4,dominios)
# >>> dominios
# {1: [], 2: [], 3: [], 4: []}

# >>> dominios = {1:[1,2,3,4],2:[3,4],3:[1,4],4:[1,2,3,4]}
# >>> AC3(psr_n4,dominios)
# >>> dominios
# {1: [2], 2: [4], 3: [1], 4: [3]}







# ===================================================================
# Parte III: Algoritmo de b�squeda AC3
# ===================================================================


# ===================================================================
# Ejercicio 5
# ===================================================================


# Definir una funci�n parte_dominio(doms), que a partir de un diccionario doms
# en el que cada variable del problema tiene asignado un dominio de posibles
# valores (como los que obtiene el algoritmo AC-3 anterior), devuelve dos
# diccionarios obtenidos partiendo en dos el primero de los dominios que no
# sea unitario.   

# Nota: Supondremos que el diccionario doms que se recibe no tiene dominios
# vac�os y que al menos uno de los dominios no es unitario. El m�todo para
# partir en dos el dominio se deja a libre elecci�n (basta con que sea una
# partici�n en dos). 

# Ejemplo:

# >>> doms4_1={1: [2, 4], 2: [1, 4], 3: [1, 3], 4: [1, 3, 4]}
# >>> parte_dominios(doms4_1)
# ({1: [2], 2: [1, 4], 3: [1, 3], 4: [1, 3, 4]}, {1: [4], 2: [1, 4], 3: [1, 3], 4: [1, 3, 4]})







# ===================================================================
# Ejercicio 6
# ===================================================================

# Definir la funci�n b�squeda_AC3(psr), que recibiendo como entrada un psr
# (tal y como se define en el ejercicio 1), aplica el algoritmo de b�squeda
# AC-3 tal y como se define en el tema 2

# Ejemplos:

# >>> psr_nreinas4=n_reinas(4)
# >>> busqueda_AC3(psr_nreinas4)
# {1: 3, 2: 1, 3: 4, 4: 2}
# >>> psr_nreinas3=n_reinas(3)
# >>> busqueda_AC3(psr_nreinas3)
# No hay soluci�n






# ===================================================================
# Ejercicio 7
# ===================================================================



#   En este ejercicio no se pide ninguna funci�n. Tan s�lo comprobar el
# algoritmo resolviendo diversas instancias del problema de las 
# n_reinas. Para visualizar las soluciones, puede ser �til la siguiente
# funci�n:

def dibuja_tablero_n_reinas(asig):

    def cadena_fila(i,asig):
        cadena="|"
        for j in range (1,n+1):
            if asig[i]==j:
                cadena += "X|"
            else:
                cadena += " |"
        return cadena

    n=len(asig)
    print("+"+"-"*(2*n-1)+"+")
    for i in range(1,n):
        print(cadena_fila(i,asig))
        print("|"+"-"*(2*n-1)+"|")
    print(cadena_fila(n,asig))
    print("+"+"-"*(2*n-1)+"+")

# Ejemplos:


# >>> dibuja_tablero_n_reinas(busqueda_AC3(n_reinas(4)))
# +-------+
# | | |X| |
# |-------|
# |X| | | |
# |-------|
# | | | |X|
# |-------|
# | |X| | |
# +-------+

# >>> dibuja_tablero_n_reinas(busqueda_AC3(n_reinas(6)))
# +-----------+
# | | | | |X| |
# |-----------|
# | | |X| | | |
# |-----------|
# |X| | | | | |
# |-----------|
# | | | | | |X|
# |-----------|
# | | | |X| | |
# |-----------|
# | |X| | | | |
# +-----------+

# >>> dibuja_tablero_n_reinas(busqueda_AC3(n_reinas(8)))
# +---------------+
# | | | | | | | |X|
# |---------------|
# | | | |X| | | | |
# |---------------|
# |X| | | | | | | |
# |---------------|
# | | |X| | | | | |
# |---------------|
# | | | | | |X| | |
# |---------------|
# | |X| | | | | | |
# |---------------|
# | | | | | | |X| |
# |---------------|
# | | | | |X| | | |
# +---------------+

# >>> dibuja_tablero_n_reinas(busqueda_AC3(n_reinas(14)))
# +---------------------------+
# | | | | | | | | | | | | | |X|
# |---------------------------|
# | | | | | | | | | | | |X| | |
# |---------------------------|
# | | | | | | | | | |X| | | | |
# |---------------------------|
# | | | | | | | |X| | | | | | |
# |---------------------------|
# | | |X| | | | | | | | | | | |
# |---------------------------|
# | | | | |X| | | | | | | | | |
# |---------------------------|
# | |X| | | | | | | | | | | | |
# |---------------------------|
# | | | | | | | | | | |X| | | |
# |---------------------------|
# |X| | | | | | | | | | | | | |
# |---------------------------|
# | | | | | |X| | | | | | | | |
# |---------------------------|
# | | | | | | | | | | | | |X| |
# |---------------------------|
# | | | | | | | | |X| | | | | |
# |---------------------------|
# | | | | | | |X| | | | | | | |
# |---------------------------|
# | | | |X| | | | | | | | | | |
# +---------------------------+



