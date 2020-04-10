#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ==========================================================
# Ampliación de Inteligencia Artificial. Tercer curso.
# Grado en Ingeniería Informática - Tecnologías Informáticas
# Curso 2019-20
# Ejercicio de programación
# ===========================================================

# -----------------------------------------------------------
# NOMBRE: Luis 
# APELLIDOS: Galocha Domínguez
# -----------------------------------------------------------

import random


# Escribir el código Python de las funciones que se piden en el
# espacio que se indica en cada ejercicio.

# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS FUNCIONES QUE SE
# PIDEN (aquellas funciones con un nombre distinto al que se pide en el
# ejercicio NO se corregirán).

# ESTE ENTREGABLE SUPONE 1.25 PUNTOS DE LA NOTA TOTAL

# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: la realización de los ejercicios es un
# trabajo personal, por lo que deben completarse por cada estudiante de manera
# individual.  La discusión con los compañeros y el intercambio de información
# DE CARÁCTER GENERAL con los compañeros se permite, pero NO AL NIVEL DE
# CÓDIGO. Igualmente el remitir código de terceros, obtenido a través
# de la red o cualquier otro medio, se considerará plagio.

# Cualquier plagio o compartición de código que se detecte significará
# automáticamente la calificación de CERO EN LA ASIGNATURA para TODOS los
# alumnos involucrados, independientemente de otras medidas de carácter
# DISCIPLINARIO que se pudieran tomar. Por tanto a estos alumnos NO se les
# conservará, para futuras convocatorias, ninguna nota que hubiesen obtenido
# hasta el momento.
# *****************************************************************************



# Lo que sigue es la implementación de la clase HMM vista en la práctica 2,
# que representa de manera genérica un modelo oculto de Markov.

class HMM(object):
    """Clase para definir un modelo oculto de Markov"""

    def __init__(self,estados,mat_ini,mat_trans,observables,mat_obs):
        """El constructor de la clase recibe una lista con los estados, otra
        lista con los observables, un diccionario representado la matriz de
        probabilidades de transición, otro diccionario con la matriz de
        probabilidades de observación, y otro con las probabilidades de
        inicio. Supondremos (no lo comprobamos) que las matrices son 
        coherentes respecto de la  lista de estados y de observables."""
        
        self.estados=estados
        self.observables=observables
        self.a={(si,sj):ptrans
                for (si,l) in zip(estados,mat_trans)
                for (sj,ptrans) in zip(estados,l)}
        self.b={(si,vj):pobs
                for (si,l) in zip(estados,mat_obs)
                for (vj,pobs) in zip(observables,l)}
        self.pi=dict(zip(estados,mat_ini))

# Las variables ej1_hmm y ej2_hmm son objetos de la clase HMM, representando
# respectivamente los ejemplos de modelo oculto de Markov que se dan en las
# diapositivas:

ej1_hmm=HMM(["c","f"],
            [0.8,0.2],
            [[0.7,0.3],[0.4,0.6]],
            [1,2,3],   
            [[0.2,0.4,0.4],[0.5,0.4,0.1]])
            

ej2_hmm=HMM(["l","no l"],
            [0.5,0.5],
            [[0.7, 0.3], [0.3,0.7]],
            ["u","no u"],   
            [[0.9,0.1],[0.2,0.8]])





# ========================================================
# Ejercicio 1
# ========================================================

# El algoritmo de Viterbi se define como sigue:

# Entrada: un modelo oculto de Markov y una secuencia
#          de observaciones, o_1, ..., o_t, 
# Salida: La secuencia de estados más probable, dadas las
#         observaciones. 

# Este algoritmo está explicado en el tema 2 de teoría:

# Inicio: nu(1,si) = b(i)(o1)pi(i) para 1 <= i <= n
#         pr(1,si) = null
# Para k desde 2 a t:
#    Para j desde 1 a n:
#         nu(k,sj) = b(j)(ok) * max([a(i,j) * nu(k-1, si) 
#                                    para 1 <= i <= n]) 
#         pr(k,sj) = argmax([a(i,j) * nu(k-1, si) para 1 <= i <= n])
# Hacer s = argmax([nu(t,si) para 1 <= i <= n])
# Devolver la secuencia de estados que lleva hasta s, usando para ello los
#         punteros almacenados en pr.  


# Se pide: 

# Implementar la función viterbi que use el algoritmo anterior a
# partir de un modelo oculto de Markov y una lista de observaciones,
# calcule la lista: [s_1, ..., s_t] con la sucesión de estados más
# probables usando adecuadamente el algoritmo de Viterbi.

# Ejemplos:


# >>> viterbi(ej1_hmm,[3,1,3,2])
# ['c', 'c', 'c', 'c']

# >>> viterbi(ej2_hmm,["u","u","no u"])
# ['l', 'l', 'no l']



# INDICACIÓN: Como ayuda, se proporciona la siguiente función viterbi_pre, que sería una
# versión preliminar de la función que se pide. Esta función preliminar sólo
# calcula los valores nu_k del algoritmo, pero hay que modificarla para
# incluir la "infraestructura" necesaria y poder obtener la secuencia de
# estados más probable.

def viterbi_pre(hmm,observaciones):
        """Versión pre-Viterbi que calcula los nu_k"""
        nu_list=[hmm.b[(e,observaciones[0])]*hmm.pi[e] for e in hmm.estados]
        for o in observaciones[1:]:

            nu_list=[hmm.b[(e,o)]*max(hmm.a[(e1,e)]*nu 
                                                for (e1,nu) in zip(hmm.estados,nu_list))
                                            for e in hmm.estados]
        return nu_list


def viterbi(modelo, observaciones):
    nu_list = [modelo.b[(i, observaciones[0])] * modelo.pi[i] for i in modelo.estados]
    anteriores = [[None for i in modelo.estados]]
    for o in observaciones[1:]:
        nu = []
        antr = []
        for j in modelo.estados:
            (v, pr) = max([(modelo.a[i, j] * p, modelo.estados.index(i))
                           for i, p in zip(modelo.estados, nu_list)])
            nu.append(modelo.b[j,o] * v)
            antr.append(pr)
        nu_list = nu
        anteriores.append(antr)         
    indices = [nu_list.index(max(nu_list))]
    i = len(observaciones) - 1
    while i > 0:
        elemento = anteriores[i][indices[0]]
        indices.insert(0,elemento)
        i = i - 1
    solucion = [modelo.estados[i] for i in indices]
    return solucion 

#print(ej2_hmm.a)
print(viterbi(ej2_hmm,["u","u","no u"]))






# ========================================================
# Ejercicio 2
# ========================================================


# Se pide ahora definir un algoritmo de muestreo para modelos ocultos de
# Markov. Es decir una función muestreo_hmm(hmm,n), que recibiendo un modelo
# oculto de Markov y un número natural n, genera una secuencia de $n$ estados
# y la correspondiente secuencia de $n$ observaciones, siguiendo las
# probabilidades  del modelo. 

# Explicamos a continuación con más detalle este algoritmo de muestreo,
# ilustrándolo con un ejemplo, suponiendo que el modelo oculto de Markov es el
# primer ejemplo que se usa en las diapositivas (el de los helados).

#   El problema es generar una secuencia de estados, con las correspondientes
# observaciones, siguiendo las probabilidades del modelo. Supondremos que
# disponemos de un generador de números aleatorios entre 0 y 1, con
# probabilidad uniforme (es decir, el random de python). Vamos a generar una
# secuencia de 3 estados y la correspondiente secuencia de 3 observaciones.

#   El primer estado ha de ser generado siguiendo el vector de probabilidades
# iniciales. En este caso pi_1=P(c)=0.8 y pi_2=P(f)=0.2. Supongamos que
# al generar un número aleatorio, obtenemos $0.65$. Esto significa que el
# primer estado en nuestra secuencia es $c$.

#   Ahora tenemos que generar la observación correspondiente, y para ello
# usamos las probabilidades de la matriz de observaciones. Puesto que el estado
# actual es c, las probabilidades de generar cada observable son: b_1(1) =
# P(1|c) = 0.2, b_1(2) = P(2|c) = 0.4 y b_1(3) = P(3|c) = 0.4. Si
# obtenemos aleatoriamente el número 0.53, eso significa que la observación
# correspondiente es 2, ya que es la primera de las observaciones cuya
# probabilidad acumulada (0.2+0.4) supera a 0.53.

#   Generamos ahora el siguiente estado. Para ello usamos las probabilidades
# de la matriz de transición. Como el estado actual es c, las probabilidades de
# que cada estado sea generado a continuación son: a_11 = P(c|c) = 0.7 y
# a_12 = P(f|c) = 0.3. Si aleatoriamente obtenemos el número 0.82,
# significa que hemos obtenido f como siguiente estado.

#   Para la siguientes observaciones y estados, procedemos de la misma
# manera. Por ejemplo, si el siguiente número aleatorio que obtenemos es
# 0.29, la observación correspondiente es 1. Si a continuación obtenemos
# los números aleatorios 0.41 y 0.12, el estado siguiente es f y la
# observación sería 1. En resumen, hemos generado la secuencia de estados
# [c,f,f] con la correspondiente secuencia de observaciones [2,1,1].

# Ejemplos (téngase en cuenta que la salida está sujeta a aleatoriedad):


# >>> muestreo_hmm(ej1_hmm,10)  
# [['c', 'c', 'f', 'f', 'c', 'c', 'c', 'c', 'c', 'c'],
#  [2, 1, 1, 1, 3, 2, 1, 2, 3, 1]]

# >>> muestreo_hmm(ej1_hmm,10) 
# [['c', 'c', 'f', 'f', 'f', 'f', 'c', 'c', 'c', 'c'],
#  [1, 1, 1, 1, 3, 1, 2, 3, 2, 3]]


# >>> muestreo_hmm(ej2_hmm,7)
# [['l', 'l', 'l', 'l', 'no l', 'l', 'l'],
#  ['u', 'u', 'u', 'u', 'no u', 'u', 'u']]

# >>> muestreo_hmm(ej2_hmm,7) 
# [['no l', 'no l', 'no l', 'no l', 'no l', 'no l', 'l'],
#  ['no u', 'no u', 'u', 'no u', 'no u', 'no u', 'u']]



def  muestreo_hmm(modelo, n): 
    r = random.random()
    ac = 0
    for e in modelo.estados:
        ac = ac + modelo.pi[e]
        if r < ac:
            sol = [[e], []]
            break 
    r = random.random()
    ac = 0
    for o in modelo.observables:
        ac = ac + modelo.b[sol[0][0], o]
        if r < ac:
            sol[1].append(o)
            break

    for i in range(1,n):
        r = random.random()
        ac = 0
        for e in modelo.estados:
            ac = ac + modelo.a[sol[0][i - 1],e]
            if r < ac:
                sol[0].append(e)
                estado = e
                break 
        r = random.random()
        ac = 0
        for o in modelo.observables:
            ac = ac + modelo.b[estado, o]
            if r < ac:
                sol[1].append(o)
                break
    return sol


#print(muestreo_hmm(ej1_hmm,10))





# ========================================================
# Ejercicio 3
# ========================================================

# Vamos ahora a aplicar las dos funciones anteriores para experimentar sobre
# un problema simple de localización de robots que se mueve en una cuadrícula.
# Esta aplicación está descrita en la sección 15.3.2 del libro "Artificial
# Intelligence: A Modern Approach (3rd edition)" de S. Russell y P. Norvig.

# Supongamos que tenemos la siguiente lista de strings, que representa una
# cuadrícula bidimensional, sobre la que se desplaza un robot:

#     ["ooooxoooooxoooxo",
#      "xxooxoxxoxoxoxxx",
#      "xoooxoxxoooooxxo",
#      "ooxoooxooooxoooo"]

# Aquí la "x" representa una casilla bloquedada, y la "o" representa una
# casilla libre en la que puede estar el robot. 

#   El robot puede iniciar su movimiento en cualquiera de las casillas libres,
# con igual probabilidad. En cada instante, el robot se mueve de la casilla en
# la que está a una contigua: al norte, al sur, al este o al oeste, siempre que
# dicha casilla no esté bloqueda. El movimiento del robot está sujeto a
# incertidumbre, pero sabemos que se puede mover con igual probabilidad a cada
# casilla vecina no bloquedada.

#   Desgraciadamente, el robot no nos comunica en qué casilla se encuentra en
# cada instante de tiempo, ni nosotros podemos observarlo. Lo único que el
# robot puede observar en cada casilla son las direcciones hacia las que
# existen obstáculos (es decir, casillas bloqueadas o paredes). Por ejemplo, una
# observación "NS" representa que el robot ha detectado que desde la casilla
# en la que está, al norte y al sur no pueda transitar, pero que sí puede
# hacerlo a las casillas que están al este y al oeste.

#   Para acabar de complicar la cosa, los sensores de obstáculos que tiene el
# robot no son perfectos, y están sujetos a una probabilidad de error.
# Supondremos que hay una probabilidad epsilon de que la detección de
# obstáculo en una dirección sea errónea (y por tanto, hay una probabilidad
# 1-epsilon de que sea correcta). Supondremos también que los errores en
# cada una de las cuatro direcciones son independientes entre sí. Esto nos
# permite calcular la probabilidad de las observaciones dados los estados, como
# ilustramos a continuación.

#   Por ejemplo, supongamos que X y E son, respectivamente, las variables
# aleatorias que indican la casilla en la que está el robot y la observación
# que realiza el robot. Supongamos también que c es una casilla que hacia el
# norte y el este tiene obstáculos, y que tiene casillas transitables al sur y
# al oeste. Si por ejemplo el robot informara que existen obstáculos al sur y
# al este, la probabilidad de esto sería 

#     P(E=SE|X=c) = (epsilon)^2 * (1-epsilon)^2 

# (ya que habría errado en dos direcciones, norte y sur, y acertado en otras
# dos, este y oeste). 

# Por el contrario, la probabilidad de que en ese mismo estado el robot
# informara de obstáculos al norte, sur y este, sería 

#     P(E=NSE|X=c) = epsilon * (1-epsilon)^3 

# (ya que habría errado en una dirección y acertado en tres).




# Se pide:

# Definir una clase Robot, subclase de HMM, cuyo constructor reciba una lista
# de strings del estilo de la del ejemplo anterior, y un error epsilon, generando a
# partir de la misma un objeto de la clase HMM. Importante: se pide hacerlo de 
# manera genérica, no solo para la cuadrícula del ejemplo. 

# Aplicar el algoritmo de Viterbi a varias secuencias de observaciones del robot,
# para estimar las correspondientes secuencias de casillas más probables por
# las que ha pasado el robot, en la cuadrícula del ejemplo.

# Hacer lo mismo para alguna otra cuadrícula, distinta de la del ejemplo. 

# NOTAS: 

# - Representar los estados por pares de coordenadas, en el que la (0,0) sería
#   la casilla de arriba a la izquierda. 
# - Las observaciones las representamos por una tupla (i1,i2,i3,i4), en el que 
#   sus elementos son 0 ó 1, donde 0 indica que no se ha detectado obstáculo, 
#   y 1, indica que sí, respectivamente en  el N,S, E y O (en ese orden). 
#   Por ejemplo (1,1,0,0) indica que se detecta obstáculo en el N y en el S.
#   y (0,0,1,0) indica que se detecta obstáculo solo en el E.    
# - Por simplificar, supondremos que no hay casillas aisladas. 


# Ejemplo de HMM generado para una cuadrícula básica:
    
cuadr0=["ooo",
        "oxo",
        "ooo"]

# >>> robot0=Robot(cuadr0,0.1)

# >>> robot0.estados
# [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]

# >>> robot0.observables

#[(0, 0, 0, 0),(0, 0, 0, 1),(0, 0, 1, 0),(0, 0, 1, 1),(0, 1, 0, 0),
# (0, 1, 0, 1),(0, 1, 1, 0),(0, 1, 1, 1),(1, 0, 0, 0),(1, 0, 0, 1),
# (1, 0, 1, 0),(1, 0, 1, 1),(1, 1, 0, 0),(1, 1, 0, 1),(1, 1, 1, 0),
# (1, 1, 1, 1)]

# >>> robot0.pi 
# {(0, 0): 0.125, (0, 1): 0.125, (0, 2): 0.125, (1, 0): 0.125,
#  (1, 2): 0.125, (2, 0): 0.125, (2, 1): 0.125, (2, 2): 0.125}

# >>> robot0.a
 
#{((0, 0), (0, 0)): 0, ((0, 0), (0, 1)): 0.5, ((0, 0), (0, 2)): 0,
# ((0, 0), (1, 0)): 0.5,((0, 0), (1, 2)): 0, ((0, 0), (2, 0)): 0,
# ((0, 0), (2, 1)): 0, ((0, 0), (2, 2)): 0,
# ((0, 1), (0, 0)): 0.5, ((0, 1), (0, 1)): 0, ((0, 1), (0, 2)): 0.5,
# ((0, 1), (1, 0)): 0, ((0, 1), (1, 2)): 0, ((0, 1), (2, 0)): 0,
# ((0, 1), (2, 1)): 0, ((0, 1), (2, 2)): 0,
# ((0, 2), (0, 0)): 0, ((0, 2), (0, 1)): 0.5,
# ... Continúa .....

# >>> robot0.b
#{((0, 0), (0, 0, 0, 0)): 0.008100000000000001,
# ((0, 0), (0, 0, 0, 1)): 0.07290000000000002,
# ((0, 0), (0, 0, 1, 0)): 0.0009000000000000002,
# ((0, 0), (0, 0, 1, 1)): 0.008100000000000001,
#  ... Continúa ....



# -----------

class Robot(HMM):
    def __init__(self, cuadrante, error):
        def observacion_real(estado):
            return [0 if (estado[0] + i, estado[1]) in validos(estado[0], estado[1]) else 1  
            for i in [1, -1]] + [0 if (estado[0], estado[1] + i) in validos(estado[0], estado[1]) else 1  
            for i in [1, -1]]


        def validos(x,y):
            s = [(x + j, y ) for j in [1, -1] 
                    if x + j >= 0 and  x + j< len(cuadrante) and cuadrante[x + j][y] != "x"] 
            s = s + [(x, y + i) for i in [1, -1]
                    if y + i < len(cuadrante[0]) and y + i >= 0 and cuadrante[x][y + i] != "x"] 
            if s == []:
                s = [(x,y)]
            return s

    
        estados = [(i,j) for i in range(len(cuadrante)) 
                for j in range(len(cuadrante[i])) if cuadrante[i][j] != "x"]
        observables = [(int(bin(i + 16)[3]), int(bin(i + 16)[4]),int(bin(i + 16)[5]),int(bin(i + 16)[6])) 
                for i in range(16)]
        mat_ini = [1/len(estados) for _ in range(len(estados))]
        mat_trans = [[1/len(validos(estadoi[0],estadoi[1]))
            if estadoj in validos(estadoi[0], estadoi[1]) else 0 
            for estadoj in estados]
            for estadoi in estados]
        mat_obs = [[error ** len([0 for (i,j) in zip(observacion_real(estado),observable) if i != j]) 
            * (1-error) ** len([0 for (i,j) in zip(observacion_real(estado),observable) if i == j]) 
            for observable in observables] 
            for estado in estados]
        HMM.__init__(self, estados, mat_ini, mat_trans, observables, mat_obs)
        
# Ejemplo de uso de Viterbi en la cuadrícula del ejemplo

cuadr_rn=     ["ooooxoooooxoooxo",
               "xxooxoxxoxoxoxxx",
               "xoooxoxxoooooxxo",
               "ooxoooxooooxoooo"]

robot_rn=Robot(cuadr_rn,0.15)

# Secuencia de 7 observaciones:
seq_rn1=[(1, 1, 0, 0), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1),
         (1, 1, 0, 0),(0, 1, 1, 0),(1, 1, 0, 0)]

# Usando Viterbi, estimamos las casillas por las que ha pasado:

# >>> viterbi(robot_rn,seq_rn1)
# [(3, 14), (3, 13), (3, 12), (3, 13), (3, 14), (3, 15), (3, 14)]

print(viterbi(robot_rn, seq_rn1))   




# ========================================================
# Ejercicio 4
# ========================================================

# Realizar experimentos para ver cómo de buenas son las secuencias que se
# obtienen con el algoritmo de Viterbi que se ha implementado. Para ello, una
# manera podría ser la siguiente: generar una secuencia de estados y la
# correspondiente secuencia de observaciones usando el algoritmo de
# muestreo. La secuencia de observaciones obtenida se puede usar como entrada
# al algoritmo de Viterbi y comparar la secuencia obtenida con la secuencia de
# estados real que ha generado las observaciones. Se pide ejecutar con varios
# ejemplos y comprobar cómo de ajustados son los resultados obtenidos. Para
# medir el grado de coincidencia entre las dos secuencias de estados, calcular
# la proporción de estados coincidentes, respecto del total de estados de la
# secuencia.


# Por ejemplo:

# Función que calcula el porcentaje de coincidencias:
def compara_secuencias(seq1,seq2):
    return sum(x==y for x,y in zip(seq1,seq2))/len(seq1)


# Generamos una secuencia de 20 estados y observaciones
# >>> seq_e,seq_o=muestreo_hmm(rn_hmm,20)

# >>> seq_o 
# [(0, 0, 1, 1), (0, 1, 1, 0), (1, 1, 0, 0),....]

# >>> seq_e
# [(2, 5),(3, 5), (3, 4), (3, 3), (3, 4), ....]
 
# >>> seq_estimada=viterbi(rn_hmm,seq_o)

# >>> seq_estimada
# [(2, 5),(3, 5),(3, 4),(3, 3),(3, 4),(3, 5),...]
 
# Vemos, cuántas coincidencias hay, proporcinalmente al total de estados de la 
# secuencia:
    
# >>> compara_secuencias(seq_e,seq_estimada)
# 0.95

# -----------------------------------

# Para mecanizar esta experimentación, definir una función

#     experimento_hmm_robot(cuadrícula,epsilon,n,m) 

# que genera el HMM correspondiente a la cuadrícula y al epsilon, y realiza 
# m experimentos, como se ha descrito:
    
# - generar en cada uno de ellos una secuencia de n observaciones y estados 
#  (con muestreo_hmm)
# - con la secuencia de observaciones, llamar a viterbi para estimar la 
#   secuencia de estados más probable
# - calcular qué proporción de coincidencias hay entre la secuencia de estados real 
#   y la que ha estimado viterbi 
# Y devuelvela media de los m experimentos. 

# Experimentar al menos con la cuadrícula del ejemplo y con varios valores de
# n, con varios valores de epsilon y con un m suficientemente grande para que 
# la media devuelta sea significativa del rendimiento del algoritmo. 

def experimento_hmm_robot(cuadricula,epsilon,n,m):
    robot = Robot(cuadricula, epsilon)
    puntuaciones = []
    for i in range(m):
        muestra = muestreo_hmm(robot, 4)
        estimacion = viterbi(robot, muestra[1])
        comparacion = compara_secuencias(muestra[0], estimacion)
        puntuaciones.append(comparacion)
    return sum(puntuaciones)/m

print(experimento_hmm_robot(cuadr_rn, 0.15, 5, 100))
print(experimento_hmm_robot(cuadr_rn, 0.1, 10, 100))
print(experimento_hmm_robot(cuadr_rn, 0.15, 15, 100))
print(experimento_hmm_robot(cuadr_rn, 0.1, 20, 100))
